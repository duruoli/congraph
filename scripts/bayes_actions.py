#!/usr/bin/env python3
"""Bayes env v0 actions a and their observation channels J(a) (see BAYES_ENV.md section 4.1, anchor 1.5).

An action = one standard orderable unit (CPT code), defined a priori in scripts/bayes_actions.csv, NOT clustered
from what doctors order together (that co-ordering is the policy under study). Two standard units become one action
only when this lab reports them in the same tube >= 95% of the time both ways; add-ons the doctor orders (differential, urine
microscopy) are their own actions with `requires` = the base they ride on; a reflex test the lab adds on its own
(manual smear review) is no action: its results ride on the base's event (J_reflex, reflex_rate). Every lab itemid belongs to exactly one
action, so a tube (specimen_id) splits into its actions without ambiguity; micro = one test per specimen, after the
method merges of scripts/bayes_items.csv (C. diff) and the bundle rows of bayes_actions.csv.

Outputs (restricted MIMIC-derived, runs/ only) in runs/bayes_env/v0/actions/:
  actions.csv     one row per action: definition, J(a) = the o items it can reveal, how often it is done
  events.parquet  one row per admission x specimen x action: t, day, items revealed (all / o only)
  summary.md      coverage + the checks below

  PYTHONPATH="$PWD" /opt/anaconda3/envs/congraph/bin/python scripts/bayes_actions.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw_data/ibd_mimiciv_3_1"
V0 = ROOT / "runs/bayes_env/v0"
OUT = V0 / "actions"
HCT = 51221
MIN_ADM = 30  # an action enters the candidate set when done in >= this many admissions
NOT_ACTION = {"meta", "dup"}


def cohort():
    c = pd.read_parquet(RAW / "cohort")
    return c[c.is_primary_ibd_diagnosis][["hadm_id", "admittime"]].drop_duplicates("hadm_id")


def definitions():
    d = pd.read_csv(ROOT / "scripts/bayes_actions.csv", dtype=str).fillna("")
    lab = d[d.kind.ne("micro")].assign(itemid=lambda x: x["items"].str.split()).explode("itemid")
    lab = lab.assign(itemid=lab.itemid.astype(int))[["itemid", "action"]]
    lab_all = lab
    lab = lab[~(lab.action.eq("hct"))]  # Hct alone is decided per tube below; the itemid itself belongs to cbc
    dup = lab.itemid[lab.itemid.duplicated()]
    assert dup.empty, f"itemid in two actions: {sorted(dup)}"
    micro = d[d.kind.eq("micro")].assign(unit=lambda x: x["items"].str.split(";")).explode("unit")
    micro = micro.assign(unit="micro:" + micro.unit.str.strip())[["unit", "action"]]
    return d, lab.set_index("itemid").action, lab_all, micro.set_index("unit").action


def fallbacks():
    """o-side rule shared with bayes_data.py: a fallback item stands in for `into` when the draw lacks it."""
    r = pd.read_csv(ROOT / "scripts/bayes_items.csv")
    return r[r.action.eq("fallback")].set_index("unit").into.to_dict()


# ---------------------------------------------------------------- events
def lab_events(coh, lab_map, reflex):
    l = pd.read_parquet(RAW / "labs", columns=["hadm_id", "specimen_id", "itemid", "label", "charttime"])
    l = l[l.hadm_id.isin(coh.hadm_id)]
    l = l.assign(action=l.itemid.map(lab_map).fillna("?unmapped"))
    # a tube whose only blood-count item is the hematocrit = the stand-alone H/H check, not a CBC
    cbc = l[l.action.eq("cbc")].groupby("specimen_id").itemid.agg(set)
    alone = cbc.index[cbc.map(lambda s: s == {HCT})]
    l.loc[l.specimen_id.isin(alone) & l.itemid.eq(HCT), "action"] = "hct"
    # a reflex test is the lab's follow-up, not an order: its results ride on the base action's event
    l["action"] = l.action.replace(reflex)
    unmapped = (l[l.action.eq("?unmapped")].groupby(["itemid", "label"]).hadm_id.nunique()
                .sort_values(ascending=False).rename("admissions").reset_index())
    rows = (len(l), int(l.action.eq("?unmapped").sum()))
    l = l[~l.action.isin(NOT_ACTION | {"?unmapped"}) & ~l.itemid.map(lab_map).isin(NOT_ACTION)]
    l = l.assign(item="lab:" + l.itemid.astype(str), specimen="lab:" + l.specimen_id.astype(str))
    return l, unmapped, rows


def micro_events(coh, micro_map):
    m = pd.read_parquet(RAW / "microbiology", columns=["hadm_id", "micro_specimen_id", "charttime", "chartdate",
                                                        "spec_type_desc", "test_name"])
    m = m[m.hadm_id.isin(coh.hadm_id)]
    rules = pd.read_csv(ROOT / "scripts/bayes_items.csv")
    merge = rules[rules.action.eq("merge")].set_index("unit").into
    unit = "micro:" + m.spec_type_desc.fillna("?") + "|" + m.test_name.fillna("?")
    item = unit.map(merge).fillna(unit)
    # a cancelled test is still an action taken; it just reveals nothing (terrain_units drops it from o)
    return m.assign(item=item, action=item.map(micro_map).fillna(item.str.removeprefix("micro:")),
                    charttime=m.charttime.fillna(m.chartdate),
                    specimen="micro:" + m.micro_specimen_id.astype(str))


def events(coh, l, m, kept, fb):
    both = pd.concat([l[["hadm_id", "specimen", "action", "charttime", "item"]],
                      m[["hadm_id", "specimen", "action", "charttime", "item"]]])
    ev = (both.groupby(["hadm_id", "specimen", "action"])
          .agg(t=("charttime", "min"), items=("item", lambda s: sorted(set(s)))).reset_index())
    # with its twin in the same event the fallback is a copy; alone it is that twin's observation -> a set does both
    ev["o_items"] = ev["items"].map(lambda s: sorted({fb.get(i, i) for i in s} & kept))
    ev = ev.merge(coh, on="hadm_id")
    ev["day"] = ((ev.t - ev.admittime).dt.total_seconds() // 86400).clip(lower=-1).astype("Int64")
    return ev.drop(columns="admittime").sort_values(["hadm_id", "t", "specimen", "action"]).reset_index(drop=True)


# ---------------------------------------------------------------- action table
def action_table(d, lab_all, reflex, m, ev, kept, fb, n):
    items = pd.Series(("lab:" + lab_all.itemid.astype(str)).map(lambda i: fb.get(i, i)).values,
                      index=lab_all.action.values).loc[lambda s: s.isin(kept)]
    is_rx = items.index.isin(reflex.index)
    lab_J = items[~is_rx].groupby(level=0).agg(lambda s: sorted(set(s)))
    # J_reflex = what the base action SOMETIMES also reveals, at the lab's discretion (reflex_rate of its events)
    J_rx = items[is_rx].rename(index=reflex.to_dict()).groupby(level=0).agg(lambda s: sorted(set(s)))
    micro_J = m[m.item.isin(kept)].groupby("action").item.agg(lambda s: sorted(set(s)))
    J = pd.concat([lab_J, micro_J]).groupby(level=0).agg(lambda x: sorted(set(sum(x, []))))
    stats = ev.groupby("action").agg(admissions=("hadm_id", "nunique"), events=("specimen", "size"))
    a = stats.join(d.set_index("action")[["zh", "kind", "requires", "standard", "note"]], how="left")
    a["kind"] = a.kind.fillna("micro")
    a["J"] = a.index.map(lambda x: J.get(x, []))
    a["n_J"] = a.J.map(len)
    a["J_reflex"] = a.index.map(lambda x: ";".join(J_rx.get(x, [])))
    rx_items = set(sum(J_rx.tolist(), []))
    rate = ev.assign(rx=ev.o_items.map(lambda s: bool(rx_items & set(s)))).groupby("action").rx.mean()
    a["reflex_rate"] = a.index.map(lambda x: round(rate[x], 3) if x in J_rx.index else np.nan)
    a["coverage"] = (a.admissions / n).round(3)
    a["in_space"] = a.n_J.gt(0) & a.admissions.ge(MIN_ADM)
    a["J"] = a.J.map(";".join)  # micro names contain spaces
    return (a.reset_index().rename(columns={"index": "action"})
            .sort_values(["in_space", "admissions"], ascending=False).reset_index(drop=True))


# ---------------------------------------------------------------- summary
def summary(a, ev, unmapped, rows, kept, n):
    sp = a[a.in_space]
    ev_sp = ev[ev.action.isin(sp.action)]
    orphan = sorted(set(kept) - set(";".join(list(sp.J) + list(sp.J_reflex)).split(";")))
    L = ["# Bayes env v0 actions", "",
         f"Cohort: primary-IBD admissions, n = {n:,}. Definitions: `scripts/bayes_actions.csv` "
         f"(standard CPT unit; merged only when >= 95% same tube both ways).", "",
         "## Coverage", "",
         f"- Lab rows mapped to an action or to meta: {1 - rows[1] / rows[0]:.2%} of {rows[0]:,}; "
         f"unmapped itemids: {len(unmapped)} "
         f"(top: {', '.join(unmapped.label.head(8))}).",
         f"- Actions seen: {len(a)}; candidate set (reveals >= 1 o item, done in >= {MIN_ADM} admissions): "
         f"**{len(sp)}** (lab {int((sp.kind != 'micro').sum())}, micro {int((sp.kind == 'micro').sum())}).",
         f"- Events: {len(ev):,}; in the candidate set {len(ev_sp):,} ({len(ev_sp) / len(ev):.1%}).",
         f"- o items not revealed by any candidate action: {orphan or 'none'}.", "",
         "## Candidate actions", "",
         "| action | | kind | standard | admissions | events / adm | J(a) size |", "|---|---|---|---|---:|---:|---:|"]
    L += [f"| `{r.action}` | {r.zh if isinstance(r.zh, str) else ''} | {r.kind} | "
          f"{r.standard if isinstance(r.standard, str) else ''} | {r.admissions:,} | "
          f"{r.events / r.admissions:.1f} | {r.n_J} |" for r in sp.itertuples()]

    # checks
    per_tube = ev[ev.specimen.str.startswith("lab:")].groupby("specimen").action.agg(set)
    L += ["", "## Checks", "", "**Add-ons without their base in the same tube** (should be ~0):", ""]
    for r in a[a.kind.eq("addon")].itertuples():
        has = per_tube[per_tube.map(lambda s: r.action in s)]
        L.append(f"- `{r.action}` needs `{r.requires}`: {int((~has.map(lambda s: r.requires in s)).sum())} "
                 f"of {len(has):,} tubes")
    for r in a[a.reflex_rate.notna()].itertuples():
        L.append(f"- reflex: `{r.action}` also reveals {len(r.J_reflex.split(';'))} smear-type items in "
                 f"{r.reflex_rate:.1%} of its events (not a candidate action of its own)")
    L += ["", "**Partial panels** (event revealed < 80% of the action's o items; a lab may skip a result):", ""]
    full = ev_sp.merge(sp[["action", "n_J"]], on="action")
    full = full[full.n_J.gt(1)].assign(frac=lambda x: x.o_items.map(len) / x.n_J)
    for act, g in full.groupby("action"):
        L.append(f"- `{act}`: {(g.frac < 0.8).mean():.1%} of {len(g):,}")
    k = per_tube.map(len).value_counts().sort_index()
    L += ["", "**Actions per lab tube** (one tube often carries several orders): "
          + ", ".join(f"{i}: {v:,}" for i, v in k.items()) + ".", ""]
    same_time = ev_sp.groupby(["hadm_id", "t"]).action.nunique().value_counts().sort_index()
    L += ["**Candidate actions at the same timestamp** (step 6 compares a choice against the menu; several at "
          "once = a set choice): " + ", ".join(f"{i}: {v:,}" for i, v in same_time.items()) + "."]
    return "\n".join(L) + "\n"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    coh = cohort()
    n = len(coh)
    kept = set(pd.read_csv(V0 / "items.csv").query("kept").item)
    d, lab_map, lab_all, micro_map = definitions()
    reflex = d[d.kind.eq("reflex")].set_index("action").requires
    fb = fallbacks()
    l, unmapped, rows = lab_events(coh, lab_map, reflex)
    m = micro_events(coh, micro_map)
    ev = events(coh, l, m, kept, fb)
    a = action_table(d, lab_all, reflex, m, ev, kept, fb, n)
    a.to_csv(OUT / "actions.csv", index=False)
    ev.to_parquet(OUT / "events.parquet", index=False)
    unmapped.to_csv(OUT / "unmapped_items.csv", index=False)
    (OUT / "summary.md").write_text(summary(a, ev, unmapped, rows, kept, n))
    print(f"{len(a)} actions ({int(a.in_space.sum())} candidates), {len(ev):,} events -> {OUT}")


if __name__ == "__main__":
    main()
