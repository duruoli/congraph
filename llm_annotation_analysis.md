# Deviate + Confirmed
## Insight 1: The diagnosis is already settled, so the question has moved on
Doctors route by whatever question is still open, and once “is it disease X?” is answered—by labs, by prior imaging, or by an overall clinical picture, they jump to the test that answers the next question vs. The rubric routes imaging by disease ("appendicitis → US first")
The single largest pattern (~55 of 119): 
pancreatitis is already locked by a lipase 10–80× normal, so US/MRCP is ordered not to confirm pancreatitis but to find the biliary cause (e.g. 26860125, 25995993)
appendicitis is clinically near-certain, so CT is ordered to stage perforation/abscess, which US can‘t do (20279299); 
cholecystitis of the wall is settled, so imaging turns to the bile duct when labs are cholestatic (24636219, 26586156).
Pearl: certainty about the diagnosis re-aiming imaging at severity → etiology→ complications, each of which may call for a different modality than rubrics. Route by current clinical question, not disease label.

## Insight 2: A non-diagnostic or low-quality report is not a negative; inadequacy licenses pushing past the rubric's "stop.”
The rubric treats a completed first-line test as terminal. Doctors read the adequacy of the study and refuse to let an inconclusive scan close the workup
~30+ cases the reasoning: “appendix not visualized,” “US technically limited,” “limited study,” “sludge vs. non-shadowing stones,” “outside CT very limited given body habitus.” Each time, the next imaging step is verified 
24478777, 23562407 appendix-not-seen→CT found it; 
29234985, 27163978 limited-US→repeat dedicated US)
Pearl: the rubric needs a study-adequacy gate — “negative” must be conditioned on “and the study could actually have seen it.” Absence of evidence from an inadequate test ≠ evidence of absence

## Insight 3: Discordance between two data streams is itself the alarm; conflicting signals drive the next (correct) deviation.
The doctor trusts the mismatch over the rubric's single-axis logic
Sepsisphysiology with a benign belly (WBC 30 + soft abdomen →26586156; perforated appendix but soft, non-tender → 20279299); 
high pretest probability with normal labs or a blunted immune response (normal WBC but classic signs, transplant/immunosuppressed → 24613821, 29487535); 
sky-high lipase with a normal-looking pancreas on CT (22470405); cholestatic labs with a non-dilated duct (27426621). 
I.e., When two streams disagree, the doctor orders tests to resolve the conflict
Pearl: clinical–test discordance should drop the certainty score and trigger further imaging, exactly where the rubric (reading one axis) would stop. This is the certainty-trigger's sharpest input.

### How they handle it: 
they note it in information_gap/reasoning (“clinical-radiologic discordance”、“dissociation”、“discordant finding”) and write the expected finding as a fork; they even pick the modality for its power to break the tie (US “because CT is insensitive to microlithiasis")
 Did belief change? 
A — diagnosis stays, question moves: when one stream is overwhelming (perforation on CT, lipase 80×), they keep the diagnosis (top_branch unchanged, often high) and re-attribute the mismatch to severity / etiology / a known test limitation
B — localization collapses, belief broadens: when both streams are ambiguous and conflict (sepsis + benign belly; high suspicion + blunted host), the differential goes wide (other 0.77; appendicitis kept at only 0.45). Here belief genuinely diffuses, i.e., more flat.

## Conclusion
All three are forms of "the rubric stopped reasoning too early" 

Rubric totally routes by disease (1)
Rubric trusts a test it never quality-checked (2)
Rubric reads one axis at a time (3)
The doctor‘s edge is keeping the workup open on the right question, the real report quality, and the conflict between signals

---

# Disconfirmed groups (the symmetric signal: when NOT to keep going, and when a negative is not a failure)

> **Updated 2026-06-30 to the clean pre-admission cut (real MIMIC charttime).** This supersedes the earlier 542-step / FD-55 / DD-43 taxonomy, which was contaminated by post-intervention monitoring scans (the old "Caveat 2" below — now resolved).
>
> **Two timing-filtered datasets now coexist** (both from `scripts/filter_deviation_by_timing.py`, real `charttime`/`admittime`):
> - **A. intervention-cut** → `belief_deviation_filtered.csv` (**502 kept**, drops only `post_intervention`; FD=52 / DD=36). Used for agent-training so legitimate severity-staging survives.
> - **B. pre-admission-cut** → `belief_deviation_preadmission.csv` (**285 kept**, `timing_role=='pre_admission'`, i.e. `charttime < admittime`; **FD=29 / DD=24**). The purest diagnostic-phase view — drops the admitted-but-pre-intervention grey zone (119), same-day (98), and post-intervention (40). This is the cut the disconfirm analysis below is drawn on.
>
> **Label caveat (unchanged, critical):** `follow` is *computed* (doctor action == rubric-traversal recommendation on the belief). The LLM annotation **never saw the rubric**, so `follow` = *convergence, not obedience* — the reconstructed reasoning is the doctor's own logic. ⇒ high FD does NOT mean "the rubric misled the doctor." certainty_update: ALL "down".
>
> **Disconfirm group (B-cut) = 53** (DD deviate+disconfirmed 24, FD follow+disconfirmed 29). **34/53 abd-US + 5 MRCP = 74% are biliary-tree interrogations**; pancreatitis 28, cholecystitis 17 dominate. The question for the agent: which disconfirms are **unavoidable** (test was right, a negative is the informative answer → teach calibration) vs **real holes** (should not have followed / deviated → teach the brake).

## The 3 disconfirm findings (symmetric to the deviate+confirmed alarm/loosening findings)

### Finding 1 (dominant, ~39–51/53): most "disconfirms" are RULE-OUT BY DESIGN — the negative is the product, not a failure.
Biochemistry (lipase 10–280× / cholestatic LFTs / hyperbilirubinemia) makes "gallstone / CBD obstruction" the leading etiology; the doctor orders US→MRCP to detect-or-rule-out choledocholithiasis *because it gates ERCP*. Pancreatitis 27/28 disconfirms are exactly this "image to find the gallstone/CBD-stone etiology → none found." Crucially the doctor is **NOT confirming a leading Dx** — they are **EXCLUDING an *actionable* alternative** (gallstone pancreatitis → would mandate ERCP/cholecystectomy). A clean negative changes management (no intervention) = the deliverable.
Pearl: separate **"test-to-CONFIRM-my-Dx"** from **"test-to-EXCLUDE-an-actionable-alternative."** For the latter: pre-register a high P(negative), read the negative as **SUCCESS**, and drop the **etiology-belief** — NOT decision-confidence. The rubric defect is **representational** (no "expected-negative / rule-out" node, no pretest layer), so a naïve agent misreads the negative as surprise/failure.

### Finding 2 (the "over-imaging brake" mostly EVAPORATES as a timing artifact): what survives is narrow.
The old DD-1 "defensive re-scan past a valid STOP" (10/43) shrank once real charttime removed monitoring scans — its cleanest instances (27993727 already-had-appendectomy, 28684468, 21849575) were **post_intervention** and are now gone; the remaining candidates (25414251 / 28122817 / 24440089) are legitimate `assess_severity` staging, NOT errors. The **real** residual brake = the **8 `appropriateness=partial` cases**: re-asking an *already-answered actionable question* with a **weaker modality**. Signature traps:
- **"CBD-dilated-but-no-stone"** — dilation ≠ stone (post-chole / sphincterotomy / passed-stone): 24922174, 24050288, 26756106.
- **"US-after-CT for cholecystitis"** — US systematically under-reads (Murphy absent, wall not thick after CT already characterized it): 27194914, 28306018, 22023307.
Pearl: once **one adequate study** answers the actionable question, **suppress the confirmatory second modality**. This is the whole brake — it is NOT "stop scanning" in general.

### Finding 3 (KEEP the deviation — the outlier-patch signal): disconfirm ≠ wrong test; the broad net that misses its stated target is exactly how the real outlier disease gets caught.
Local prediction failed yet the deviation was diagnostically superior. 4 clean cases: 25444703 (CT-for-pancreatitis-edema → 50 mm pancreatic-head **mass**), 28238173 (CT-for-appendicitis → true acute **cholecystitis**), 28672604 (US/CT-for-postop-complication → advanced **liver disease**), 20501678 (CT-for-appe/stone → **duodenal perforation**). Prediction disconfirmed, but the scan found the REAL unexpected pathology = exactly the outlier "patch" the project targets.
Pearl: the reward must separate **disconfirmed-PREDICTION** from **wrong-ACTION**; never penalize the net that overturns the working Dx. Symmetric to the confirm+deviate "loosening": there the deviation found MORE, here disconfirmation **REDIRECTED**.

## Synthesis — the reframed brake
- **Calibration / rule-out-by-design** (Finding 1): test correct, the negative IS the informative answer → teach realistic pretest + drop the *etiology* belief + STOP re-escalating. NOT a deviation trigger, NOT an error.
- **The real brake, narrowed** (Finding 2): NOT "stop scanning" — that lesson was largely a post-intervention monitoring artifact. The real brake is **"stop EXPECTING A POSITIVE / stop re-escalating once the actionable cause is excluded one adequate time."** Concretely: suppress the confirmatory *second, weaker* modality after one adequate study.
- **Anti-brake guardrail** (Finding 3): the broad net that catches the outlier must NOT be suppressed → separate disconfirmed-prediction from wrong-action.

**Pairing with the deviate+confirmed insights:** those said "the rubric stops reasoning too early" → the agent needs *loosening*. The disconfirm findings add the symmetric **brake**: "a negative is not a failure" (Finding 1) and "one adequate look is enough" (Finding 2), guarded by "don't suppress the net that overturns the Dx" (Finding 3). A good agent needs BOTH — the evidence-based loosening AND these brakes — otherwise loosening just becomes over-testing.

---

# Methodological note — the DD-3 / Finding-3 reward split (still stands)

> The earlier "Caveat 2" (post-intervention scans contaminating DD-1 / FD-2 / FD-3) is **RESOLVED**: real MIMIC `charttime`/`admittime` arrived 2026-06-28 and the pre-admission cut above is drawn on genuine timestamps, not text-hints. The one methodological correction that survives is on how Finding 3 (old DD-3) is rewarded.

**"Caught the truth" is the WRONG reward justification; split `aimed-at-truth` vs `incidental-catch`.** The Finding-3 cases are not homogeneous:
- **panc 25444703** (CT→50 mm pancreatic-head mass): CT imaged the *right organ* (lipase→pancreas) and found the real pathology there; disconfirm (mass≠edema) correctly drove CTA→EUS-FNA. **Systematic — keep.**
- **chole 28238173** (broad CT under failed localization→cholecystitis): localization was impossible (obese, unreliable exam) so a whole-abdomen CT was ordered by design; the net caught the real source. **Policy-validated — keep.**
- **appe 28672604** (CT chasing post-chole liver/biliary disease): the dominant finding was advanced liver disease; the *discharge-dx* appendicitis appeared only as a report footnote ("incidental thickened appendix"). **Genuinely incidental — the truth-catch was luck, not reasoning.**

⇒ Do **not** reward Finding-3 because it "caught the truth" (that is a global-outcome hit, which the project's local-validation rule forbids — correct outcomes can come from luck). Reward it iff the deviation was *locally appropriate* (addressed the stated gap) and **aimed at** the region/question where the truth was found. 28672604 should be judged on its actual aim (rule out post-op complication / characterize liver), not on the incidental appendix. **Action: sub-split `aimed-at-truth` vs `incidental-catch`; only the former is the outlier-patch signal.**

**Provenance / reproduce:** enrichment + dumps in scratchpad `groupB_disconfirm.py` / `groupB_enriched.csv` / `dump53.py`; counts verified from `results/annotation_experiment/full/belief_deviation_preadmission.csv` (285 rows, FD=29 / DD=24). Timing pipeline: `experiments/annotation/timing.py` + `scripts/build_timing_table.py --source-dir data/raw_data/mimic_source` → `timing_roles.csv` (pre_admission 285 / post_admission_diagnostic 119 / same_day 98 / post_intervention 40) → `scripts/filter_deviation_by_timing.py`.
