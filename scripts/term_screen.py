"""Label-blind first-pass screen of frequent terms: keep / maybe / drop.

Input  runs/term_frequency/terms.csv (from term_frequency_overview.py).
Output runs/term_frequency/screen.csv (+ screen.md). The case/control columns are
deliberately NOT carried over: screening must not see the label, or the later
per-term AUC is inflated by selection.

Rule: report text (words / 2-word phrases) defaults to DROP unless listed below
(template, technique, generic words); structured items (lab/drug/order/rad/micro)
default to KEEP unless listed (duplicates, routine care, carriers).
Edit the `final` column in screen.csv by hand; blank = accept `auto`.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MIN_PATIENTS = 80  # ~10% of admissions

# ---- report text: findings / states / diseases -> keep
KEEP_TEXT = {
    # single words
    "effusion", "pneumothorax", "thickening", "obstruction", "lymphadenopathy", "dilatation", "edema",
    "atelectasis", "hydronephrosis", "enlarged", "consolidation", "crohn", "pneumonia", "stranding",
    "fracture", "mass", "colitis", "enhancement", "calcifications", "infection", "cyst", "abscess",
    "collection", "inflammatory", "aneurysm", "dilation", "inflammation", "opacity", "ascites", "opacities",
    "lytic", "ulcerative", "hernia", "fistula", "stones", "flare", "infectious", "fever", "congestion",
    "hypodensities", "malignancy", "thrombosis", "narrowing", "distention", "distended", "nodule",
    "hemorrhage", "cysts", "masses", "sclerotic", "pain", "atherosclerotic", "degenerative", "reactive",
    "opacification", "dilated", "effusions",
    # 2-word phrases
    "pleural effusion", "no free", "free fluid", "wall thickening", "free air", "pericardial effusion",
    "no pneumothorax", "abdominal pain", "crohn disease", "bowel obstruction", "ductal dilatation",
    "biliary dilatation", "fluid pelvis", "inguinal lymphadenopathy", "pulmonary edema", "fluid collection",
    "peripancreatic stranding", "focal consolidation", "atherosclerotic disease", "aortic aneurysm",
    "ulcerative colitis", "air fluid", "free intraperitoneal", "mesenteric lymphadenopathy", "dilated loops",
    "degenerative changes", "intraperitoneal air", "bibasilar atelectasis", "evidence obstruction",
    "vascular congestion", "bilateral pleural", "fluid levels", "history crohn", "no hydronephrosis",
    "no peripancreatic", "no pleural", "no pericardial", "acute fracture", "pleural effusions",
    "effusion pneumothorax", "biliary dilatation", "dilatation gallbladder", "abdominal aortic",
}
# ---- report text: anatomy alone / course & certainty modifiers / devices -> maybe
MAYBE_TEXT = {
    # anatomy without a state
    "bowel", "colon", "liver", "pancreas", "gallbladder", "kidneys", "bladder", "spleen", "stomach", "rectum",
    "ileum", "appendix", "sigmoid", "biliary", "duct", "bile", "mesenteric", "peripancreatic", "pancreatic",
    "intrahepatic", "extrahepatic", "portal", "colonic", "mucosal", "uterus", "prostate",
    "small bowel", "terminal ileum", "large bowel", "sigmoid colon", "bowel loops", "loops small", "bowel wall",
    "bowel gas", "gas pattern", "right lower", "lower quadrant", "air", "fluid", "gas", "free",
    # course / severity / certainty
    "mild", "moderate", "severe", "new", "unchanged", "stable", "worsening", "improved", "increased",
    "decreased", "persistent", "chronic", "acute", "recent", "minimal", "extensive", "diffuse", "active",
    "concerning", "suspicious", "worrisome", "likely", "possible", "concern", "nonspecific", "interval change",
    # history / devices
    "status post", "picc", "picc line", "catheter", "tube", "lungs clear",
}
# ---- structured items: drop (duplicates, routine care, IV carriers)
DROP_STRUCT = {
    "lab:hematocrit_low", "lab:red_blood_cells_low", "lab:rdw_sd_high", "lab:anion_gap_low",
    "lab:urea_nitrogen_low", "lab:urine_mucous_abn", "lab:hyaline_casts_high", "lab:absolute_basophil_count_low",
    "lab:eosinophils_low", "lab:absolute_eosinophil_count_low", "lab:absolute_neutrophil_count_high",
    "lab:absolute_lymphocyte_count_low", "lab:absolute_monocyte_count_high", "lab:rbc_high", "lab:pt_high",
    "lab:iron_binding_capacity_total_low", "lab:po2_high", "lab:monocytes_low",
    "drug:sodium_chloride_0_9_flush", "drug:0_9_sodium_chloride", "drug:bag", "drug:5_dextrose", "drug:vial",
    "drug:ns", "drug:iso_osmotic_dextrose", "drug:heparin_flush_10_units_ml", "drug:sodium_chloride_0_9",
    "drug:influenza_vaccine_quadrivalent", "drug:d5_1_2ns", "drug:0_9_sodium_chloride_mini_bag_plus",
    "drug:sw", "drug:sterile_water", "drug:soln", "drug:lidocaine_1_for_picc_midline_insertions",
    "drug:multivitamins", "drug:multivitamins_w_minerals", "drug:vitamin_d", "drug:glucose_gel",
    "order:general_care/vitals_monitoring", "order:nutrition/diet_order", "order:general_care/activity",
    "order:general_care/other", "order:general_care/code_status", "order:general_care/precautions",
    "order:consults/social_work", "order:consults/physical_therapy", "order:consults/nursing",
    "order:consults/discharge_followup_appointment", "order:general_care/therapeutic_devices_dvt_prophylaxis",
    "order:iv_therapy/iv_access", "order:iv_therapy/iv_access_request",
    "micro:mrsa_screen",
}
# ---- structured items: maybe (supportive / symptomatic, not disease-directed)
MAYBE_STRUCT = {
    "drug:acetaminophen", "drug:heparin", "drug:ondansetron", "drug:potassium_chloride", "drug:magnesium_sulfate",
    "drug:lactated_ringers", "drug:lorazepam", "drug:pantoprazole", "drug:senna", "drug:omeprazole",
    "drug:docusate_sodium", "drug:trazodone", "drug:neutra_phos", "drug:polyethylene_glycol",
    "drug:diphenhydramine", "drug:folic_acid", "drug:ramelteon", "drug:calcium_gluconate", "drug:bisacodyl",
    "drug:levothyroxine_sodium", "drug:thiamine", "drug:simethicone", "drug:lidocaine_5_patch",
    "drug:heparin_sodium", "drug:famotidine", "drug:atorvastatin", "drug:gabapentin", "drug:potassium_phosphate",
    "drug:calcium_carbonate", "drug:dextrose_50", "drug:glucagon", "drug:aspirin", "drug:metoprolol_tartrate",
    "drug:albuterol_0_083_neb_soln", "drug:albuterol_inhaler", "drug:insulin",
    "lab:mchc_low", "lab:mch_low", "lab:mch_high", "lab:mcv_high", "lab:chloride_high", "lab:chloride_low",
    "lab:ph_low", "lab:pco2_high", "lab:anion_gap_high", "lab:urea_nitrogen_high", "lab:protein_high",
    "lab:bacteria_abn", "lab:poikilocytosis_abn", "lab:wbc_high",
    "rad:chest_portable_ap", "rad:chest_pa_lat", "rad:chest_port_line_placement",
    "order:iv_therapy/iv_fluids", "order:radiology/general_xray", "order:cardiology/ecg", "order:general_care/telemetry",
}


def auto_label(term, category):
    if category.startswith("report"):
        return "keep" if term in KEEP_TEXT else "maybe" if term in MAYBE_TEXT else "drop"
    return "drop" if term in DROP_STRUCT else "maybe" if term in MAYBE_STRUCT else "keep"


def main():
    out = ROOT / "runs/term_frequency"
    df = pd.read_csv(out / "terms.csv")
    df = df[df.patients_mentioning >= MIN_PATIENTS]
    df = df[["term", "category", "patients_mentioning", "days_per_mentioning_patient"]].copy()
    df["auto"] = [auto_label(t, c) for t, c in zip(df.term, df.category)]
    df["final"] = ""
    df = df.sort_values(["auto", "category", "patients_mentioning"], ascending=[True, True, False])
    df.to_csv(out / "screen.csv", index=False)

    counts = df.groupby(["category", "auto"]).size().unstack(fill_value=0)
    lines = [f"# 高频词初筛（≥{MIN_PATIENTS} 次住院提到；不看标签）", "",
             "| 类别 | " + " | ".join(counts.columns) + " |", "|---|" + "---:|" * len(counts.columns)]
    lines += [f"| {c} | " + " | ".join(str(v) for v in row) + " |" for c, row in counts.iterrows()]
    lines.append("")
    for lab in ("keep", "maybe"):
        lines += [f"## {lab}（{(df.auto == lab).sum()}）", ""]
        for cat, g in df[df.auto == lab].groupby("category"):
            lines += [f"**{cat}** " + " · ".join(f"{t}({n})" for t, n in zip(g.term, g.patients_mentioning)), ""]
    (out / "screen.md").write_text("\n".join(lines))
    print(df.auto.value_counts().to_string())


if __name__ == "__main__":
    main()
