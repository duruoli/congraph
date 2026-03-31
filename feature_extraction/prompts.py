"""prompts.py — LLM prompt templates for all extraction steps.

Each template is a plain string; variables are filled with str.format().
All prompts instruct the model to return a JSON object only.
"""

# ---------------------------------------------------------------------------
# Step 0: HPI + Physical Examination
# ---------------------------------------------------------------------------

STEP0_SYSTEM = """\
You are a clinical NLP assistant that extracts structured features from \
patient admission notes.
Extract ONLY information that is EXPLICITLY stated or STRONGLY implied by the \
text. Do NOT infer, guess, or hallucinate.
Respond with a valid JSON object only — no markdown, no commentary.\
"""

STEP0_USER_TEMPLATE = """\
Extract clinical features from the patient notes below.

=== PATIENT HISTORY / HPI ===
{patient_history}

=== PHYSICAL EXAMINATION ===
{physical_exam}

Return a JSON object with EXACTLY the following keys.

PAIN CHARACTERISTICS (infer from HPI / history of present illness):
  "pain_location"             : string — primary pain location.
                                Must be one of:
                                "RLQ" | "RUQ" | "LLQ" | "LUQ" |
                                "Epigastric" | "Periumbilical" | "Pelvic" |
                                "General_Abdomen" | "Other"
  "pain_migration_to_RLQ"     : bool — pain that began elsewhere and then
                                migrated to the right lower quadrant.
  "epigastric_radiating_to_back" : bool — epigastric pain radiating to the back
                                (classic for pancreatitis).
  "bowel_habit_change"        : bool — change in bowel habits (diarrhea,
                                constipation, altered frequency).
  "symptom_duration_over_72h" : bool — symptoms present for more than 72 hours.

HPI SYMPTOM FLAGS (from history / chief complaint):
  "anorexia"                  : bool — loss of appetite / anorexia.
  "nausea_vomiting"           : bool — nausea and/or vomiting.
  "alcohol_history"           : bool — significant alcohol use history
                                (heavy use or history of alcohol abuse).
  "gallstone_history"         : bool — prior known gallstones or cholelithiasis.
  "prior_diverticular_disease": bool — prior history of diverticulosis or
                                diverticulitis.

PHYSICAL EXAMINATION FINDINGS (from exam section only, not HPI):
  "murphys_sign"              : bool — Murphy's sign positive on palpation.
  "RLQ_tenderness"            : bool — right lower quadrant tenderness on
                                palpation (including TTP RLQ, guarding RLQ,
                                Rovsing's sign, psoas/obturator signs).
  "rebound_tenderness"        : bool — rebound tenderness present
                                (Blumberg's sign, "+rebound").
  "RUQ_mass"                  : bool — palpable mass in right upper quadrant.
  "peritoneal_signs"          : bool — signs of peritoneal irritation
                                (rigidity, involuntary guarding, rebound).
  "impaired_mental_status"    : bool — altered consciousness or confusion
                                (somnolent, confused, GCS < 15).

DEMOGRAPHICS (from patient history):
  "age_gt_60"                 : bool — patient is older than 60 years.
                                If age is redacted or unknown, return false.
  "is_female_reproductive_age": bool — patient is female AND of reproductive
                                age (approximately 15–50 years old).
                                If sex or age is unknown, return false.

Return only the JSON object.\
"""


# ---------------------------------------------------------------------------
# Imaging prompts — shared system message
# ---------------------------------------------------------------------------

IMAGING_SYSTEM = """\
You are a clinical radiology NLP assistant that extracts structured findings \
from radiology reports.
Extract ONLY findings that are EXPLICITLY stated in the report text.
Do NOT infer, guess, or hallucinate.
Respond with a valid JSON object only — no markdown, no commentary.\
"""


# ---------------------------------------------------------------------------
# Ultrasound
# ---------------------------------------------------------------------------

US_USER_TEMPLATE = """\
Extract findings from the abdominal ultrasound report below.

=== ULTRASOUND REPORT ===
{report}

Return a JSON object with EXACTLY these keys (all boolean):
  "US_appendix_inflamed"      : appendix is visualized and shows signs of
                                inflammation (non-compressible, enlarged >6 mm,
                                hyperemic, with periappendiceal changes).
  "US_gallstones"             : gallstones / cholelithiasis present in the
                                gallbladder or biliary system.
  "US_GB_wall_thickening"     : gallbladder wall thickening > 4 mm.
  "US_pericholecystic_fluid"  : pericholecystic fluid (fluid around the
                                gallbladder) identified.
  "US_sonographic_murphys"    : sonographic Murphy's sign positive (maximal
                                tenderness directly over the visualized
                                gallbladder under the probe).
  "pleural_effusion_on_imaging": pleural effusion identified (if chest was
                                incidentally visualized).

Return only the JSON object.\
"""


# ---------------------------------------------------------------------------
# CT Abdomen/Pelvis
# ---------------------------------------------------------------------------

CT_USER_TEMPLATE = """\
Extract findings from the CT report below.

=== CT REPORT ===
{report}

Return a JSON object with EXACTLY these keys:

Boolean keys:
  "CT_appendicitis_positive"  : CT confirms appendicitis (inflamed / enlarged
                                appendix, periappendiceal fat stranding,
                                appendicolith).
  "CT_perforation_abscess"    : appendix-related perforation, abscess, or
                                phlegmon visible on CT.
  "CT_cholecystitis_positive" : CT confirms cholecystitis (gallbladder wall
                                thickening, pericholecystic fluid, hyperemia).
  "CT_GB_severe_findings"     : CT shows gangrenous or emphysematous
                                cholecystitis (gas in gallbladder wall/lumen,
                                intraluminal membranes, gallbladder perforation).
  "CT_diverticulitis_confirmed": CT confirms diverticulitis (colonic
                                diverticula + wall thickening + pericolonic
                                fat stranding), WITHOUT complications.
  "CT_diverticulitis_complicated": CT shows complications of diverticulitis
                                (abscess, perforation, fistula, obstruction).
  "CT_phlegmon"               : pericolic phlegmon / inflammatory mass without
                                discrete fluid collection (Hinchey Ia).
  "CT_abscess_lt_3cm"         : localized pericolonic abscess < 3 cm
                                (Hinchey Ib).
  "CT_abscess_ge_3cm"         : distant or pelvic abscess ≥ 3 cm
                                (Hinchey II).
  "CT_purulent_peritonitis"   : free peritoneal air / purulent peritonitis
                                without fecal contamination (Hinchey III).
  "CT_fecal_peritonitis"      : fecal peritonitis / free perforation with
                                fecal contamination (Hinchey IV).
  "CT_pancreatitis_positive"  : CT confirms pancreatitis (pancreatic and/or
                                peripancreatic inflammation, stranding, edema).
  "pleural_effusion_on_imaging": pleural effusion identified on CT.
  "cholecystitis_additional_imaging_positive": CT confirms cholecystitis
                                (gallbladder wall thickening, pericholecystic
                                fluid, hyperemia, or other cholecystitis
                                findings), whether used as primary or secondary
                                imaging modality.
  "has_organ_dysfunction"     : any organ dysfunction evident from CT or
                                clinical context described in the report
                                (e.g. renal failure, respiratory failure,
                                cardiovascular compromise, hepatic failure).
  "local_complications_pancreatitis": local complications of pancreatitis
                                visible on CT (peripancreatic fluid collections,
                                pancreatic necrosis, walled-off necrosis).
  "organ_failure_transient"   : report or clinical context mentions transient
                                organ failure that resolved within 48 h
                                (self-resolving; Moderately Severe AP criterion).
  "organ_failure_persistent"  : report or clinical context explicitly mentions
                                persistent organ failure lasting > 48 h
                                (Severe AP criterion).

Numeric key:
  "CTSI_score"                : float 0.0–10.0 — Modified CT Severity Index
                                score if explicitly stated in the report;
                                otherwise 0.0.

Return only the JSON object.\
"""


# ---------------------------------------------------------------------------
# HIDA / MRCP / MRI
# ---------------------------------------------------------------------------

HIDA_MRCP_MRI_USER_TEMPLATE = """\
Extract findings from the imaging report below.

Modality: {modality}

=== REPORT ===
{report}

Return a JSON object with EXACTLY these keys (all boolean):
  "cholecystitis_additional_imaging_positive": this imaging study (HIDA scan,
                                MRCP, or MRI) confirms cholecystitis or
                                biliary pathology consistent with cholecystitis
                                (e.g. non-visualised gallbladder on HIDA,
                                biliary obstruction on MRCP, GB inflammation
                                on MRI).
  "pleural_effusion_on_imaging": pleural effusion identified in this study.

Return only the JSON object.\
"""


# ---------------------------------------------------------------------------
# Chest Radiograph
# ---------------------------------------------------------------------------

CHEST_USER_TEMPLATE = """\
Extract findings from the chest imaging report below.

=== CHEST IMAGING REPORT ===
{report}

Return a JSON object with EXACTLY this key (boolean):
  "pleural_effusion_on_imaging": pleural effusion present on chest imaging
                                (unilateral or bilateral).

Return only the JSON object.\
"""
