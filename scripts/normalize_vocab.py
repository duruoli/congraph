"""Direction-2 step 1 (API-free): converge the NARRATIVE 名词库 (anatomy) and
状态库 (state) into a canonical vocabulary + raw->canonical map, using
hand-authored medical rules instead of an LLM/API.

Scope decisions (rubric_update.md §0/§6c + session):
  * narrative pieces only (source_test NOT in {laboratory, microbiology}).
  * NUMERIC states are noise for the 状态库 exactly like labs -- and vitals folded
    into the physical-exam pass ARE numeric (temp 97.8, RR 18, "100% on room air").
    So anatomy==vital_sign pieces and any purely-numeric state route to __NUMERIC__
    (their clinical content lives in finding_status + value_unit, not the vocab).
  * POLARITY is preserved for states (never merge X with not-X / normal).

Method = ordered keyword/regex rules (anatomy is antonym-free -> safe to route by
keyword; state uses adverb-strip + negation detection + base-descriptor rules).
First matching rule wins. A fallback keeps the (normalized) term as its own
canonical so nothing is silently dropped; those are reported as `residual` for
iterative rule-tightening.

Output: results/vocab/{anatomy,state}_vocab.json, {anatomy,state}_map.json,
        vocab_summary.md
"""
from __future__ import annotations

import glob
import json
import os
import re
from collections import Counter, defaultdict

EV_DIR = "results/evidence_pieces"
OUT_DIR = "results/vocab"
LAB_SOURCES = {"laboratory", "microbiology"}

# ===========================================================================
# collect narrative terms  (also track which anatomy each state co-occurs with,
# so we can drop vital_sign states)
# ===========================================================================

def collect():
    anat = Counter()
    state = Counter()
    attr = Counter()               # attribute axis (narrative only) -> the required(S) dims
    state_under_vital = Counter()  # states that appear on a vital_sign anatomy
    for f in glob.glob(os.path.join(EV_DIR, "*.jsonl")):
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            for p in r.get("pieces", []):
                if p.get("source_test") in LAB_SOURCES:
                    continue
                a = str(p.get("anatomy", "")).strip().lower()
                s = str(p.get("state", "")).strip().lower()
                at = str(p.get("attribute", "")).strip().lower()
                if a:
                    anat[a] += 1
                if s:
                    state[s] += 1
                    if a == "vital_sign":
                        state_under_vital[s] += 1
                if at:
                    attr[at] += 1
    return anat, state, attr, state_under_vital


# ===========================================================================
# ANATOMY rules   (regex on raw term) -> (canonical, system, core)
# ordered: specific families first; first match wins.
# core = in the 4-disease diagnostic field of view.
# ===========================================================================

A = "anatomy"
# system groups
GI, HB, PANC, URO, GYN, MGU, VASC, PERI, LN, SPL, ADR = (
    "gastrointestinal", "hepatobiliary", "pancreas", "urinary", "gynecologic",
    "male_gu", "vascular", "peritoneum_mesentery", "lymphatic", "spleen", "adrenal")
CARD, PULM, MSK, NEURO, PSYCH, HEENT, SKIN, DEV, SYS, BREAST, ENDO = (
    "cardiac", "pulmonary", "musculoskeletal", "neurologic", "psychiatric",
    "heent", "integument", "device_line", "systemic", "breast", "endocrine")

CORE_SYS = {GI, HB, PANC, URO, GYN, MGU, VASC, PERI, LN, SPL, ADR}

# (pattern, canonical, system).  core = system in CORE_SYS.
ANATOMY_RULES: list[tuple[str, str, str]] = [
    # --- devices / lines / tubes / drains (big non-core tail) ---
    (r"picc|central (venous|line|catheter)|\bij (catheter|line|central)|dobbhoff|dobhoff|"
     r"\bng ?tube|\bngt\b|nasogastric|orogastric|\bet tube\b|endotracheal|endogastric|"
     r"enteric tube|feeding tube|dialysis catheter|port-?a-?cath|pacemaker|pacer|pacing|"
     r"\baicd\b|\bpicc\b|\bport\b|drain|catheter|\bline\b|\bleads?\b|\bstent|nephrostomy|"
     r"cholecystostomy|tracheostomy|\bpeg tube|colostomy|\bostomy|sternotomy|sternal wire|"
     r"support (tubes|lines)|indwelling|monitoring and support|lines and tubes|vascular access|"
     r"\btube\b|\bsheath\b|\bwires?\b|\bdevices?\b", "device/line", DEV),

    # --- pancreas ---
    (r"pancreatic duct|pancreatic dorsal duct|pancreatic ventral duct|main pancreatic|duct of|major papilla|ampulla|sphincter of oddi", "pancreatic duct", PANC),
    (r"peripancreat|periduoden|peripancreatic", "peripancreatic region", PANC),
    (r"pancrea", "pancreas", PANC),

    # --- gallbladder & biliary ---
    (r"cystic duct", "cystic duct", HB),
    (r"common bile duct|\bcbd\b|common (hepatic )?duct|common duct|\bchd\b", "common bile duct", HB),
    (r"hepatic duct", "hepatic duct", HB),
    (r"intrahepatic (bile |biliary )?duct|intrahepatic ducts|central (biliary|hepatic|intrahepatic) duct|peripheral intrahepatic", "intrahepatic bile ducts", HB),
    (r"biliary|bile duct|biliary tree|biliary system|biliary tract|biliary ductal|central biliary", "biliary tree", HB),
    (r"gallbladder|gall bladder|\bgb\b|cholecyst|pericholecystic|perigallbladder", "gallbladder", HB),

    # --- liver ---
    (r"\bliver|hepatic (parenchyma|lobe|segment|artery|arteries|vein|vasculature|flexure|portal)|hepatic (right|left)|hepato|portal (triad|tract)|perihepatic|gerota", "liver", HB),

    # --- appendix / cecum / terminal ileum / RLQ ---
    (r"appendix|appendice|periappendice", "appendix", GI),
    (r"cecum|cecal|ileocec", "cecum", GI),
    (r"terminal ileum", "terminal ileum", GI),

    # --- colon / diverticula / rectum ---
    (r"diverticul", "diverticulum", GI),
    (r"sigmoid|descending colon|ascending colon|transverse colon|hepatic flexure|colon|colonic|rectosigmoid|large bowel|large intestine|paracolic|pericolic|right colon|left colon", "colon", GI),
    (r"rectum|rectal|anorect|perirectal|rectal (stump|vault)|external hemorrhoid|hemorrhoid|anus|anal", "rectum", GI),

    # --- small bowel / stomach / duodenum / esophagus / general bowel ---
    (r"duoden", "duodenum", GI),
    (r"small bowel|small intestine|jejun|ileum|distal small bowel|proximal small bowel", "small bowel", GI),
    (r"stomach|gastric|gastr(o|ic)|antrum|pylor|gastroduoden|gastroesophageal|hiatal hernia|hiatus", "stomach", GI),
    (r"esophag", "esophagus", GI),
    (r"bowel|loops of bowel|bowel loops|enteric|gastrointestinal|\bgi tract|aerodigestive|intestin", "bowel", GI),

    # --- spleen ---
    (r"splee?n|splenic|perisplenic|splenule", "spleen", SPL),

    # --- adrenal ---
    (r"adrenal", "adrenal gland", ADR),

    # --- urinary ---
    (r"kidney|renal|nephro|perinephric|perirenal|pararenal|collecting system|calyx|calyces|renal pelvis|renal (cortex|cortices|parenchyma|transplant|artery|arteries|vein)", "kidney", URO),
    (r"ureter", "ureter", URO),
    (r"bladder|urinary|urethra|urovesical|ureterovesical|colovesic", "bladder", URO),

    # --- gynecologic ---
    (r"uter|endometri|myometri|cervix|cervical os|vagina|fallopian|adnexa|adnexal|ovar|cul-de-sac|retrouterine|placenta|amniotic|iud|intrauterine|paravesical|pelvic sidewall", "female_pelvis", GYN),

    # --- male GU ---
    (r"test[ie]s|testi|testicle|scrotum|scrotal|epididym|prostate|seminal|penis|varicocele", "male_pelvis", MGU),

    # --- vascular: arterial ---
    (r"aorta|aortic (arch|knob|annulus|valve)|aortoiliac|infrarenal|celiac|superior mesenteric artery|\bsma\b|iliac arter|renal arter|hepatic arter|splenic artery|gastroduodenal artery|mesenteric (artery|arteries|vessel|vasculature)|carotid|coronary|pulmonary arter|femoral arter|brachial artery|radial artery|great vessels|major.*(vessel|arter)|abdominal (aorta|arter)", "arterial", VASC),
    # --- vascular: venous ---
    (r"portal vein|main portal vein|\bivc\b|inferior vena cava|superior vena cava|splenic vein|mesenteric vein|superior mesenteric vein|\bsmv\b|renal vein|hepatic vein|portal (vein|confluence)|jugular vein|subclavian vein|axillary vein|brachial vein|basilic vein|cephalic vein|iliac vein|popliteal vein|venous|\bveins?\b|varices", "venous", VASC),
    (r"vascula|vessels?|angio", "vasculature", VASC),

    # --- peritoneum / mesentery / retroperitoneum / abdominal wall ---
    (r"peritone|peritoneal cavity|free fluid|ascit", "peritoneum", PERI),
    (r"mesenter|mesentery|omentum|omental", "mesentery", PERI),
    (r"retroperitoneum|retroperitoneal", "retroperitoneum", PERI),
    (r"abdominal wall|ventral (hernia|abdominal)|inguinal (hernia|region)|umbilical hernia|hernia|incision|port site|laparoscop|surgical (scar|incision|site)|\bwound\b|drain site", "abdominal wall", PERI),

    # --- lymph nodes ---
    (r"lymph node|lymphadenopath|\blad\b|adenopathy|nodal", "lymph nodes", LN),

    # --- generic abdomen / pelvis regions (place AFTER organ-specific) ---
    (r"epigast|periumbilic|\bruq\b|\bllq\b|\brlq\b|\bluq\b|right (lower|upper) quadrant|left (lower|upper) quadrant|right upper quadrant|hypogastr|suprapubic|flank|subcostal|subhepatic|subphrenic|subdiaphragmatic|perihilar|abdominal (region|cavity|skin)|intra-?abdominal|abdomen|abdominopelvic|paracolic gutter|gutter", "abdomen/pelvis region", GI),
    (r"pelvi|deep pelvis|pelvic (cavity|structures|organs|wall|loops)", "pelvis", GYN),

    # --- cardiac ---
    (r"heart|cardiac|cardiomediastin|cardiopulmonary|myocard|pericard|mitral|aortic valve|ventricle|atrium|atrial|precordium|point of maximal|jugular venous pressure|cardiac silhouette", "heart", CARD),

    # --- pulmonary / pleura / mediastinum / chest ---
    (r"\blung|pulmonary|pleura|pleural|mediastin|\bhila\b|hilum|hilar|airway|airspace|"
     r"bronch|trachea|interstiti|atelectasis|lingula|apices|apex|costophrenic|"
     r"hemidiaphragm|diaphragm|chest wall|hemithorax|thorax|thoracic (structure|cage)|"
     r"retrocardiac|\blobes?\b|(lower|upper|middle|mid) lobe|lung (base|field|parenchyma|apex|volume)|"
     r"\bchest\b|chest cage|\bzone\b|lung zone|\bbase\b|bibasilar|biapical|subpleural", "lungs", PULM),

    # --- musculoskeletal ---
    (r"spine|vertebr|\bl\d|\bt\d|\bc\d|disc\b|spinal|lumbar|thoracic spine|cervical spine|sacro|ligamentum|paraspinal|\brib|sternum|clavicle|scapula|humerus|humeral|femur|femoral (region|head)|patella|knee|hip|ankle|foot|feet|toe|hand|wrist|elbow|shoulder|acromio|glenohumeral|iliac bone|ilium|pelvic bone|bony|bone|osseous|skeletal|skeleton|joint|musculoskeletal|muscle|muscul|iliopsoas|tendon|rotator cuff|marrow|calvari|skull|extremit|\bleg|\barm|thigh|calf|calves|buttock|metatars|\bmtp\b|plantar|acetabul", "musculoskeletal", MSK),

    # --- neuro ---
    (r"brain|cerebr|cerebell|intracranial|cranial nerve|\bcn \b|spinal cord|frontal lobe|parietal|ventricles?|white matter|meninges|sella|cistern|sulci|gyri|nervous system|neuro|neurolog|motor system|sensory|peripheral nervous|cognit|gait|mental status", "nervous system", NEURO),

    # --- psych ---
    (r"psychiatr|mood|affect|neuropsych", "psychiatric", PSYCH),

    # --- HEENT / neck / thyroid / sinuses / eyes ---
    (r"thyroid", "thyroid", ENDO),
    (r"scler|conjunctiv|pupil|\beyes?\b|extraocular|\beomi|orbit|globe|retina|lens|fundi|ocular|visual|tympanic|external auditory|\bears?\b|\bnose|nares|nasal|nasopharyn|oropharyn|oral (cavity|mucosa|intake)|\bmouth|lips|palate|dentition|throat|pharyn|larynx|vocal|salivary|parotid|sinus|mastoid|ethmoid|sphenoid|maxillary|paranasal|\bneck|\bhead\b|heent|normocephalic|forehead|face|facial|cochlea|meatus|perioral", "heent", HEENT),

    # --- integument ---
    (r"\bskin|integument|mucous membrane|mucus membrane|mucosa\b|subcutaneous|hair|nails|ecchymos|rash|nail", "integument", SKIN),

    # --- breast ---
    (r"breast", "breast", BREAST),

    # --- small stragglers (place before systemic catch-all) ---
    (r"periportal|porta hepatis|portal (region|triad)", "liver", HB),
    (r"dorsalis pedis|radial pulse|peripheral pulse|femoral pulse|posterior tibial|pulses?\b", "arterial", VASC),
    (r"fetus|fetal|gestational|placenta", "female_pelvis", GYN),
    (r"\bback\b|lower back|low back|mid back|upper back|midback|paraspinal|flank", "musculoskeletal", MSK),
    (r"soft tissue|soft tissues", "integument", SKIN),
    (r"umbilicus|groin|inguinal", "abdominal wall", PERI),
    (r"speech|vestibular", "nervous system", NEURO),
    (r"stool|feces|feculent|vomitus|urine|bloodstream|\bblood\b", "systemic/constitutional", SYS),
    (r"midline structures?|\bmidline\b|left side|right side|midline", "systemic/constitutional", SYS),

    # --- systemic / exam domains / constitutional ---
    (r"vital_?sign|vital signs?|body habitus|whole body|\bbody\b|constitutional|general appearance|general\b|systemic|respiratory\b|respiratory (tract|system)|cardiovascular\b|genitourinary\b|gastrointestinal\b|endocrine|hematolog|immune|lymphatic system|reproductive|integumentary|exposure history|family( history)?|surgical history|patient\b|study\b|image quality|study quality", "systemic/constitutional", SYS),
]

ANATOMY_RE = [(re.compile(p), c, s) for p, c, s in ANATOMY_RULES]


def norm_anatomy(term: str):
    for rx, canon, sysg in ANATOMY_RE:
        if rx.search(term):
            return canon, sysg, (sysg in CORE_SYS)
    return None  # -> fallback residual


# ===========================================================================
# STATE normalization: numeric/vitals drop, adverb strip, negation, base rules
# ===========================================================================

# purely numeric / vital-sign VALUES (temp, RR, HR, SpO2, BP) -> not a descriptor
_NUMERIC = re.compile(
    r"^\s*[<>]?\s*\d+(\.\d+)?\s*(%|°?f|bpm|breaths?/min|mmhg|/min|#/hpf|k/ul|"
    r"mg/dl|on room air|beats?/min)?\s*[+-]?\s*$")  # trailing +/- = reflex/pulse grade
_VITAL_PHRASE = re.compile(
    r"\bon room air\b|\bbreaths?/min\b|\bbpm\b|°f|\bmmhg\b|% ?on ?ra\b")

# non-informative process words -> drop from vocabulary
_DROP = re.compile(
    r"^(measured|seen|noted|identified|documented|recorded|obtained|demonstrated|"
    r"evaluated|assessed|imaged|visualized as such|in place|removed|remain in place|"
    r"remains? in place|status post .*|s/p .*|interval removal|not (documented|recorded|"
    r"obtained|assessed|evaluated|reported)|conventional|nonspecific|non-specific|"
    r"otherwise .*|per (report|history)|history of|present \(history of\))$")

_SEVERITY = re.compile(
    r"\b(mild(ly)?|moderate(ly)?|marked(ly)?|slight(ly)?|minimal(ly)?|mildy|severe(ly)?|"
    r"trace|small( amount)?( of)?|large|borderline|very|grossly|diffusely|focally|"
    r"minimally|top[- ]?normal|top normal|probable|probably|possible|possibly|likely|"
    r"faint|subtle|significantly|appreciable|overtly?|frankly?|extensive|multiple|"
    r"numerous|scattered|patchy)\b")

_NEG = re.compile(r"^(no|non-?|not|without|free of|absence of|negative for)\b|-free$")

# base descriptor rules on the ADVERB-STRIPPED term. (pattern, base_canonical).
# polarity is added as a prefix by the caller: negated -> "no <base>".
STATE_RULES: list[tuple[str, str]] = [
    # normal family: genuine normals + exam boilerplate + GENERIC negatives only.
    # Specific-finding negations ("no wall thickening","no mass","no free fluid") are
    # deliberately NOT here -- they flow to their polarity canonical ("no thickened",
    # "no mass/lesion", "no fluid/collection") via the base rules + _NEG.
    (r"^(normal.*|unremarkable.*|within normal.*|grossly (normal|unremarkable).*|wnl|"
     r"regular( rate)?( and rhythm)?|regular rhythm|clear.*|no acute distress|no distress|"
     r"well[- ]?(perfused|expanded|aerated|appearing).*|warm.*|supple|moist|dry|anicteric|"
     r"no scleral icterus|perrl.?a?|eomi|normocephalic.*|atraumatic|alert\b.*|"
     r"fluent|intact.*|grossly intact|nonfocal|non-?focal|comfortable|afebrile|vss|"
     r"vital signs stable|stable.*|nondistended|non-?distended|soft.*|smooth|sharp|"
     r"anteverted|palpable|appropriate.*|top[- ]?normal|upper limits of normal|"
     r"oriented (x ?\d|to |person|place|time).*|a&?ox ?\d|aaox ?\d|"
     r"no clubbing.*|equal.*reactive.*|s1.*s2.*|.*heart sounds?|preserved|patent.*|"
     r"normoactive|active\b|moving all extremities|cn ii.*intact|clean.*intact|"
     r"symmetric.*|homogeneous|homogenous|no acute.*|no abnormal.*|no textural.*|"
     r"no evidence of abnormal.*|no organomegaly|no lymphadenopath.*|no adenopathy|no lad\b|"
     r"midline|at midline|no change|unchanged.*|no new.*|resolved|improved|interval-improved|"
     r"otherwise (normal|clear|unremarkable).*)$", "normal"),
    # visualization / adequacy (carries inadequacy signal)
    (r"not (well |clearly |fully |definitively )?(visualized|seen|evaluated|assessed|"
     r"documented|imaged)|not visualized|obscured.*|partially (obscured|imaged|collapsed)|"
     r"limited\b|difficult to assess|assessment limited|poorly (seen|visualized)|"
     r"suboptimal|not well seen|largely obscured.*", "non-visualized"),
    # size / caliber
    (r"dilat", "dilated"),
    (r"distend", "distended"),
    (r"enlarg", "enlarged"),
    (r"prominent|engorg|ectatic|tortuous|accentuated|crowded|prominence", "prominent"),
    (r"contracted|decompress|collapsed|atrophic|atrophy|small\b|shrunken|nonobstruct", "contracted/small"),
    (r"thicken|thickened|wall edema|edematous wall", "thickened"),
    # wall / texture / density
    (r"echogenic|hyperechoic|steatos|fatty", "echogenic"),
    (r"hypoattenuat|hypodens|hypoechoic|low attenuation", "hypoattenuating"),
    (r"heterogen", "heterogeneous"),
    (r"calcif|calcium", "calcified"),
    (r"stranding|inflammatory (change|fat)|fat stranding|hyperem|inflam", "inflammatory stranding"),
    (r"edema|edematous|congested|congestion", "edema"),
    (r"necros|necrotic|devitaliz|nonenhanc|non-enhanc|hypoperfus", "necrosis/nonenhancement"),
    (r"enhanc", "enhancing"),
    # fluid / collection / air
    (r"free fluid|fluid[- ]filled|fluid collection|ascit|effusion|pericholecystic fluid|"
     r"free air|pneumoperitoneum|pneumobilia|air-?fluid|abscess|phlegmon|fluid\b", "fluid/collection"),
    # stones / sludge
    (r"stone|cholelith|choledocholith|calcul|sludge|shadowing|gallstone", "stones/sludge"),
    # patency / flow
    (r"hepatopetal|hepatofugal|antegrade|patent|flow (demonstrated|present)|appropriate.*flow|"
     r"widely patent|flow\b", "patent/flow"),
    (r"non-?compressible|noncompressible", "non-compressible"),
    (r"compressible", "compressible"),
    # tenderness / exam (PE)
    (r"tender|tenderness", "tender"),
    (r"guard", "guarding"),
    (r"rebound", "rebound"),
    (r"rigid|peritone|involuntary", "peritoneal signs"),
    # misc morphology
    (r"irregular|nodular|lobulated|masslike|mass\b|lesion|nodule|cyst", "mass/lesion"),
    (r"opacit|opacif|consolidat|infiltrate", "opacity/consolidation"),
    (r"obstruct|obstruction", "obstruction"),
    (r"perforat|rupture|dehisc", "perforation"),
    (r"icter|jaundice", "jaundice"),
    (r"equivocal|indeterminate|cannot (be )?exclud|possibl|questionable|borderline", "equivocal"),
    (r"cardiomegaly", "cardiomegaly"),
    (r"atelectas", "atelectasis"),
    (r"tachycard|hypertens|hypotens|bradycard|febrile|tachypn", "vital-abnormal"),
    (r"obese|obesity|body habitus|large body habitus|overweight", "obese/habitus"),
    # directional magnitude (non-polar; keep as-is)
    (r"increas|elevat|interval-increas|worsen|progress|hyperdynamic", "increased"),
    (r"decreas|diminish|reduc|interval-decreas|hypoactive|paucity|\blow\b|low-normal", "decreased"),
    # plainly present / adequately seen
    (r"^visualized$|adequately (seen|visualized|imaged)|well (seen|visualized|demonstrated)|"
     r"flow demonstrated", "present"),
    # generic presence
    (r"present|positive|likely present|demonstrated|noted", "present"),
    (r"absent|surgically absent|none|no |negative|not identified|none identified|"
     r"not present|resected|appendectomy|hysterectomy|cholecystectomy", "absent"),
]
STATE_RE = [(re.compile(p), c) for p, c in STATE_RULES]


def norm_state(term: str):
    t = term.strip().lower()
    if not t:
        return None
    if _NUMERIC.match(t) or _VITAL_PHRASE.search(t):
        return "__NUMERIC__", "numeric_vital", False
    if _DROP.match(t):
        return "__DROP__", "process_word", False
    negated = bool(_NEG.search(t))
    core = _SEVERITY.sub(" ", t).strip()  # strip severity adverbs
    core = re.sub(r"\s+", " ", core).strip(" ,.")
    if not core:  # term was ONLY degree words ("mild","severe","trace") -> no finding
        return "__DROP__", "degree_only", False
    probe = core
    for rx, base in STATE_RE:
        if rx.search(probe):
            # the 'normal' bucket already encodes polarity; don't double-negate it
            if base in ("normal", "absent", "present", "non-visualized", "non-compressible"):
                return base, "state", True
            canon = f"no {base}" if negated else base
            return canon, "state", True
    return None  # residual


# ===========================================================================
# ATTRIBUTE normalization  (raw property -> canonical dim, tagged by AXIS)
# ---------------------------------------------------------------------------
# `attribute` is the CORE axis of the adequacy gate: the atomic covered/required
# unit is the (anatomy, attribute) PAIR, and `required(S)` / `sought_dimensions`
# ARE attribute dims (GB x {wall/stones/pericholecystic-fluid/sonographic-Murphy}).
# So this converges the demand+supply property vocabulary. Axes:
#   morphology  = imaging/exam CHARACTERIZATION dims  (CORE = what a study covers;
#                 the required(S)/covered(S) dims the adequacy gate reads)
#   symptom     = patient-reported symptoms          (demand/trigger side)
#   sign        = elicited physical-exam signs        (demand/trigger side)
#   vital       = vital-sign dimension names          (value in finding_status+unit)
#   constitutional = general appearance / mental / hydration / habitus
#   history     = comorbidity / PMH / surgical / social / family history (path C)
#   device      = lines / tubes / hardware position & presence (non-core)
# core = axis in {morphology}. First matching rule wins; misses -> residual.
# ===========================================================================

MORPH, SYMP, SIGN, VITAL, CONST, HIST, DEV = (
    "morphology", "symptom", "sign", "vital", "constitutional", "history", "device")
CORE_ATTR_AXES = {MORPH}

# (pattern, canonical, axis).  ordered: specific/device/history first, then vitals,
# morphology, signs, symptoms, constitutional; broad fallbacks last.
ATTRIBUTE_RULES: list[tuple[str, str, str]] = [
    # --- devices / lines / tubes / hardware / foreign bodies (position & presence) ---
    (r"foreign bod|foreign object|radiopaque foreign", "foreign-body", DEV),
    (r"\btip\b|tip position|distal tip|side port|distance (above|from) carina|"
     r"course and (caliber|position)|course/position|position$|\bplacement\b|repositioning|"
     r"advancement|retraction recommend|recommended (adjustment|repositioning)", "device-position", DEV),
    (r"picc|central (venous |line |catheter)|central line|dobb?hoff|nasogastric|\bng\b tube|"
     r"\bng\b (tube|catheter|position)|feeding tube|gastrostomy|endotracheal|\bet tube\b|"
     r"enteric tube|tracheostomy|\bcatheter\b|\bdrain\b|drainage catheter|percutaneous (tube|drain)|"
     r"\bstent\b|stent graft|\bgraft\b|\bimplant|prosthes|pacemaker|pacer|\baicd\b|\bicd\b|"
     r"hardware|surgical (clip|staple|scar|instrument|hardware)|\bclips?\b|\bstaples?\b|"
     r"\bwires?\b|sternotomy wire|\bfilter\b|\bport\b|portacath|coils?|\bcuff\b|\bsuture|"
     r"\btubing\b|monitoring and support|external device|device (presence|type)|\bdevice\b|"
     r"line (placement|presence|position|sepsis)|\bline\b|\btube\b|\bdrains?\b|dressing|drain sponge", "device/hardware", DEV),

    # --- history / comorbidity / surgical / social / family (path C) -----------
    (r"\bhistory\b|\bpmh\b|\bpsh\b|\bh/o\b|\bprior\b|status post|\bs/p\b|"
     r"family history|\(pmh\)|episodes?( of)?|recent (illness|infection|travel|gi)|\brecurrent\b|"
     r"\bpast\b|previous", "comorbidity/pmh", HIST),
    (r"tobacco|smoking|\betoh\b|alcohol|illicit drug|recreational drug|\biv drug\b|drug (use|abuse)|"
     r"substance", "social-history", HIST),
    (r"surger|surgical|cholecystectomy|appendectomy|hysterectomy|nephrectomy|colectomy|lobectomy|"
     r"mastectomy|tonsillectomy|splenectomy|\bresection\b|bypass|laparoscop|\btah\b|\bbso\b|"
     r"salpingo|amputation|laminectomy|arthroscopy|sphincterotomy|\bercp\b|colonoscopy|biopsy|"
     r"transplant|hernia repair|gastric (band|bypass)|lap band|cesarean|tubal ligation", "surgical-history", HIST),
    (r"hypertens|hyperlipid|hypercholesterol|dyslipid|diabet|\bcopd\b|asthma|coronary artery|\bcad\b|"
     r"\bchf\b|congestive heart|heart failure|atrial fibrillation|\bafib\b|cardiomyopathy|"
     r"valv|\bmi\b|myocardial infarct|arrhythmia|chronic kidney|\bckd\b|renal (insufficiency|failure)|"
     r"end.stage renal|cirrhosis|\bgerd\b|reflux (disease|esophagitis)|hypothyroid|hyperthyroid|"
     r"\bcancer\b|malignan|carcinoma|lymphoma|melanoma|adenoma|neoplasm|sarcoma|"
     r"depress|anxiety|bipolar|schizo|\bptsd\b|psychotic|panic (disorder|attack)|"
     r"obstructive sleep|\bosa\b|osteoarthritis|osteoporosis|osteopenia|\barthritis\b|arthropath|"
     r"dementia|alzheimer|parkinson|seizure|epilep|\bhiv\b|\baids\b|hepatitis|peptic ulcer|"
     r"pancreatitis|diverticulosis|diverticular disease|nephrolithiasis|cholelithiasis|gallstone|"
     r"\bfibroids?\b|endometriosis|\bibd\b|crohn|ulcerative colitis|irritable bowel|\bibs\b|"
     r"\bpvd\b|peripheral vascular|\bdvt\b|deep vein|pulmonary (embol|hypertension)|stroke|\bcva\b|"
     r"\btia\b|anemia|thalassemia|sickle cell|coagulopath|\bobesity\b|\bobese\b|\bgout\b|psoriasis|"
     r"fibromyalgia|hypothyroidism|cholecystitis|cholangitis|choledocholithiasis|nephropathy|"
     r"disease|disorder|syndrome|infection|\bulcer\b|colitis|gastritis|esophagitis|cystitis|"
     r"pneumonia|tuberculos|dermatitis|\bpolyps? history|\bstd|sexually transmitted|allerg|"
     r"cholecystectomy history", "comorbidity/pmh", HIST),

    # --- vitals (dimension names; value lives in finding_status+unit) ----------
    (r"^temperature|tactile fever(?!s)|febrile", "temperature", VITAL),  # 'fever(symptom)' handled below
    (r"heart[_ ]?rate|\bpulse rate\b|\brate_?rhythm\b|^rate$|tachycard|bradycard", "heart-rate", VITAL),
    (r"blood[_ ]?pressure|^bp$|systolic (blood pressure|bp)|hypotension|orthostatic", "blood-pressure", VITAL),
    (r"\bspo2\b|oxygen (saturation|requirement)|\bo2 sat|supplemental oxygen|oxygenation", "oxygen-saturation", VITAL),
    (r"respiratory[_ ]?rate|\brespirations?\b|tachypn|respiratory (rate|status)", "respiratory-rate", VITAL),
    (r"^rhythm$|rate and rhythm|cardiac rhythm|\brhythm\b|conduction|ectopy", "cardiac-rhythm", VITAL),

    # --- MORPHOLOGY (imaging/exam characterization dims = CORE required(S)) -----
    # size / caliber / distension / dilatation (one caliber dim)
    (r"^size|caliber|diameter|\blength\b|\bwidth\b|\bvolume|dimension|distension|distention|"
     r"dilat|enlarg|engorg|expansion|\bbulk\b|contracted|atrophy|\bswelling\b|"
     r"transverse diameter|anteroposterior diameter|tapering|distal taper|\bcurvature\b|\bfullness\b|"
     r"size and (contour|configuration|caliber)|size/|configuration and size|contour/size", "size/caliber", MORPH),
    # wall / thickness
    (r"\bwall\b|thickness|thicken|mural (edema|thicken)|wall (edema|echogenicity|integrity|tone|vascularity|hyperemia)|"
     r"cortical thickness|stripe thickness|fascial thickening|soft tissue thickening|pelvi-infundibular", "wall/thickness", MORPH),
    # stones / calculi / calcification
    (r"stones?|calcul|calcif|\bcalcium\b|appendicolith|gallstone|cholelith|choledocholith|"
     r"nephrolith|phlebolith|milk of calcium|echogenic (focus|shadowing structure)|"
     r"comet tail|wall.echo.shadow|ring.down|shadowing|luminal (stone|sludge)|sludge", "stones/calculi", MORPH),
    # fluid / collection / ascites / effusion / hematoma
    (r"pericholecystic (fluid|free fluid|fluid collection)|perigallbladder fluid|free fluid|"
     r"\bfluid\b|fluid[- ]?(collection|filled|filling|level|density|component|overload|wave)|"
     r"free (pelvic |intraperitoneal )?fluid|\bascit|effusion|\bcollection|abscess|phlegmon|"
     r"perihepatic fluid|peripancreatic fluid|periportal (edema|fluid)|adjacent fluid|"
     r"drainable|surrounding free fluid|intra-?abdominal fluid|septation|cystic (collection|structure|area)|"
     r"seroma|leak|drainage character|purulent", "fluid/collection", MORPH),
    # hemorrhage / hematoma
    (r"hemorrhage|hematoma|sentinel clot|\bclot\b|intraluminal thrombus|thromb|\bembol|"
     r"hemorrhagic (material|cyst)|bleed(?!ing (disorder|ulcer))", "hemorrhage/thrombus", MORPH),
    # fat-stranding / inflammation
    (r"fat.?strand|\bstrand|inflammatory (change|extension)|inflammation|hyperem|phlegmonous|"
     r"periappendiceal (fat|inflammation)|pericolonic fat|peripancreatic (fat|inflammatory|strand)|"
     r"post.?inflammatory|surrounding inflammatory", "fat-stranding/inflammation", MORPH),
    # echogenicity / echotexture / density / signal / attenuation
    (r"echogenic|echotextur|\bechot|hypoechoic|hyperechoic|attenuat|hypodens|hyperdens|"
     r"\bdensity\b|signal( intensity)?|\bsignal|t1 signal|t2 |precontrast signal|diffusion restrict|"
     r"steatos|fat (content|fraction|infiltrat|sparing|plane)|\btexture\b|density/texture|"
     r"heterogen|homogen|grayscale|corticomedullary|parenchym|\bhaziness\b|lucency|opacif", "echogenicity/density", MORPH),
    # enhancement / perfusion / vascularity / contrast excretion
    (r"enhanc|\bperfusion|vascular(ity|ization| flow| structure| congestion| marking)|"
     r"contrast (excretion|opacification|reflux|extension|administration|extravasation)|excretion|"
     r"color flow|nephrogram|hyperenhance|nonenhancing|internal vascularity|warmth", "enhancement/vascularity", MORPH),
    # patency / flow / waveforms / doppler
    (r"patenc|\bflow\b|flow direction|waveform|resistive index|resistive indi|doppler|augmentation|"
     r"upstrokes|spectral doppler|venous (flow|drainage)|arterial (flow|waveform)|blood flow|"
     r"vascular (waveform|coiling)|jets?|\bstenosis\b|occlusion|dissection|aneurysm|pseudoaneurysm|"
     r"origin at|amplitude|pulse (quality|amplitude)|tortuosit", "patency/flow", MORPH),
    # mass / lesion / nodule / cyst / polyp / filling defect
    (r"\bmass|\blesions?\b|nodul|\bcysts?\b|\bcystic\b|\bpolyp|filling defect|focal (mass|lesion|nodule)|"
     r"space.occupying|tumor|fibroid|granuloma|\bfnh\b|adenom|\bipmn\b|pseudocyst|solid (mass|lesion|renal)|"
     r"concerning (lesion|finding)|suspicious (lesion|nodule)|worrisome|destructive lesion|"
     r"lytic|sclerotic|blastic|osteolytic|osteoblastic|\bfoci\b|hyperdense (focus|structure)|"
     r"hypodense (focus|material)|hypodensit|echogenic (material|structure|areas)|internal (contents|pigtail)|"
     r"renal (mass|lesion)|hepatic (cysts|lesion)|intrahepatic lesion|new.*(lesion|nodule)", "mass/lesion", MORPH),
    # air / gas / pneumoperitoneum / pneumatosis
    (r"free (air|gas|intra)|pneumoperitoneum|pneumatos|pneumobil|extraluminal (air|gas)|extra-?luminal air|"
     r"subcutaneous gas|\bgas\b|gas pattern|gas pocket|air.?fluid level|air.?bubble|\bair\b|"
     r"air bronchogram|subdiaphragmatic air|intramural (air|gas)|portal (venous )?gas|pneumaturia", "air/gas", MORPH),
    # compressibility
    (r"compressib|non-?compressib|\bcompression\b(?! fracture)", "compressibility", MORPH),
    # obstruction / stricture / transition point / ileus
    (r"obstruct|\bstricture|\bileus\b|transition point|closed.loop|bowel obstruction|fecal (load|matter|loading)|"
     r"obstructing (mass|stone|lesion)|impacted stone", "obstruction", MORPH),
    # perforation / rupture
    (r"perforat|rupture|dehisc|\bfistula\b|extraluminal (contrast|oral)|contrast leak|"
     r"free (intra-?abdominal|intraperitoneal) (air|gas)", "perforation", MORPH),
    # hernia
    (r"\bhernia|incarcerat|reducib|hiatal", "hernia", MORPH),
    # lymphadenopathy
    (r"lymphadenopath|adenopathy|lymph node|\bnodal\b|\blad\b", "lymphadenopathy", MORPH),
    # hydronephrosis / collecting system
    (r"hydronephro|hydroureter|pelvicali|pelvi-?caly|renal pelvis (fullness|dilat)|caliectasis|"
     r"collecting system|calyceal", "hydronephrosis", MORPH),
    # pulmonary: opacity/consolidation, atelectasis, pleural effusion, pneumothorax, edema, nodule
    (r"pneumothorax", "pneumothorax", MORPH),
    (r"atelectas", "atelectasis", MORPH),
    (r"pleural (effusion|thickening)|\beffusions?\b", "pleural-effusion", MORPH),
    (r"pulmonary (edema|vascular|congestion|opacif|nodule)|vascular congestion|vascular redistribution|"
     r"interstitial (edema|markings|lines|opacit|pattern|abnormal)|\bb lines\b|kerley|"
     r"cephalization|upper zone redistribution|volume overload|anasarca|peripheral edema|"
     r"congest", "pulmonary-edema/congestion", MORPH),
    (r"pulmonary nodule|lung nodule|\bnodules?\b", "mass/lesion", MORPH),  # generic nodule -> mass/lesion
    (r"opacit|opacif|consolidat|infiltrat|airspace|ground.?glass|air bronchogram|aeration|"
     r"\bhaz|bronchovascular|peribronchial|mucous plug|mucus plug|bronchieffect", "opacity/consolidation", MORPH),
    # generic edema (wall/soft-tissue/peripheral) after pulmonary-specific
    (r"\bedema\b|edematous|periportal edema|mural edema|surrounding edema", "edema", MORPH),
    # necrosis / ischemia / infarct
    (r"necros|devitaliz|ischemi|infarct|nonenhanc|hypoperfus|non-?viab", "necrosis/ischemia", MORPH),
    # organomegaly
    (r"organomegaly|hepatomegaly|splenomegaly|cardiomegaly|megaly", "organomegaly", MORPH),
    # generic imaging appearance / contour / morphology / architecture / structure
    (r"appearance|\bcontour|\bmorpholog|\bshape\b|configuration|architecture|\borientation\b|"
     r"\bstructure|gross (morpholog|abnormal)|overall|sonographic (appearance|sign|finding|abnormal|penetration|sign)|"
     r"visuali[sz]|visibility|evaluab|evaluation (quality|adequacy)|image quality|motion artifact|"
     r"\bpresence\b|presence and position|identification|\bposition\b|position and orientation|"
     r"corticomedullary|\bparenchyma\b|\bcortex\b|cortical (appearance|margin|echo)|\bmargins?\b|"
     r"border|definition|\bsurface\b|\bcontents?\b|luminal content|intraluminal (content|material|structure)|"
     r"\bcourse\b|\bcaliber\b|\bsymmetry\b|\balignment\b|\bposture\b|\bnumber\b|\bcount\b|integrity|"
     r"\bfindings?\b|\babnormalit|acute (abnormalit|finding|process|disease)|focal (abnormalit|finding|deficit)|"
     r"prominence|\bblunting\b|\bsilhouette\b|cardiomediastinal|mediastinal (contour|silhouette)|"
     r"degenerative (change|disc|disease|joint)|spondyl|osteophyt|endplate|disc (disease|herniation|bulge)|"
     r"\bfracture|\bdeformit|\bmineralization|bone (density|structure)|marrow|\bfusion\b|"
     r"soft tissue|granulation|scarring|induration|adhes|inflammatory changes|inflammatory bowel(?! disease)|"
     r"acute cholecystitis signs|signs of (cholecystitis|infection|obstruction)|special sign|"
     r"pericholecystic (edema|stranding|enhancement)|mucosa|submucosa|\bfollicl|"
     r"peristal|\bmotility\b|contraction|involvement|extension|encasement|encroach|"
     r"drainage$|drainage (procedure|tubes)|\btorsion\b", "appearance/contour", MORPH),

    # --- SIGNS (elicited physical exam) ----------------------------------------
    (r"rebound tenderness|\brebound\b|rebound[/ ]guard", "rebound", SIGN),
    (r"tender|tenderness|shake tenderness|percussion tenderness|\bcva\b tenderness|"
     r"costovertebral angle tenderness|jar tenderness|cervical motion tenderness", "tenderness", SIGN),
    (r"guard|rigidit|involuntary guard|voluntary guard", "guarding/rigidity", SIGN),
    (r"peritoneal sign|peritonit|peritoneal", "peritoneal-signs", SIGN),
    (r"murphy|rovsing|psoas sign|obturator sign|special sign|ultrasonographic (murphy )?sign|"
     r"sonographic sign|courvoisier", "special-sign", SIGN),
    (r"bowel sounds|bowel movements?(?! change| frequency)|peristalsis", "bowel-sounds", SIGN),
    (r"auscultation|breath sounds|\brales\b|\brhonchi\b|ronchi|\bwheez|\bcrackle|crepitation|"
     r"adventitious sounds|air (entry|movement)|air (?:entry|movement)|extra sounds|"
     r"work of breathing|respiratory (effort|distress|pattern)|inspiratory (effort|volume)|"
     r"breathing (effort|pattern|difficulty)|degree of inspiration|inspiration (level|effort)|"
     r"excursion|\bfremitus\b", "lung-auscultation", SIGN),
    (r"heart sounds?|\bmurmur|\bgallop|\brubs?\b|\bs1\b|\bs2\b|\bclick\b|\bpmi\b|\bjvp\b|"
     r"jugular venous|point of maximal", "heart-auscultation", SIGN),
    (r"\bpulses?\b|distal pulses|perfusion/warmth|perfusion and warmth|dorsalis pedis", "pulses", SIGN),
    (r"icter|jaundice|\bsclera\b|scleral", "jaundice/icterus", SIGN),
    (r"\brash|erythema|ecchymos|petechia|purpura|\bskin\b|skin (color|change|condition|lesion)|"
     r"pigment|striae|telangiect|pallor|\bcyanos|clubbing|\bwound\b|wound (appearance|status)|"
     r"\bincision|\bscars?\b|\bulcers?\b|\bulceration|blister|bruis|turgor|open wound", "skin/wound", SIGN),
    (r"organomegaly|palpable (mass|masses)|palpab", "organomegaly", MORPH),

    # --- SYMPTOMS (patient-reported) -------------------------------------------
    (r"\bpain\b|pain (radiation|character|location|quality|severity|onset|duration|trigger|"
     r"migration|localization|relief|recurrence|response|history|episode|association|exacerbation|"
     r"with|relation|positional|inhibiting|trend)|chest pain|abdominal pain|back pain|epigastric|"
     r"colic|discomfort|\bcramp|dysmenorrh|referred pain|angina|dermatomal|burning(?! sensation)|"
     r"\bache|pressure(?!$)|tenesmus|dyspareunia", "pain", SYMP),
    (r"nausea|vomit|\bemesis\b|dry heav|retch|wretch|bilious|regurgitat|hematemesis|indigestion|"
     r"heartburn|\breflux|acid reflux|\bgerd symptom", "nausea/vomiting", SYMP),
    (r"\bfevers?\b|\bchills?\b|\brigors?\b|night sweats|\bsweats?\b|diaphor|shaking chills|"
     r"subjective fever|hot/cold|flushing|cold sweat|shakes", "fever/chills", SYMP),
    (r"diarrhea|constipat|bloat|\bflatus\b|bowel (movement|habit|pattern|changes?)|stool (color|"
     r"consistency|frequency|character|change|burden|output|passage|incontinence)|obstipation|"
     r"\bstool\b|feces|loose stool|watery|fecal (urgency|urgency)|change in (stool|bowel)|"
     r"gas (pain|passage)|passing gas|\bbloating\b", "bowel-habit", SYMP),
    (r"melena|hematochezia|\bbrbpr\b|bright red blood|blood (in |per )?(stool|rectum|bowel|emesis|vomit)|"
     r"occult blood|guaiac|guaic|\bheme\b|rectal bleed|\bgi bleed|gastrointestinal bleed|bloody (stool|vomit|"
     r"bowel|diarrhea)|gross blood|\bmucoid\b|blood and mucus|hemoptysis|epistaxis|variceal", "gi-bleeding", SYMP),
    (r"dysuria|hematuria|urinary (frequency|urgency|retention|symptom|incontinence|catheter|changes?|habits?)|"
     r"\bhesitancy\b|burning (on |with )?urinat|dark urine|urine (color|output|hcg)|voiding|urinat|"
     r"pneumaturia|\bpolyuria\b|urological symptom|particulate matter in urine|choluria|difficulty urinating", "urinary-symptom", SYMP),
    (r"shortness of breath|\bdyspnea\b|\bsob\b|dyspnea on exertion|\bcough\b|hemoptys|orthopnea|"
     r"paroxysmal nocturnal|\bhypoxia\b|breathing|air hunger|wheezing|pleuris|sore throat|"
     r"nasal congestion|rhinorrhea|\buri\b|nasal", "respiratory-symptom", SYMP),
    (r"appetite|anorexia|weight (loss|gain|change)|\bpo intake|oral intake|decreased appetite|"
     r"loss of appetite|diet tolerance|\bintake\b|nutrition|fluid intake|\bhiccup|dysphagia|"
     r"odynophagia|pain with (eating|food|po|fatty|meals?|spicy)|early satiety|dehydration", "appetite/intake", SYMP),
    (r"headache|dizz|lightheaded|light.?headed|vertigo|syncope|presyncope|pre-?syncope|\bweakness\b|"
     r"numbness|tingling|paresthes|vision|visual|blurry|diplopia|\bspeech\b|dysarthr|aphasia|"
     r"confusion|\bseizure\b|tremor|focal (weakness|numbness|deficit|neuro)|gait|paralysis|"
     r"sensorimotor|sensation|neurolog|motor (function|strength)|\bfalls?\b|balance|dysphag", "neuro-symptom", SYMP),
    (r"fatigue|malaise|lethargy|weakness|exhaustion|energy level|somnolence|general well|"
     r"palpitat|diaphoresis|weight-related|night sweat|bleeding disorder(?! history)", "constitutional-symptom", SYMP),

    # --- CONSTITUTIONAL (general appearance / mental / hydration / habitus) -----
    (r"^general|general (appearance|assessment|well)|general_?appearance|\bdistress\b|\bcomfort|"
     r"well.?(appearing|being)|overall (assessment|vitals?)|vital_?signs?_?overall|"
     r"posture/behavior|appearance, behavior|acute distress|no acute", "general-appearance", CONST),
    (r"mental status|alertness|orientat|\bmentation\b|consciousness|level of consciousness|\bmood\b|"
     r"\baffect\b|\bbehavior\b|cognit|\bjudgment\b|\battention\b|responsive|arousal|awake|"
     r"neurological status|cognitive|altered mental|mental status changes", "mental-status", CONST),
    (r"moisture|\bturgor\b|hydration|mucous membrane|mucus membrane|\bmoist\b|\bdry\b(?! heav)", "hydration", CONST),
    (r"body habitus|\bhabitus\b|\bobesity\b|\bobese\b|\boverweight\b|morbid obes|body mass|\bweight\b|\bheight\b", "body-habitus", CONST),
    (r"\btone\b|\bstrength\b|muscle (bulk|tone)|\bbulk\b|range of motion|\bmobility\b|ambulat|"
     r"dorsiflex|reflexes?|plantar reflex|rectal tone|sphincter tone|motor|\bgait\b", "musculoskeletal-exam", SIGN),
]
ATTRIBUTE_RE = [(re.compile(p), c, ax) for p, c, ax in ATTRIBUTE_RULES]


def norm_attribute(term: str):
    t = term.strip().lower()
    if not t:
        return None
    for rx, canon, axis in ATTRIBUTE_RE:
        if rx.search(t):
            return canon, axis, (axis in CORE_ATTR_AXES)
    return None  # residual


# ===========================================================================
# apply + write
# ===========================================================================

def build_field(field, counter, normfn, drop_terms=None):
    drop_terms = drop_terms or set()
    mapping = {}
    residual = Counter()
    canon_freq = Counter()
    canon_sys = {}
    canon_surface = defaultdict(set)
    for term, n in counter.items():
        if term in drop_terms:
            mapping[term] = "__NUMERIC__"
            canon_freq["__NUMERIC__"] += n
            canon_sys["__NUMERIC__"] = "numeric_vital"
            continue
        res = normfn(term)
        if res is None:
            residual[term] += n
            mapping[term] = term  # self-map; flagged residual
            canon_freq[term] += n
            canon_sys.setdefault(term, "__residual__")
            canon_surface[term].add(term)
            continue
        canon, sysg, _core = res
        mapping[term] = canon
        canon_freq[canon] += n
        canon_sys[canon] = sysg
        canon_surface[canon].add(term)
    return mapping, residual, canon_freq, canon_sys, canon_surface


def write_field(field, counter, mapping, residual, canon_freq, canon_sys,
                canon_surface, core_axes=CORE_SYS):
    os.makedirs(OUT_DIR, exist_ok=True)
    special = {"__NUMERIC__", "__DROP__"}
    vocab = []
    for canon, fr in canon_freq.most_common():
        vocab.append({
            "canonical": canon,
            "system": canon_sys.get(canon, ""),
            "core": canon_sys.get(canon, "") in core_axes,
            "freq": fr,
            "n_surface": len(canon_surface.get(canon, [])),
            "residual": canon_sys.get(canon, "") == "__residual__",
        })
    json.dump(vocab, open(os.path.join(OUT_DIR, f"{field}_vocab.json"), "w"),
              ensure_ascii=False, indent=2)
    json.dump({"field": field, "raw_distinct": len(counter),
               "raw_total": sum(counter.values()), "map": mapping,
               "raw_freq": dict(counter)},
              open(os.path.join(OUT_DIR, f"{field}_map.json"), "w"),
              ensure_ascii=False, indent=2)
    # stats
    n_real = sum(1 for v in vocab if not v["residual"] and v["canonical"] not in special)
    resid_occ = sum(residual.values())
    print(f"[{field}] raw_distinct={len(counter)} raw_total={sum(counter.values())}")
    print(f"  canonical concepts (real)={n_real} | residual terms={len(residual)} "
          f"({resid_occ} occ, {resid_occ/sum(counter.values())*100:.1f}%)")
    return vocab, residual


def write_summary(avocab, aresid, svocab, sresid, a_total, s_total,
                  tvocab=None, tresid=None, t_total=0):
    special = {"__NUMERIC__", "__DROP__"}
    L = []
    L.append("# Narrative vocabulary (名词库 / 状态库 / 属性库) — v1 converged\n")
    L.append("Built by `scripts/normalize_vocab.py` (API-free, hand-authored medical "
             "rules) from `results/evidence_pieces/*.jsonl`, NARRATIVE pieces only "
             "(labs/micro excluded — their attribute IS the analyte name, already "
             "canonical via itemid→label). Raw→canonical maps live in "
             "`results/vocab/{anatomy,state,attribute}_map.json`.\n")
    # anatomy
    a_real = [v for v in avocab if not v["residual"] and v["canonical"] not in special]
    a_core = sum(v["freq"] for v in a_real if v["core"])
    L.append(f"## 名词库 (anatomy): {len(a_real)} canonical concepts, "
             f"{a_total} occurrences\n")
    L.append(f"Residual (rule-miss, self-mapped): {sum(aresid.values())} occ "
             f"({sum(aresid.values())/a_total*100:.1f}%). "
             f"Core diagnostic mass: {a_core}/{a_total} = {a_core/a_total*100:.0f}%.\n")
    L.append("| canonical | system | core | freq | surface forms |")
    L.append("|---|---|:-:|--:|--:|")
    for v in sorted(a_real, key=lambda x: (not x["core"], -x["freq"])):
        L.append(f"| {v['canonical']} | {v['system']} | {'✓' if v['core'] else ''} "
                 f"| {v['freq']} | {v['n_surface']} |")
    # state
    s_real = [v for v in svocab if not v["residual"] and v["canonical"] not in special]
    numeric = next((v["freq"] for v in svocab if v["canonical"] == "__NUMERIC__"), 0)
    dropped = next((v["freq"] for v in svocab if v["canonical"] == "__DROP__"), 0)
    L.append(f"\n## 状态库 (state): {len(s_real)} canonical descriptors, "
             f"{s_total} occurrences\n")
    L.append(f"POLARITY preserved (X vs `no X` are distinct). "
             f"`__NUMERIC__` (numeric vitals, read via finding_status+value_unit): "
             f"{numeric} occ. `__DROP__` (degree-only / process words): {dropped} occ. "
             f"Residual: {sum(sresid.values())} occ "
             f"({sum(sresid.values())/s_total*100:.1f}%).\n")
    L.append("| canonical | freq | surface forms |")
    L.append("|---|--:|--:|")
    for v in sorted(s_real, key=lambda x: -x["freq"]):
        L.append(f"| {v['canonical']} | {v['freq']} | {v['n_surface']} |")
    # attribute
    if tvocab is not None:
        t_real = [v for v in tvocab if not v["residual"] and v["canonical"] not in special]
        t_core = sum(v["freq"] for v in t_real if v["core"])
        L.append(f"\n## 属性库 (attribute): {len(t_real)} canonical dims, "
                 f"{t_total} occurrences\n")
        L.append("`attribute` is the CORE axis of the adequacy gate — the atomic "
                 "covered/required unit is the `(anatomy, attribute)` PAIR, and "
                 "`required(S)` / `sought_dimensions` ARE attribute dims. **core = "
                 "`morphology` axis** = the imaging/exam characterization dims a study "
                 "COVERS (the required(S)/covered(S) dims the gate reads); the other "
                 "axes (symptom / sign / vital / constitutional / history / device) are "
                 "the demand-trigger side, slotted present/absent.\n")
        L.append(f"Morphology (adequacy-covered) mass: {t_core}/{t_total} = "
                 f"{t_core/t_total*100:.0f}%. "
                 f"Residual (rule-miss, self-mapped — mostly the rare comorbidity/"
                 f"disease-name tail): {sum(tresid.values())} occ "
                 f"({sum(tresid.values())/t_total*100:.1f}%).\n")
        L.append("| canonical | axis | core | freq | surface forms |")
        L.append("|---|---|:-:|--:|--:|")
        for v in sorted(t_real, key=lambda x: (not x["core"], x["system"], -x["freq"])):
            L.append(f"| {v['canonical']} | {v['system']} | {'✓' if v['core'] else ''} "
                     f"| {v['freq']} | {v['n_surface']} |")
    open(os.path.join(OUT_DIR, "vocab_summary.md"), "w").write("\n".join(L) + "\n")
    print(f"wrote {OUT_DIR}/vocab_summary.md")


def main():
    anat, state, attr, state_under_vital = collect()
    # NOTE: we do NOT globally drop states that co-occur with vital_sign. Pure numeric
    # vitals ("18","97.8","100% on room air") are handled by norm_state's _NUMERIC by
    # CONTENT; polysemous descriptors ("elevated","low") are legitimate states elsewhere
    # (elevated hemidiaphragm) and must survive. Per-piece "this is a vital, read it as
    # numeric" is the anatomy slot's job (vital_sign -> systemic), not the state map's.
    print(f"vital_sign co-occurring states (NOT globally dropped): {len(state_under_vital)} "
          f"distinct; numeric ones handled by content")

    am, ares, af, asys, asurf = build_field(A, anat, norm_anatomy)
    avocab, aresid = write_field(A, anat, am, ares, af, asys, asurf)

    sm, sres, sf, ssys, ssurf = build_field("state", state, norm_state)
    svocab, sresid = write_field("state", state, sm, sres, sf, ssys, ssurf)

    tm, tres, tf, tsys, tsurf = build_field("attribute", attr, norm_attribute)
    tvocab, tresid = write_field("attribute", attr, tm, tres, tf, tsys, tsurf,
                                 core_axes=CORE_ATTR_AXES)

    # residual dumps for iteration
    with open(os.path.join(OUT_DIR, "anatomy_residual.tsv"), "w") as fh:
        for t, n in aresid.most_common():
            fh.write(f"{n}\t{t}\n")
    with open(os.path.join(OUT_DIR, "state_residual.tsv"), "w") as fh:
        for t, n in sresid.most_common():
            fh.write(f"{n}\t{t}\n")
    with open(os.path.join(OUT_DIR, "attribute_residual.tsv"), "w") as fh:
        for t, n in tresid.most_common():
            fh.write(f"{n}\t{t}\n")

    write_summary(avocab, aresid, svocab, sresid,
                  sum(anat.values()), sum(state.values()),
                  tvocab=tvocab, tresid=tresid, t_total=sum(attr.values()))


if __name__ == "__main__":
    main()
