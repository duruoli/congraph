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
                if a:
                    anat[a] += 1
                if s:
                    state[s] += 1
                    if a == "vital_sign":
                        state_under_vital[s] += 1
    return anat, state, state_under_vital


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
     r"no scleral icterus|perrl.?a?|eomi|normocephalic.*|atraumatic|alert.*|oriented.*|"
     r"fluent|intact.*|grossly intact|nonfocal|non-?focal|comfortable|afebrile|vss|"
     r"vital signs stable|stable.*|nondistended|non-?distended|soft.*|smooth|sharp|"
     r"anteverted|palpable|appropriate.*|top[- ]?normal|upper limits of normal|"
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


def write_field(field, counter, mapping, residual, canon_freq, canon_sys, canon_surface):
    os.makedirs(OUT_DIR, exist_ok=True)
    special = {"__NUMERIC__", "__DROP__"}
    vocab = []
    for canon, fr in canon_freq.most_common():
        vocab.append({
            "canonical": canon,
            "system": canon_sys.get(canon, ""),
            "core": canon_sys.get(canon, "") in CORE_SYS,
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


def write_summary(avocab, aresid, svocab, sresid, a_total, s_total):
    special = {"__NUMERIC__", "__DROP__"}
    L = []
    L.append("# Narrative vocabulary (名词库 / 状态库) — v1 converged\n")
    L.append("Built by `scripts/normalize_vocab.py` (API-free, hand-authored medical "
             "rules) from `results/evidence_pieces/*.jsonl`, NARRATIVE pieces only "
             "(labs/micro excluded). Raw→canonical maps live in "
             "`results/vocab/{anatomy,state}_map.json`.\n")
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
    open(os.path.join(OUT_DIR, "vocab_summary.md"), "w").write("\n".join(L) + "\n")
    print(f"wrote {OUT_DIR}/vocab_summary.md")


def main():
    anat, state, state_under_vital = collect()
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

    # residual dumps for iteration
    with open(os.path.join(OUT_DIR, "anatomy_residual.tsv"), "w") as fh:
        for t, n in aresid.most_common():
            fh.write(f"{n}\t{t}\n")
    with open(os.path.join(OUT_DIR, "state_residual.tsv"), "w") as fh:
        for t, n in sresid.most_common():
            fh.write(f"{n}\t{t}\n")

    write_summary(avocab, aresid, svocab, sresid,
                  sum(anat.values()), sum(state.values()))


if __name__ == "__main__":
    main()
