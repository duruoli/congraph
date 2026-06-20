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
~30+ cases the reasoning: “appendix not visualized,” “US technically limited,” “limited study,” “sludge vs. non-shadowing stones,” “outside CT very limited given body habitus.” Each time, the next imaging step is vindicated 
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

> Source: 430 rrn-aligned judged steps in `results/annotation_experiment/full/belief_deviation_analysis.csv`.
> Two subsets: **Follow + Disconfirmed (FD) = 55**, **Deviate + Disconfirmed (DD) = 43**.
> **Label caveat (critical):** `follow` is *computed* (doctor action == rubric-traversal recommendation on the belief). The LLM annotation **never saw the rubric**, so `follow` = *convergence, not obedience* — the reconstructed reasoning is the doctor's own logic. ⇒ high FD does NOT mean "the rubric misled the doctor."
> appropriateness: FD yes 50 / partial 5; DD yes 30 / partial 13. certainty_update: ALL "down".
> The question for the agent: which disconfirms are **unavoidable** (test was right, a negative is the informative answer → teach calibration) vs **real holes** (should not have followed / deviated → teach the override/brake).

## Follow + Disconfirmed (55) — three patterns

### FD-1 (UNAVOIDABLE, dominant 54/55): the biliary-cause hunt that comes back clean is the correct call, not a mistake.
Biochemistry (lipase 10–280× / cholestatic LFTs / hyperbilirubinemia) makes "gallstone / CBD obstruction" the leading hypothesis; the doctor (independently — and the rubric happens to agree) orders US→MRCP to detect-or-rule-out choledocholithiasis *because it gates ERCP*. Duct comes back clean (20129489, 21775506, 25233319, 24677490, 26349833 s3). You cannot know ex-ante which lipase is gallstone-driven → imaging is mandatory and the negative IS the informative answer (removes ERCP, redirects to alcohol / metabolic / drug / idiopathic).
Pearl: this disconfirm is a *planned-for branch, not a hole*. The rubric's defect is **representational** — no "expected-negative / rule-out" outcome node, no pretest layer — so a naïve agent reads the negative as surprise/failure. Agent should: (i) pre-register a realistic P(negative biliary); (ii) use the disconfirm to drop the **belief** (P(gallstone etiology)↓), NOT its decision-confidence; (iii) **stop re-escalating** — one clean duct is enough (repeat-MRCP-after-clean-CBD is the low-yield tail).

### FD-2 (REAL HOLE): stale anatomy — imaging an organ that is already removed / drained / resolved.
US still hunting gallstones in a post-cholecystectomy patient (21061497); US of an already-drained gallbladder (20334898); follow-up of a collection that already resolved (29573603 s4). Disconfirm is *guaranteed* because the premise is dead.
Pearl: rubric / naïve-follow has **no current-state gate**. Agent must check intervention/anatomy state before imaging → skip the futile test. A clean should-not-follow pattern.

### FD-3 (REAL HOLE / redundant): sonographic confirmation of CT-suspected cholecystitis systematically fails its own criteria; and "dilation ⟹ stone" fails post-intervention.
31/55 expected cholecystitis signs; CT already showed GB distension/stranding, rubric still sends US to "confirm Murphy / wall / pericholecystic fluid," repeatedly absent (22023307, 24115267, 25633498, 27194914, 28306018). Plus the "dilated duct, no stone" sub-cluster (24050288, 24270186, 24922174) — post-chole / sphincterotomy / passed-stone give dilation without a target.
Pearl: once CT characterized the organ, a second confirmatory modality is low-yield; condition the implicit "dilation ⟹ stone" rule on biliary-intervention history. Agent: down-weight redundant confirmation.

## Deviate + Disconfirmed (43) — three patterns

### DD-1 (REAL HOLE, cleanest should-NOT-deviate, 10/43): over-imaging past a valid rubric STOP.
Rubric already in terminal_confirmed / terminal_low_risk / terminal_excluded / blocked ("no more imaging"); the doctor re-scans anyway — almost always a repeat CT for a feared complication (necrosis / abscess / perforation / pseudocyst) that is **absent**, disease often stable or improving (28684468 s3, 25414251 s4, 28122817 s3, 21849575 s4, 24440089 s2; 27993727 s2 re-scanned a patient who had **already had the appendectomy**).
Pearl: here the rubric's stop was RIGHT and the deviation WRONG — defensive / anxiety re-scan. The patch must learn to **suppress** this, not imitate it. (appropriateness is "partial" far more often here than in FD — the data already flags these as questionable.)

### DD-2 (UNAVOIDABLE, mirrors FD-1, 32/43 biliary): escalating the modality to chase the same biliary cause still comes up empty.
The "skip US, go straight to CT/MRCP" deviation (cf. deviate+confirmed) — but when the etiology is truly non-obstructive, the bigger test finds no cause either (21238215, 20418179, 26349833 s1, 26472405, 25444703, 25616232).
Pearl: going-bigger is not a fix for a clean biliary tree — the gap is the **etiology**, not test resolution. Agent must not treat modality-escalation as the response to a negative biliary look.

### DD-3 (KEEP THIS deviation — the outlier-patch signal): disconfirmed ≠ wrong test; broad nets miss the stated target but catch the truth.
Local prediction failed yet the deviation was diagnostically superior: 28238173 (CT expecting appendicitis → found the true acute cholecystitis), 25444703 (expected pancreatic edema → found a 50 mm cystic pancreatic-head mass), 28672604 (expected post-op biliary complication → found advanced intrinsic liver disease). (+ competing-organ tail 29794234 ovaries, 26860125 renal allograft — normal but defensible broad triage.)
Pearl: a "disconfirmed" deviation must NOT be auto-penalized — the broad CT net IS the outlier-patch behavior the project targets. The reward must separate **disconfirmed-prediction** from **wrong-action**.

## Synthesis — the disconfirm taxonomy the agent must learn
- **Unavoidable / calibration** (FD-1, DD-2): test correct, negative = the informative answer → teach realistic pretest + belief-update + STOP re-escalating. NOT a deviation trigger, NOT an error.
- **Real hole → should-not-follow / should-not-deviate** (FD-2 stale anatomy, FD-3 redundant confirmation, DD-1 over-imaging past a stop): teach current-state gate, redundancy down-weight, respect valid terminal stops.
- **Keep despite disconfirm** (DD-3): broad net that catches the truth → do not penalize; this is the patch.

**Pairing with the deviate+confirmed insights:** those said "the rubric stops reasoning too early" → the agent needs *loosening*. The disconfirm groups add the symmetric **brake**: "the doctor sometimes reasons too long / on a dead premise" (FD-2, DD-1) and "a negative is not a failure" (FD-1, DD-2). A good agent needs BOTH — the evidence-based loosening AND these brakes — otherwise loosening just becomes over-testing.
