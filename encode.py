import json
from typing import Any, Dict, List, Tuple
from openai import OpenAI

# pip install openai
# export OPENAI_API_KEY="your_key"

MODEL_NAME = "gpt-4o"  # or gpt-4.1-mini for lower cost
INPUT_PATH = "state_text.json"
OUTPUT_PATH = "state_trajectories.json"

client = OpenAI()

SYSTEM_PROMPT = """
You are a Clinical Data Scientist specializing in Electronic Health Record (EHR) encoding.
Your task is to maintain a state vector for a patient undergoing a diagnostic workup.
You must synthesize new evidence into the existing clinical state while maintaining consistency.

### CONTEXT
You are tracking a patient's diagnostic trajectory.
- Previous State Vector: {{previous_state_json}}
- New Observation: {{new_raw_text}} (This could be HPI, a batch of Labs, or a Radiology Report)

### ENCODING INSTRUCTIONS
Update the following 5 dimensions based on the New Observation.
If the observation contains no information for a dimension, retain the value from the Previous State.

1) Systemic_Stress (Scalar 0.0-1.0)
- Increase based on signs of infection/stress (WBC > 10.0, Fever, Tachycardia, High Bilirubin).
- 0.0 = perfect health; 1.0 = critical illness/sepsis.

2) Topographic_Focus (Categorical)
- [RUQ, LUQ, RLQ, LLQ, Epigastric, Periumbilical, Pelvic, Chest, General_Abdomen, Other]

3) Organ_Stress_Index (Object)
- {Liver: [0-1], Kidney: [0-1], Pancreas: [0-1]}
- High Lipase -> Pancreas; High Creatinine -> Kidney.

4) Structural_Findings (Binary Flags)
- stone_detected (0/1)
- wall_thickening (0/1)
- fluid_collection (0/1)
- organ_dilation (0/1)

5) Modality_History (List)
- Append modality of current observation if it is a test
  (e.g., "CT_Abdomen", "US_Abdomen", "Lab_Panel").

### OUTPUT FORMAT
Return ONLY a valid JSON object representing the updated state. No prose.
""".strip()

STATE_JSON_SCHEMA = {
    "name": "clinical_state",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "Systemic_Stress": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "Topographic_Focus": {
                "type": "string",
                "enum": [
                    "RUQ", "LUQ", "RLQ", "LLQ", "Epigastric",
                    "Periumbilical", "Pelvic", "Chest", "General_Abdomen", "Other"
                ]
            },
            "Organ_Stress_Index": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "Liver": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "Kidney": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "Pancreas": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                },
                "required": ["Liver", "Kidney", "Pancreas"]
            },
            "Structural_Findings": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "stone_detected": {"type": "integer", "enum": [0, 1]},
                    "wall_thickening": {"type": "integer", "enum": [0, 1]},
                    "fluid_collection": {"type": "integer", "enum": [0, 1]},
                    "organ_dilation": {"type": "integer", "enum": [0, 1]},
                },
                "required": ["stone_detected", "wall_thickening", "fluid_collection", "organ_dilation"]
            },
            "Modality_History": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "Systemic_Stress",
            "Topographic_Focus",
            "Organ_Stress_Index",
            "Structural_Findings",
            "Modality_History",
        ],
    },
}


def default_state() -> Dict[str, Any]:
    return {
        "Systemic_Stress": 0.0,
        "Topographic_Focus": "Other",
        "Organ_Stress_Index": {"Liver": 0.0, "Kidney": 0.0, "Pancreas": 0.0},
        "Structural_Findings": {
            "stone_detected": 0,
            "wall_thickening": 0,
            "fluid_collection": 0,
            "organ_dilation": 0,
        },
        "Modality_History": [],
    }


def infer_modality(modality_key: str, region: str) -> str:
    def norm(x: str, default: str = "Other") -> str:
        x = (x or "").strip()
        if not x:
            return default
        # Normalize to stable token: letters/numbers + underscores
        return "_".join(x.split())

    key_part = norm(modality_key)       # e.g., "Carotid_ultrasound"
    region_part = norm(region)          # e.g., "Abdomen"
    return f"{key_part}_{region_part}"  # e.g., "CT_Abdomen"


def build_first_observation(patient: Dict[str, Any]) -> Tuple[str, str]:
    hpi = patient.get("hpi", "")
    labs = patient.get("lab_tests", {})
    lines = []
    for k, v in labs.items():
        if isinstance(v, list):
            v = "; ".join(map(str, v))
        lines.append(f"- {k}: {v}")
    obs = f"HPI:\n{hpi}\n\nLAB PANEL:\n" + "\n".join(lines)
    return "Lab_Panel", obs


def build_radiology_observations(patient: Dict[str, Any]) -> List[Tuple[str, str]]:
    seq: List[Tuple[str, str]] = []
    radiology = patient.get("radiology", [])
    for r in radiology:
        exam_name = r.get("exam_name", "")
        region = r.get("region", "")
        report = r.get("report", "")
        modality = r.get("modality", "other")
        modality_region = infer_modality(modality, region)
        obs = f"MODALITY: {exam_name}\nREGION: {region}\nREPORT:\n{report}"
        seq.append((modality_region, obs))
    return seq


def update_state(previous_state: Dict[str, Any], new_observation: str, current_modality: str) -> Dict[str, Any]:
    user_payload = {
        "previous_state_json": previous_state,
        "new_raw_text": new_observation,
        "current_modality": current_modality
    }

    resp = client.responses.create(
        model=MODEL_NAME,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": STATE_JSON_SCHEMA["name"],
                "strict": STATE_JSON_SCHEMA["strict"],
                "schema": STATE_JSON_SCHEMA["schema"],
            }
        },
        temperature=0
    )
    return json.loads(resp.output_text)


def build_trajectory_for_patient(patient_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    states: List[Dict[str, Any]] = []
    state = default_state()  # reset fresh for each patient

    # State 0: HPI + all labs
    mod0, obs0 = build_first_observation(patient_obj)
    state = update_state(state, obs0, mod0)
    states.append(state)

    # State 1..N: each radiology report sequentially
    for modality, obs in build_radiology_observations(patient_obj):
        state = update_state(state, obs, modality)
        states.append(state)

    return states


def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_trajectories: Dict[str, List[Dict[str, Any]]] = {}
    for patient_id, patient_obj in data.items():
        try:
            traj = build_trajectory_for_patient(patient_obj)
            all_trajectories[patient_id] = traj
            print(f"[OK] patient={patient_id}, num_states={len(traj)}")
        except Exception as e:
            print(f"[ERR] patient={patient_id}: {e}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_trajectories, f, ensure_ascii=False, indent=2)

    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()