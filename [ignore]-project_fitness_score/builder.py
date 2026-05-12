import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------
# Fixed vocabularies (stable dimensions)
# ----------------------------
REGIONS = [
    "RUQ", "LUQ", "RLQ", "LLQ", "Epigastric",
    "Periumbilical", "Pelvic", "Chest", "General_Abdomen", "None", "Other"
]

MODALITIES = ['CTU_Abdomen', 'CT_Abdomen', 'CT_Chest', 'CT_Head', 'CT_Neck', 'CT_Spine', 'Drainage_Abdomen', 'ERCP_Abdomen', 'Fluoroscopy_Chest', 'Fluoroscopy_Hip', 'Lab_Panel', 'MRA_Chest', 'MRCP_Abdomen', 'MRE_Abdomen', 'MRI_Abdomen', 'MRI_Chest', 'MRI_Head', 'MRI_Spine', 'PTC_Abdomen', 'Radiograph_Abdomen', 'Radiograph_Ankle', 'Radiograph_Chest', 'Radiograph_Foot', 'Radiograph_Hip', 'Radiograph_Knee', 'Radiograph_Spine', 'Radiograph_Venous', 'Ultrasound_Abdomen', 'Ultrasound_Chest', 'Ultrasound_Neck', 'Ultrasound_Scrotum', 'Ultrasound_Venous', 'Upper_GI_Series_Abdomen']

# Policy actions: all non-lab modalities + dataset-specific terminal actions
TERMINAL_ACTIONS = [
    "DIAGNOSE_CHOLECYSTITIS",
    "DIAGNOSE_PANCREATITIS",
    "DIAGNOSE_DIVERTICULITIS",
]

ACTION_MODALITIES = [m for m in MODALITIES if m != "Lab_Panel"] + TERMINAL_ACTIONS
ACTION2ID = {a: i for i, a in enumerate(ACTION_MODALITIES)}
ID2ACTION = {i: a for a, i in ACTION2ID.items()}


@dataclass
class ILDataset:
    X: np.ndarray          # [N, 42]
    y: np.ndarray          # [N]
    patient_ids: np.ndarray  # [N]
    action_names: List[str]


def state_to_vector(s: Dict) -> np.ndarray:
    v: List[float] = []

    # 1) Systemic stress (1)
    v.append(float(s.get("Systemic_Stress", 0.0)))

    # 2) Topographic focus one-hot (11)
    region_vec = np.zeros(len(REGIONS), dtype=np.float32)
    region = s.get("Topographic_Focus", "Other")
    if region in REGIONS:
        region_vec[REGIONS.index(region)] = 1.0
    v.extend(region_vec.tolist())

    # 3) Organ stress (3)
    osi = s.get("Organ_Stress_Index", {})
    v.append(float(osi.get("Liver", 0.0)))
    v.append(float(osi.get("Kidney", 0.0)))
    v.append(float(osi.get("Pancreas", 0.0)))

    # 4) Structural findings (4)
    sf = s.get("Structural_Findings", {})
    v.append(float(sf.get("stone_detected", 0)))
    v.append(float(sf.get("wall_thickening", 0)))
    v.append(float(sf.get("fluid_collection", 0)))
    v.append(float(sf.get("organ_dilation", 0)))

    # 5) Modality history multi-hot (23)
    mh = np.zeros(len(MODALITIES), dtype=np.float32)
    for m in s.get("Modality_History", []):
        if m in MODALITIES:
            mh[MODALITIES.index(m)] = 1.0
    v.extend(mh.tolist())

    return np.asarray(v, dtype=np.float32)  # shape (42,)


def extract_next_action(curr_state: Dict, next_state: Dict) -> str:
    h1 = curr_state.get("Modality_History", [])
    h2 = next_state.get("Modality_History", [])

    # Expected: next history is prefix + exactly one new modality
    if len(h2) != len(h1) + 1 or h2[:len(h1)] != h1:
        raise ValueError(f"Invalid transition histories:\n{h1}\n->\n{h2}")
    return h2[-1]


def build_il_dataset(
    path_json: str,
    include_terminal_diagnose: bool = True,
    terminal_action: str = "DIAGNOSE_CHOLECYSTITIS",
    dataset_tag: str = "default",
) -> ILDataset:
    with open(path_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    X_list, y_list, pid_list = [], [], []

    for pid, traj in data.items():
        if not traj:
            continue

        # prefix pid to avoid accidental cross-dataset ID collisions
        pid_global = f"{dataset_tag}:{pid}"

        for t in range(len(traj)):
            s_t = traj[t]
            if t < len(traj) - 1:
                a_name = extract_next_action(s_t, traj[t + 1])
            else:
                if not include_terminal_diagnose:
                    continue
                a_name = terminal_action

            if a_name not in ACTION2ID:
                continue

            X_list.append(state_to_vector(s_t))
            y_list.append(ACTION2ID[a_name])
            pid_list.append(pid_global)

    X = np.stack(X_list).astype(np.float32)
    y = np.asarray(y_list, dtype=np.int64)
    pids = np.asarray(pid_list)
    return ILDataset(X=X, y=y, patient_ids=pids, action_names=ACTION_MODALITIES)

def build_il_dataset_mixed(specs, include_terminal_diagnose: bool = True) -> ILDataset:
    """
    specs: list of tuples
      (dataset_tag, path_json, terminal_action)
    """
    X_all, y_all, pid_all = [], [], []

    for dataset_tag, path_json, terminal_action in specs:
        ds = build_il_dataset(
            path_json=path_json,
            include_terminal_diagnose=include_terminal_diagnose,
            terminal_action=terminal_action,
            dataset_tag=dataset_tag,
        )
        X_all.append(ds.X)
        y_all.append(ds.y)
        pid_all.append(ds.patient_ids)

    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    pids = np.concatenate(pid_all, axis=0)
    return ILDataset(X=X, y=y, patient_ids=pids, action_names=ACTION_MODALITIES)


def patient_level_split(
    dataset: ILDataset, test_ratio: float = 0.2, seed: int = 42
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    unique_pids = np.unique(dataset.patient_ids)
    rng.shuffle(unique_pids)

    n_test = max(1, int(len(unique_pids) * test_ratio))
    test_pids = set(unique_pids[:n_test])

    is_test = np.array([pid in test_pids for pid in dataset.patient_ids], dtype=bool)
    is_train = ~is_test

    X_train, y_train = dataset.X[is_train], dataset.y[is_train]
    X_test, y_test = dataset.X[is_test], dataset.y[is_test]
    return (X_train, y_train), (X_test, y_test)


def make_class_weights(y_train: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.bincount(y_train, minlength=n_classes).astype(np.float32)
    counts = np.clip(counts, 1.0, None)
    weights = counts.sum() / (n_classes * counts)
    return weights.astype(np.float32)