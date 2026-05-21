"""KNN top-N similar patients by step-0 (or step-k) features.

Independent from FeatureSimulator: that indexes per-transition; we want
per-patient. We L2-distance on encoded features against the training pool
and return each neighbour's *full* actual test_sequence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from knn.empirical_scorer import encode_features


@dataclass
class TrainPatient:
    patient_id: str
    disease: str
    vec_step0: np.ndarray
    test_sequence: list[str]   # all completed test_keys in order, skipping the HPI step


class PatientKNN:
    """Look up nearest training patients by encoded feature L2 distance."""

    def __init__(self, train_dict: dict[str, dict[str, list[dict]]]) -> None:
        self.entries: list[TrainPatient] = []
        for disease, patients in train_dict.items():
            for pid, steps in patients.items():
                if not steps:
                    continue
                seq = [s["test_key"] for s in steps[1:] if s.get("test_key")]
                self.entries.append(
                    TrainPatient(
                        patient_id=str(pid),
                        disease=disease,
                        vec_step0=encode_features(steps[0]["features"]),
                        test_sequence=seq,
                    )
                )
        if not self.entries:
            self._mat = None
        else:
            self._mat = np.stack([e.vec_step0 for e in self.entries], axis=0)

    def top_n(self, query_features: dict, n: int = 5) -> list[tuple[TrainPatient, float]]:
        if self._mat is None:
            return []
        q = encode_features(query_features)
        dists = np.linalg.norm(self._mat - q, axis=1)
        idx = np.argsort(dists)[:n]
        return [(self.entries[i], float(dists[i])) for i in idx]
