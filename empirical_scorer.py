"""empirical_scorer.py  —  KNN-based empirical diagnosis scorer

Encodes patient feature dicts into fixed-length float32 vectors, then answers
KNN queries over a reference corpus to return log P(disease | neighbors).

This replaces the mimic_score() stub in diagnosis_distribution.py once a
training corpus is available (here: the held-out training split from the real
patient JSON files).

Usage
-----
    from empirical_scorer import EmpiricalScorer

    scorer = EmpiricalScorer(k=15)
    scorer.fit([(features_dict, "appendicitis"), (features_dict2, "pancreatitis"), ...])

    log_p = scorer.score("appendicitis", query_features)

Feature encoding
----------------
    Boolean features    → 0.0 / 1.0
    pain_location       → one-hot over VALID_PAIN_LOCATIONS (9 dims)
    CTSI_score          → divided by 10 → [0.0, 1.0]
    tests_done          → one binary flag per VALID_TESTS entry (7 dims)

Distance metric: L2 (Euclidean) over the encoded vector.

Scoring: Laplace-smoothed log P(disease | K nearest neighbors)
    log((count_d + α) / (K + α × |diseases|))
where α = smoothing (default 1.0).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from feature_schema import default_features, VALID_PAIN_LOCATIONS, VALID_TESTS

DISEASES: tuple[str, ...] = (
    "appendicitis",
    "cholecystitis",
    "diverticulitis",
    "pancreatitis",
)

# ── feature keys that are plain booleans in the schema ────────────────────────
_BOOL_KEYS: list[str] = []
_FLOAT_KEYS: list[str] = []
_SKIP_KEYS: frozenset[str] = frozenset({"pain_location", "tests_done", "CTSI_score"})

def _build_key_lists() -> None:
    """Populate _BOOL_KEYS and _FLOAT_KEYS from the schema defaults."""
    defaults = default_features()
    for key, val in defaults.items():
        if key in _SKIP_KEYS:
            continue
        if isinstance(val, bool):
            _BOOL_KEYS.append(key)
        elif isinstance(val, float):
            _FLOAT_KEYS.append(key)

_build_key_lists()

# Total vector length:
#   len(_BOOL_KEYS) + len(VALID_PAIN_LOCATIONS) + 1 (CTSI) + len(VALID_TESTS) + len(_FLOAT_KEYS)
_VECTOR_DIM = (
    len(_BOOL_KEYS)
    + len(VALID_PAIN_LOCATIONS)
    + 1
    + len(VALID_TESTS)
    + len(_FLOAT_KEYS)
)


# ---------------------------------------------------------------------------
# Feature encoding
# ---------------------------------------------------------------------------

def encode_features(features: dict) -> np.ndarray:
    """
    Encode a feature dict into a fixed-length float32 vector.

    The encoding order is deterministic:
        [bool_features | pain_location_onehot | CTSI_score_norm | tests_done_flags | other_floats]
    """
    vec: list[float] = []
    defaults = default_features()

    # 1. Boolean features (sorted for determinism)
    for key in _BOOL_KEYS:
        val = features.get(key, defaults[key])
        vec.append(1.0 if val else 0.0)

    # 2. pain_location one-hot
    loc = features.get("pain_location", defaults["pain_location"])
    for valid_loc in VALID_PAIN_LOCATIONS:
        vec.append(1.0 if loc == valid_loc else 0.0)

    # 3. CTSI_score normalised to [0, 1]
    ctsi = features.get("CTSI_score", 0.0)
    vec.append(float(ctsi) / 10.0)

    # 4. tests_done — one binary flag per VALID_TESTS entry
    done_set: set[str] = set(features.get("tests_done", []))
    for test in VALID_TESTS:
        vec.append(1.0 if test in done_set else 0.0)

    # 5. Other float-typed features
    for key in _FLOAT_KEYS:
        val = features.get(key, defaults[key])
        vec.append(float(val))

    assert len(vec) == _VECTOR_DIM, f"Encoding dim mismatch: {len(vec)} != {_VECTOR_DIM}"
    return np.array(vec, dtype=np.float32)


# ---------------------------------------------------------------------------
# EmpiricalScorer
# ---------------------------------------------------------------------------

class EmpiricalScorer:
    """
    KNN-based empirical scorer.

    Parameters
    ----------
    k          : number of nearest neighbors
    smoothing  : Laplace smoothing parameter α (default 1.0)
    """

    def __init__(self, k: int = 15, smoothing: float = 1.0) -> None:
        self.k = k
        self.smoothing = smoothing
        self._X: Optional[np.ndarray] = None   # shape (N, dim)
        self._labels: list[str] = []

    # ── training ──────────────────────────────────────────────────────────

    def fit(self, patients: list[tuple[dict, str]]) -> "EmpiricalScorer":
        """
        Fit on a list of (feature_dict, disease_label) pairs.

        Parameters
        ----------
        patients : list of (features, disease) where disease ∈ DISEASES
        """
        if not patients:
            return self
        self._X = np.stack([encode_features(f) for f, _ in patients])
        self._labels = [label for _, label in patients]
        return self

    # ── inference ─────────────────────────────────────────────────────────

    def score(self, disease: str, features: dict) -> float:
        """
        Return log P(disease | K nearest neighbors) using Laplace smoothing.

        Returns 0.0 if the scorer has not been fitted yet.
        """
        if self._X is None or not self._labels:
            return 0.0

        q = encode_features(features)                  # (dim,)
        dists = np.linalg.norm(self._X - q, axis=1)   # (N,)

        k = min(self.k, len(self._labels))
        nn_idx = np.argpartition(dists, k - 1)[:k]
        nn_labels = [self._labels[i] for i in nn_idx]

        count = nn_labels.count(disease)
        n_classes = len(DISEASES)
        log_p = math.log(
            (count + self.smoothing) / (k + self.smoothing * n_classes)
        )
        return log_p

    # ── diagnostics ───────────────────────────────────────────────────────

    @property
    def is_fitted(self) -> bool:
        return self._X is not None

    @property
    def n_train(self) -> int:
        return len(self._labels)

    def label_distribution(self) -> dict[str, int]:
        """Return counts of each disease label in the training corpus."""
        dist: dict[str, int] = {d: 0 for d in DISEASES}
        for label in self._labels:
            dist[label] = dist.get(label, 0) + 1
        return dist
