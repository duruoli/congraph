#!/usr/bin/env python3
"""Dependency-free integrity checks for the Track A ACR corpus."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "acr_normative"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    corpus = json.loads((DATA / "acr_topics.json").read_text())
    rows = [json.loads(line) for line in (DATA / "acr_actions.jsonl").read_text().splitlines()]
    manifest = json.loads((DATA / "sources" / "manifest.json").read_text())
    assert corpus["schema_version"] == "1.1.0"
    assert corpus["ranking_policy"] == {
        "primary_metric": "final_rating",
        "direction": "higher_is_more_appropriate",
        "range": [1, 9],
        "tie_policy": "Preserve ties; do not infer a unique path from equal ratings.",
        "non_ranking_fields": [
            "appropriateness_category", "strength_of_evidence",
            "median_rating", "final_tabulations",
        ],
    }
    assert len(corpus["topics"]) == 4
    assert {t["topic_id"] for t in corpus["topics"]} == {20, 21, 126, 132}
    variants = [v for topic in corpus["topics"] for v in topic["variants"]]
    actions = [a for variant in variants for a in variant["actions"]]
    rationales = [r for topic in corpus["topics"] for r in topic["rationales"]]
    assert len(variants) == 17
    assert len(actions) == len(rows) == 141
    assert len(rationales) == 90
    assert len({a["action_id"] for a in actions}) == 141
    assert all(a["rationale_ids"] for a in actions)
    assert all(isinstance(a["provenance"]["page"], int) for a in actions)
    assert all(len(a["final_tabulations"]) == 9 for a in actions)
    assert all(a["evidence_references"] or a["strength_of_evidence"] == "Expert Consensus" for a in actions)
    assert all(set(v["context"]) == {
        "clinical_state", "imaging_history", "modifiers", "decision_stage"
    } for v in variants)
    assert all(set(v["context"]["clinical_state"]) == {
        "presentation", "condition", "severity_or_complication"
    } for v in variants)
    assert all(set(v["context"]["imaging_history"]) == {
        "prior_test", "prior_result", "source_phrases"
    } for v in variants)
    assert all(set(v["context"]["modifiers"]) == {
        "population", "timing", "constraints_or_confounders"
    } for v in variants)
    assert all(set(v["context"]["decision_stage"]) == {
        "imaging_stage", "encounter_status", "source_phrase"
    } for v in variants)
    assert all("fever" not in v["context"]["clinical_state"]["presentation"]
               for v in variants if "abrupt change in fever curve" in v["variant_text"])
    first_time = next(v for v in variants if "First time presentation" in v["variant_text"])
    assert first_time["context"]["modifiers"]["population"] == []
    assert first_time["context"]["decision_stage"]["encounter_status"] == ["first time presentation"]
    assert Counter(v["context"]["decision_stage"]["imaging_stage"] for v in variants) == {
        "initial": 10, "next": 3, "unspecified": 4,
    }
    relationships = [r for v in variants for r in v["action_relationships"]]
    assert Counter(r["relationship"] for r in relationships) == {
        "equivalent_alternatives": 4, "complementary": 4,
    }
    for variant in variants:
        valid_ids = {a["action_id"] for a in variant["actions"]}
        assert all(set(r["action_ids"]).issubset(valid_ids) for r in variant["action_relationships"])
    assert Counter(a["appropriateness_category"] for a in actions) == {
        "May be appropriate": 70,
        "Usually not appropriate": 43,
        "Usually appropriate": 27,
        "May be appropriate (Disagreement)": 1,
    }
    for action in actions:
        category = action["appropriateness_category"]
        assert "rating" not in action
        rating = action["final_rating"]
        assert (rating <= 3 and category == "Usually not appropriate") or (
            4 <= rating <= 6 and category.startswith("May be appropriate")
        ) or (rating >= 7 and category == "Usually appropriate")
    forbidden = {"assumption", "question", "coverage", "a_t", "q_t", "c_t"}
    for value in (corpus, rows):
        stack = [value]
        while stack:
            item = stack.pop()
            if isinstance(item, dict):
                assert forbidden.isdisjoint({str(k).lower() for k in item})
                stack.extend(item.values())
            elif isinstance(item, list):
                stack.extend(item)
    for source in manifest["sources"]:
        path = DATA / source["file"]
        assert path.is_file()
        assert path.stat().st_size == source["bytes"]
        assert digest(path) == source["sha256"]
    print("PASS: 4 topics, 17 variants, 141 actions, 90 rationales; provenance and source hashes valid.")


if __name__ == "__main__":
    main()
