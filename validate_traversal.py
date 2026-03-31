"""validate_traversal.py

Runs every toy-patient scenario through the traversal engine and
asserts that the terminal nodes match the expected values in
toy_patients.EXPECTED_TERMINALS.

Usage:
    python validate_traversal.py           # compact pass/fail summary
    python validate_traversal.py --verbose # full per-step report
"""

from __future__ import annotations

import sys

from traversal_engine import run_full_traversal, print_traversal_report
from toy_patients import SCENARIOS, EXPECTED_TERMINALS


# ---------------------------------------------------------------------------
# Disease that owns each scenario (used to look up the right sub-rubric result)
# ---------------------------------------------------------------------------

_SCENARIO_DISEASE: dict[str, str] = {
    "C1_cholecystitis_mild":            "cholecystitis",
    "C2_cholecystitis_moderate":        "cholecystitis",
    "P1_pancreatitis_mild":             "pancreatitis",
    "P2_pancreatitis_severe":           "pancreatitis",
    "D1_diverticulitis_uncomplicated":  "diverticulitis",
    "D2_diverticulitis_hinchey2":       "diverticulitis",
}


def run_validation(verbose: bool = False) -> bool:
    """
    Returns True if every assertion passes, False otherwise.
    """
    total = 0
    passed = 0
    failed_cases: list[str] = []

    for scenario_name, steps in SCENARIOS.items():
        disease = _SCENARIO_DISEASE[scenario_name]
        expected_steps = EXPECTED_TERMINALS.get(scenario_name, {})

        for step_name, features in steps.items():
            total += 1
            full = run_full_traversal(features)
            result = full.diseases[disease]

            expected_terminal = expected_steps.get(step_name, "NOT_IN_EXPECTED")
            actual_terminal = result.terminal_node   # None = in progress

            ok = (actual_terminal == expected_terminal)

            status = "PASS" if ok else "FAIL"
            line = (
                f"  [{status}] {scenario_name} / {step_name}"
                f"  expected={expected_terminal!r}  got={actual_terminal!r}"
            )
            print(line)

            if verbose and not ok:
                print_traversal_report(full)

            if ok:
                passed += 1
            else:
                failed_cases.append(f"{scenario_name}/{step_name}")

    print(f"\n{'─'*60}")
    print(f"  Results: {passed}/{total} passed")
    if failed_cases:
        print(f"  Failed : {', '.join(failed_cases)}")
    print(f"{'─'*60}\n")

    return passed == total


def demo_one(scenario_name: str, step_name: str) -> None:
    """Print the full traversal report for one scenario/step combination."""
    steps = SCENARIOS[scenario_name]
    features = steps[step_name]
    full = run_full_traversal(features)
    print_traversal_report(full)


if __name__ == "__main__":
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    demo = "--demo" in sys.argv

    if demo:
        # Quick visual demo: C1 cholecystitis at step2 (terminal expected)
        print("\n── DEMO: C1_cholecystitis_mild / step2 ─────────────────────────")
        demo_one("C1_cholecystitis_mild", "step2")
        print("\n── DEMO: P2_pancreatitis_severe / step3 ─────────────────────────")
        demo_one("P2_pancreatitis_severe", "step3")
    else:
        success = run_validation(verbose=verbose)
        sys.exit(0 if success else 1)
