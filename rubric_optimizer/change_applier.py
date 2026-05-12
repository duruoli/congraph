"""change_applier.py — Apply and revert LLM-proposed code changes.

Two public functions:

    apply_change(change)  → (success: bool, error: str)
        Replaces change.old_code with change.new_code in change.target_file.
        Does a verbatim substring replacement (safe: verifies exactly one match).

    revert_change(change) → (success: bool, error: str)
        Runs `git checkout <target_file>` from the project root.
        Returns (True, "") on success, (False, error_message) on failure.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_PKG  = Path(__file__).parent
_ROOT = _PKG.parent
for _p in [str(_ROOT), str(_PKG)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from trial_log import ChangeRecord

_ALLOWED_FILES = {"rubric_graph.py", "diagnosis_distribution.py"}


# ---------------------------------------------------------------------------
# Indentation correction helpers
# ---------------------------------------------------------------------------

def _add_indent(code: str, indent: int) -> str:
    """Add `indent` spaces to every non-empty line."""
    if indent == 0:
        return code
    prefix = " " * indent
    lines = code.split("\n")
    return "\n".join(prefix + line if line else line for line in lines)


def _find_with_indent_correction(text: str, old_code: str) -> tuple[str, int] | None:
    """
    YAML `|` block scalars strip the common leading whitespace from every line.
    This means code that lives 8 spaces deep in a function body (inside `edges = [`)
    will have those 8 spaces removed before we search for it in the file.

    This function tries adding 0, 4, 8, 12, 16, 20 spaces back to every line
    and returns (corrected_old_code, indent_added) on the first match, or None.
    """
    # Strip trailing newline YAML always appends
    base = old_code.rstrip("\n")

    for indent in range(0, 25, 4):
        candidate = _add_indent(base, indent)
        # Try exact match, then with trailing newline
        for probe in (candidate, candidate + "\n"):
            if probe in text:
                return probe, indent

    return None


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------

def apply_change(change: ChangeRecord) -> tuple[bool, str]:
    """
    Replace change.old_code with change.new_code in change.target_file.

    Returns (True, "") on success, (False, error_message) on failure.

    Safety checks
    -------------
    - target_file must be rubric_graph.py or diagnosis_distribution.py
    - old_code must appear exactly once in the file
    - YAML `|` block scalars strip leading indentation; this function
      automatically detects the stripped indent level and adds it back
      to both old_code and new_code before the replacement.
    """
    if change.target_file not in _ALLOWED_FILES:
        return False, (
            f"Unsafe target_file {change.target_file!r}. "
            f"Must be one of {_ALLOWED_FILES}."
        )

    path = _ROOT / change.target_file
    if not path.exists():
        return False, f"File not found: {path}"

    text     = path.read_text(encoding="utf-8")
    old_code = change.old_code
    new_code = change.new_code

    if old_code not in text:
        # 1. Try stripping the trailing newline YAML always appends
        stripped = old_code.rstrip("\n")
        if stripped and stripped in text:
            old_code = stripped
        else:
            # 2. Try re-adding the indentation YAML stripped
            result = _find_with_indent_correction(text, old_code)
            if result is None:
                return False, (
                    f"old_code not found in {change.target_file} "
                    f"(tried verbatim + trailing-newline strip + indent correction "
                    f"0/4/8/12/16/20 spaces).\n"
                    f"Searched for:\n{old_code!r}\n\n"
                    f"Hint: old_code must be an exact substring of the file, "
                    f"character-for-character."
                )
            old_code, indent_added = result
            # Apply the same indent correction to new_code so the replacement
            # is syntactically valid at the same nesting depth.
            new_code = _add_indent(new_code.rstrip("\n"), indent_added)
            # Persist corrected values so callers (e.g. revert) see them
            change.old_code = old_code
            change.new_code = new_code

    count = text.count(old_code)
    if count > 1:
        return False, (
            f"old_code matches {count} locations in {change.target_file} "
            f"— replacement would be ambiguous. Provide more context."
        )

    new_text = text.replace(old_code, new_code, 1)
    path.write_text(new_text, encoding="utf-8")
    return True, ""


# ---------------------------------------------------------------------------
# Revert
# ---------------------------------------------------------------------------

def revert_change(change: ChangeRecord) -> tuple[bool, str]:
    """
    Revert change.target_file to its last committed state using git checkout.

    Returns (True, "") on success, (False, error_message) on failure.
    """
    if change.target_file not in _ALLOWED_FILES:
        return False, f"Unsafe target_file {change.target_file!r}."

    try:
        result = subprocess.run(
            ["git", "checkout", change.target_file],
            cwd         = str(_ROOT),
            capture_output = True,
            text        = True,
        )
        if result.returncode != 0:
            msg = (result.stderr or result.stdout).strip()
            return False, msg or f"git checkout exited {result.returncode}"
        return True, ""
    except FileNotFoundError:
        return False, "git not found — is it installed and on PATH?"
    except Exception as e:
        return False, str(e)
