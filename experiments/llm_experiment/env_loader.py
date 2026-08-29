"""Load OPENROUTER_API_KEY / TINKER_API_KEY from a repo-root env file if not already in env."""
from __future__ import annotations
import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_ENV_FILE = _ROOT / ".openrouter_env"
_TINKER_ENV_FILE = _ROOT / ".tinker_env"


def load_tinker_key() -> str:
    """Mirror of load_openrouter_key for TINKER_API_KEY (read .tinker_env, never source it)."""
    if os.environ.get("TINKER_API_KEY"):
        return os.environ["TINKER_API_KEY"]
    if _TINKER_ENV_FILE.exists():
        for line in _TINKER_ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                if k.strip() == "TINKER_API_KEY":
                    v = v.strip().strip('"“”').strip("'‘’")
                    os.environ["TINKER_API_KEY"] = v
                    return v
    raise RuntimeError(f"TINKER_API_KEY not in env and not found in {_TINKER_ENV_FILE}")


def load_openrouter_key() -> str:
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    if _ENV_FILE.exists():
        for line in _ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                if k.strip() == "OPENROUTER_API_KEY":
                    v = v.strip().strip('"“”').strip("'‘’")
                    os.environ["OPENROUTER_API_KEY"] = v
                    return v
    raise RuntimeError(
        f"OPENROUTER_API_KEY not in env and not found in {_ENV_FILE}"
    )
