"""Print this OpenRouter key's spend (daily/weekly/monthly/all-time) + remaining limit.

Usage: python3 scripts/check_openrouter_usage.py
Key is loaded via experiments.llm_experiment.env_loader (.openrouter_env).
"""
import sys, json, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.llm_experiment.env_loader import load_openrouter_key


def _get(url: str, key: str) -> dict:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)
    except Exception as e:
        body = e.read().decode("utf-8", "ignore") if hasattr(e, "read") else ""
        return {"_error": str(e), "_body": body}


def main():
    key = load_openrouter_key()
    d = _get("https://openrouter.ai/api/v1/auth/key", key).get("data", {})
    if not d:
        print("could not read key usage:", _get("https://openrouter.ai/api/v1/auth/key", key))
        return
    limit = d.get("limit")
    remaining = d.get("limit_remaining")
    print(f"OpenRouter key {d.get('label')}")
    print(f"  today   : ${d.get('usage_daily', 0):.2f}")
    print(f"  week    : ${d.get('usage_weekly', 0):.2f}")
    print(f"  month   : ${d.get('usage_monthly', 0):.2f}")
    print(f"  all-time: ${d.get('usage', 0):.2f}")
    if limit is not None:
        print(f"  limit   : ${limit:.2f}  (remaining ${remaining:.2f})")
    else:
        print(f"  limit   : none set  (all-time spend ${d.get('usage', 0):.2f})")


if __name__ == "__main__":
    main()
