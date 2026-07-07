"""Model-family chat-template compatibility for the certainty-agent scripts.

The SFT / eval / inspector scripts run on a SWAPPABLE base model — Qwen2.5-7B was the
placeholder; medgemma-27b-text-it (a Gemma-3 model) is the successor. Two template facts
bite when you swap the base, and this module makes both fail loud instead of silent:

  1. assistant-only loss (train_lora_qwen.py). TRL's `assistant_only_loss=True` needs the
     chat template to delimit the assistant span with a `{% generation %}...{% endgeneration %}`
     block (transformers detects it via the regex below). Qwen2.5's stock template has it;
     stock Gemma-3 / medgemma does NOT. Training on the un-marked template would either error
     or train on the WHOLE prompt (the ~13.5k-char rubric drowns the loss signal). For a
     Gemma-family base we swap in configs/gemma3_assistant_loss_template.jinja, which is the
     stock Gemma-3 template with a generation block wrapped around the model turn.

  2. system role. Gemma-3 has no native system turn; its template folds a leading system
     message into the first user turn. Both Qwen and Gemma templates accept a leading system
     message, so inference is unaffected — but we assert the round-trip so a bad swap (a base
     whose template rejects the system role) fails immediately.

Detection uses the SAME regex transformers uses, so whitespace-controlled tags
(`{%- generation -%}`) count as present.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GEMMA_ASSIST_TEMPLATE = ROOT / "configs" / "gemma3_assistant_loss_template.jinja"

# same pattern transformers uses to decide a template supports return_assistant_tokens_mask
_GENERATION_RE = re.compile(r"\{\%-?\s*generation\s*-?\%\}")


def is_gemma(model_name: str = "", tok=None) -> bool:
    """Gemma-family = medgemma / gemma id, or a template that speaks Gemma turn tokens."""
    name = (model_name or "").lower()
    if "gemma" in name:  # matches "gemma" and "medgemma"
        return True
    tmpl = (getattr(tok, "chat_template", None) or "") if tok is not None else ""
    return "<start_of_turn>" in tmpl


def supports_assistant_mask(tok) -> bool:
    tmpl = getattr(tok, "chat_template", None) or ""
    return bool(_GENERATION_RE.search(tmpl))


def ensure_assistant_loss_template(tok, model_name: str = "", template_path=None) -> bool:
    """Guarantee the tokenizer template can mask the assistant span for assistant_only_loss.

    Returns True if a generation-marked template was swapped in, False if the stock template
    already supports it. Raises SystemExit for a base whose family we can't auto-fix (pass an
    explicit --chat-template, or turn assistant_only_loss off to train on the full sequence).
    """
    if supports_assistant_mask(tok):
        return False
    path = (Path(template_path) if template_path
            else GEMMA_ASSIST_TEMPLATE if is_gemma(model_name, tok) else None)
    if path is None or not path.exists():
        raise SystemExit(
            f"[chat-compat] base '{model_name}' chat template has no {{% generation %}} marker, "
            f"so assistant_only_loss cannot delimit the target. Pass --chat-template <path> to a "
            f"generation-marked template (see configs/gemma3_assistant_loss_template.jinja for the "
            f"Gemma-3 pattern), or --no-assistant-only-loss to train on the full sequence.")
    tok.chat_template = path.read_text()
    return True


def validate_assistant_mask(tok, messages) -> None:
    """Assert the (possibly-swapped) template yields a NON-TRIVIAL assistant mask on a real
    example: at least one masked prompt token (0) AND one kept assistant token (1). Catches a
    broken template BEFORE the expensive training run rather than silently learning on the
    wrong tokens. `messages` = one training example's [system, user, assistant]."""
    enc = tok.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_assistant_tokens_mask=True)
    mask = enc.get("assistant_masks")
    if not mask or 1 not in mask or 0 not in mask:
        raise SystemExit(
            "[chat-compat] assistant-token mask is trivial (all 0s or all 1s): the chat template "
            "does not correctly delimit the assistant turn, so assistant_only_loss would train on "
            "the wrong tokens. Fix the {% generation %} block in the template.")
    kept = sum(mask)
    print(f"[chat-compat] assistant mask OK — {kept}/{len(mask)} tokens are the assistant target")


def assert_roundtrip(tok, messages) -> None:
    """Inference-side sanity: the template applies to [system, user, ...] and adds a generation
    prompt. Fails loud on a base whose template rejects the system role or otherwise breaks."""
    try:
        tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    except Exception as e:  # noqa: BLE001 — surface any template failure as a clear abort
        raise SystemExit(f"[chat-compat] chat template failed to apply to the messages: {e}")
