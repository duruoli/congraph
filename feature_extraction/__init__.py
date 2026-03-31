"""feature_extraction — Algorithmic and LLM-based feature extraction.

Submodules
----------
algo_extractor   Extracts structured features from lab JSON + PE vitals regex.
llm_extractor    Extracts free-text features via GPT-4o (OpenAI API).
prompts          Prompt templates for all LLM extraction steps.
pipeline         Orchestrates both extractors into a step-by-step sequence.

Typical usage
-------------
    from openai import OpenAI
    from feature_extraction.pipeline import extract_patient_steps, extract_all_patients
"""

import os
import sys

# Ensure the workspace root (parent of this package) is on sys.path so that
# feature_schema, clinical_session, etc. can be imported from submodules.
_pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)
