"""callai — one pipeline for otrcallai and connectviewcallai.

    recording → audio (split channels, normalise) → ASR per channel → dialogue
              → segregate (issue / sub_issue from taxonomy.yaml)
              → respond (OTR only: playbook-grounded answer, optional TTS)

Everything model-related sits behind a small backend class so the same code runs on a
Kaggle T4 today and on a vLLM box later.
"""

from .pipeline import Pipeline, PipelineConfig  # noqa: F401

__version__ = "0.1.0"
