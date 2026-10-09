# Files that are not in git

Everything needed to run is in this repo **except** the files below. They are too large for GitHub (100 MB limit)
or are built from real caller transcripts, so they ship in **CallAI_OTR_portal.zip**
(SHA-256 4931343121294b7a0103b8ef44999d6d49bfcdaceb247c2699cbd4e269705088). Copy them from the zip
(paths are the same inside `CallAI_OTR/`) before running:

| Path | Size | What it is | Without it |
|---|---|---|---|
| `models/whisper_otr_ft/model.safetensors` | 1.5 GB | fine-tuned Whisper weights (the tokenizer/config files next to it are in git) | transcription cannot run |
| `code/models/gemini_cls.pkl` | 127 MB | category / sub-category classifier trained on Gemini's portal labels | falls back to the older LLM categoriser (a WARNING is printed): ~35% sub-category agreement with Gemini instead of ~53% |
| `code/data/otr/` | 5 MB | labelled example calls (caller transcripts) | only the fallback categoriser needs them |
| `code/out/emb_cache/rows/` | 5 MB | bge-m3 embeddings of those examples | only the fallback categoriser needs them |

Already in git: `code/models/turns_model.pkl`, `turns_v2.pkl`, `language_clf.pkl`, `agent_score.json`,
`romanize_lexicon.json`, `turns_eval.txt`. llama.cpp, Qwen3-4B and bge-m3 are downloaded by `python setup.py`.

**Quick check without a GPU** (works on a Mac): copy `gemini_cls.pkl` into `code/models/`, then
`pip install -r requirements.txt` and `python test_classifier.py`. It classifies three made-up sample calls on the CPU.
The full pipeline (Whisper + llama-server) needs a Linux or Windows machine with an NVIDIA GPU (tested on a T4).
