# Model and data files (not in git)

This branch holds the code only. The trained models and the labelled example data contain information derived from
caller recordings, and the Whisper model is 1.5 GB, so they are shipped separately in **CallAI_OTR_portal.zip**
(SHA-256 4931343121294b7a0103b8ef44999d6d49bfcdaceb247c2699cbd4e269705088). Copy these folders from the zip into
this folder before running `python setup.py`:

| From the zip (CallAI_OTR/...) | Put here | What it is |
|---|---|---|
| models/whisper_otr_ft/ | models/whisper_otr_ft/ | fine-tuned Whisper (transcription) |
| code/models/ | code/models/ | category classifier, speaker roles, agent-score calibration, language, romaniser lexicon |
| code/data/otr/ | code/data/otr/ | labelled example calls (fallback categoriser only) |
| code/out/emb_cache/ | code/out/emb_cache/ | pre-computed embeddings of those examples |

Everything else (llama.cpp, Qwen3-4B, bge-m3) is downloaded by `python setup.py`. See README.md.
