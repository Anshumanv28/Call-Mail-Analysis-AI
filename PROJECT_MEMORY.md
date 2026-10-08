# CallAI OTR: project memory

Shared, persistent context for every session and every agent (Claude Code, subagents, other AI tools, humans).
**Read this before working. Update it after any change** (see "How to keep this file" at the bottom).
This folder is a **git repository** (since 2026-10-08, branch `main`, no remote yet). Git records *what* changed;
the Change log below records *why*, and context git cannot hold.

Last updated: 2026-10-08

---

## 1. What the project is

An offline call-analysis package for the **UPESSC OTR (One Time Registration) helpline** in Uttar Pradesh, built to
fill the **Verification Portal's** AI columns. It replaces or sits beside the portal's current Gemini-based output.
It runs entirely on one GPU server: no external API, no call data leaves the machine.

Input: call recordings (.wav). Output, using the portal's own column names: Transcription (AGENT / CANDIDATE / SYSTEM
turns), Call Category, Sub-Category, Tags, AI Insight, Summary, Call Resolved, Agent Score (1-10), Audio Rating,
Language, Resolution Source, plus extra keys (`category_id`, `sub_category_id`, `confidence`, `needs_review`,
`score_items`, `agent_score_raw`, `error`).

Full user-facing docs are in `README.md`. This file holds the context README does not.

## 2. Pipeline (otr_pipeline.py, `OTRPipeline`)

1. Silero VAD: speech segments of 25 s or less (`callai.audio.speech_chunks`)
2. Fine-tuned Whisper-turbo `models/whisper_otr_ft`: romanised Hindi / Hinglish per segment; `callai.loops.collapse_loops` removes repetition loops
3. Speaker roles: `callai.turns2.TurnLabeller2` (words + voice features, `code/models/turns_v2.pkl`); falls back to `callai.turns.TurnLabeller` (`turns_model.pkl`, words only)
4. Category + sub-category: Qwen3-4B-Instruct-2507 Q4_K_M via llama.cpp `llama-server`, with 6 retrieved human-labelled calls (bge-m3, `callai.retrieve.ExamplePool`, recipe `RETRIEVAL` constant). Goes through `callai.Pipeline` / `segregate.py`
5. QA: Qwen3-4B scores 5 rubric items (greeting, understanding, correctness vs `playbooks.yaml`, next_steps, closing; each 0-2) plus Call Resolved, AI Insight, Summary, Language. `QA_SYSTEM` prompt in `otr_pipeline.py`
6. Agent Score is calibrated to Gemini's scale with `code/models/agent_score.json`; Language uses `code/models/language_clf.pkl`; Audio Rating is a signal rule
7. Fewer than 20 words gives "Audio Unclear"; a download or decode failure gives "Recording Unavailable"

Batch mode runs the stages in sequence (transcribe all, free Whisper, then the LLM with 8 slots), because everything
together does not fit 24 GB. The API keeps both loaded with 4 slots.

## 3. Entry points

| File | Purpose |
|---|---|
| `otr_pipeline.py` | `OTRPipeline`: the whole pipeline; `process_many(paths)`, `close()` |
| `serve.py` | stdlib HTTP API on port 8700: `/health`, `/process`, `/process_batch` (50 max); a lock serialises GPU work |
| `run_batch.py` | portal call-report .xlsx in, `<report>_CallAI.xlsx` out (same 31 columns, sheet "Calls"); resumable via `work/` |
| `setup.py` | one-time download of llama.cpp (tag `b11064`), Qwen3-4B (sha256-pinned) and bge-m3, plus a GPU check |

## 4. Layout and which files are live

- `models/whisper_otr_ft/`: fine-tuned Whisper (do not modify)
- `code/callai/`: library. **Used by the OTR pipeline:** audio, loops, turns2, turns, llm, retrieve, pipeline,
  segregate, taxonomy, normalise. **Carried over and not on the OTR path:** asr (faster-whisper / Qwen3-ASR /
  IndicConformer backends), dialogue (stereo per-channel, from ConnectView), route, polish, hinglish, romanize,
  devanagari, respond, reply_prompt, tts, voices, windows. Check imports before assuming a module is unused.
- `code/products/otr/`: `taxonomy.yaml` (5 categories / 196 sub-categories, from "Innovatiview-Complaint-Taxonomy.pdf"),
  `groups.yaml`, `product.yaml` (classifier context and disambiguation rules), `playbooks.yaml` (approved answers).
  `taxonomy_v1.yaml`, `playbooks_connectview_backup.yaml` and `kb_extra_lines.json` are not referenced by code.
- `code/data/otr/`: retrieval pool (`pool_transcripts.jsonl`, `pool_human_labels.csv`). This is the pool the model
  learns from; it must never contain calls that are being evaluated.
- `code/models/`: turns_v2.pkl, turns_model.pkl, agent_score.json, language_clf.pkl, romanize_lexicon.json, turns_eval.txt
- `code/out/emb_cache/rows/`: cached bge-m3 embeddings of the pool. **Rebuild it if the pool changes.**
- `docs/reference/`: reference material from other projects, kept as-is for comparison (do not edit)
- Created at runtime: `llama/`, `hf/`, `models/*.gguf`, `logs/`, `work/`

**Not in git (see `.gitignore`), so these must be copied separately when deploying or cloning:**
- `models/whisper_otr_ft/model.safetensors` (1.5 GB, over GitHub's 100 MB limit). The config and tokenizer files are tracked.
- `code/data/otr/` (real caller transcripts and labels, so personal data) and `code/out/emb_cache/` (embeddings derived from them). The user chose to keep these out of git.
- Everything created by `setup.py` or at run time, plus call-report .xlsx files and recordings.

## 5. Context not visible in the code

- This folder is a **deployment package** cut from a larger development workspace. Training and fitting scripts
  referenced in docstrings (`scripts/build_turns.py`, `build_turns_v2.py`, `fit_agent_score.py`, `build_lexicon.py`,
  `kb_playbooks.py`, `kb_bot.py`, `build_reply_data.py`, `voice_role_model.py`) are **not here**. Earlier work ran
  on Kaggle T4s. The `callai` library was shared with a sibling product, **ConnectView** (`connectviewcallai`).
- The reference labels are the portal's **Gemini** outputs. Several fields (Agent Score, Language) are deliberately
  fitted to match Gemini's habits, not an absolute ideal.
- Version pins matter: `scikit-learn==1.9.1` (pickled models), `transformers==4.57.6`, the Qwen GGUF sha256 (accuracy
  numbers were measured with that exact file). Retrain or re-measure if any of these change.
- Target hardware is an AWS g5.xlarge (A10G 24 GB); 16 GB works with `OTR_LLM_SLOTS=2`. Python 3.11 to 3.13, Windows or Linux.
- Development machine is macOS (no CUDA), so the GPU path cannot be run end to end locally.
- Privacy: recordings and transcripts hold callers' personal data. Keep `work/` and `logs/` on the server and never
  send call content to an external service.

## 6. Current accuracy (see README "Accuracy" for detail)

| Field | Result |
|---|---|
| Transcription | 24.5% words differ from Gemini's transcripts (held out) |
| Category / group / exact sub-category | 87.6% / 77.9% / 52.8% on 300 human-labelled calls (Gemini: 82.1 / 67.9 / 36.6) |
| Agent Score vs Gemini | same 49%, within 1 point 91% (baseline "always 7": 46 / 88) |
| Call Resolved vs Gemini | 82% agree |
| Speaker roles | 80.6% word agreement (v1 73.9%, ceiling 88.1%) |
| Language vs Gemini | about 68% |
| Throughput | T4: 631 calls in 107 min (about 6 per minute) |

## 6b. Comparison baseline: Gemini cost

`docs/reference/gemini-cost-breakdown-2026-10-08.md` (copied from oasis-call-center/docs) has the portal's current
Gemini 2.5 Flash cost. This is the figure to compare CallAI's server cost against. Key numbers (₹88.40/$):
- Per call: ₹0.45 average / ₹0.53 higher-end for a 1-minute call, plus about ₹0.25 per extra minute. Real recordings
  average about 2 min, so about ₹0.66 to ₹0.78 per analysed call. "Audio Unclear" calls are paid in full.
- 12 to 21 Sep: about ₹265 a day for calls (about 425 answered calls a day); volume has since roughly doubled (about 886 a day, 25 Sep to 7 Oct).
- Gemini has refused every request since 25 Sep 2026 because the spend cap was reached.
- For comparison, CallAI throughput on a T4 is about 6 calls a minute end to end (section 6).

## 7. Known limits and open items

- Sub-category is the weakest field (196 close classes); `needs_review` routes uncertain calls to a person
- Audio Rating says "Good" more often than Gemini (it cannot hear background noise)
- Hindi vs Hinglish language agreement with Gemini is only about 68%
- Very short turns ("haan ji") often get the wrong speaker role
- No automated tests in the package

## 8. Change log

Newest first. Each entry gives the date, what changed, why, and the files touched.
Entries marked *(reconstructed)* were inferred from file modification times on 2026-10-08, before this log
existed, so treat their wording as approximate.

### 2026-10-08
- `git init` (branch `main`, no remote yet; the user will add one later) and added `.gitignore`. It excludes the Whisper weights, caller data (`code/data/otr/`, `code/out/`), setup/runtime outputs, recordings, call reports, Python/editor files and `.env`. Initial commit made.
- Copied `gemini-cost-breakdown-2026-10-08.md` from `/Users/archon/oasis-call-center/docs/` to `docs/reference/` as the cost baseline for comparing CallAI with Gemini (summary in section 6b).
- Created `PROJECT_MEMORY.md` (this file), with `CLAUDE.md` and `AGENTS.md` pointing to it, so all sessions and agents share context.
- *(reconstructed)* Speaker roles v2 retrained (`code/models/turns_v2.pkl` 13:36, `turns_eval.txt` 14:58; `callai/turns2.py` 11:24). Result: 80.6% word role agreement.
- *(reconstructed)* `callai/llm.py` updated (15:03).
- *(reconstructed)* `setup.py` and `requirements.txt` updated together (15:27): dependency pins and installer.
- *(reconstructed)* `kb_extra_lines.json` updated (16:45). It is not used by the runtime pipeline.
- *(reconstructed)* Agent Score calibration refit (`agent_score.json` 17:23) and language classifier refit (`language_clf.pkl` 17:25), then `otr_pipeline.py` (17:25) and `README.md` (17:26) updated to match.
- *(reconstructed)* `serve.py` and `run_batch.py` last edited (11:17).

### 2026-10-06
- *(reconstructed)* `callai/retrieve.py` updated (retrieval of few-shot examples).

### 2026-10-01
- *(reconstructed)* `kb_devanagari_lexicon.json`, `callai/devanagari.py`, `callai/reply_prompt.py`, `callai/windows.py` updated (reply-bot and TTS side; not on the OTR scoring path).

### 2026-09-30
- *(reconstructed)* `playbooks.yaml` (approved answers) and `product.yaml` rebuilt; ConnectView playbooks backed up to `playbooks_connectview_backup.yaml`.

### 2026-09-25 / 26
- *(reconstructed)* Taxonomy (`taxonomy.yaml`, `taxonomy_v1.yaml`, `groups.yaml`) and retrieval pool (`pool_transcripts.jsonl`, `pool_human_labels.csv`) created.

---

## How to keep this file

Every agent or session that changes the project must, before finishing:
1. Add a dated entry at the top of **Change log** (date, what changed, why, files touched, any new metric).
2. Update the other sections if they are no longer true (pipeline, live files, accuracy, open items).
3. Update "Last updated".

Use absolute dates (YYYY-MM-DD). Keep entries short. Record decisions and the reasons for them, including approaches
that were tried and rejected, because those are the hardest to recover later.
