# CallAI OTR: call analysis package for the Verification Portal

Analyses UPESSC OTR helpline recordings and returns the portal's own columns: transcript with AGENT / CANDIDATE
turns, Call Category, Sub-Category, AI Insight, Summary, Call Resolved, Agent Score (1-10), Audio Rating, Language.
Everything runs on one GPU server. No external API is called and no call data leaves the machine.

## What happens to a recording

```
recording (.wav, any sample rate)
  1. Silero VAD splits it into speech segments (<= 25 s, roughly one speaker turn each)
  2. fine-tuned Whisper-turbo (models/whisper_otr_ft) transcribes every segment (Hindi / Hinglish, romanised)
  3. speaker roles: each segment -> AGENT / CANDIDATE / SYSTEM (IVR) from its words + voice features
     (code/models/turns_v2.pkl; falls back to the words-only turns_model.pkl if v2 is absent)
  4. category + sub-category: a classifier trained on ~7,200 calls the portal's Gemini labelled (text SVM + bge-m3
     embeddings + nearest labelled calls, combined by a gradient-boosted re-ranker; code/models/gemini_cls.pkl)
  5. QA: Qwen3-4B scores the agent on a fixed rubric against the official approved answer for that sub-category
     -> Agent Score, Call Resolved, AI Insight, Summary, Language
  6. Audio Rating from signal statistics (words per speech-minute, empty segments)
```

Calls with fewer than 20 transcribed words get Call Category "Audio Unclear"; recordings that cannot be downloaded
or decoded get "Recording Unavailable" (both also in Tags, Sub-Category blank), as the portal does today.

## Requirements

- NVIDIA GPU with 24 GB (tested: AWS g5.xlarge, A10G). 16 GB works for the API with OTR_LLM_SLOTS=2.
- CUDA driver 12.4 or newer, Python 3.11, 3.12 or 3.13, Windows or Linux x64.
- About 15 GB of disk (Whisper 1.6 GB, Qwen3-4B 2.5 GB, bge-m3 2.3 GB, llama.cpp, Python packages).

## Setup (once)

```
unzip CallAI_OTR_portal.zip && cd CallAI_OTR
python -m venv .venv && . .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python setup.py           # downloads llama.cpp (CUDA), Qwen3-4B (sha256-checked) and bge-m3; checks the GPU
```

## Use 1: HTTP API (portal integration)

```
python serve.py --port 8700
```

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | /health | | `{"ok": true}` |
| POST | /process | `{"call_id": "123", "recording_url": "https://.../x.wav"}` | one result object |
| POST | /process_batch | `{"calls": [{"call_id", "recording_url"}, ...]}` (max 50) | `{"results": [...]}` same order |

Requests are processed one at a time (a lock guards the GPU); send batches for throughput. A call takes a few
seconds of GPU time once both models are loaded; start-up loads them (about 1 minute).

Example result:

```json
{
  "call_id": "123",
  "Call Category": "Registration & Portal",
  "Sub-Category": "Mobile OTP Not Received",
  "Tags": "",
  "AI Insight": "OTP not received on registered mobile",
  "Summary": "The caller is not getting the OTP for OTR login. The agent asked them to retry after 30 minutes ...",
  "Call Resolved": "Yes",
  "Agent Score": 8,
  "Audio Rating": "Good",
  "Language": "Hindi",
  "Transcription": "AGENT: namaskar UPESSC helpline ...\nCANDIDATE: sir mera OTP nahi aa raha ...",
  "Resolution Source": "AI",
  "Bug Category": "", "Bug Description": "",
  "category_id": "registration_and_portal", "sub_category_id": "mobile_otp_not_received", "confidence": 0.86, "needs_review": false,
  "score_items": {"greeting": 2, "understanding": 2, "correctness": 2, "next_steps": 1, "closing": 1}
}
```

## Use 2: a whole call report (fastest per call)

```
python run_batch.py --report call-report-2026-09-25-to-2026-10-06.xlsx   [--out result.xlsx] [--limit N]
```

Reads the portal's call-report export (needs its recording URL column), downloads the recordings (16 at a time),
transcribes everything, frees Whisper, then runs the LLM with 8 parallel slots, and writes
`<report>_CallAI.xlsx`: the same 31 columns as the portal export, sheet "Calls", with our columns filled in.
It is resumable: progress is kept in `work/` (transcripts.jsonl, results.jsonl), so re-running the same command
continues where it stopped.

## Use 3: from Python

```python
from otr_pipeline import OTRPipeline
p = OTRPipeline()
rows = p.process_many(["a.wav", "b.wav"])      # one dict per recording, same keys as the API result
p.close()                                      # stops the llama-server it started
```

## Output fields

| Portal column | How it is produced | Values |
|---|---|---|
| Transcription | Whisper per segment, romanised, one line per turn | `AGENT: ...` / `CANDIDATE: ...` / `SYSTEM: ...` |
| Call Category / Sub-Category | classifier trained on Gemini's portal labels | the OTR taxonomy (code/products/otr/taxonomy.yaml) |
| Agent Score | 5 rubric items + call resolved + audio, calibrated to Gemini's scale | 1-10 (mostly 7-8, like the portal) |
| Call Resolved | Qwen3-4B | Yes / Partial / No |
| AI Insight | Qwen3-4B, caller's core issue | <= 8 words, English |
| Summary | Qwen3-4B | 1-2 English sentences |
| Language | word classifier fitted on Gemini's labels (Qwen3-4B decides English) | Hindi / Hinglish / English / "Hindi, English" |
| Audio Rating | rule on words per speech-minute and empty segments | Good / Moderate / Poor |
| Resolution Source | constant | AI |

Agent Score: Qwen3-4B rates the agent on five items (each 0 / 1 / 2): greeting and courtesy; understanding the
caller's problem; correctness against the approved answer (code/products/otr/playbooks.yaml); clear next steps;
proper closing. Gemini's portal scores use a different scale (almost always 7 or 8, lower when the call was not
resolved or the audio is poor), so the items, Call Resolved, Audio Rating and call length are mapped to Gemini's scale
by a small model fitted on 629 calls Gemini scored (code/models/agent_score.json). `score_items` and
`agent_score_raw` (the plain 1-10 item sum) are returned too, so a score can be explained.

Extra keys (not portal columns): `category_id`, `sub_category_id`, `confidence` (0-1), `needs_review` (true when
confidence < 0.4: route these to a human), `alternatives` (the next two most likely sub-categories), `score_items`,
`agent_score_raw`, `error`.

## Accuracy (measured)

- Transcription: on 300 held-out calls 24.1% of words differ from the portal's Gemini transcripts (both are imperfect;
  most differences are spelling of romanised Hindi).
- Category / sub-category, same answer as Gemini on 631 later calls (25 Sep report, transcribed by this package, never
  used in training): category 79.1%, sub-category 53.2%; Gemini's label is among our top 3 on 76.5%. With confidence
  >= 0.5 (64% of calls) sub-category agreement is 66%. (The previous LLM-based categoriser: 75.9% / 35.3%.)
  Agreement with Gemini by confidence (same 631 calls); set OTR_REVIEW_CONF to choose which calls go to a person:

  | confidence | share of calls | sub-category = Gemini | category = Gemini |
  |---|---|---|---|
  | all | 100% | 53% | 79% |
  | >= 0.4 (default review threshold) | 81% | 60% | 81% |
  | >= 0.6 | 47% | 70% | 85% |
  | >= 0.7 | 30% | 78% | 86% |
  | >= 0.8 | 17% | 83% | 89% |
  | >= 0.9 | 8% | 86% | 94% |
- Agent Score vs Gemini's on 629 calls (cross-validated): same score 49%, within 1 point 91% (always answering 7
  would give 46% / 88%). Call Resolved: same answer as Gemini on 82% of calls.
- Speaker roles, on 300 test calls against the portal's Gemini transcript: 80.6% of words get the same speaker
  (words-only model: 73.9%). Ceiling 88.1%, because some speech segments contain both speakers. Of the agent's
  words, 77% are labelled AGENT; of the caller's words, 84% are labelled CANDIDATE. Details: `code/models/turns_eval.txt`.

## Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| OTR_LLM_SLOTS | 8 (batch), 4 (API) | parallel LLM requests; lower it on a 16 GB GPU |
| OTR_REVIEW_CONF | 0.4 | calls whose category confidence is below this get needs_review = true |
| OTR_LLAMA_PORT | 8080 | port of the llama-server the package starts on 127.0.0.1 |
| CALLAI_LLM_URL | | use an already-running OpenAI-compatible llama-server instead of starting one |
| HF_HOME | ./hf | where bge-m3 is cached |

## Package layout

```
CallAI_OTR/
  otr_pipeline.py  serve.py  run_batch.py  setup.py  requirements.txt  README.md
  models/whisper_otr_ft/           fine-tuned Whisper-turbo
  code/callai/                     pipeline library (categoriser, retrieval, speaker roles, romaniser)
  code/products/otr/               taxonomy, groups, prompts, approved answers (playbooks.yaml)
  code/data/otr/                   human-labelled example calls used for retrieval
  code/models/                     speaker-role models, romanisation lexicon
  code/out/emb_cache/rows/         pre-computed bge-m3 embeddings of the example calls
  llama/ models/Qwen3-4B...gguf hf/   created by setup.py
  logs/ work/                      created at run time
```

## Limits

- Hindi / Hinglish / English only. Other languages get a transcript of poor quality.
- Throughput on one T4 (16 GB, slower than the A10G): 631 calls (avg 2.3 min) in 107 min, i.e. ~6 calls/min end to end. The API took 38 s for a first single call and 11 s for a 2-call batch.
- Audio Rating is a signal rule (words per speech-minute, empty segments): it flags dead / garbled calls as Poor but
  does not hear background noise the way Gemini does, so it says Good more often than Gemini (Gemini: 71% Good).
- Language: Gemini's Hindi vs Hinglish choice is hard to reproduce from a transcript; ours agrees on ~68% of calls.
- Speaker roles are automatic: a few lines per call can carry the wrong role, mostly very short turns ("haan ji").
- Sub-category is the hardest field (196 classes, many close). Use `needs_review` to send uncertain calls to a person.
- Transcripts and recordings contain callers' personal data: keep `work/` and `logs/` on the server.
