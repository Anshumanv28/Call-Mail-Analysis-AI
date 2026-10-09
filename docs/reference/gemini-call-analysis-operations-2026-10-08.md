# What happens when a call is analysed with Gemini

Prepared 8 October 2026. This explains, step by step, everything the portal
does with Gemini for one call: what is sent, what Gemini is asked to do and
what comes back. The costs are in
[gemini-cost-breakdown-2026-10-08.md](gemini-cost-breakdown-2026-10-08.md);
this file points to the matching rows there.

**In short:** each answered call with a recording is **one paid request** to
Gemini (`gemini-2.5-flash`). The recording and the instructions go in
together. One answer comes back with all ten results listed under Step 3.
Every other step around that request is free.

---

## The whole operation at a glance

| # | Step | Does it talk to Gemini? | Paid? |
|---|---|---|---|
| 1 | Call arrives and is queued for analysis | No | No |
| 2 | Checks that decide whether the call needs Gemini at all | No | No |
| 3a | Recording is streamed from the IVR's storage to Gemini | Yes, file upload | No |
| 3b | **The analysis request: Gemini listens and answers** | **Yes** | **Yes, the only paid step** |
| 3c | The uploaded recording is deleted from Gemini | Yes, file delete | No |
| 4 | The answer is checked and tidied | No | No |
| 5 | Results are saved and shown on the screens | No | No |
| 6 | Retries, if something went wrong | Repeats 3a–3c | Yes, each repeat of 3b |
| 7 | Hourly bug-category clean-up (not per call) | Yes | Yes, small and rare |

---

## Step 1: The call is queued

When the IVR reports a finished call that has a recording link, the portal
saves the call and puts it in the analysis queue straight away. Calls also
enter the queue when an admin presses Re-analyse or Retry, or uses the bulk
"retry failed" action.

A background worker picks calls off the queue, up to **5 at a time** by
default, and checks for new work every 5 seconds.

## Step 2: Checks before anything is sent (free)

These calls never reach Gemini and **cost nothing**:

| Situation | What the call is marked |
|---|---|
| Missed call (nobody answered) | Not analysed; it has no recording |
| Call lasted 0 seconds | No audio |
| Recording link is not an audio file | Recording insufficient |
| Call shorter than 10 seconds | **Call too Short** |
| Recording file is empty (0 bytes) | No audio |
| Recording file is over 100 MB | Failed, not retried |
| Recording not found, call less than 90 minutes old | Asked for again later, because the IVR may still be uploading it |
| Recording not found, call older than that | **Recording Unavailable** |

*Cost doc reference:* "Missing recordings and missed calls never reach Gemini
and cost nothing."

## Step 3: The Gemini request

### 3a. Upload the recording (free)

The recording is streamed directly from the IVR's storage into Gemini's file
store. It is never saved on our server. Google does not charge for this
upload. The audio is billed only when Gemini listens to it in 3b.

### 3b. The analysis request (the paid step)

**What is sent in:**

| Input | Size | Cost doc row |
|---|---|---|
| The recording | 32 tokens per second of audio, so 1,920 for 1 minute | "Listening to 1 minute of recording" |
| Instructions, the full category list (5 categories, 196 sub-categories) and the bug-category list | about 4,809 tokens, the same for every call | "Instructions + category list sent with every call" |

**What Gemini is asked to do.** These are the ten tasks in the instructions,
all answered in one reply:

| # | Task | What comes back | Cost doc row |
|---|---|---|---|
| 1 | **Transcription.** Write out the whole call, labelling each turn CANDIDATE, AGENT or SYSTEM (IVR voice or hold music). Hindi and Hinglish are written in Roman letters. | The transcript | "Writing the transcript" (the largest part of the answer, about 69%) |
| 2 | **Tagging.** First decide whether the words can be made out. If yes, give one tag (category + sub-category) for each separate issue the caller raised, up to a limit, with the main issue first. | Category, sub-category, tags | "Category, sub-category, tags" |
| 3 | **Summary.** Two sentences: what the candidate asked, and what the agent did or said. | Summary | "Writing the summary" |
| 4 | **AI Insight.** A 4–5 word label for the caller's issue. | Insight | "Insight + issues/bugs noted" |
| 5 | **Bug detection.** Note any portal or software fault the caller reported, or "-" if none. | Bugs | "Insight + issues/bugs noted" |
| 6 | **Agent score.** Score the agent from 1 to 10 on professionalism, understanding, accuracy, resolution, clarity and call handling. | Agent score | "Resolution verdict + agent score" |
| 7 | **Call resolved.** Did the agent give the caller a response? First write what the agent told them and any question left unanswered, then answer Yes, No or Partial. | Resolution, with its evidence | "Resolution verdict + agent score" |
| 8 | **Audio quality.** Rate the recording Good, Moderate or Poor and list any problems (noise, drops, echo). | Audio quality | "Audio-quality rating + language" |
| 9 | **Language.** List the languages spoken. | Languages | "Audio-quality rating + language" |
| 10 | **Bug category.** If a bug was found, put it in one of the existing bug categories, or "Uncategorised". | Bug category | "Insight + issues/bugs noted" |

**Audio Unclear.** If the words cannot be made out, Gemini skips tasks 3–7
and returns fixed values plus an audio-quality rating and whatever partial
transcript it could manage. **This is still billed in full**, because Gemini
had to listen to the whole recording to decide.

**Settings on the request:**
- The answer must be a single JSON object.
- "Thinking" is switched off, so no extra reasoning tokens are billed.
- At most 8,192 tokens can come back.

**Typical size of one request** (from the cost doc):

| | Average | Higher end |
|---|---|---|
| 1-minute call | ~7,415 tokens, **₹0.45** | ~7,570 tokens, ₹0.48 (+10% retries = **₹0.53**) |
| 2-minute call (the real average recording) | **₹0.68** | **₹0.78** |

### 3c. Delete the uploaded recording (free)

Once Gemini has answered, the portal deletes its copy of the recording from
Gemini's file store, whether the analysis succeeded or failed.

## Step 4: Checks on Gemini's answer (free)

The answer is checked before anything is saved:
- **Readable answer.** If it isn't valid JSON, or is empty, the request is
  tried again.
- **Looping transcript.** If the transcript repeats itself over and over, the
  call is failed and **not** retried, since the same recording would loop
  again.
- **Categories.** A category or sub-category that is not on the official
  list is replaced with "Uncategorised" or "Other", so made-up names never
  reach the reports.
- **Tags.** Snapped to the list in the same way. Every call keeps at least one
  tag.
- **Resolution.** Held to the evidence Gemini gave. A "No" is corrected when
  Gemini's own evidence shows the agent did respond.
- **Token counts.** Recorded for the cost report.

## Step 5: Results saved (free)

Everything from Step 3b is saved to the call's analysis record. Category, tags
and resolution are also copied onto the call itself, so the Call Report,
filters, exports and Daily Report can use them.

## Step 6: When something goes wrong

Retries happen at two levels:

| Level | When | How many times | Cost |
|---|---|---|---|
| Inside one job | Network error, timeout, an unreadable answer | Up to 3 tries, a few seconds apart; each try uploads again | Each try that gets an answer from Gemini is billed |
| The queue | Any temporary failure | Up to 5 attempts in total, waiting 30 s, 2 min, 8 min, then 30 min | Same as above |

- **Rate limit (Gemini answers "too busy").** The request is sent once to the
  back-up model, `gemini-2.5-flash-lite`. Since 12 Sep every completed request
  has used 2.5 Flash.
- **Final failure.** After the last attempt the call is marked **failed**.
  Failed calls are never retried automatically; someone has to press Retry or
  requeue them.

*Cost doc reference:* this is the **"+10% retry allowance"** in the
higher-end figures. That 10% is an estimate, not a measurement.

## Step 7: Hourly bug-category clean-up (not per call)

Once an hour a separate job looks for analysed calls whose bug was marked
"Uncategorised". It sends Gemini a short text list of those bug descriptions,
with no audio, and asks it to place each in an existing bug category or create
a new one. It runs only when such calls exist, and handles many calls in one
request. "Thinking" is on for this job, since it uses Google's defaults. It
is rare and small, so the per-call figures leave it out.

A similar hourly job that invents new call categories exists in the code, but
it is **switched off** (dynamic categories are off), so it costs nothing.

---

## Where this lives in the code

| Step | File |
|---|---|
| 1 Queue | `backend/src/routes/webhook.js`, `enqueueRecording` in `backend/src/workers/analysisWorker.js` |
| 2 Checks | `processRecord` in `analysisWorker.js`; empty/oversize/missing file checks in `uploadAudioToGemini` (`backend/src/services/geminiService.js`) |
| 3a–3c Gemini request | `uploadAudioToGemini` and `categorizeRecording` in `geminiService.js` |
| 4 Answer checks | `categorizeRecording`, `resolvedFromEvidence`, `snapTags` in `geminiService.js` |
| 5 Save | `processRecord` in `analysisWorker.js` |
| 6 Retries | `categorizeRecording` (3 tries); `backoffSeconds` / `scheduleRetryOrFail` in `analysisWorker.js` (5 attempts) |
| 7 Bug clean-up | `backend/src/workers/bugCategoryWorker.js` |
| Token log | `backend/src/lib/geminiUsageLog.js` |

**Note:** Gemini has refused every request since 25 Sep 2026 because the
Google project's spend cap was reached. This describes how the analysis runs
once Gemini is working again.
