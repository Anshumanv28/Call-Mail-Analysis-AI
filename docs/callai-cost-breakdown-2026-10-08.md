# CallAI OTR cost: per call and per mail (instead of Gemini)

Prepared 8 October 2026, as the counterpart of
[reference/gemini-cost-breakdown-2026-10-08.md](reference/gemini-cost-breakdown-2026-10-08.md) and
[reference/gemini-call-analysis-operations-2026-10-08.md](reference/gemini-call-analysis-operations-2026-10-08.md).
Same currency rate (₹88.40 per $). GST, if the invoice adds it, comes on top.

**The main difference from Gemini.** Gemini is paid per request, so a call costs the same whether there are 10 calls
a day or 10,000. CallAI runs on our own rented GPU server, paid **per hour the server is switched on**, busy or idle.
The models are free: Whisper (MIT), Qwen3-4B (Apache 2.0), bge-m3 (MIT) and llama.cpp. So the cost per call depends
mostly on **how many hours a day the server runs**, not on what each call contains. Section 3 is therefore the part to
decide on. Sections 1 and 2 give the per-call and per-mail figures in the same layout as the Gemini doc.

## Prices used

AWS, Mumbai region (ap-south-1), Linux, billed per second:

| Server | GPU | $/hour | ₹/hour |
|---|---|---|---|
| g4dn.xlarge | NVIDIA T4, 16 GB | $0.579 | ₹51.18 |
| g5.xlarge | NVIDIA A10G, 24 GB | $1.208 | ₹106.79 |
| g4dn.xlarge, 1-year commitment (estimate) | T4 | ~$0.365 | ~₹32.2 |
| g4dn.xlarge, spot (changes hourly; can be taken back by AWS) | T4 | ~$0.27–0.42 | ~₹24–37 |
| Disk: 60 GB gp3 (models, code, working files) | | ~$5.5 a month | ~₹16 a day |

- The on-demand prices are DoiT's listing for ap-south-1. Check them in the AWS console before budgeting.
- The 1-year price assumes the same ~37% discount AWS gives in us-east-1 ($0.526 → $0.331). The Mumbai figure was not looked up.
- Downloading recordings into AWS is free, and nothing of size leaves the server.
- The disk is billed even while the server is stopped.

**Throughput used** (measured, README "Limits"): one T4 processed **631 calls averaging 2.3 minutes in 107 minutes**,
end to end, in batch mode. That is about 354 calls an hour, or **10.2 seconds of server time per call**. Throughput
on the A10G has not been measured. It should be faster than the T4, but no figure for it is used here.

---

## 1. Per call (unit: a 1-minute call)

Each call goes through steps on our server. There is no paid request and no token bill. The only cost is the
server's time. The table maps each step to the Gemini doc's "Part of the work" rows.

| Part of the work | What does it here | Gemini row it replaces | Grows with call length? |
|---|---|---|---|
| Listening to the recording | Silero VAD (CPU), then fine-tuned Whisper-turbo (GPU) on each speech segment | Listening to the recording + Writing the transcript | Yes |
| AGENT / CANDIDATE / SYSTEM roles | Small scikit-learn model on words + voice features (CPU) | (part of the transcript) | Yes, slightly |
| Finding similar past calls | bge-m3 embeddings of the transcript, nearest 6 labelled calls | Instructions + category list | No |
| Category, sub-category | Qwen3-4B request 1 (llama.cpp on the GPU) | Category, sub-category, tags | Slightly (longer prompt) |
| Score, resolved, insight, summary, language | Qwen3-4B request 2, with the approved answer for that sub-category | Summary; Insight; Resolution verdict + agent score; language | Slightly |
| Agent Score calibration, language choice, Audio Rating | Small fitted models and a signal rule (CPU, milliseconds) | Audio-quality rating + language | No |

**One 1-minute call, step by step (T4 server at ₹51.18 an hour, busy the whole time):**

| Part of the work | Size (1-minute call) | Average server time | Average ₹ |
|---|---|---|---|
| Download the recording, VAD, voice features | CPU, runs alongside the GPU | ~0 | ~₹0.000 |
| Listening + writing the transcript (Whisper on the GPU) | 1 min of audio → ~680 characters (~225 tokens) | ~1.8 s | ₹0.026 |
| Speaker roles, similar-call search (bge-m3) | transcript in | ~0.1 s | ₹0.001 |
| Category + sub-category (Qwen3-4B request 1) | ~11,500 tokens in: category list + rules ~10,900 (the same for every call), 6 example calls ~300, transcript ~225; ~200 tokens out | ~4.0 s | ₹0.057 |
| Score, resolved, insight, summary, language (Qwen3-4B request 2) | ~950 tokens in: rubric ~300, approved answer ~400, transcript ~225; ~120 tokens out | ~1.5 s | ₹0.021 |
| Score calibration, language choice, Audio Rating | small fitted models, CPU | ~0 | ₹0.000 |
| *One call* | | *~7.4 s* | *₹0.105* |
| Re-run allowance (+10%) | | | ₹0.011 |
| **Per 1-minute call** | | | **₹0.10 ($0.0012) average · ₹0.16 ($0.0018) higher end** |

- Prompt sizes are measured from the code and data (classifier prompt 38,096 characters; transcripts ~676 characters per
  minute of audio; approved answers average 1,493 characters). Tokens are estimated from characters.
- **Measured:** the total, 10.2 s of server time for a 2.3-minute call (631 calls on a T4). **Estimated:** how those
  seconds split between the steps (Whisper ~1.8 s a minute; LLM ~5.5 s a call). The split has not been timed on the GPU.
- "Higher end" (₹0.16) is the planning figure. It assumes none of the 10.2 s shrinks for a shorter call (₹0.145), plus
  10% for re-runs. The lowest it could be, if all the time scaled with length, is ₹0.06.
- **Each extra minute adds about ₹0.03** (more Whisper work and a longer transcript in both prompts).
- **2-minute call** (the real average recording): about **₹0.14** average, **₹0.16** higher end (the measured 2.3-minute figure + 10%, so it cannot be higher).
- These figures assume the server does nothing but analyse calls while it is on. Idle hours are extra (section 3).
  As a per-call price for a whole working setup: nightly batch ≈ **₹0.14 a 1-minute call** (adds start-up and disk, ~₹0.03);
  a server on 12 hours a day ≈ ₹0.71 a call; on 24 hours ≈ ₹1.40 a call. In these last two, call length makes almost no
  difference, because the server is paid for either way.

- "Higher end" adds 10% for calls that have to be run again, the same allowance the Gemini doc uses. Re-runs are much
  rarer here than Gemini retries, because nothing goes over the network except the recording download. No 90th
  percentile figure exists, because per-call timings have not been logged.
- **Audio Unclear** (fewer than 20 words): Whisper still runs, but both LLM requests are skipped, so these calls
  cost less than a normal call. Gemini bills them in full.
- **Recording Unavailable**, missed calls and calls under 10 s cost nothing, as with Gemini.

**Not produced yet** (the Gemini answer has them; the portal columns are left blank). Adding them is a few more
fields in LLM request 2, which would add well under ₹0.01 a call:
- Bug detection and Bug Category (Gemini tasks 5 and 10, plus the hourly bug clean-up)
- More than one tag per call (Gemini task 2)
- The list of audio problems (noise, drops, echo)

## 2. Per mail (estimate: no mail pipeline exists yet)

This package analyses calls only. A mail analysis would run on the **same server and the same Qwen3-4B**: a text
request in place of Gemini's. The figures below are estimates from typical llama.cpp speeds for a 4B model on a T4,
not measurements.

| Part of the work | Gemini tokens (avg) | On our server |
|---|---|---|
| Reading the chain + instructions | 5,242 in | ~2–3 s of prompt reading |
| Writing drafts, summary, category, etc. | ~300 out | ~1–2 s, shared with other requests running in parallel |
| **Per run, at full use** | | **~3–5 s, ₹0.04–0.07** |
| **Per incoming mail** (~1.6 runs, as in the Gemini doc) | | **₹0.07–0.11** |

Caveats, which matter more than the number:
- **Attachments.** Gemini reads images and scanned PDFs. Qwen3-4B reads text only, so attachments would need OCR or a
  small vision model added. That costs more GPU time, and how much has not been estimated.
- **Quality.** A 4B model writing reply drafts for agents has not been tested. Call categorisation was tested (section
  6 of PROJECT_MEMORY.md); mail was not.
- **It has to be live.** Agents need drafts while they work, so mail needs the server switched on during working
  hours. That decides the server cost much more than the seconds per mail (section 3).

## 3. Cost per day, at today's volume

Volume from the Gemini doc: **~886 answered calls a day** (25 Sep to 7 Oct), recordings averaging ~2 minutes, and
**~290–450 mail runs a day**. GPU work per day: calls ~2.2–2.5 hours, mail ~0.3–0.6 hours (estimate). One T4 can
do ~8,500 calls a day at most in batch mode, so one server is plenty at today's volume.

| How the server runs | Results arrive | ₹ per day | ₹ per call | vs Gemini |
|---|---|---|---|---|
| **Gemini (reference)**, calls + mail | within minutes | **₹662–786** | ₹0.68–0.78 | |
| A. T4, on-demand, **always on** (live API, calls + mail) | within minutes | ₹1,244 | ₹1.40 | ~1.6–1.9× more |
| B. T4, 1-year commitment, always on | within minutes | ~₹790 | ~₹0.89 | about the same |
| C. T4, on-demand, **12 hours a day** (live in working hours; night calls done next morning) | working hours | ₹630 | ₹0.71 | about the same or a little less |
| D. T4, on-demand, **one nightly batch for calls** (~2.75 h incl. start-up); no mail | next morning | ₹157 (higher end ₹171) | ₹0.18–0.19 | **~4× less (calls only)** |
| E. As D, on spot servers (`run_batch.py` is resumable, so an interruption only delays it) | next morning | ₹82–118 | ₹0.09–0.13 | ~6–8× less (calls only) |
| F. A10G, on-demand, always on | within minutes | ₹2,579 | ₹2.91 | ~3.3–3.9× more |

- Per call = ₹ per day ÷ 886. In rows A to C and F, mail is covered by the same server and adds nothing.
- All rows include the ₹16 a day for the disk.
- The Gemini row is ₹0.68–0.78 × 886 for calls, plus ₹60–95 for mail, from the Gemini doc.

**What this means:**
- **Running the server around the clock on-demand costs more than Gemini at today's volume.** Most of those hours
  are idle. On-demand always-on (row A) breaks even with Gemini at about **1,800 calls a day**, roughly twice today's
  volume, which had already doubled from September.
- **Batch is where CallAI is clearly cheaper**: about ₹160 a day against Gemini's ₹600–700 for calls. The trade-off is
  that results arrive the next morning instead of within minutes.
- A practical mix: run the server in working hours (row C) for live mail and same-day calls. It costs about the same
  as Gemini, has no spend cap to hit, and keeps call data on our server.
- The A10G is only worth it if the T4 turns out too slow. At today's volume it is not needed.

## 3b. Recommended setup for the real requirements

Requirements: **instant transcript and analysis** as soon as a recording arrives; the helpline runs **9 am to 6 pm**.

- A T4 server (g4dn.xlarge, on-demand) is **warm from 8:50 am to 6:30 pm**, about 9.7 hours, switched on and off by an
  EventBridge schedule. It runs `serve.py`; the portal sends each new call to `/process` instead of to Gemini.
  The 30 minutes after closing catch the last calls' recordings.
- **Outside those hours** (late recording uploads, mail): the queue + Lambda wakes the server when work is waiting, with
  a ~2-minute start-up delay. Results are not instant then, but nobody is on shift.
- Time per call once the recording is in: about 10–20 s.

| | ₹ per day | ₹ per call (886/day) | ₹ per month (30 days) |
|---|---|---|---|
| Server warm 9.7 h (₹51.18/h) | ₹495 | | |
| Disk | ₹16 | | |
| Out-of-hours wake-ups (a few, ~₹4 each) | ~₹10 | | |
| **CallAI total, calls + mail** | **~₹520** | **~₹0.59** | **~₹15,600** |
| Gemini, calls + mail | ₹662–786 | ₹0.68–0.78 (calls) | ₹19,900–23,600 |
| **Saving** | **₹140–265** | | **₹4,300–8,000 (about 20–35%)** |

- A 1-year commitment does not help here: it bills all 24 hours (~₹773 a day), which is more than 9.7 on-demand hours.
- If the helpline is closed on some days, keep the server off then: both CallAI and Gemini cost nothing on those days.
- Cost per call keeps falling as volume grows: about 1.6 calls a minute on average today, and one T4 can handle about
  6 a minute in batch mode.



- Engineering time to build the mail pipeline and the missing call fields (bugs, multi-tags), and to keep the server
  running (updates, monitoring).
- Spare server capacity for peaks. At ~886 calls a day one T4 is far from full, so this is not needed now.
- GST.

## 5. How to firm up these numbers

1. Run `run_batch.py` on the target server with one day's call report. Its log (`logs/run_batch.log`) timestamps each
   stage, which gives the Whisper vs LLM split and replaces the ranges in section 1.
2. Repeat on an A10G if the T4 looks too slow.
3. Look up the 1-year commitment and spot prices for g4dn.xlarge in ap-south-1 in the AWS console.
4. For mail: build a prototype on Qwen3-4B, time it on ~100 real mails, and compare its drafts with Gemini's.

Sources: [DoiT, g4dn.xlarge ap-south-1](https://www.doit.com/compute/spot/ap-south-1/g4dn.xlarge),
[DoiT, g5.xlarge ap-south-1](https://www.doit.com/compute/spot/ap-south-1/g5.xlarge),
[Vantage, g4dn.xlarge](https://instances.vantage.sh/aws/ec2/g4dn.xlarge),
[Vantage, g5.xlarge](https://instances.vantage.sh/aws/ec2/g5.xlarge). Throughput: README.md "Limits".
Volumes and Gemini figures: the two reference docs above.
