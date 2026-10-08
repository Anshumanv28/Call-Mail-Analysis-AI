# Gemini 2.5 Flash cost: per call and per mail

Prepared 8 October 2026. Each table gives two figures:
- the **average**, which is what a typical request actually cost;
- the **higher end**, the realistic figure to plan with. It is not the
  minimum.

## Prices used

Google's prices for `gemini-2.5-flash`, per 1 million tokens:

| What is sent or returned | USD | ₹ (at ₹88.40 per $) |
|---|---|---|
| Text and images sent in | $0.30 | ₹26.52 |
| Audio sent in | $1.00 | ₹88.40 |
| Text returned | $2.50 | ₹221.00 |

₹88.40/$ is the rate implied by Google's own bill for 22–25 Sep. GST, if the
invoice adds it, comes on top.

**"Average"** is the mean of the measured requests, with nothing added for
retries.

**"Higher end" means:**
- each request is sized at the 90th percentile of real requests, so 9 in 10
  cost less;
- plus 10% for requests that have to be sent again.

Token sizes were measured on live data from 4,021 call analyses and 1,440 mail
analyses that succeeded between 12 and 22 Sep 2026.

---

## Per call (unit: a 1-minute call)

Each answered call is **one request** to Gemini. In that one request, Gemini
listens to the recording and returns the transcript, summary, category,
resolution verdict and agent score. Transcription, generation and decisions
are not billed separately. The rows below show what each part costs within
that one answer.

| Part of the work | Average tokens | Average ₹ | Higher-end tokens | Higher-end ₹ |
|---|---|---|---|---|
| Listening to 1 minute of recording (audio in, 32 tokens a second) | 1,920 | ₹0.170 | 1,920 | ₹0.170 |
| Instructions + category list sent with every call (text in) | 4,809 | ₹0.128 | 4,809 | ₹0.128 |
| Writing the transcript | ~476 | ₹0.105 | ~582 | ₹0.129 |
| Writing the summary | ~71 | ₹0.016 | ~87 | ₹0.019 |
| Category, sub-category, tags | ~75 | ₹0.017 | ~91 | ₹0.020 |
| Insight + issues/bugs noted | ~32 | ₹0.007 | ~39 | ₹0.009 |
| Audio-quality rating + language | ~21 | ₹0.005 | ~26 | ₹0.006 |
| Resolution verdict + agent score | ~10 | ₹0.002 | ~12 | ₹0.003 |
| *One request* | *~7,415* | *₹0.45* | *~7,570* | *₹0.48* |
| Retry allowance (+10%) | | — | | ₹0.05 |
| **Per 1-minute call** | | **₹0.45 ($0.0051)** | | **₹0.53 ($0.0060)** |

- **Each extra minute adds about ₹0.25 at the higher end** (₹0.23 on
  average). That is ₹0.17 more audio, plus ₹0.06 for a longer transcript,
  plus the retry allowance.
- The instructions and category list cost the same ₹0.13 however short the
  call is.
- The output tokens are measured on calls with 1–2 minutes of audio: 686 on
  average, 838 at the 90th percentile. The split between the output parts is measured
  from the length of each field Gemini returned, so it is approximate.
- **What counts is the length of the recording, not the conversation.** In the
  12–22 Sep sample the recordings Gemini received averaged about **2 minutes**.
  In a spot check of 7 calls, the recording covered almost the whole call (all
  but 20–70 seconds of it), not just the part spent talking to an agent. At
  2 minutes, a call costs about **₹0.78** at the higher end (₹0.53 + ₹0.25),
  or **₹0.68** on average (₹0.45 + ₹0.23).
- The 32-tokens-a-second rate was checked against 7 real recordings, and
  Gemini's token count matched their length exactly.
- Recordings Gemini can't make out ("Audio Unclear") are still paid in full.
  Missing recordings and missed calls never reach Gemini and cost nothing.

## Per mail

A mail analysis **run** happens when a mail arrives, and again each time we
send a reply. Each run re-reads the sender's whole chain, plus any
attachments. In one request it returns the category, summary, resolution
verdict and draft replies for the agent.

| Part of the work | Average tokens | Average ₹ | Higher-end tokens | Higher-end ₹ |
|---|---|---|---|---|
| Reading the chain + instructions (text and attachments in) | 5,242 | ₹0.139 | 5,980 | ₹0.159 |
| Writing draft replies for the agent | ~226 | ₹0.050 | ~391 | ₹0.086 |
| Checking attachments / documents | ~27 | ₹0.006 | ~46 | ₹0.010 |
| Writing the summary | ~15 | ₹0.003 | ~27 | ₹0.006 |
| Category, sub-category, tags | ~16 | ₹0.004 | ~28 | ₹0.006 |
| Insight, issues, requested action | ~11 | ₹0.002 | ~19 | ₹0.004 |
| Resolution verdict, language | ~4 | ₹0.001 | ~6 | ₹0.001 |
| *One request* | *~5,540* | *₹0.21* | *~6,500* | *₹0.27* |
| Retry allowance (+10%) | | — | | ₹0.03 |
| **Per run** | | **₹0.21 ($0.0023)** | | **₹0.30 ($0.0034)** |

- **Per incoming mail: about ₹0.49 at the higher end** (₹0.34 on average).
  Each incoming mail leads to about 1.6 runs: its own, plus the re-read after
  our reply. Mail with no reply costs ₹0.30 (₹0.21 on average).
- Mails under the minimum length ("Email too Short") are skipped and cost
  nothing.
- Since 24 Sep the mail instructions also check for a closing thank-you. That
  adds a few hundred tokens to each run, which is under ₹0.01.

---

## Average cost as measured, 12–21 Sep (real call lengths)

These are the earlier figures. They were measured on real calls as they came
in, so they use the actual recording length of about 2 minutes, not the
1-minute unit.

**One call analysis**

| Part | Tokens | Rate (per million tokens) | Cost |
|---|---|---|---|
| Audio of the recording | 3,900 | ₹88.40 | ₹0.345 |
| Text (instructions + category list) | 4,809 | ₹26.52 | ₹0.128 |
| Output (the answer) | 832 | ₹221.00 | ₹0.184 |
| **Total** | **≈ 9,541** | | **≈ ₹0.66** |

**One mail analysis run**

| Part | Tokens | Rate (per million tokens) | Cost |
|---|---|---|---|
| Input (the sender's whole chain + instructions) | 5,252 | ₹26.52 | ₹0.139 |
| Output (the answer; thinking is off, so 0 thinking tokens) | 301 | ₹221.00 | ₹0.067 |
| **Total** | **≈ 5,553** | | **≈ ₹0.21** |

**Per item**
- Per analysed call: ≈ ₹0.66. This matches the 2-minute average above
  (₹0.68).
- Per answered call: ≈ ₹0.63, since some answered calls never reach Gemini.
- Per call offered: ≈ ₹0.20, since missed calls cost nothing.
- Per incoming mail: ≈ ₹0.21–0.33. The higher figure includes the re-read
  after each of our replies.
- 12–21 Sep total: about ₹265 a day for calls and ₹60–95 a day for mail, so
  roughly ₹330–360 a day in all. Volumes have about doubled since: about 886
  answered calls a day from 25 Sep to 7 Oct, against about 425 then.

---

## Notes

- "Thinking" is switched off for call and mail analysis, so it adds nothing to
  the figures above.
- Gemini has refused every request since 25 Sep because the Google project's
  spend cap was reached. These are the costs once it is working again.
- Sources:
  - prices: `backend/src/config/geminiPricing.js`;
  - token sizes: stored usage in `call_analysis` and `email_analysis` on live
    Atlas, read only;
  - model and settings: the live server's environment on 8 Oct 2026.
