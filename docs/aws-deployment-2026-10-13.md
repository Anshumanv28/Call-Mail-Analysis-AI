# CallAI on AWS: first deployment, 13 October 2026

The package ran on a real AWS GPU server with real helpline calls: 264 calls processed end to end, about
**7.4 to 8 seconds per call**, models ready **20 seconds** after start. In queue mode this costs about **₹4,700 a
month on demand or ₹1,900 on spot**, against about ₹19,500 a month for Gemini.

## Server used

| Item | Value |
|---|---|
| Instance | g4dn.xlarge: NVIDIA T4 16 GB, 4 vCPU, 16 GB RAM, Mumbai (ap-south-1) |
| OS | Ubuntu 22.04 (plain image, no GPU driver) |
| Disk | 60 GB gp3 (root partition had to be grown from 8 GB) |
| Installed | NVIDIA driver 570 server, Python 3.12 (uv), torch 2.14 + CUDA 12.6, llama.cpp b11064 built from source |
| Install time | about 1 hour: 5 min packages and models, 25 min llama.cpp build, the rest checks |

Everything is now one script: `bash deploy/install_ubuntu.sh` (see README.md, Setup).

## Measured

| What | Result |
|---|---|
| Calls processed | 264 real calls from the 25 Sep report, average 2.6 minutes |
| Time per call | 7.4 to 8 s end to end (download, Whisper, speaker roles, category, QA) |
| Batch of 32 calls | 226 to 271 s |
| Model load after start | about 20 s |
| Errors | none in the final run; out-of-memory batches went back to the queue and succeeded on retry |
| GPU memory | llama-server ~5 GB, Whisper + bge-m3 up to ~9 GB, T4 has 14.5 GB usable |

## AWS prices (Mumbai, official AWS price files, 12 Oct 2026)

| Instance | GPU | On-demand Linux | Spot Linux | Note |
|---|---|---|---|---|
| **g4dn.xlarge** | T4 16 GB | **$0.579 / h** | **~$0.19 / h** | cheapest GPU, enough for ~450 calls an hour |
| g6.xlarge | L4 24 GB | $0.966 / h | | not needed at this volume |
| g5.xlarge | A10G 24 GB | $1.208 / h | ~$0.51 / h | not needed at this volume |

Spot prices change hourly. A spot server can be taken back by AWS with 2 minutes' notice; the queue worker
returns unfinished calls to the queue, so nothing is lost.

## Monthly cost at today's volume (about 800 answered calls a day, ₹88.4 per $)

GPU time per day: 800 calls × ~7.7 s = 1.7 h, plus about 1 h of start-ups (every 15 minutes in working hours).
Disk: 60 GB gp3, about ₹485 a month.

| How it runs | Results in | On demand | Spot |
|---|---|---|---|
| **Queue mode** (on only while calls wait) | ~20 min | **~₹4,700** | **~₹1,900** |
| On all working day (10 h) | seconds | ~₹15,800 | ~₹5,500 |
| Gemini (reference, calls + mail) | minutes | ~₹19,500 | |

## Problems found and fixed

1. **Prebuilt llama.cpp did not run on Ubuntu 22.04** (`GLIBC_2.38 not found`): it is built for Ubuntu 24.04.
   `setup.py` now detects this and `deploy/build_llama_from_source.sh` builds the same version for the T4.
2. **GPU out of memory** with llama-server sized for the old LLM categoriser (16k tokens per slot, 7.5 GB). Now 8k
   per slot when the classifier is present (5 GB); PyTorch uses expandable memory; embeddings run in small chunks.
3. **Call Resolved** from the LLM agreed with Gemini on only 8% of calls (it answers Partial where Gemini says Yes).
   A small calibration (`code/models/resolved_cal.json`, fitted on 629 Gemini-labelled calls) brings it to 95%.
   An earlier README figure of 82% was a misreading and has been corrected.

## Next

1. Keep the installed server: save it as an AMI so the next start takes minutes, not an hour.
2. SQS queue `callai-calls` with a dead-letter queue, and the Lambda + 15-minute schedule (`deploy/README.md`).
3. Portal: send each answered call to the queue; accept results at one URL (or read them from S3).
4. Back-fill the ~12,500 calls unanalysed since 25 Sep: about 27 GPU hours, about ₹1,400 on demand.
