"""CallAI OTR - one pipeline for the portal.

recording -> fine-tuned Whisper (per speech segment) -> AGENT / CANDIDATE turns -> category + sub-category
(Qwen3-4B, knn_hybrid over labelled calls) -> agent score, call resolved, AI insight, summary, language (Qwen3-4B with
a fixed rubric and the official answer for that sub-category) -> audio rating (signal rule).
The output keys are the portal's own column names (see README.md "Output fields").

    from otr_pipeline import OTRPipeline
    p = OTRPipeline()
    rows = p.process_many(["a.wav", "b.wav"])          # list of dicts, one per recording, same order
    p.close()                                          # stops the llama-server it started

Batch mode runs the stages one after the other (transcribe everything, free the GPU, then the LLM), because
Whisper + bge-m3 + an 8-slot llama-server together do not fit a 24 GB GPU. The API (serve.py) keeps both loaded
and processes few calls at a time, which fits.
"""
from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent
CODE = ROOT / "code"
MODELS = ROOT / "models"
os.environ.setdefault("HF_HOME", str(ROOT / "hf"))
sys.path.insert(0, str(CODE))

import callai.audio as A  # noqa: E402
from callai.audio import speech_chunks  # noqa: E402
from callai.turns2 import voice_features  # noqa: E402
from callai.loops import collapse_loops  # noqa: E402

LLAMA_PORT = int(os.environ.get("OTR_LLAMA_PORT", "8080"))
SLOTS = int(os.environ.get("OTR_LLM_SLOTS", "8"))
os.environ.setdefault("CALLAI_LLM_RETRY_S", "300")      # a dead llama-server fails the run in 5 min, not 2 h
RETRIEVAL = "data/otr/pool_transcripts.jsonl,data/otr/pool_human_labels.csv,6"   # the measured recipe
MIN_WORDS = 20                                   # fewer words: "Audio Unclear" (greetings, dropped calls)
SPECIAL = ("Audio Unclear", "Recording Unavailable")
REVIEW_CONF = float(os.environ.get("OTR_REVIEW_CONF", "0.4"))   # below this, needs_review = true

QA_SYSTEM = (
    "You are a quality analyst for the UPESSC One Time Registration (OTR) helpline in Uttar Pradesh. Calls are in "
    "Hindi/Hinglish; the transcript is romanised and machine-made, so ignore small word errors. AGENT = helpline "
    "agent, CANDIDATE = caller (speaker labels are automatic and may be wrong on a few lines; judge by content).\n"
    "Score the AGENT on five items, each 0, 1 or 2:\n"
    "greeting: greeted politely and stayed courteous (2), curt (1), rude or no greeting (0)\n"
    "understanding: understood the caller's real problem, asked what was needed (2), partly (1), not at all (0)\n"
    "correctness: answer consistent with the APPROVED ANSWER below (2), partly or vague (1), wrong / misleading (0); "
    "if no approved answer applies, judge plausibility\n"
    "next_steps: clear next steps or resolution (2), vague (1), none (0)\n"
    "closing: confirmed the caller is satisfied / closed the call properly (2), abrupt (1), none (0)\n"
    "call_resolved: Yes = the caller's question was answered or the fix explained; Partial = partly, or escalated "
    "with clear next steps; No = unanswered, wrong answer, or the call dropped.\n"
    "insight: the caller's core issue in at most 8 English words (e.g. \"Husband's name mismatch with Aadhaar\").\n"
    "summary: 1-2 English sentences: what the caller needed and what the agent said.\n"
    "language: the language spoken (Hindi, Hinglish, English, or \"Hindi, English\").")
QA_SCHEMA = {"type": "object", "additionalProperties": False,
             "required": ["greeting", "understanding", "correctness", "next_steps", "closing", "call_resolved",
                          "insight", "summary", "language"],
             "properties": {**{k: {"type": "integer", "enum": [0, 1, 2]} for k in
                               ("greeting", "understanding", "correctness", "next_steps", "closing")},
                            "call_resolved": {"type": "string", "enum": ["Yes", "Partial", "No"]},
                            "insight": {"type": "string", "maxLength": 80},
                            "summary": {"type": "string", "maxLength": 400},
                            "language": {"type": "string", "enum": ["Hindi", "Hinglish", "English", "Hindi, English"]}}}


class OTRPipeline:
    def __init__(self, device: str | None = None, batch: int = 32, threads: int = 4):
        import torch
        self.torch = torch
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.dev == "cuda" else torch.float32
        torch.set_num_threads(1 if self.dev == "cuda" else max(1, (os.cpu_count() or 4) // 2))
        self.batch, self.threads = batch, threads
        self._tl = threading.local()
        A._silero = self._silero                     # one Silero VAD per thread (it keeps state inside the model)
        self._silero()                               # fail now rather than fall back to the energy VAD
        self.whisper = None
        self.llama = None                            # the llama-server process we started, if any
        self.llm = None
        self.pipe = None
        if (CODE / "models" / "turns_v2.pkl").exists():         # words + voice (better AGENT / CANDIDATE split)
            from callai.turns2 import TurnLabeller2
            self.turns = TurnLabeller2.load(CODE / "models" / "turns_v2.pkl")
        else:                                                   # words only
            from callai.turns import TurnLabeller
            self.turns = TurnLabeller.load(CODE / "models" / "turns_model.pkl")
        self.voice = (CODE / "models" / "turns_v2.pkl").exists()
        import yaml
        self.playbooks = yaml.safe_load((CODE / "products/otr/playbooks.yaml").read_text(encoding="utf-8")) or {}
        cal = CODE / "models" / "agent_score.json"            # maps our rubric to Gemini's 1-10 scale
        self.score_cal = json.loads(cal.read_text(encoding="utf-8")) if cal.exists() else None
        lang = CODE / "models" / "language_clf.pkl"           # Hindi / Hinglish the way Gemini labels it
        self.lang_clf = pickle.loads(lang.read_bytes()) if lang.exists() else None
        gpath = CODE / "models" / "gemini_cls.pkl"            # category / sub-category trained on Gemini's labels
        self.gcls = None
        if gpath.exists():
            from callai.gemini_cls import GeminiClassifier
            self.gcls = GeminiClassifier.load(gpath)
        else:
            print("WARNING: code/models/gemini_cls.pkl not found - categories fall back to the older LLM categoriser "
                  "(about 35% sub-category agreement with Gemini instead of 53%). Copy it from CallAI_OTR_portal.zip; "
                  "see MODELS.md.", file=sys.stderr, flush=True)
        tax = yaml.safe_load((CODE / "products/otr/taxonomy.yaml").read_text(encoding="utf-8"))
        self.cat_id = {i["name"]: i["id"] for i in tax["issues"]}
        self.sub_id = {s["name"]: s["id"] for i in tax["issues"] for s in i["sub_issues"]}

    # ---------------------------------------------------------------- audio -> segments
    def _silero(self):
        v = getattr(self._tl, "v", None)
        if v is None:
            from silero_vad import get_speech_timestamps, load_silero_vad
            v = self._tl.v = (load_silero_vad(), get_speech_timestamps)
        return v

    def _load_whisper(self):
        if self.whisper is None:
            from transformers import WhisperForConditionalGeneration, WhisperProcessor
            self.proc = WhisperProcessor.from_pretrained(MODELS / "whisper_otr_ft")
            self.whisper = WhisperForConditionalGeneration.from_pretrained(
                MODELS / "whisper_otr_ft", torch_dtype=self.dtype).to(self.dev).eval()

    def free_whisper(self):
        self.whisper = None
        if self.dev == "cuda":
            self.torch.cuda.empty_cache()

    def _prepare(self, path: str | Path):
        """One read, one resample: raw 16 kHz audio for Whisper, its RMS-normalised copy for the VAD,
        and each segment's voice features for the speaker-role model."""
        try:
            raw, sr = sf.read(str(path), dtype="float32", always_2d=True)
            x = raw.mean(1)
            dur = len(x) / sr
            x16 = resample_poly(x, 2, 1).astype("float32") if sr == 8000 else (
                resample_poly(x, 16000, sr).astype("float32") if sr != 16000 else x)
            segs = speech_chunks(A.normalise(x16), 16000)           # <= 25 s each, roughly one speaker turn
            audio = [x16[int(s * 16000):int(e * 16000)] for s, e in segs]
            feats = [voice_features(y) for y in audio] if self.voice else [None] * len(audio)
            return [(round(s, 2), round(e, 2)) for s, e in segs], audio, feats, dur, None
        except Exception as e:  # noqa: BLE001 - reported per call
            return [], [], [], 0.0, repr(e)

    def _decode(self, audio: list) -> list[str]:
        out = []
        for i in range(0, len(audio), self.batch):
            fe = self.proc.feature_extractor(audio[i:i + self.batch], sampling_rate=16000, return_tensors="pt", device=self.dev)
            x = fe.input_features.to(self.dev, self.dtype)
            with self.torch.no_grad():
                ids = self.whisper.generate(input_features=x, language="hindi", task="transcribe", max_new_tokens=220, num_beams=1)
            out += [collapse_loops(t.strip()) for t in self.proc.batch_decode(ids, skip_special_tokens=True)]
        return out

    def transcribe_many(self, paths: list, on_done=None) -> list[dict]:
        """{file, duration_s, speech_s, segments: [{start, end, text}], dialogue, transcript (AGENT/CANDIDATE), error}"""
        self._load_whisper()
        results = [None] * len(paths)
        block = 32
        with ThreadPoolExecutor(self.threads) as ex:
            nxt = [ex.submit(self._prepare, p) for p in paths[:block]]
            for b0 in range(0, len(paths), block):
                cur = [f.result() for f in nxt]
                nxt = [ex.submit(self._prepare, p) for p in paths[b0 + block:b0 + 2 * block]]
                audio = [a for _, segs_audio, _, _, _ in cur for a in segs_audio]
                texts = self._decode(audio) if audio else []
                k = 0
                for j, (segs, segs_audio, feats, dur, err) in enumerate(cur):
                    tx = texts[k:k + len(segs_audio)]
                    k += len(segs_audio)
                    seg_rows = [{"start": s, "end": e, "text": t, "feat": f} for (s, e), t, f in zip(segs, tx, feats) if t.strip()]
                    transcript = self.turns.transcript(seg_rows) if seg_rows else ""
                    for sr_ in seg_rows:
                        del sr_["feat"]
                    r = {"file": Path(paths[b0 + j]).stem, "duration_s": round(dur, 1),
                         "speech_s": round(sum(e - s for s, e in segs), 1), "segments": seg_rows,
                         "dialogue": " ".join(s["text"] for s in seg_rows),
                         "transcript": transcript,
                         "n_segments": len(segs), "error": err}
                    results[b0 + j] = r
                    if on_done:
                        on_done(r)
        return results

    # ---------------------------------------------------------------- LLM (llama-server)
    def start_llm(self):
        if self.llm is not None:
            return
        url = os.environ.get("CALLAI_LLM_URL")
        if not url:                                   # start our own llama-server on the GPU
            exe = next(iter(sorted((ROOT / "llama").rglob("llama-server.exe" if os.name == "nt" else "llama-server"))), None)
            gguf = MODELS / "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
            if exe is None or not gguf.exists():
                raise RuntimeError("llama-server or the Qwen3-4B model is missing: run python setup.py")
            log = open(ROOT / "logs" / "llama-server.log", "ab")
            self.llama = subprocess.Popen([str(exe), "-m", str(gguf), "--host", "127.0.0.1", "--port", str(LLAMA_PORT),
                                           "-c", str(16384 * SLOTS), "-np", str(SLOTS), "-ngl", "99", "-fa", "on",
                                           "-ctk", "q8_0", "-ctv", "q8_0", "--jinja", "--temp", "0", "-t", "2", "--cache-ram", "0"],
                                          stdout=log, stderr=subprocess.STDOUT)
            url = f"http://127.0.0.1:{LLAMA_PORT}/v1"
            import httpx
            for _ in range(120):
                try:
                    if httpx.get(f"http://127.0.0.1:{LLAMA_PORT}/health", timeout=3).status_code == 200:
                        break
                except Exception:  # noqa: BLE001
                    pass
                if self.llama.poll() is not None:
                    raise RuntimeError("llama-server exited: see logs/llama-server.log")
                time.sleep(2)
        os.environ["CALLAI_LLM_URL"] = url
        os.environ.setdefault("CALLAI_LLM_PARALLEL", str(SLOTS))
        from callai.llm import make_llm
        self.llm = make_llm("llama")
        if self.gcls is not None:                     # categories come from the Gemini-trained classifier
            return
        cwd = os.getcwd()
        os.chdir(CODE)                                # product files and the retrieval spec are relative to code/
        try:
            from callai import Pipeline, PipelineConfig
            self.pipe = Pipeline(PipelineConfig(product="otr", asr="whisper", llm="llama", classify_mode="knn_hybrid",
                                                retrieval=RETRIEVAL, answer=False))
            self.pipe.segregator                      # build it now (loads the labelled pool)
        finally:
            os.chdir(cwd)

    def _llama_log_tail(self) -> str:
        if self.llama is None:
            return ""
        state = "running" if self.llama.poll() is None else f"exited with code {self.llama.returncode}"
        try:
            tail = (ROOT / "logs" / "llama-server.log").read_text(encoding="utf-8", errors="replace")[-3000:]
        except OSError:
            tail = ""
        return f"\nllama-server {state}; end of logs/llama-server.log:\n{tail}"

    def close(self):
        if self.llama is not None:
            self.llama.terminate()
            self.llama = None

    def _qa_messages(self, r: dict, sub_id: str) -> list[dict]:
        approved = str(self.playbooks.get(sub_id) or self.playbooks.get("default") or "")[:1500]
        transcript = (r.get("transcript") or r.get("dialogue") or "")[:7000]
        return [{"role": "system", "content": QA_SYSTEM},
                {"role": "user", "content": f"Sub-category: {sub_id}\nAPPROVED ANSWER:\n{approved}\n\nTRANSCRIPT:\n{transcript}"}]

    def analyse_many(self, rows: list[dict]) -> list[dict]:
        """Category / sub-category + QA fields for transcribed rows (in place; also returned)."""
        from callai.llm import extract_json
        from callai.retrieve import ExamplePool
        self.start_llm()
        talk = [r for r in rows if not r.get("error") and len((r.get("dialogue") or "").split()) >= MIN_WORDS]
        if talk:
            enc = ExamplePool.__new__(ExamplePool)    # embed once, batched; the categorizer then reads the cache
            enc._model_name, enc._model = "BAAI/bge-m3", None
            enc._cached_encode([r["dialogue"] for r in talk])
            enc._model = None
            if self.gcls is not None:
                E = np.vstack([np.load(ExamplePool.CACHE_DIR / f"{ExamplePool._row_key(r['dialogue'])}.npy") for r in talk])
                mu = None
                if "muril" in self.gcls.models and (CODE / "models" / "gemini_muril").exists():
                    from callai.gemini_cls import muril_scores
                    mu = muril_scores(CODE / "models" / "gemini_muril", [r["dialogue"] for r in talk], self.gcls.labels, self.dev)
                for r, p in zip(talk, self.gcls.predict([r["dialogue"] for r in talk], E.astype(np.float32), mu)):
                    r["segregation"] = {"issue_name": p["cat"], "sub_issue_name": p["sub"],
                                        "issue": self.cat_id.get(p["cat"], ""), "sub_issue": self.sub_id.get(p["sub"], ""),
                                        "confidence": p["confidence"], "needs_review": p["confidence"] < REVIEW_CONF,
                                        "alternatives": [c for c, _ in p["top"][1:3]]}
            else:
                seg_rows = [{"file": r["file"], "dialogue": r["dialogue"], "duration_s": r["duration_s"]} for r in talk]
                cwd = os.getcwd()
                os.chdir(CODE)
                try:
                    self.pipe.segregate(seg_rows)
                except Exception as e:  # noqa: BLE001
                    raise RuntimeError(f"LLM failed: {e!r}" + self._llama_log_tail()) from e
                finally:
                    os.chdir(cwd)
                for r, s in zip(talk, seg_rows):
                    r["segregation"] = s.get("segregation") or {}
            try:
                replies = self.llm.chat_batch([self._qa_messages(r, r["segregation"].get("sub_issue", "")) for r in talk],
                                              max_new_tokens=400, json_schema=QA_SCHEMA)
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(f"LLM failed: {e!r}" + self._llama_log_tail()) from e
            for r, rep in zip(talk, replies):
                r["qa"] = extract_json(rep) or {}
        return [self.portal_fields(r) for r in rows]

    # ---------------------------------------------------------------- portal columns
    @staticmethod
    def audio_rating(r: dict) -> str:
        words = len((r.get("dialogue") or "").split())
        speech_min = max(1e-6, (r.get("speech_s") or 0) / 60)
        empty = 1 - len(r.get("segments") or []) / max(1, r.get("n_segments") or 0)
        if words < MIN_WORDS or words / speech_min < 60 or empty >= 0.3:
            return "Poor"
        return "Good" if words / speech_min >= 100 and empty <= 0.1 else "Moderate"

    def agent_score(self, r: dict, items: list[int], resolved: str, audio: str) -> int:
        """Gemini-scale Agent Score: calibrated on calls Gemini scored (scripts/fit_agent_score.py, same features)."""
        raw = max(1, sum(items))
        c = self.score_cal
        if not c:
            return raw
        words = len((r.get("dialogue") or "").split())
        x = np.array([float(v) for v in items] + [sum(items), resolved == "Yes", resolved == "Partial", audio == "Good",
                                                   audio == "Poor", np.log1p(words), np.log1p(float(r.get("duration_s") or 0))])
        z = (x - np.array(c["mu"])) / np.array(c["sd"])
        logits = np.array(c["coef"]) @ z + np.array(c["intercept"])
        prob = np.exp(logits - logits.max())
        prob /= prob.sum()
        return int(round(float(prob @ np.array(c["classes"]))))   # expected Gemini score, rounded

    def language(self, r: dict, llm_says: str) -> str:
        """The LLM reads a romanised transcript and calls nearly every call Hinglish; Gemini (hearing the audio) says
        Hindi for two in three. A word classifier fitted on Gemini's labels decides, except for English calls."""
        if llm_says == "English" or self.lang_clf is None:
            return llm_says
        return str(self.lang_clf.predict([r.get("dialogue") or ""])[0])

    def portal_fields(self, r: dict) -> dict:
        base = {"file": r.get("file"), "Resolution Source": "AI", "Transcription": r.get("transcript") or r.get("dialogue") or "",
                "Audio Rating": self.audio_rating(r) if not r.get("error") else "", "Call Category": "", "Sub-Category": "",
                "Tags": "", "AI Insight": "", "Summary": "", "Call Resolved": "", "Agent Score": "", "Language": "",
                "Bug Category": "", "Bug Description": ""}
        if r.get("error"):
            base.update({"Call Category": "Recording Unavailable", "Tags": "Recording Unavailable", "error": r["error"]})
            return base
        seg, qa = r.get("segregation"), r.get("qa") or {}
        if not seg:
            base.update({"Call Category": "Audio Unclear", "Tags": "Audio Unclear"})
            return base
        items = [int(qa.get(k, 0) or 0) for k in ("greeting", "understanding", "correctness", "next_steps", "closing")]
        base.update({
            "Call Category": seg.get("issue_name", ""), "Sub-Category": seg.get("sub_issue_name", ""),
            "AI Insight": qa.get("insight", ""), "Summary": qa.get("summary") or seg.get("summary", ""),
            "Call Resolved": qa.get("call_resolved", ""),
            "Agent Score": self.agent_score(r, items, qa.get("call_resolved", ""), base["Audio Rating"]) if qa else "",
            "Language": self.language(r, qa.get("language", "")),
            # not portal columns: kept for review and audit
            "category_id": seg.get("issue", ""), "sub_category_id": seg.get("sub_issue", ""),
            "confidence": seg.get("confidence", ""), "needs_review": seg.get("needs_review", ""),
            "alternatives": seg.get("alternatives", []),
            "score_items": dict(zip(("greeting", "understanding", "correctness", "next_steps", "closing"), items)),
            "agent_score_raw": max(1, sum(items)) if qa else ""})
        return base

    def process_many(self, paths: list) -> list[dict]:
        rows = self.transcribe_many(paths)
        return self.analyse_many(rows)
