"""ASR backends. Both return the same Segment list so the rest of the pipeline is
backend-blind. Pick with PipelineConfig.asr = "whisper" | "qwen".

whisper : faster-whisper, large-v3-turbo by default. Hindi forced (auto-detect guesses
          English on Hinglish and then hallucinates), its own VAD, a domain prompt.
qwen    : Qwen3-ASR (0.6B / 1.7B). No timestamps of its own, so we chunk each channel
          with the energy VAD and transcribe the chunks in one batch. Hindi yes, Gujarati no.
indic   : AI4Bharat IndicConformer-600M (22 Indian languages incl. Hindi and Gujarati, MIT).
          Same VAD chunking; language per call ("hi", "gu", ...). CPU-friendly.
"""

from __future__ import annotations

import dataclasses
import time

import numpy as np

from . import audio as A

DOMAIN_PROMPT = (
    "हेलो सर, control room से बोल रहा हूँ। internet, server, portal, login, camera, "
    "CCTV, video, recording, biometric, fingerprint, laptop, dongle, exam centre, "
    "candidate, roll number, error, ticket."
)


@dataclasses.dataclass
class Segment:
    channel: int  # 0 = L, 1 = R
    start: float
    end: float
    text: str


@dataclasses.dataclass
class ASRResult:
    segments: list[Segment]
    backend: str
    seconds: float  # wall-clock spent

    def channel_text(self, ch: int) -> str:
        return " ".join(s.text for s in self.segments if s.channel == ch).strip()


class WhisperASR:
    name = "whisper"

    def __init__(self, model: str = "large-v3-turbo", device: str = "auto", compute_type: str | None = None,
                 language: str = "hi", beam_size: int = 5):
        from faster_whisper import WhisperModel

        if compute_type is None:
            compute_type = "float16" if _cuda() else "int8"
        if device == "auto":
            device = "cuda" if _cuda() else "cpu"
        self.model = WhisperModel(model, device=device, compute_type=compute_type)
        self.language = language
        self.beam_size = beam_size

    def transcribe(self, aud: A.Audio) -> ASRResult:
        t0 = time.time()
        segs: list[Segment] = []
        for ch, x in enumerate(aud.channels):
            it, _ = self.model.transcribe(
                x,
                language=self.language,
                beam_size=self.beam_size,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 400},
                condition_on_previous_text=False,
                initial_prompt=DOMAIN_PROMPT,
                no_speech_threshold=0.6,
                log_prob_threshold=-1.0,
            )
            for s in it:
                text = s.text.strip()
                if text and not _hallucinated(text):
                    segs.append(Segment(ch, float(s.start), float(s.end), text))
        return ASRResult(segs, self.name, time.time() - t0)


class QwenASR:
    name = "qwen"

    def __init__(self, model: str = "Qwen/Qwen3-ASR-1.7B", language: str | None = "Hindi",
                 batch_size: int = 16, context: str | None = None):
        import torch
        from qwen_asr import Qwen3ASRModel

        dtype = torch.float16 if _cuda() else torch.float32
        if not _cuda():
            batch_size = min(batch_size, 4)  # on CPU big batches pad to the longest chunk: 16 was 40% slower than 4
        # second GPU when there is one, so the LLM can own cuda:0 untouched
        device = ("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0") if _cuda() else "cpu"
        self.model = Qwen3ASRModel.from_pretrained(
            model, dtype=dtype, device_map=device,
            max_inference_batch_size=batch_size, max_new_tokens=256,
        )
        self.language = language
        self.batch_size = batch_size
        self.context = context or ""  # vocabulary hint passed to the model with every chunk (domain terms, spellings)

    def _jobs(self, aud: A.Audio) -> list[tuple[int, float, float, np.ndarray]]:
        jobs = []
        for ch, x in enumerate(aud.channels):
            for s, e in A.speech_chunks(x, aud.sr):
                jobs.append((ch, s, e, A.slice_audio(x, s, e, aud.sr)))
        return jobs

    def _run(self, clips: list[tuple[np.ndarray, int]]) -> list[str]:
        out: list[str] = []
        for i in range(0, len(clips), self.batch_size):
            results = self.model.transcribe(audio=clips[i : i + self.batch_size], language=self.language,
                                            context=self.context)
            if not isinstance(results, list):
                results = [results]
            for r in results:
                text = (getattr(r, "text", None) or "").strip()
                if self.context and text and _echoes(text, self.context):
                    text = ""  # the model repeated the vocabulary hint instead of transcribing (short / silent chunk)
                out.append(text)
        return out

    def transcribe(self, aud: A.Audio) -> ASRResult:
        return self.transcribe_many([aud])[0]

    def transcribe_many(self, auds: list[A.Audio]) -> list[ASRResult]:
        """Chunks from several files go through the model together — the recordings
        are short, so per-file batches were mostly half empty."""
        t0 = time.time()
        jobs = [(k, j) for k, aud in enumerate(auds) for j in self._jobs(aud)]
        texts = self._run([(j[3], auds[k].sr) for k, j in jobs]) if jobs else []
        per: list[list[Segment]] = [[] for _ in auds]
        for (k, (ch, s, e, _)), text in zip(jobs, texts):
            if text and not _hallucinated(text):
                per[k].append(Segment(ch, s, e, text))
        dt = (time.time() - t0) / max(1, len(auds))
        return [ASRResult(segs, self.name, dt) for segs in per]


class IndicConformerASR:
    """AI4Bharat IndicConformer through transformers (trust_remote_code). One language per
    call; for mixed Hindi/Gujarati traffic pick the centre's language per product."""

    name = "indic"

    def __init__(self, model: str = "ai4bharat/indic-conformer-600m-multilingual", language: str = "hi",
                 decoding: str = "rnnt"):
        import torch
        from transformers import AutoModel

        self.torch = torch
        self.model = AutoModel.from_pretrained(model, trust_remote_code=True)
        self.language = language
        self.decoding = decoding  # "ctc" (faster) | "rnnt" (usually more accurate)

    def _one(self, clip: np.ndarray, sr: int) -> str:
        wav = self.torch.from_numpy(np.ascontiguousarray(clip, dtype=np.float32))[None, :]
        out = self.model(wav, self.language, self.decoding)
        if isinstance(out, (list, tuple)):
            out = out[0] if out else ""
        return str(out).strip()

    def transcribe(self, aud: A.Audio) -> ASRResult:
        t0 = time.time()
        segs: list[Segment] = []
        for ch, x in enumerate(aud.channels):
            for s, e in A.speech_chunks(x, aud.sr):
                text = self._one(A.slice_audio(x, s, e, aud.sr), aud.sr)
                if text and not _hallucinated(text):
                    segs.append(Segment(ch, s, e, text))
        return ASRResult(segs, self.name, time.time() - t0)


def make_asr(kind: str, **kw):
    if kind == "whisper":
        return WhisperASR(**kw)
    if kind == "qwen":
        return QwenASR(**kw)
    if kind == "indic":
        return IndicConformerASR(**kw)
    if kind == "route":  # language-routed: Qwen for Hinglish, whisper in-language for Tamil/Telugu/Kannada/Gujarati...
        from .route import RoutedASR

        return RoutedASR(**kw)
    raise ValueError(f"unknown ASR backend {kind!r}")


def _echoes(text: str, context: str) -> bool:
    """True when a transcription is mostly the context string itself."""
    words = [w.strip(",.।") for w in text.lower().split()]
    if len(words) < 4:
        return False
    cw = set(w.strip(",.।") for w in context.lower().split())
    return sum(1 for w in words if w in cw) / len(words) >= 0.6


def _cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _hallucinated(text: str) -> bool:
    """Whisper on silence produces a few well-known phrases; drop them."""
    t = text.strip().lower()
    junk = ("subscribe", "thank you for watching", "धन्यवाद।", "please subscribe", "©", "♪")
    if any(j in t for j in junk):
        return True
    words = [w.strip(",.।?!") for w in t.split()]
    words = [w for w in words if w]
    if len(words) >= 6 and len(set(words)) <= 2:
        return True  # one word over and over
    # a short phrase repeated back to back (Whisper's decoder loop): if any 2-4 word
    # n-gram covers more than half the text, it is a loop
    for n in (2, 3, 4):
        if len(words) < 3 * n:
            continue
        grams = [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
        top = max(set(grams), key=grams.count)
        if grams.count(top) * n > 0.5 * len(words) and grams.count(top) >= 3:
            return True
    return False
