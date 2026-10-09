"""Language-routed ASR: detect the language of the louder channel, then transcribe with the
model that can read it.

  hi / en / ur (Hinglish)            -> Qwen3-ASR, Hindi forced (best on these calls)
  ta / te / kn / ml / gu / mr / bn   -> faster-whisper large-v3-turbo in that language
  anything else                      -> Qwen3-ASR, Hindi forced

Whisper's detector reads code-mixed Hinglish as "en" most of the time, so "en" is treated as
Hinglish, not routed away. Only a confident (p >= 0.5) non-Hindi Indian language is routed.
"""

from __future__ import annotations

import time

import numpy as np

from . import audio as A
from .asr import ASRResult, QwenASR, Segment, WhisperASR, _hallucinated

ROUTED = {"ta", "te", "kn", "ml", "gu", "mr", "bn", "pa", "or"}


class RoutedASR:
    name = "route"

    def __init__(self, qwen_kwargs: dict | None = None, whisper_model: str = "large-v3-turbo", min_prob: float = 0.5):
        from faster_whisper import WhisperModel

        self.qwen = QwenASR(**(qwen_kwargs or {}))
        self.detector = WhisperModel(whisper_model, device="cpu", compute_type="int8")
        self.whisper_model = whisper_model
        self._whisper: dict[str, WhisperASR] = {}
        self.min_prob = min_prob
        self.last_route: dict[str, tuple[str, float]] = {}

    def detect(self, aud: A.Audio) -> tuple[str, float]:
        ch = max(aud.channels, key=lambda x: float(np.sqrt(np.mean(x**2))))
        try:
            lang, prob, _ = self.detector.detect_language(audio=ch[: 30 * aud.sr])
            return str(lang), float(prob)
        except Exception:
            return "hi", 0.0

    def _whisper_for(self, lang: str) -> WhisperASR:
        if lang not in self._whisper:
            w = WhisperASR.__new__(WhisperASR)
            w.model = self.detector  # reuse the loaded weights
            w.language = lang
            w.beam_size = 5
            self._whisper[lang] = w
        return self._whisper[lang]

    def transcribe(self, aud: A.Audio) -> ASRResult:
        lang, prob = self.detect(aud)
        self.last_route[aud.path] = (lang, prob)
        if lang in ROUTED and prob >= self.min_prob:
            res = self._whisper_for(lang).transcribe(aud)
            return ASRResult(res.segments, f"route:{lang}", res.seconds)
        res = self.qwen.transcribe(aud)
        return ASRResult(res.segments, "route:hi", res.seconds)

    def transcribe_many(self, auds: list[A.Audio]) -> list[ASRResult]:
        out: list[ASRResult | None] = [None] * len(auds)
        qwen_idx = []
        for i, aud in enumerate(auds):
            lang, prob = self.detect(aud)
            self.last_route[aud.path] = (lang, prob)
            if lang in ROUTED and prob >= self.min_prob:
                res = self._whisper_for(lang).transcribe(aud)
                out[i] = ASRResult(res.segments, f"route:{lang}", res.seconds)
            else:
                qwen_idx.append(i)
        if qwen_idx:
            for i, res in zip(qwen_idx, self.qwen.transcribe_many([auds[i] for i in qwen_idx])):
                out[i] = ASRResult(res.segments, "route:hi", res.seconds)
        return out  # type: ignore[return-value]
