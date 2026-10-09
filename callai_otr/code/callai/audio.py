"""Audio loading, channel handling, normalisation and a light energy VAD.

The ConnectView recordings are stereo 16 kHz with the two parties on separate channels,
so we never diarise: each channel is transcribed on its own and the segments are merged
by time into a dialogue.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000
SILENT_RMS = 10 ** (-55 / 20)  # -55 dBFS: below this a channel is treated as silent, never normalised up
MIN_SNR_DB = 6.0  # a VAD chunk must sit this far above the channel's noise floor to be transcribed


@dataclasses.dataclass
class Audio:
    channels: list[np.ndarray]  # each float32 mono at SR
    sr: int
    path: str

    @property
    def duration(self) -> float:
        return len(self.channels[0]) / self.sr

    @property
    def stereo(self) -> bool:
        return len(self.channels) == 2


def load(path: str | Path) -> Audio:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)  # (n, ch)
    if sr != SR:
        data = _resample(data, sr, SR)
        sr = SR
    chans = [np.ascontiguousarray(data[:, i]) for i in range(data.shape[1])]
    if len(chans) > 2:  # unexpected multichannel: keep the first two
        chans = chans[:2]
    # A "stereo" file whose channels are identical is really mono; collapse it so the
    # dialogue does not show every line twice.
    if len(chans) == 2 and _same(chans[0], chans[1]):
        chans = [chans[0]]
    return Audio(channels=[normalise(c) for c in chans], sr=sr, path=str(path))


def _resample(data: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(sr_in, sr_out)
    return resample_poly(data, sr_out // g, sr_in // g, axis=0).astype(np.float32)


def _same(a: np.ndarray, b: np.ndarray) -> bool:
    n = min(len(a), len(b), SR * 30)
    if n == 0:
        return True
    return float(np.max(np.abs(a[:n] - b[:n]))) < 1e-4


def normalise(x: np.ndarray, target_rms: float = 0.06, peak: float = 0.97) -> np.ndarray:
    """RMS-normalise a telephone channel so the quiet party is not lost by the ASR."""
    x = x.astype(np.float32)
    rms = float(np.sqrt(np.mean(x**2))) if len(x) else 0.0
    if rms < SILENT_RMS:  # a dead or near-silent channel: boosting it turns line noise into "speech"
        return x
    y = x * (target_rms / rms)
    m = float(np.max(np.abs(y)))
    if m > peak:
        y = y * (peak / m)
    return y


def vad_chunks(
    x: np.ndarray,
    sr: int = SR,
    frame_ms: int = 30,
    min_silence_ms: int = 500,
    min_speech_ms: int = 250,
    max_chunk_s: float = 25.0,
    pad_ms: int = 150,
    min_snr_db: float = MIN_SNR_DB,
) -> list[tuple[float, float]]:
    """Energy-based utterance chunking. Returns (start_s, end_s) pairs.

    Deliberately simple: the threshold adapts to the channel's own noise floor, so a
    quiet caller on a noisy line still splits into utterances. Whisper brings its own
    VAD; this is for backends that do not (Qwen3-ASR)."""
    n = int(sr * frame_ms / 1000)
    if len(x) < n:
        return [(0.0, len(x) / sr)] if len(x) else []
    frames = x[: len(x) - len(x) % n].reshape(-1, n)
    energy = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
    floor = np.percentile(energy, 20)
    thr = max(floor * 3.0, 0.004)
    speech = energy > thr

    min_sil = max(1, min_silence_ms // frame_ms)
    min_sp = max(1, min_speech_ms // frame_ms)
    chunks: list[tuple[int, int]] = []
    i, start, silence = 0, None, 0
    for i, s in enumerate(speech):
        if s:
            if start is None:
                start = i
            silence = 0
        elif start is not None:
            silence += 1
            if silence >= min_sil:
                if i - silence - start >= min_sp:
                    chunks.append((start, i - silence + 1))
                start, silence = None, 0
    if start is not None and len(speech) - start >= min_sp:
        chunks.append((start, len(speech)))

    out: list[tuple[float, float]] = []
    pad = pad_ms / 1000
    for a, b in chunks:
        # noise gate: the loud part of the chunk must clear the noise floor by min_snr_db,
        # otherwise it is line noise / breathing / crosstalk that the ASR would turn into words
        peak = float(np.percentile(energy[a:b], 90)) if b > a else 0.0
        if 20 * np.log10(peak / (floor + 1e-9) + 1e-9) < min_snr_db:
            continue
        s, e = a * frame_ms / 1000, b * frame_ms / 1000
        # split over-long runs so no chunk exceeds max_chunk_s
        while e - s > max_chunk_s:
            out.append((max(0.0, s - pad), s + max_chunk_s))
            s += max_chunk_s
        out.append((max(0.0, s - pad), min(len(x) / sr, e + pad)))
    return out


_SILERO = None


def _silero():
    """Silero VAD (2 MB, MIT): a neural speech detector, far better than energy on line noise,
    breathing and crosstalk. Loaded once; None when the package is not installed."""
    global _SILERO
    if _SILERO is None:
        try:
            from silero_vad import get_speech_timestamps, load_silero_vad

            _SILERO = (load_silero_vad(), get_speech_timestamps)
        except Exception:
            _SILERO = False
    return _SILERO or None


def speech_chunks(x: np.ndarray, sr: int = SR, max_chunk_s: float = 25.0, pad_ms: int = 150,
                  min_speech_ms: int = 250, min_silence_ms: int = 500) -> list[tuple[float, float]]:
    """Utterance chunks for ASR backends without their own VAD. Silero when available,
    the energy VAD otherwise. Same (start_s, end_s) contract as vad_chunks.

    Opt-in variants via the CALLAI_CHUNKING environment variable (default unchanged):
      gentle  softer speech threshold, 300 ms padding, and neighbouring chunks less than 1 s apart merged
              (up to max_chunk_s): short quiet words between chunks (hai, ji, sir) are no longer dropped;
      whole   the whole recording as one chunk (Qwen3-ASR accepts up to 1200 s), no edges at all."""
    import os

    mode = os.environ.get("CALLAI_CHUNKING", "").strip().lower()
    if mode == "whole":
        dur = len(x) / sr
        step = 1100.0
        return [(t, min(dur, t + step)) for t in np.arange(0.0, max(dur, 0.01), step)]
    if mode == "gentle":
        return _gentle_chunks(x, sr, max_chunk_s)
    v = _silero()
    if v is None or len(x) < sr // 10:
        return vad_chunks(x, sr, max_chunk_s=max_chunk_s, pad_ms=pad_ms)
    import torch

    model, get_ts = v
    ts = get_ts(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)), model, sampling_rate=sr,
                threshold=0.5, min_speech_duration_ms=min_speech_ms, min_silence_duration_ms=min_silence_ms)
    out: list[tuple[float, float]] = []
    pad = pad_ms / 1000
    for t in ts:
        s, e = t["start"] / sr, t["end"] / sr
        while e - s > max_chunk_s:
            out.append((max(0.0, s - pad), s + max_chunk_s))
            s += max_chunk_s
        out.append((max(0.0, s - pad), min(len(x) / sr, e + pad)))
    return out


def _gentle_chunks(x: np.ndarray, sr: int, max_chunk_s: float, merge_gap_s: float = 1.0,
                   pad_s: float = 0.3, threshold: float = 0.35) -> list[tuple[float, float]]:
    v = _silero()
    if v is None or len(x) < sr // 10:
        return vad_chunks(x, sr, max_chunk_s=max_chunk_s, pad_ms=int(pad_s * 1000))
    import torch

    model, get_ts = v
    ts = get_ts(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)), model, sampling_rate=sr,
                threshold=threshold, min_speech_duration_ms=150, min_silence_duration_ms=300)
    dur = len(x) / sr
    spans = [(t["start"] / sr, t["end"] / sr) for t in ts]
    merged: list[list[float]] = []
    for s0, e0 in spans:
        if merged and s0 - merged[-1][1] < merge_gap_s and e0 - merged[-1][0] <= max_chunk_s:
            merged[-1][1] = e0
        else:
            merged.append([s0, e0])
    out: list[tuple[float, float]] = []
    for s0, e0 in merged:
        s0, e0 = max(0.0, s0 - pad_s), min(dur, e0 + pad_s)
        while e0 - s0 > max_chunk_s:
            out.append((s0, s0 + max_chunk_s))
            s0 += max_chunk_s - 1.0               # 1 s overlap so a word on the cut is heard whole once
        out.append((s0, e0))
    return out


def slice_audio(x: np.ndarray, start_s: float, end_s: float, sr: int = SR) -> np.ndarray:
    return x[int(start_s * sr) : int(end_s * sr)]
