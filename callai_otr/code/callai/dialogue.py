"""Merge per-channel ASR segments into one time-ordered dialogue.

Speakers are labelled by channel (A = left, B = right). We do not guess which one is the
caller; the classifier sees the whole conversation and does not need to know. If the
vendor confirms the channel order, set PipelineConfig.speaker_names.
"""

from __future__ import annotations

from .asr import ASRResult, Segment


def build(res: ASRResult, names: tuple[str, str] = ("A", "B")) -> str:
    segs = sorted(res.segments, key=lambda s: (s.start, s.channel))
    lines: list[str] = []
    last_ch, buf = None, []
    for s in segs:
        if s.channel != last_ch and buf:
            lines.append(f"{names[last_ch]}: {' '.join(buf)}")
            buf = []
        buf.append(s.text)
        last_ch = s.channel
    if buf:
        lines.append(f"{names[last_ch]}: {' '.join(buf)}")
    return "\n".join(lines)


def per_channel(res: ASRResult, names: tuple[str, str] = ("A", "B")) -> dict[str, str]:
    return {names[ch]: res.channel_text(ch) for ch in (0, 1) if any(s.channel == ch for s in res.segments)}


def word_count(res: ASRResult) -> int:
    return sum(len(s.text.split()) for s in res.segments)


__all__ = ["build", "per_channel", "word_count", "Segment"]
