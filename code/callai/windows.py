"""Cut a call into decode windows that overlap, and stitch the text back without repeats.

Windows used to abut: each ended where the next began. Words sitting on that seam were lost by
whichever side clipped them — 30% of all missed words landed exactly on a boundary although
boundaries are only 7.4% of word positions, about 4 points of word error.

Overlapping the windows means a seam word is inside both, so it survives. The cost is that the
overlap is transcribed twice, which `stitch` removes by finding where the tail of one window and the
head of the next say the same thing.

    wins = windows_for(segments, overlap=0.8)     # (start, end) pairs, deliberately overlapping
    text = stitch(window_texts)                   # one transcript, no duplicated seam

Both sides must agree: training windows and inference windows have to be cut the same way, or the
model is asked at serving time for something it never saw.
"""
from __future__ import annotations

import re

MAX_WIN = 28.0
MIN_CUT_GAP = 0.8
PAD = 0.3
MAX_LEN = 30.0          # the encoder's own limit
WORD = re.compile(r"[^\W\d_]+|\d+", re.UNICODE)


def windows_for(segs: list[dict], overlap: float = 0.8) -> list[tuple[float, float]]:
    """Group speech segments into windows, each extended by `overlap` seconds at both seams."""
    groups, cur = [], []
    for i, s in enumerate(segs):
        if cur:
            start = segs[cur[0]]["start"]
            gap = s["start"] - segs[cur[-1]]["end"]
            if s["end"] - start > MAX_WIN or (gap >= MIN_CUT_GAP and s["end"] - start > 12.0):
                groups.append(cur)
                cur = []
        cur.append(i)
    if cur:
        groups.append(cur)

    out = []
    for k, idx in enumerate(groups):
        lo, hi = segs[idx[0]]["start"], segs[idx[-1]]["end"]
        lo = max(0.0, lo - (PAD + (overlap if k else 0.0)))
        hi = hi + PAD + (overlap if k + 1 < len(groups) else 0.0)
        if hi - lo > MAX_LEN:                      # keep the end: a clipped tail is what we are fixing
            lo = hi - MAX_LEN
        out.append((round(lo, 2), round(hi, 2)))
    return out


def _words(t: str) -> list[str]:
    return WORD.findall((t or "").lower())


def stitch(texts: list[str], max_probe: int = 14) -> str:
    """Join window texts, dropping the part of each that repeats the end of the previous one."""
    kept: list[str] = []
    prev_tail: list[str] = []
    for t in texts:
        raw = (t or "").strip()
        if not raw:
            continue
        w = _words(raw)
        if not kept:
            kept.append(raw)
            prev_tail = w[-max_probe:]
            continue
        drop = 0
        for n in range(min(max_probe, len(w), len(prev_tail)), 1, -1):
            if prev_tail[-n:] == w[:n]:            # the seam said the same thing twice
                drop = n
                break
        if drop:
            toks = list(WORD.finditer(raw))
            raw = raw[toks[drop - 1].end():].lstrip(" ,.।-") if drop <= len(toks) else ""
        if raw.strip():
            kept.append(raw.strip())
            prev_tail = _words(" ".join(kept[-1:]))[-max_probe:] or prev_tail
    return " ".join(kept).strip()
