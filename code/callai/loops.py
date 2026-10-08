"""Collapse decoder repetition loops in ASR text (Whisper-style "sir, sir, sir, sir, ..." / "primary primary ...").

A 1-4 token phrase repeated back to back more than `keep` times is cut down to `keep` repetitions. Measured on the
300 held-out OTR calls with the fine-tuned Whisper: keep=2 takes words-differing-from-Gemini from 28.4% to 24.5%
(extra words 7.2% -> 4.1%); keep=1 is worse (removes legitimate "haan haan").
"""
from __future__ import annotations

import re

_KEY = re.compile(r"[^\w']")


def collapse_loops(text: str, keep: int = 2, max_n: int = 4) -> str:
    toks = (text or "").split()
    key = [_KEY.sub("", t.lower()) for t in toks]
    out, i = [], 0
    while i < len(toks):
        best = None
        for n in range(1, max_n + 1):
            if i + n > len(toks):
                break
            reps = 1
            while i + (reps + 1) * n <= len(toks) and key[i + reps * n:i + (reps + 1) * n] == key[i:i + n]:
                reps += 1
            if reps > keep and (best is None or reps * n > best[0] * best[1]):
                best = (reps, n)
        if best:
            reps, n = best
            out += toks[i:i + keep * n]
            i += reps * n
        else:
            out.append(toks[i])
            i += 1
    return " ".join(out)
