"""Deterministic ASR-confusion fixes: a per-product lookup of spellings the ASR actually
produces for known domain terms, applied to the transcript before classification.

    products/<name>/asr_fixes.yaml
      fixes:
        - heard: ["सिक्सटी परसेंट", "सीसीटीवी परसेंट", "सीसीटीवी परसना"]
          meant: "CCTV person"

No model, no invention: only strings that appear in the list are touched, longest first.
The raw transcript is kept alongside (`dialogue_raw`).
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml


class Normaliser:
    def __init__(self, pairs: list[tuple[str, str]]):
        # longest patterns first so "सीसीटीवी स्पोक" wins over "स्पोक"
        pairs = sorted(pairs, key=lambda p: -len(p[0]))
        self.rules = [(re.compile(re.escape(h)), m) for h, m in pairs]

    @classmethod
    def load(cls, path: str | Path | None) -> "Normaliser | None":
        if not path or not Path(path).exists():
            return None
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        pairs = []
        for f in raw.get("fixes", []):
            heard = f.get("heard") or []
            if isinstance(heard, str):
                heard = [heard]
            for h in heard:
                pairs.append((h, f["meant"]))
        return cls(pairs) if pairs else None

    def apply(self, text: str) -> tuple[str, int]:
        n = 0
        for rx, meant in self.rules:
            text, k = rx.subn(meant, text)
            n += k
        return text, n
