"""Retrieval of labelled example calls for dynamic few-shot prompting.

    pool = ExamplePool.build(transcripts_jsonl, labels_csv)     # embeds every labelled call once
    pool.nearest(dialogue, k=6) -> [{"excerpt", "sub_issue", "why"}, ...]

Embeddings: BAAI/bge-m3 (multilingual, handles Devanagari + Hinglish), cosine similarity.
The pool is the labelled set the model may learn from; never the set being scored.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

EXCERPT = 160


class ExamplePool:
    def __init__(self, texts: list[str], labels: list[str], notes: list[str], model: str = "BAAI/bge-m3"):
        self._model_name, self._model = model, None   # loaded on demand: a fully cached pool never needs it
        self.texts, self.labels, self.notes = texts, labels, notes
        self.emb = self._cached_encode(texts)

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        return self._model

    @model.setter
    def model(self, m):
        self._model = m

    CACHE_DIR = Path(__file__).resolve().parent.parent / "out" / "emb_cache" / "rows"

    @staticmethod
    def _row_key(text: str) -> str:
        import hashlib

        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:20]

    def _cached_encode(self, texts: list[str]):
        """Embeddings cached per transcript (out/emb_cache/rows/<sha1>.npy): any pool, subset or
        re-ordering reuses what was embedded before; only never-seen transcripts hit the model."""
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        keys = [self._row_key(t) for t in texts]
        out = [None] * len(texts)
        missing = []
        for i, k in enumerate(keys):
            f = self.CACHE_DIR / f"{k}.npy"
            if f.exists():
                try:
                    out[i] = np.load(f)
                except Exception:  # noqa: BLE001 - a half-written file is re-embedded
                    missing.append(i)
            else:
                missing.append(i)
        if missing:
            uniq = list(dict.fromkeys(texts[i] for i in missing))
            emb = self.model.encode(uniq, normalize_embeddings=True, batch_size=8, show_progress_bar=False)
            by_text = dict(zip(uniq, emb))
            for i in missing:
                v = np.asarray(by_text[texts[i]], dtype=np.float32)
                out[i] = v
                tmp = self.CACHE_DIR / f"{keys[i]}.tmp.npy"
                np.save(tmp, v)
                tmp.replace(self.CACHE_DIR / f"{keys[i]}.npy")
        return np.vstack(out).astype(np.float32)

    @classmethod
    def build(cls, transcripts: str | Path, labels: str | Path, confident_only: bool = True, model: str = "BAAI/bge-m3"):
        rows = {json.loads(l)["file"]: json.loads(l)["dialogue"] for l in Path(transcripts).read_text(encoding="utf-8").split(chr(10)) if l.strip()}
        g = pd.read_csv(labels, dtype=str).fillna("")
        if confident_only:
            g = g[g.confidence != "low"]
        texts, labs, notes = [], [], []
        for _, r in g.iterrows():
            d = rows.get(r.file, "")
            if len(d.split()) <= 3:
                continue
            texts.append(d)
            labs.append(r.sub_issue)
            notes.append(r.note)
        return cls(texts, labs, notes, model)

    def knn(self, dialogue: str, k: int = 3, exclude_text: str | None = None, vec=None) -> dict:
        """Nearest-neighbour label: the (distance-weighted) majority label of the k nearest
        labelled calls, whether the k neighbours were unanimous, and their mean similarity.
        Unanimous k=3 neighbours were right 96 % of the time on the held-out set."""
        q = self._cached_encode([dialogue])[0] if vec is None else vec   # cached: pre-warmed calls never load the model
        sims = self.emb @ q
        order = [i for i in np.argsort(-sims) if exclude_text is None or self.texts[i] != exclude_text][:k]
        weights: dict[str, float] = {}
        for i in order:
            weights[self.labels[i]] = weights.get(self.labels[i], 0.0) + max(float(sims[i]), 1e-6)
        label = max(weights, key=weights.get)
        return {"label": label, "unanimous": len(weights) == 1, "sim": float(np.mean([sims[i] for i in order])),
                "neighbours": [self.labels[i] for i in order]}

    def nearest(self, dialogue: str, k: int = 6, exclude_text: str | None = None) -> list[dict]:
        q = self._cached_encode([dialogue])[0]
        sims = self.emb @ q
        order = np.argsort(-sims)
        out = []
        for i in order:
            if exclude_text is not None and self.texts[i] == exclude_text:
                continue
            out.append({"excerpt": self.texts[i].replace("\n", " | ")[:EXCERPT], "sub_issue": self.labels[i],
                        "why": self.notes[i][:60], "sim": float(sims[i])})
            if len(out) >= k:
                break
        return out
