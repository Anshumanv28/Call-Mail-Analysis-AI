"""Speaker roles v2 (AGENT / CANDIDATE / SYSTEM) from words + voice. Trained by scripts/build_turns_v2.py.

Every segment needs "text" and "feat" = voice_features(segment audio at 16 kHz). The agent's headset and the
caller's phone sound different, so within one call the voice features split the segments into two speakers;
the words decide which speaker is the agent.

    from callai.turns2 import TurnLabeller2, voice_features
    seg["feat"] = voice_features(audio16k[int(s * 16000):int(e * 16000)])
    tl = TurnLabeller2.load()
    tl.transcript(segs)       # "AGENT: namaskar ...\nCANDIDATE: sir mera ..."
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from .romanize import Romanizer
from .turns import features, viterbi

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "turns_v2.pkl"
NF = 42


def voice_features(y: np.ndarray, sr: int = 16000) -> list[float]:
    """20 MFCC means + 20 MFCC stds + log energy + spectral centroid (kHz). Same code as the Kaggle worker."""
    import librosa

    y = np.asarray(y, dtype=np.float32)
    if len(y) < 1600:
        y = np.pad(y, (0, 1600 - len(y)))
    m = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    e = float(np.log(np.mean(y ** 2) + 1e-9))
    c = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    return [round(float(v), 3) for v in list(m.mean(1)) + list(m.std(1))] + [round(e, 3), round(c / 1000, 3)]


def dense_features(segs: list[dict], tlogp: np.ndarray) -> np.ndarray:
    """One row per segment. tlogp: (n, 3) text-model log-probs in order AGENT, CANDIDATE, SYSTEM."""
    n = len(segs)
    F = np.array([s.get("feat") or [0.0] * NF for s in segs], dtype=np.float32)
    has = float(bool(segs and segs[0].get("feat")))
    mu, sd = F.mean(0), F.std(0) + 1e-3
    Z = (F - mu) / sd if n > 1 else np.zeros_like(F)
    lo = tlogp[:, 1] - tlogp[:, 0]                          # text log-odds CANDIDATE vs AGENT
    D = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).mean(-1)) if n > 1 else np.zeros((1, 1))
    side = np.zeros(n)
    share = np.zeros(n)
    if n >= 4 and has:
        from sklearn.cluster import KMeans

        k = KMeans(2, n_init=4, random_state=0).fit_predict(Z)
        for c in (0, 1):
            if (k == c).any() and (k != c).any():
                side[k == c] = lo[k == c].mean() - lo[k != c].mean()
                share[k == c] = (k == c).mean()
    knn = np.zeros(n)
    if n >= 3 and has:
        for i in range(n):
            order = [j for j in np.argsort(D[i]) if j != i][:3]
            knn[i] = lo[order].mean()
    rows = []
    for i in range(n):
        start, end = float(segs[i].get("start", 0)), float(segs[i].get("end", 0))
        prev_end = float(segs[i - 1].get("end", 0)) if i else 0.0
        pl = tlogp[i - 1] if i else np.zeros(3)
        nl = tlogp[i + 1] if i < n - 1 else np.zeros(3)
        dp = D[i, i - 1] if i else -1.0
        dn = D[i, i + 1] if i < n - 1 else -1.0
        rows.append(np.concatenate([
            tlogp[i], pl, nl, F[i], Z[i],
            [dp, dn, side[i], share[i], knn[i], lo[i] - knn[i], i / max(1, n - 1), float(i == 0), float(i == n - 1),
             n, end - start, max(0.0, start - prev_end), len((segs[i].get("text") or "").split()), has]]))
    return np.array(rows, dtype=np.float32)


class TurnLabeller2:
    def __init__(self, m: dict, romanizer: Romanizer | None = None):
        self.m = m
        self.rom = romanizer or Romanizer.load()
        self.classes = m["classes"]
        self.gb_idx = [list(m["gb"].classes_).index(c) for c in self.classes]

    @classmethod
    def load(cls, path: str | Path = MODEL, romanizer: Romanizer | None = None) -> "TurnLabeller2":
        return cls(pickle.loads(Path(path).read_bytes()), romanizer)

    def _text_logp(self, segs, texts):
        P = self.m["tclf"].predict_proba(self.m["tvec"].transform(features(segs, texts)))
        out = np.full((len(segs), 3), 1e-6)
        for j, c in enumerate(self.m["tclf"].classes_):
            out[:, self.classes.index(c)] = P[:, j]
        return np.log(np.clip(out, 1e-6, 1))

    def roles(self, segs: list[dict], texts: list[str] | None = None) -> list[str]:
        if not segs:
            return []
        texts = texts if texts is not None else [self.rom.text(s.get("text", "")) for s in segs]
        P = self.m["gb"].predict_proba(dense_features(segs, self._text_logp(segs, texts)))[:, self.gb_idx]
        path = viterbi(np.log(np.clip(P, 1e-6, 1)), self.m["trans"], self.m["start"])
        return [self.classes[j] for j in path]

    def transcript(self, segs: list[dict]) -> str:
        segs = [s for s in (segs or []) if isinstance(s, dict) and (s.get("text") or "").strip()]
        texts = [self.rom.text(s["text"]).strip() for s in segs]
        lines: list[list[str]] = []
        for role, t in zip(self.roles(segs, texts), texts):
            if not t:
                continue
            if lines and lines[-1][0] == role:
                lines[-1][1] += " " + t
            else:
                lines.append([role, t])
        return chr(10).join(f"{r}: {t}" for r, t in lines)
