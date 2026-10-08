"""Speaker turns in the portal's layout (CANDIDATE: / AGENT: / SYSTEM:) for our ASR output.

Our ASR splits a call into VAD segments (about as many as the portal has turns). Each segment gets a role
from a classifier trained on calls where the portal's Gemini transcript says who spoke (scripts/build_turns.py):
the segment's words, its neighbours' words, its position, length and the pause before it, then a Viterbi pass
with learned role-to-role transitions. Consecutive segments with the same role are merged into one turn.

    from callai.turns import TurnLabeller
    tl = TurnLabeller.load()
    print(tl.transcript(row["segments"]))       # "CANDIDATE: hello sir ...\nAGENT: ji bataiye ..."
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np

from .romanize import Romanizer

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "turns_model.pkl"
ROLES = ["CANDIDATE", "AGENT", "SYSTEM"]
_W = re.compile(r"[a-z0-9']+")


def seg_words(text: str) -> list[str]:
    return _W.findall((text or "").lower())


def features(segs: list[dict], texts: list[str]) -> list[dict]:
    """One feature dict per segment. texts = romanized segment texts."""
    n = len(segs)
    words = [seg_words(t) for t in texts]
    out = []
    for i in range(n):
        f = {}
        for w in words[i]:
            f["w=" + w] = 1.0
        for a, b in zip(words[i], words[i][1:]):
            f["b=" + a + "_" + b] = 1.0
        if words[i]:
            f["first=" + words[i][0]] = 1.0
            f["last=" + words[i][-1]] = 1.0
        for off, tag in ((-1, "p"), (1, "n")):
            j = i + off
            if 0 <= j < n:
                for w in words[j][:12]:
                    f[tag + "=" + w] = 0.5
            else:
                f[tag + "=NONE"] = 1.0
        start, end = float(segs[i].get("start", 0)), float(segs[i].get("end", 0))
        prev_end = float(segs[i - 1].get("end", 0)) if i else 0.0
        f["pos=%d" % min(i, 4)] = 1.0
        f["rpos=%d" % int(10 * i / max(1, n))] = 1.0
        f["islast"] = 1.0 if i == n - 1 else 0.0
        f["nw=%d" % min(len(words[i]), 12)] = 1.0
        f["dur=%d" % min(int(end - start), 10)] = 1.0
        f["gap=%d" % min(int(2 * max(0.0, start - prev_end)), 6)] = 1.0
        out.append(f)
    return out


def viterbi(logp: np.ndarray, trans: np.ndarray, start: np.ndarray) -> list[int]:
    n, k = logp.shape
    score = start + logp[0]
    back = np.zeros((n, k), dtype=int)
    for i in range(1, n):
        cand = score[:, None] + trans
        back[i] = cand.argmax(0)
        score = cand.max(0) + logp[i]
    path = [int(score.argmax())]
    for i in range(n - 1, 0, -1):
        path.append(int(back[i][path[-1]]))
    return path[::-1]


class TurnLabeller:
    def __init__(self, vec, clf, trans, start, romanizer: Romanizer | None = None):
        self.vec, self.clf, self.trans, self.start = vec, clf, trans, start
        self.rom = romanizer or Romanizer.load()
        self.classes = list(clf.classes_)

    @classmethod
    def load(cls, path: str | Path = MODEL, romanizer: Romanizer | None = None) -> "TurnLabeller":
        d = pickle.loads(Path(path).read_bytes())
        return cls(d["vec"], d["clf"], d["trans"], d["start"], romanizer)

    def roles(self, segs: list[dict], texts: list[str] | None = None) -> list[str]:
        if not segs:
            return []
        texts = texts if texts is not None else [self.rom.text(s.get("text", "")) for s in segs]
        X = self.vec.transform(features(segs, texts))
        logp = np.log(np.clip(self.clf.predict_proba(X), 1e-6, 1))
        path = viterbi(logp, self.trans, self.start)
        return [self.classes[j] for j in path]

    def transcript(self, segs: list[dict]) -> str:
        """Gemini-layout transcript: one line per turn, consecutive same-role segments merged."""
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


class VoiceTurnLabeller:
    """Roles from words + voice fingerprints (scripts/voice_role_model.py, models/turns_voice.pkl): a CANDIDATE vs
    AGENT decision from the text classifier's log-odds + wv x the voice classifier's log-odds, plus a within-call
    2-way clustering of the fingerprints named by the summed evidence (wc). SYSTEM (IVR prompts) is kept from the
    3-class text labeller, which the voice model does not know. Falls back to text only when a call has no
    fingerprints (out/voice_emb/<file>.npy, one row per non-empty segment)."""

    def __init__(self, tl: TurnLabeller, vm: dict, emb_dir: str | Path = ROOT / "out" / "voice_emb"):
        self.tl, self.vm, self.emb_dir = tl, vm, Path(emb_dir)
        self.rom = tl.rom

    @classmethod
    def load(cls, romanizer: Romanizer | None = None, path: str | Path = ROOT / "models" / "turns_voice.pkl") -> "VoiceTurnLabeller":
        return cls(TurnLabeller.load(romanizer=romanizer), pickle.loads(Path(path).read_bytes()))

    @staticmethod
    def _logodds(classes, P):
        ci, ai = list(classes).index("CANDIDATE"), list(classes).index("AGENT")
        return np.log(P[:, ci] + 1e-6) - np.log(P[:, ai] + 1e-6)

    def embeddings(self, file_id: str | None, n: int):
        if not file_id:
            return None
        p = self.emb_dir / f"{file_id}.npy"
        if not p.exists():
            return None
        E = np.load(p)
        return E if len(E) == n else None

    def roles(self, segs: list[dict], texts: list[str] | None = None, E=None) -> list[str]:
        texts = texts if texts is not None else [self.rom.text(s.get("text", "")) for s in segs]
        base = self.tl.roles(segs, texts)
        if E is None or len(segs) < 2:
            return base
        vm = self.vm
        lt = self._logodds(vm["tclf"].classes_, vm["tclf"].predict_proba(vm["tvec"].transform(features(segs, texts))))
        lv = self._logodds(vm["vclf"].classes_, vm["vclf"].predict_proba(E))
        s = lt + vm["wv"] * lv
        if vm["wc"] and len(segs) >= 4:
            from sklearn.cluster import KMeans

            k = KMeans(2, n_init=5, random_state=0).fit_predict(E)
            side = np.where(k == 0, 1.0, -1.0) * (1.0 if s[k == 0].sum() >= s[k == 1].sum() else -1.0)
            s = s + vm["wc"] * side
        return ["SYSTEM" if b == "SYSTEM" else ("CANDIDATE" if v >= 0 else "AGENT") for b, v in zip(base, s)]

    def transcript(self, segs: list[dict], file_id: str | None = None) -> tuple[str, bool]:
        """(Gemini-layout transcript, used_voice)."""
        segs = [s for s in (segs or []) if isinstance(s, dict) and (s.get("text") or "").strip()]
        texts = [self.rom.text(s["text"]).strip() for s in segs]
        E = self.embeddings(file_id, len(segs))
        lines: list[list[str]] = []
        for role, t in zip(self.roles(segs, texts, E), texts):
            if not t:
                continue
            if lines and lines[-1][0] == role:
                lines[-1][1] += " " + t
            else:
                lines.append([role, t])
        return chr(10).join(f"{r}: {t}" for r, t in lines), E is not None
