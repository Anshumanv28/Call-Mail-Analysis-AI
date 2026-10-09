"""Category / sub-category the way the portal's Gemini labels them (trained on Gemini's labels, not on rules).

Base models, each giving a score per sub-category:
  svm    word + char tf-idf of the transcript, linear SVM (trained on our transcript and Gemini's transcript of each call)
  emb    bge-m3 embedding of the transcript, logistic regression
  knn    bge-m3 nearest labelled calls (k=15), similarity-weighted vote shares
  muril  optional: fine-tuned MuRIL classifier probabilities (romanised Indian text), passed in by the caller
Re-ranker (gradient-boosted trees): candidates = union of every model's top 5; per candidate the models' scores,
ranks, gaps to their best, category agreement and the label's prior -> "is this Gemini's label"; best candidate wins.

    from callai.gemini_cls import GeminiClassifier
    gc = GeminiClassifier.load(path)
    gc.predict(texts, E)  -> [{"sub": name, "cat": name, "confidence": p, "top": [(name, p), ...]}, ...]
Training: scripts/train_gemini_cls.py.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import scipy.sparse as sp

K_NN = 15
TOP = int(__import__("os").environ.get("GEMCLS_TOP", "5"))


def knn_share(Etr: np.ndarray, ytr_idx: np.ndarray, Eq: np.ndarray, n_cls: int, k: int = K_NN) -> np.ndarray:
    S = Eq @ Etr.T
    k = min(k, Etr.shape[0] - 1)
    idx = np.argpartition(-S, k, axis=1)[:, :k]
    out = np.zeros((len(Eq), n_cls))
    for i in range(len(Eq)):
        w = np.maximum(S[i, idx[i]], 0) ** 3
        np.add.at(out[i], ytr_idx[idx[i]], w)
        out[i] /= out[i].sum() + 1e-9
    return out


def to_full(model, scores: np.ndarray, ci: dict, n_cls: int, fill: float) -> np.ndarray:
    out = np.full((scores.shape[0], n_cls), fill)
    for j, c in enumerate(model.classes_):
        out[:, ci[c]] = scores[:, j]
    return out


def candidate_rows(P: dict, models: list, cat_of: np.ndarray, n_cats: int, log_prior: np.ndarray, n: int):
    ranks = {m: np.argsort(np.argsort(-P[m], axis=1), axis=1) for m in models}
    best = {m: P[m].max(1) for m in models}
    catscore = {m: np.stack([P[m][:, cat_of == c].max(1) for c in range(n_cats)], 1) for m in models}
    catbest = {m: catscore[m].argmax(1) for m in models}
    R, F = [], []
    for i in range(n):
        cand = set()
        for m in models:
            cand |= set(np.argsort(-P[m][i])[:TOP].tolist())
        for j in sorted(cand):
            f = []
            for m in models:
                f += [P[m][i, j], ranks[m][i, j], P[m][i, j] - best[m][i], catscore[m][i, cat_of[j]],
                      float(catbest[m][i] == cat_of[j])]
            f += [log_prior[j], cat_of[j]]
            R.append((i, j))
            F.append(f)
    return R, np.array(F, dtype=np.float32)


class GeminiClassifier:
    def __init__(self, d: dict):
        self.__dict__.update(d)
        self.ci = {c: i for i, c in enumerate(self.labels)}

    @classmethod
    def load(cls, path: str | Path) -> "GeminiClassifier":
        return cls(pickle.loads(Path(path).read_bytes()))

    def base_scores(self, texts: list[str], E: np.ndarray | None, muril: np.ndarray | None = None) -> dict:
        n = len(self.labels)
        X = sp.hstack([self.vec_w.transform(texts), self.vec_c.transform(texts)]).tocsr()
        P = {"svm": to_full(self.svm, self.svm.decision_function(X), self.ci, n, -1e3)}
        if "emb" in self.models:
            P["emb"] = to_full(self.emb_lr, self.emb_lr.predict_proba(E), self.ci, n, 0.0)
            P["knn"] = knn_share(self.knn_E.astype(np.float32), self.knn_y, E, n)
        if "muril" in self.models:
            P["muril"] = muril if muril is not None else np.zeros((len(texts), n))
        return P

    def predict(self, texts: list[str], E: np.ndarray | None = None, muril: np.ndarray | None = None) -> list[dict]:
        if not texts:
            return []
        P = self.base_scores(texts, E, muril)
        R, F = candidate_rows(P, self.models, self.cat_of, len(self.cats), self.log_prior, len(texts))
        s = self.gb.predict_proba(F)[:, 1]
        per: list[list] = [[] for _ in texts]
        for (i, j), v in zip(R, s):
            per[i].append((float(v), self.labels[j]))
        out = []
        for cands in per:
            cands.sort(reverse=True)
            tot = sum(v for v, _ in cands) or 1.0
            top = [(c, round(v / tot, 3)) for v, c in cands[:TOP]]
            out.append({"sub": top[0][0], "cat": self.sub2cat[top[0][0]], "confidence": top[0][1], "top": top})
        return out


def muril_scores(model_dir: str | Path, texts: list[str], labels: list[str], device: str = "cuda",
                 max_len: int = 512, batch: int = 32) -> np.ndarray:
    """Probabilities of the fine-tuned MuRIL classifier (kaggle/build_cls_notebook.py), columns in `labels` order.
    Long transcripts keep their head and last 128 tokens, exactly as in training."""
    import json
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_dir = Path(model_dir)
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    if device == "cuda":
        model = model.half()
    mlab = json.loads((model_dir / "labels.json").read_text(encoding="utf-8"))
    col = {c: i for i, c in enumerate(labels)}
    seqs = []
    for t in texts:
        ids = tok(str(t), add_special_tokens=False)["input_ids"]
        room = max_len - 2
        if len(ids) > room:
            ids = ids[:room - 128] + ids[-128:]
        seqs.append([tok.cls_token_id] + ids + [tok.sep_token_id])
    out = np.zeros((len(texts), len(labels)), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(seqs), batch):
            b = seqs[i:i + batch]
            L = max(len(s) for s in b)
            ids = torch.tensor([s + [tok.pad_token_id] * (L - len(s)) for s in b], device=device)
            am = torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in b], device=device)
            P = torch.softmax(model(input_ids=ids, attention_mask=am).logits.float(), -1).cpu().numpy()
            for j, c in enumerate(mlab):
                if c in col:
                    out[i:i + len(b), col[c]] = P[:, j]
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return out
