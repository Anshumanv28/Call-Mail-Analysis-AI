"""The pipeline: recording(s) in, transcript + segregation (+ OTR answer) out.

    p = Pipeline(PipelineConfig(product="otr"))
    p.run_one("call.wav")            # dict
    p.run_many([...], out_dir="out") # writes results.jsonl + results.csv, returns rows

ASR is per file; the LLM steps are batched across files, which is where the time goes.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import time
from pathlib import Path

from . import asr as ASR
from . import audio as A
from . import dialogue as D
from .llm import make_llm
from .normalise import Normaliser
from .respond import Responder
from .segregate import Segregator
from .taxonomy import Taxonomy

ROOT = Path(__file__).resolve().parent.parent


@dataclasses.dataclass
class PipelineConfig:
    product: str = "connectview"  # products/<product>/product.yaml decides taxonomy, playbooks, answer, tts
    taxonomy: str | None = None  # resolved from the product unless given
    playbooks: str | None = None
    answer: bool | None = None  # response head on/off (otr: on, connectview: off)
    asr: str = "qwen"  # "qwen" (Qwen3-ASR-1.7B: best Hindi, ~1.5x real time on CPU) | "indic" (IndicConformer: Hindi+Gujarati) | "whisper" (fastest, roughest)
    asr_kwargs: dict = dataclasses.field(default_factory=dict)
    llm: str = "llama"  # "llama" (local llama.cpp server) | "openai" (any OpenAI-compatible server) | "hf" (in-process transformers)
    llm_kwargs: dict = dataclasses.field(default_factory=dict)
    speaker_names: tuple[str, str] = ("A", "B")
    low_confidence: float = 0.55
    context: str | None = None  # the first lines of the classification prompt; from product.yaml `context`
    respond_persona: str | None = None  # who the reply comes from; product.yaml `respond_persona`
    empty_sub_issue: str | None = None  # product.yaml `empty_sub_issue`: where a recording with no speech is filed
    classify_mode: str | None = None  # "single" | "reason" | "two_stage" (product.yaml `classify_mode`, default reason)
    asr_fixes: str | None = None  # products/<name>/asr_fixes.yaml: known ASR mis-hearings -> domain term (deterministic)
    examples: str | None = None  # yaml/csv of few-shot examples: excerpt, sub_issue[, why]
    votes: int = 1  # self-consistency samples per call (1 = greedy only)
    retrieval: str | None = None  # "<transcripts.jsonl>,<labels.csv>,<k>": nearest labelled calls as per-call examples
    focus: int = 0  # >0: detail only the sub-categories seen among this many nearest labelled calls (needs retrieval)
    tts: bool | None = None
    tts_ref_audio: str | None = None
    tts_ref_text: str | None = None

    def __post_init__(self):
        import yaml

        pdir = ROOT / "products" / self.product
        if not (pdir / "product.yaml").exists():
            raise ValueError(f"unknown product {self.product!r}: no {pdir / 'product.yaml'}")
        prod = yaml.safe_load((pdir / "product.yaml").read_text(encoding="utf-8")) or {}
        if self.taxonomy is None:
            self.taxonomy = str(pdir / prod.get("taxonomy", "taxonomy.yaml"))
        if self.playbooks is None and prod.get("playbooks"):
            self.playbooks = str(pdir / prod["playbooks"])
        if self.answer is None:
            self.answer = bool(prod.get("answer", False))
        if self.tts is None:
            self.tts = bool(prod.get("tts", False))
        if self.context is None:
            self.context = prod.get("context") or None
        if self.respond_persona is None:
            self.respond_persona = prod.get("respond_persona") or None
        if self.empty_sub_issue is None:
            self.empty_sub_issue = prod.get("empty_sub_issue") or None
        if self.classify_mode is None:
            self.classify_mode = prod.get("classify_mode") or "reason"
        if self.asr_fixes is None:
            cand = pdir / prod.get("asr_fixes", "asr_fixes.yaml")
            self.asr_fixes = str(cand) if cand.exists() else None
        if self.answer and not self.playbooks:
            raise ValueError(f"product {self.product!r} answers calls but has no playbooks")

    @property
    def mode(self) -> str:  # kept for older call sites
        return "otr" if self.answer else "connectview"


class Pipeline:
    def __init__(self, cfg: PipelineConfig | None = None, asr=None, llm=None):
        self.cfg = cfg or PipelineConfig()
        self.tax = Taxonomy.load(self.cfg.taxonomy)
        self.normaliser = Normaliser.load(self.cfg.asr_fixes)
        # Models load lazily so a batch run can hold only the ASR model, drop it, and
        # then load the LLM — that is how everything fits on one 16 GB T4.
        self._asr = asr
        self._llm = llm
        self._segregator = None
        self._responder = None
        self._tts = None

    @property
    def asr(self):
        if self._asr is None:
            self._asr = ASR.make_asr(self.cfg.asr, **self.cfg.asr_kwargs)
        return self._asr

    @property
    def llm(self):
        if self._llm is None:
            self._llm = make_llm(self.cfg.llm, **self.cfg.llm_kwargs)
        return self._llm

    @property
    def segregator(self) -> Segregator:
        if self._segregator is None:
            ex = None
            if self.cfg.examples:
                import yaml as _y

                raw = _y.safe_load(Path(self.cfg.examples).read_text(encoding="utf-8"))
                ex = raw.get("examples", raw) if isinstance(raw, dict) else raw
            retriever = None
            focus_pool, focus_groups = None, None
            if self.cfg.retrieval:
                from .retrieve import ExamplePool

                tr, lb, k = (self.cfg.retrieval.split(",") + ["6"])[:3]
                pool = ExamplePool.build(tr, lb)
                retriever = lambda d, _p=pool, _k=int(k): _p.nearest(d, _k, exclude_text=d)  # noqa: E731
                knn = lambda d, _p=pool: _p.knn(d, 3, exclude_text=d)  # noqa: E731
                if self.cfg.focus:
                    focus_pool = pool
                    gp = ROOT / "products" / self.cfg.product / "groups.yaml"
                    if gp.exists():
                        import yaml as _y

                        raw = _y.safe_load(gp.read_text(encoding="utf-8"))
                        focus_groups = {mm: g["id"] for g in raw["groups"] for mm in g["members"]}
            else:
                knn = None
            self._segregator = Segregator(self.tax, self.llm, self.cfg.low_confidence, self.cfg.context,
                                          self.cfg.empty_sub_issue, self.cfg.classify_mode, ex, self.cfg.votes,
                                          retriever=retriever, knn=knn, focus_pool=focus_pool, focus_k=self.cfg.focus,
                                          focus_groups=focus_groups)
        return self._segregator

    @property
    def responder(self) -> Responder | None:
        if not self.cfg.answer:
            return None
        if self._responder is None:
            self._responder = Responder(self.llm, self.cfg.playbooks, self.cfg.respond_persona)
        return self._responder

    @property
    def tts(self):
        if not (self.cfg.tts and self.cfg.answer):
            return None
        if self._tts is None:
            from .tts import IndicF5TTS

            if not (self.cfg.tts_ref_audio and self.cfg.tts_ref_text):
                raise ValueError("tts=True needs tts_ref_audio and tts_ref_text")
            self._tts = IndicF5TTS(self.cfg.tts_ref_audio, self.cfg.tts_ref_text)
        return self._tts

    def release_asr(self) -> None:
        """Free the ASR model (GPU memory) once every file is transcribed."""
        self._asr = None
        import gc

        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

    # ------------------------------------------------------------------ stages
    def transcribe(self, path: str | Path) -> dict:
        aud = A.load(path)
        return self._row(path, aud, self.asr.transcribe(aud))

    def transcribe_group(self, paths: list[str | Path]) -> list[dict]:
        """Several files at once when the ASR backend batches across files."""
        if not hasattr(self.asr, "transcribe_many"):
            return [self.transcribe(p) for p in paths]
        auds = [A.load(p) for p in paths]
        return [self._row(p, a, r) for p, a, r in zip(paths, auds, self.asr.transcribe_many(auds))]

    @staticmethod
    def _row(path, aud, res) -> dict:
        return {
            "file": Path(path).name,
            "path": str(path),
            "duration_s": round(aud.duration, 1),
            "stereo": aud.stereo,
            "asr_backend": res.backend,
            "asr_seconds": round(res.seconds, 1),
            "dialogue": D.build(res),
            "channels": D.per_channel(res),
            "words": D.word_count(res),
            "segments": [dataclasses.asdict(s) for s in res.segments],
        }

    def segregate(self, rows: list[dict]) -> None:
        if self.normaliser:  # known mis-hearings -> domain terms; the raw text is kept in dialogue_raw
            for r in rows:
                if "dialogue_raw" not in r:
                    r["dialogue_raw"] = r["dialogue"]
                    r["dialogue"], r["asr_fixes_applied"] = self.normaliser.apply(r["dialogue"])
        segs = self.segregator.run([r["dialogue"] for r in rows], [r["duration_s"] for r in rows])
        for r, s in zip(rows, segs):
            r["segregation"] = s.to_dict()

    def respond(self, rows: list[dict], out_dir: Path | None = None) -> None:
        if not self.responder:
            return
        from .segregate import Segregation

        segs = [Segregation(**{k: v for k, v in r["segregation"].items()}) for r in rows]
        resps = self.responder.run(segs)
        if self.tts:
            from .tts import to_devanagari

            dev = to_devanagari(self.llm, [x.text for x in resps])
            out_dir = Path(out_dir or ROOT / "data" / "out" / "audio")
            out_dir.mkdir(parents=True, exist_ok=True)
            for r, x, d in zip(rows, resps, dev):
                try:
                    x.audio_path = self.tts.synth(d, out_dir / (Path(r["file"]).stem + "_reply.wav"))
                except Exception as e:  # TTS must never sink the run
                    x.audio_path = None
                    r["tts_error"] = str(e)
        for r, x in zip(rows, resps):
            r["response"] = x.to_dict()

    # ------------------------------------------------------------------ drivers
    def run_one(self, path: str | Path, out_dir: str | Path | None = None) -> dict:
        row = self.transcribe(path)
        self.segregate([row])
        self.respond([row], Path(out_dir) if out_dir else None)
        return row

    def run_many(self, paths: list[str | Path], out_dir: str | Path = ROOT / "data" / "out",
                 log_every: int = 25, asr_only: bool = False, free_asr: bool = False) -> list[dict]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict] = []
        t0 = time.time()
        # transcripts from an earlier run are reused, so an LLM failure never costs the ASR time twice
        ckpt = out_dir / "transcripts.jsonl"
        done: dict[str, dict] = {}
        if ckpt.exists():
            for line in ckpt.read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                if r.get("asr_backend") == self.cfg.asr and not r.get("error"):
                    done[r["file"]] = r
        if done:
            print(f"[asr] reusing {len(done)} transcripts from {ckpt}", flush=True)
        todo_paths = [p for p in paths if Path(p).name not in done]
        rows.extend(done[Path(p).name] for p in paths if Path(p).name in done)
        paths = todo_paths
        G = 6  # files per ASR group
        for g in range(0, len(paths), G):
            group = paths[g : g + G]
            try:
                rows.extend(self.transcribe_group(group))
            except Exception:  # fall back to one at a time so one bad file does not sink six
                for p in group:
                    try:
                        rows.append(self.transcribe(p))
                    except Exception as e2:
                        rows.append({"file": Path(p).name, "path": str(p), "error": f"asr: {e2}", "dialogue": "",
                                     "duration_s": 0.0, "words": 0})
            k = min(g + G, len(paths))
            if (g // G) % max(1, log_every // G) == 0 or k == len(paths):
                print(f"[asr] {k}/{len(paths)}  {time.time() - t0:.0f}s", flush=True)
        # the checkpoint keeps every good transcript ever written to this out_dir, not just this call's paths:
        # callers that feed the same out_dir chunk by chunk (Kaggle notebooks) must not lose earlier chunks
        current = {Path(p).name for p in paths} | {r["file"] for r in rows}
        kept = [r for name, r in done.items() if name not in current]
        with ckpt.open("w", encoding="utf-8") as f:
            for r in rows + kept:
                f.write(json.dumps(r, ensure_ascii=False) + chr(10))
        if kept:
            print(f"[asr] checkpoint keeps {len(kept)} earlier transcripts alongside this call's {len(rows)}", flush=True)
        timings = {"n": len(rows), "asr_new": len(paths), "asr_s": round(time.time() - t0, 1),
                   "audio_s": round(sum(r.get("duration_s", 0) for r in rows), 1)}
        if free_asr:
            self.release_asr()
        if not asr_only:
            t1 = time.time()
            self.segregate(rows)
            timings["segregate_s"] = round(time.time() - t1, 1)
            print(f"[segregate] {len(rows)} calls  {timings['segregate_s']:.0f}s", flush=True)
            if self.responder:
                t2 = time.time()
                self.respond(rows, out_dir / "audio")
                timings["respond_s"] = round(time.time() - t2, 1)
                print(f"[respond] {len(rows)} calls  {timings['respond_s']:.0f}s", flush=True)
        write_results(rows, out_dir)
        (out_dir / "timings.json").write_text(json.dumps(timings, indent=1), encoding="utf-8")
        return rows


# ---------------------------------------------------------------------- output
def write_results(rows: list[dict], out_dir: Path) -> None:
    out_dir = Path(out_dir)
    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    cols = ["file", "duration_s", "words", "issue", "sub_issue", "issue_name", "sub_issue_name", "confidence",
            "needs_review", "resolved_on_call", "caller", "purpose", "candidates", "knn_label", "knn_unanimous",
            "summary", "evidence", "note",
            "response", "dialogue", "error"]
    with (out_dir / "results.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            s = r.get("segregation", {})
            w.writerow({**r, **{k: s.get(k) for k in ("issue", "sub_issue", "issue_name", "sub_issue_name",
                                                       "confidence", "needs_review", "resolved_on_call",
                                                       "caller", "purpose", "candidates", "knn_label",
                                                       "knn_unanimous", "summary", "evidence", "note")},
                        "response": (r.get("response") or {}).get("text")})
