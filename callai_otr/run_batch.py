"""Batch: the portal's call-report export in, the same report out with every AI column filled by CallAI.

    python run_batch.py --report call-report-2026-10-06-to-2026-10-06.xlsx [--out filled.xlsx] [--limit N]

Steps (resumable: run the same command again after any stop - finished calls are kept in work/):
  1. download the recordings (Recording URL column), 16 at a time          -> work/rec/
  2. transcribe every call on the GPU, AGENT/CANDIDATE turns               -> work/transcripts.jsonl
  3. free the GPU from Whisper, start llama-server, categorize + score     -> work/results.jsonl
  4. write the report back: same sheet ("Calls"), same 31 columns, same row order
Rows without a recording keep the portal's values. A recording that cannot be downloaded -> "Recording Unavailable".
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--report", required=True)
ap.add_argument("--out")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--work", default=str(ROOT / "work"))
a = ap.parse_args()

W = Path(a.work)
REC = W / "rec"
REC.mkdir(parents=True, exist_ok=True)
(ROOT / "logs").mkdir(exist_ok=True)
t0 = time.time()


def log(m: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {m}"
    print(line, flush=True)
    with (ROOT / "logs" / "run_batch.log").open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


rep = pd.read_excel(a.report, sheet_name=0)
url = rep["Recording URL"].fillna("").astype(str).str.strip()
fid = url.str.split("/").str[-1].str.split("?").str[0].str.rsplit(".", n=1).str[0].where(url != "", "")
todo = [(f, u) for f, u in zip(fid, url) if f]
todo = list(dict.fromkeys(todo))
if a.limit:
    todo = todo[:a.limit]
log(f"{len(rep)} report rows, {len(todo)} recordings")

# ---- 1. download
client = httpx.Client(timeout=120, follow_redirects=True)


def get(item):
    f, u = item
    dst = REC / f"{f}.wav"
    if dst.exists():
        return f, True
    for attempt in range(4):
        try:
            with client.stream("GET", u) as r:
                r.raise_for_status()
                tmp = dst.with_name(dst.name + ".part")
                with tmp.open("wb") as fh:
                    for chunk in r.iter_bytes(1 << 20):
                        fh.write(chunk)
            tmp.replace(dst)
            return f, True
        except Exception:  # noqa: BLE001
            time.sleep(2 * (attempt + 1))
    return f, False


with ThreadPoolExecutor(16) as ex:
    ok = dict(ex.map(get, todo))
log(f"downloaded {sum(ok.values())} of {len(todo)} recordings ({len(todo) - sum(ok.values())} unavailable)")

# ---- 2. transcribe (resumable)
from otr_pipeline import OTRPipeline  # noqa: E402

TX, RES = W / "transcripts.jsonl", W / "results.jsonl"


def read_jsonl(p: Path) -> dict:
    out = {}
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                out[r["file"]] = r
            except Exception:  # noqa: BLE001 - half-written last line
                pass
    return out


tx = read_jsonl(TX)
need = [f for f, _ in todo if ok.get(f) and f not in tx]
log(f"transcription: {len(tx)} done before, {len(need)} to do")
p = OTRPipeline()
if need:
    with TX.open("a", encoding="utf-8") as fh:
        def save(r):
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()
        n0 = time.time()
        for i in range(0, len(need), 500):
            p.transcribe_many([REC / f"{f}.wav" for f in need[i:i + 500]], on_done=save)
            done = min(i + 500, len(need))
            log(f"  transcribed {done}/{len(need)} ({done / max(1e-6, (time.time() - n0) / 60):.0f} calls/min)")
    tx = read_jsonl(TX)
p.free_whisper()

# ---- 3. categorize + score (resumable, in chunks)
res = read_jsonl(RES)
rows = [tx[f] for f, _ in todo if f in tx and f not in res]
log(f"analysis: {len(res)} done before, {len(rows)} to do")
try:
    with RES.open("a", encoding="utf-8") as fh:
        for i in range(0, len(rows), 400):
            for r in p.analyse_many(rows[i:i + 400]):
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()
            log(f"  analysed {min(i + 400, len(rows))}/{len(rows)}")
finally:
    p.close()
res = read_jsonl(RES)

# ---- 4. the report, filled
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE  # noqa: E402

out = rep.copy()
cols = ["Call Category", "Sub-Category", "Tags", "AI Insight", "Summary", "Bug Category", "Bug Description",
        "Call Resolved", "Agent Score", "Audio Rating", "Language", "Transcription", "Resolution Source"]
for c in cols:
    out[c] = out[c].astype(object)
for i, f in fid.items():
    if not f or (a.limit and f not in ok):
        continue
    r = res.get(f) or {"Call Category": "Recording Unavailable", "Tags": "Recording Unavailable", "Resolution Source": "AI"}
    for c in cols:
        v = r.get(c, "")
        out.at[i, c] = ILLEGAL_CHARACTERS_RE.sub("", str(v))[:32000] if isinstance(v, str) else v
dst = Path(a.out) if a.out else Path(a.report).with_name(Path(a.report).stem + "_CallAI.xlsx")
out.to_excel(dst, index=False, sheet_name="Calls")
log(f"done in {(time.time() - t0) / 60:.0f} min -> {dst}")
