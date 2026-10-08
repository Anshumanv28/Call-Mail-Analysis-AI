"""HTTP API for the portal (standard library only).

    python serve.py [--port 8700]

POST /process        {"call_id": "...", "recording_url": "https://.../x.wav"}
                     -> {"call_id", "Call Category", "Sub-Category", "Tags", "AI Insight", "Summary", "Call Resolved",
                         "Agent Score", "Audio Rating", "Language", "Transcription", "Resolution Source", ...}
POST /process_batch  {"calls": [{"call_id", "recording_url"}, ...]}   (up to 50; faster per call than one by one)
GET  /health         {"ok": true}

One request is processed at a time (the GPU holds Whisper and the LLM together; a lock serialises requests).
For whole call reports use run_batch.py instead: it is several times faster per call.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx

os.environ.setdefault("OTR_LLM_SLOTS", "4")          # API mode: Whisper stays loaded beside the LLM
from otr_pipeline import OTRPipeline  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--host", default="0.0.0.0")
ap.add_argument("--port", type=int, default=8700)
a = ap.parse_args()

(Path(__file__).resolve().parent / "logs").mkdir(exist_ok=True)
PIPE = OTRPipeline(batch=16)
PIPE.start_llm()
LOCK = threading.Lock()
TMP = Path(tempfile.mkdtemp(prefix="callai_"))


def fetch(url: str, name: str) -> Path | None:
    dst = TMP / f"{name}.wav"
    try:
        with httpx.stream("GET", url, timeout=120, follow_redirects=True) as r:
            r.raise_for_status()
            with dst.open("wb") as fh:
                for chunk in r.iter_bytes(1 << 20):
                    fh.write(chunk)
        return dst
    except Exception:  # noqa: BLE001
        return None


def run(calls: list[dict]) -> list[dict]:
    paths, keep = [], []
    out = [None] * len(calls)
    for i, c in enumerate(calls):
        p = fetch(c.get("recording_url", ""), f"c{i}_{os.getpid()}")
        if p is None:
            out[i] = {"call_id": c.get("call_id"), "Call Category": "Recording Unavailable", "Tags": "Recording Unavailable",
                      "Resolution Source": "AI"}
        else:
            paths.append(p)
            keep.append(i)
    if paths:
        with LOCK:
            rows = PIPE.process_many(paths)
        for i, r in zip(keep, rows):
            r["call_id"] = calls[i].get("call_id")
            r.pop("file", None)
            out[i] = r
    for p in paths:
        p.unlink(missing_ok=True)
    return out


class H(BaseHTTPRequestHandler):
    def _send(self, code: int, obj) -> None:
        body = json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send(200, {"ok": True}) if self.path == "/health" else self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if self.path == "/process":
                self._send(200, run([req])[0])
            elif self.path == "/process_batch":
                calls = req.get("calls") or []
                if len(calls) > 50:
                    self._send(400, {"error": "at most 50 calls per request"})
                else:
                    self._send(200, {"results": run(calls)})
            else:
                self._send(404, {"error": "not found"})
        except Exception as e:  # noqa: BLE001
            self._send(500, {"error": repr(e)})

    def log_message(self, fmt, *args):
        pass


print(f"CallAI OTR API on http://{a.host}:{a.port}  (POST /process, /process_batch; GET /health)", flush=True)
try:
    ThreadingHTTPServer((a.host, a.port), H).serve_forever()
finally:
    PIPE.close()
