"""HTTP API for the portal (standard library only).

    python serve.py [--port 8700]

POST /process        {"call_id": "...", "recording_url": "https://.../x.wav"}
                     -> {"call_id", "Call Category", "Sub-Category", "Tags", "AI Insight", "Summary", "Call Resolved",
                         "Agent Score", "Audio Rating", "Language", "Transcription", "Resolution Source", ...}
POST /process_batch  {"calls": [{"call_id", "recording_url"}, ...]}   (up to 50; faster per call than one by one)
POST /llm            {"prompt": "...", "max_tokens": 2048, "json": true}  -> {"text", "finish_reason"}
                     the server's own LLM (Qwen3-4B) for text tasks such as mail analysis (replaces Gemini text calls)
GET  /health         {"ok": true}

One request is processed at a time (the GPU holds Whisper and the LLM together; a lock serialises requests).

Env:
  CALLAI_TOKEN       if set, every request must send "Authorization: Bearer <token>" (else 401)
  OTR_IDLE_STOP_S    if > 0, switch the machine off after this many seconds with no request (the portal's dispatcher
                     starts it again when a backlog builds up); default 0 = never
  OTR_SHUTDOWN_CMD   default "sudo /sbin/shutdown -h now" (instance shutdown behaviour must be Stop)
For whole call reports use run_batch.py instead: it is several times faster per call.
"""
from __future__ import annotations

import argparse
import hmac
import json
import os
import shlex
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx

os.environ.setdefault("OTR_LLM_SLOTS", "2")          # API mode: Whisper stays loaded beside the LLM
os.environ.setdefault("OTR_LLM_CTX", "16384")       # 2 x 16k: long mail threads fit; same GPU memory as 4 x 8k
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
TOKEN = os.environ.get("CALLAI_TOKEN", "")
IDLE_STOP_S = int(os.environ.get("OTR_IDLE_STOP_S", "0") or 0)
ACTIVITY = {"last": time.time(), "busy": 0}
ACT_LOCK = threading.Lock()


def touch(delta: int) -> None:
    with ACT_LOCK:
        ACTIVITY["busy"] += delta
        ACTIVITY["last"] = time.time()


def idle_watch() -> None:
    """Switch the machine off when nothing has been asked for IDLE_STOP_S seconds."""
    while True:
        time.sleep(30)
        with ACT_LOCK:
            idle = time.time() - ACTIVITY["last"]
            busy = ACTIVITY["busy"]
        if busy == 0 and idle >= IDLE_STOP_S:
            cmd = os.environ.get("OTR_SHUTDOWN_CMD", "sudo /sbin/shutdown -h now")
            print(f"idle for {idle:.0f} s: switching off ({cmd})", flush=True)
            PIPE.close()
            subprocess.run(shlex.split(cmd), check=False)
            return


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


def llm(req: dict) -> dict:
    """One text prompt to the LLM (no lock: llama-server runs requests in its own parallel slots)."""
    prompt = str(req.get("prompt") or "")
    if not prompt.strip():
        return {"error": "empty prompt"}
    body = {"model": "local", "messages": [{"role": "user", "content": prompt}], "temperature": 0,
            "max_tokens": max(64, min(int(req.get("max_tokens") or 2048), 4096))}
    if req.get("json"):
        body["response_format"] = {"type": "json_object"}
    r = httpx.post(os.environ["CALLAI_LLM_URL"].rstrip("/") + "/chat/completions", json=body, timeout=600)
    r.raise_for_status()
    ch = r.json()["choices"][0]
    return {"text": ch["message"]["content"], "finish_reason": ch.get("finish_reason")}


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

    def _authorised(self) -> bool:
        if not TOKEN:
            return True
        given = self.headers.get("Authorization", "")
        if hmac.compare_digest(given, f"Bearer {TOKEN}"):
            return True
        self._send(401, {"error": "unauthorised"})
        return False

    def do_GET(self):
        if not self._authorised():
            return
        self._send(200, {"ok": True}) if self.path == "/health" else self._send(404, {"error": "not found"})

    def do_POST(self):
        if not self._authorised():
            return
        touch(+1)
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if self.path == "/process":
                self._send(200, run([req])[0])
            elif self.path == "/llm":
                self._send(200, llm(req))
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
        finally:
            touch(-1)

    def log_message(self, fmt, *args):
        pass


if IDLE_STOP_S > 0:
    threading.Thread(target=idle_watch, daemon=True).start()
print(f"CallAI OTR API on http://{a.host}:{a.port}  (POST /process, /process_batch, /llm; GET /health)"
      f"  token={'on' if TOKEN else 'off'}  idle stop={IDLE_STOP_S or 'off'} s", flush=True)
try:
    ThreadingHTTPServer((a.host, a.port), H).serve_forever()
finally:
    PIPE.close()
