"""Queue worker for the scheduled mode: take calls from an SQS queue, analyse them in batches, send the results to
the portal, and stop the server once the queue has been empty for a while.

    python worker.py                      settings from environment variables (below)
    python worker.py --once               process what is waiting, then exit (no shutdown)

Message body (one call per message, JSON):   {"call_id": "123", "recording_url": "https://.../x.wav"}
Results, one batch at a time, as JSON:       {"results": [{"call_id": "123", "Call Category": ..., ...}, ...]}
  - posted to OTR_RESULTS_URL (with "Authorization: Bearer <OTR_RESULTS_TOKEN>" when set), and/or
  - written to OTR_RESULTS_S3 = s3://bucket/prefix/ as one JSON file per batch.
A message is deleted only after its result was delivered. If processing or delivery fails, the batch goes back
to the queue and is retried; give the queue a dead-letter queue (maxReceiveCount 5) so a bad call cannot loop.

Environment:
  OTR_QUEUE_URL        SQS queue URL (required unless --queue-dir)
  AWS_REGION           default ap-south-1
  OTR_RESULTS_URL      portal endpoint that receives results (POST)
  OTR_RESULTS_TOKEN    optional bearer token for that endpoint
  OTR_RESULTS_S3       optional s3://bucket/prefix/ copy of every batch
  OTR_BATCH            calls per batch, default 32 (max 50)
  OTR_IDLE_STOP_S      seconds of empty queue before shutdown, default 180; 0 = never stop
  OTR_SHUTDOWN_CMD     default "sudo /sbin/shutdown -h now" (instance shutdown behaviour must be "stop")
  OTR_VISIBILITY_S     how long a taken message stays hidden from other readers, default 1800
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("OTR_LLM_SLOTS", "4")          # Whisper stays loaded beside the LLM, as in serve.py
(ROOT / "logs").mkdir(exist_ok=True)
STOP = False


def log(msg: str) -> None:
    line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with (ROOT / "logs" / "worker.log").open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


# ---------------------------------------------------------------- queues
class SQSQueue:
    def __init__(self, url: str, region: str, visibility_s: int):
        import boto3
        self.c = boto3.client("sqs", region_name=region)
        self.url, self.visibility_s = url, visibility_s

    def receive(self, n: int, wait_s: int) -> list[tuple[str, str]]:
        """Up to n messages as (handle, body). Long-polls for the first, then takes what is already there."""
        out = []
        while len(out) < n:
            r = self.c.receive_message(QueueUrl=self.url, MaxNumberOfMessages=min(10, n - len(out)),
                                       WaitTimeSeconds=0 if out else wait_s, VisibilityTimeout=self.visibility_s)
            msgs = r.get("Messages", [])
            if not msgs:
                break
            out += [(m["ReceiptHandle"], m["Body"]) for m in msgs]
        return out

    def delete(self, handles: list[str]) -> None:
        for i in range(0, len(handles), 10):
            entries = [{"Id": str(j), "ReceiptHandle": h} for j, h in enumerate(handles[i:i + 10])]
            r = self.c.delete_message_batch(QueueUrl=self.url, Entries=entries)
            if r.get("Failed"):
                log(f"delete failed for {len(r['Failed'])} messages: {r['Failed'][:2]}")

    def release(self, handles: list[str]) -> None:
        """Make the messages visible again soon, so the batch is retried."""
        for i in range(0, len(handles), 10):
            entries = [{"Id": str(j), "ReceiptHandle": h, "VisibilityTimeout": 30} for j, h in enumerate(handles[i:i + 10])]
            try:
                self.c.change_message_visibility_batch(QueueUrl=self.url, Entries=entries)
            except Exception as e:  # noqa: BLE001 - they reappear after the visibility timeout anyway
                log(f"release failed: {e!r}")


class DirQueue:
    """A folder of .json message files, for testing without AWS."""

    def __init__(self, folder: str):
        self.d = Path(folder)
        self.d.mkdir(parents=True, exist_ok=True)

    def receive(self, n: int, wait_s: int) -> list[tuple[str, str]]:
        out = []
        for f in sorted(self.d.glob("*.json"))[:n]:
            taken = f.with_suffix(".inflight")
            f.rename(taken)
            out.append((str(taken), taken.read_text(encoding="utf-8")))
        if not out and wait_s:
            time.sleep(min(wait_s, 2))
        return out

    def delete(self, handles: list[str]) -> None:
        for h in handles:
            Path(h).unlink(missing_ok=True)

    def release(self, handles: list[str]) -> None:
        for h in handles:
            if Path(h).exists():
                Path(h).rename(Path(h).with_suffix(".json"))


# ---------------------------------------------------------------- result delivery
class Sinks:
    def __init__(self, url: str | None, token: str | None, s3: str | None, folder: str | None, region: str):
        self.url, self.token, self.s3, self.folder = url, token, s3, folder
        self.s3c = None
        if s3:
            import boto3
            self.s3c = boto3.client("s3", region_name=region)
        if not (url or s3 or folder):
            raise SystemExit("no result destination: set OTR_RESULTS_URL and/or OTR_RESULTS_S3 (or --results-dir)")

    def send(self, results: list[dict]) -> None:
        body = json.dumps({"results": results}, ensure_ascii=False, default=str)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        if self.s3c is not None:
            bucket, _, prefix = self.s3[len("s3://"):].partition("/")
            key = f"{prefix.rstrip('/')}/{stamp[:8]}/{stamp}-{len(results)}.json".lstrip("/")
            self.s3c.put_object(Bucket=bucket, Key=key, Body=body.encode("utf-8"), ContentType="application/json",
                                ServerSideEncryption="AES256")
        if self.folder:
            Path(self.folder).mkdir(parents=True, exist_ok=True)
            (Path(self.folder) / f"{stamp}-{len(results)}.json").write_text(body, encoding="utf-8")
        if self.url:
            headers = {"Content-Type": "application/json"}
            if self.token:
                headers["Authorization"] = f"Bearer {self.token}"
            err = None
            for attempt in range(4):
                try:
                    r = httpx.post(self.url, content=body.encode("utf-8"), headers=headers, timeout=60)
                    if r.status_code < 300:
                        return
                    err = f"HTTP {r.status_code}: {r.text[:200]}"
                    if 400 <= r.status_code < 500 and r.status_code != 429:
                        break                                   # the portal rejected it: retrying will not help
                except httpx.HTTPError as e:
                    err = repr(e)
                time.sleep(5 * (attempt + 1))
            raise RuntimeError(f"results not delivered to {self.url}: {err}")


# ---------------------------------------------------------------- processing
def fetch(url: str, dst: Path) -> bool:
    try:
        if url.startswith("file://") or (url and Path(url).exists()):
            shutil.copyfile(url[len("file://"):] if url.startswith("file://") else url, dst)
            return True
        with httpx.stream("GET", url, timeout=120, follow_redirects=True) as r:
            r.raise_for_status()
            with dst.open("wb") as fh:
                for chunk in r.iter_bytes(1 << 20):
                    fh.write(chunk)
        return dst.stat().st_size > 1000
    except Exception:  # noqa: BLE001 - reported as Recording Unavailable
        return False


class Worker:
    def __init__(self, fake: bool = False):
        self.fake = fake
        self.pipe = None

    def pipeline(self):
        if self.pipe is None and not self.fake:
            log("loading models")
            from otr_pipeline import OTRPipeline
            self.pipe = OTRPipeline(batch=16)
            self.pipe.start_llm()
            log("models loaded")
        return self.pipe

    def process(self, calls: list[dict]) -> list[dict]:
        tmp = Path(tempfile.mkdtemp(prefix="callai_q_"))
        try:
            paths = [tmp / f"c{i}.wav" for i in range(len(calls))]
            with ThreadPoolExecutor(8) as ex:
                ok = list(ex.map(lambda cp: fetch(cp[0].get("recording_url", ""), cp[1]), zip(calls, paths)))
            out: list[dict | None] = [None] * len(calls)
            todo = [i for i, good in enumerate(ok) if good]
            for i, good in enumerate(ok):
                if not good:
                    out[i] = {"Call Category": "Recording Unavailable", "Tags": "Recording Unavailable",
                              "Resolution Source": "AI", "error": "recording could not be downloaded"}
            if todo:
                if self.fake:
                    rows = [{"Call Category": "TEST", "Sub-Category": "TEST", "Resolution Source": "AI"} for _ in todo]
                else:
                    rows = self.pipeline().process_many([paths[i] for i in todo])
                for i, r in zip(todo, rows):
                    r.pop("file", None)
                    out[i] = r
            now = datetime.now(timezone.utc).isoformat(timespec="seconds")
            for c, r in zip(calls, out):
                r["call_id"] = c.get("call_id")
                r["processed_at"] = now
            return out
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def close(self):
        if self.pipe is not None:
            self.pipe.close()


def parse(body: str) -> dict | None:
    try:
        m = json.loads(body)
        return m if isinstance(m, dict) and m.get("recording_url") else None
    except (json.JSONDecodeError, TypeError):
        return None


def run(queue, sinks: Sinks, worker: Worker, batch: int, idle_stop_s: int, shutdown_cmd: str, once: bool) -> int:
    idle_since = time.time()
    failures = 0
    while not STOP:
        msgs = queue.receive(batch, wait_s=0 if once else 20)
        if not msgs:
            if once:
                break
            if idle_stop_s and time.time() - idle_since >= idle_stop_s:
                log(f"queue empty for {idle_stop_s} s: shutting down ({shutdown_cmd})")
                worker.close()
                if shutdown_cmd:
                    subprocess.run(shlex.split(shutdown_cmd), check=False)
                return 0
            continue
        good, bad = [], []
        for h, b in msgs:
            m = parse(b)
            (good if m else bad).append((h, m))
        if bad:
            log(f"dropping {len(bad)} malformed messages (need JSON with recording_url)")
            queue.delete([h for h, _ in bad])
        if good:
            t0 = time.time()
            try:
                results = worker.process([m for _, m in good])
                sinks.send(results)
                queue.delete([h for h, _ in good])
                failures = 0
                log(f"batch of {len(good)} done in {time.time() - t0:.0f} s")
            except Exception as e:  # noqa: BLE001
                failures += 1
                log(f"batch of {len(good)} failed ({failures} in a row), returned to the queue: {e!r}")
                log(traceback.format_exc()[-1500:])
                queue.release([h for h, _ in good])
                if failures >= 3:
                    log("3 failures in a row: exiting so systemd restarts the worker with fresh models")
                    worker.close()
                    return 2
        idle_since = time.time()
    worker.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="process what is waiting, then exit without shutdown")
    ap.add_argument("--queue-dir", help="folder of .json messages instead of SQS (testing)")
    ap.add_argument("--results-dir", help="also write each batch's results to this folder")
    ap.add_argument("--fake-pipeline", action="store_true", help="skip the models (testing the queue only)")
    a = ap.parse_args()
    region = os.environ.get("AWS_REGION", "ap-south-1")
    if a.queue_dir:
        queue = DirQueue(a.queue_dir)
    else:
        url = os.environ.get("OTR_QUEUE_URL")
        if not url:
            raise SystemExit("set OTR_QUEUE_URL (or use --queue-dir)")
        queue = SQSQueue(url, region, int(os.environ.get("OTR_VISIBILITY_S", "1800")))
    sinks = Sinks(os.environ.get("OTR_RESULTS_URL"), os.environ.get("OTR_RESULTS_TOKEN"),
                  os.environ.get("OTR_RESULTS_S3"), a.results_dir, region)

    def on_term(*_):
        global STOP
        STOP = True
        log("stop requested: finishing the current batch")
    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)
    batch = max(1, min(50, int(os.environ.get("OTR_BATCH", "32"))))
    log(f"worker started: batch {batch}, idle stop {os.environ.get('OTR_IDLE_STOP_S', '180')} s")
    return run(queue, sinks, Worker(fake=a.fake_pipeline), batch, int(os.environ.get("OTR_IDLE_STOP_S", "180")),
               os.environ.get("OTR_SHUTDOWN_CMD", "sudo /sbin/shutdown -h now"), a.once)


if __name__ == "__main__":
    sys.exit(main())
