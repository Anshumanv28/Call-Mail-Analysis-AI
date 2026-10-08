"""One-time setup on the GPU server, after `pip install -r requirements.txt` (see README.md):
downloads llama.cpp (CUDA build for this OS), the Qwen3-4B model (sha256-checked) and bge-m3, then checks the GPU.

    python setup.py
"""
from __future__ import annotations

import hashlib
import os
import platform
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / "hf"))
LLAMA_TAG = "b11064"
BASE = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_TAG}/"
ASSETS = {"Windows": [f"llama-{LLAMA_TAG}-bin-win-cuda-12.4-x64.zip", "cudart-llama-bin-win-cuda-12.4-x64.zip"],
          "Linux": [f"llama-{LLAMA_TAG}-bin-ubuntu-cuda-12.8-x64.tar.gz", f"cudart-llama-{LLAMA_TAG}-bin-ubuntu-cuda-12.8-x64.tar.gz"]}
QWEN_URL = "https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
QWEN_SHA256 = "3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597"


def download(url: str, dst: Path) -> None:
    print(f"  downloading {url.rsplit('/', 1)[-1]}", flush=True)
    tmp = dst.with_name(dst.name + ".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=600) as r:
        r.raise_for_status()
        with tmp.open("wb") as fh:
            for chunk in r.iter_bytes(1 << 22):
                fh.write(chunk)
    tmp.replace(dst)


def main() -> None:
    (ROOT / "logs").mkdir(exist_ok=True)
    system = platform.system()
    # 1. llama.cpp
    llama = ROOT / "llama"
    exe = "llama-server.exe" if system == "Windows" else "llama-server"
    if not any(llama.rglob(exe)):
        print("llama.cpp (CUDA):", flush=True)
        llama.mkdir(exist_ok=True)
        for name in ASSETS[system]:
            arc = llama / name
            download(BASE + name, arc)
            if name.endswith(".zip"):
                zipfile.ZipFile(arc).extractall(llama)
            else:
                tarfile.open(arc).extractall(llama, **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))
            arc.unlink()
        server = next(llama.rglob(exe))
        for lib in list(llama.rglob("*.dll")) + list(llama.rglob("*.so*")):   # CUDA runtime next to the server
            if lib.parent != server.parent:
                shutil.copy2(lib, server.parent / lib.name)
        if system != "Windows":
            server.chmod(0o755)
    print("llama-server:", next(llama.rglob(exe)), flush=True)
    # 2. Qwen3-4B (the exact file our accuracy numbers were measured with)
    gguf = ROOT / "models" / "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
    if not gguf.exists():
        print("Qwen3-4B Q4_K_M:", flush=True)
        part = gguf.with_name(gguf.name + ".dl")
        download(QWEN_URL, part)
        h = hashlib.sha256()
        with part.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                h.update(chunk)
        if h.hexdigest() != QWEN_SHA256:
            sys.exit(f"Qwen3-4B sha256 mismatch: {h.hexdigest()}")
        part.replace(gguf)
    print("Qwen3-4B: ok", flush=True)
    # 3. bge-m3 (retrieval embeddings)
    from sentence_transformers import SentenceTransformer
    SentenceTransformer("BAAI/bge-m3", device="cpu")
    print("bge-m3: ok", flush=True)
    # 4. GPU and our Whisper
    import torch
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE - a CUDA GPU is required", flush=True)
    assert (ROOT / "models" / "whisper_otr_ft" / "model.safetensors").exists(), "models/whisper_otr_ft missing"
    print("setup complete", flush=True)


if __name__ == "__main__":
    main()
