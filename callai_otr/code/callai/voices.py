"""Voice fingerprints for ASR segments: WeSpeaker ResNet34 speaker embeddings (ONNX, CPU).

    from callai.voices import embed_call
    E = embed_call(wav_path, segments)      # (n_segments, 256), L2-normalised; cached in out/voice_emb/<id>.npy
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "out" / "voice_emb"


@lru_cache(maxsize=1)
def _session():
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download

    onnx = hf_hub_download("Wespeaker/wespeaker-voxceleb-resnet34-LM", "voxceleb_resnet34_LM.onnx")
    import os

    so = ort.SessionOptions()
    so.intra_op_num_threads = int(os.environ.get("VOICE_THREADS", "0"))   # 0 = onnxruntime default
    s = ort.InferenceSession(onnx, sess_options=so, providers=["CPUExecutionProvider"])
    return s, s.get_inputs()[0].name


def embed_segments(wav: str | Path, segs: list[dict]) -> np.ndarray:
    import librosa
    import torch
    import torchaudio

    sess, name = _session()
    y, _ = librosa.load(str(wav), sr=16000, mono=True)
    out = []
    for s in segs:
        a, b = int(float(s["start"]) * 16000), int(float(s["end"]) * 16000)
        if b - a < 16000:                           # under 1 s: widen to 1 s around the segment
            mid = (a + b) // 2
            a, b = max(0, mid - 8000), min(len(y), mid + 8000)
        x = torch.tensor(y[a:b] * 32768.0, dtype=torch.float32).unsqueeze(0)
        if x.shape[1] < 4000:
            x = torch.nn.functional.pad(x, (0, 4000 - x.shape[1]))
        fb = torchaudio.compliance.kaldi.fbank(x, num_mel_bins=80, frame_length=25, frame_shift=10, dither=0.0,
                                               sample_frequency=16000, window_type="hamming")
        fb = fb - fb.mean(0, keepdim=True)
        e = sess.run(None, {name: fb.unsqueeze(0).numpy()})[0][0]
        out.append(e / (np.linalg.norm(e) + 1e-9))
    return np.array(out, dtype=np.float32)


def embed_call(wav: str | Path, segs: list[dict], file_id: str | None = None) -> np.ndarray:
    CACHE.mkdir(parents=True, exist_ok=True)
    fid = file_id or Path(wav).stem
    cp = CACHE / f"{fid}.npy"
    if cp.exists():
        E = np.load(cp)
        if len(E) == len(segs):
            return E
    E = embed_segments(wav, segs)
    np.save(cp, E)
    return E
