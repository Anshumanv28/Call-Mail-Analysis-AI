"""Quick check of the category / sub-category classifier on CPU (no GPU, no llama-server; works on a Mac).

    pip install -r requirements.txt        (torch CPU wheel is fine here)
    python test_classifier.py              (needs code/models/gemini_cls.pkl from CallAI_OTR_portal.zip)
    python test_classifier.py "sir mera OTP nahi aa raha registration ke time"

Downloads bge-m3 (~2.3 GB) on first run. Prints the top 3 sub-categories with confidence for each sample transcript.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / "hf"))
sys.path.insert(0, str(ROOT / "code"))

SAMPLES = [   # made-up transcripts in the style of the helpline (romanised Hindi), not real calls
    "AGENT: namaskar UPESSC helpline bataiye. CANDIDATE: sir maine fees bhari thi bank se paisa kat gaya lekin form "
    "abhi bhi incomplete dikha raha hai payment pending aa raha hai. AGENT: aap 24 ghante wait kariye status update ho jayega.",
    "AGENT: ji bataiye. CANDIDATE: sir registration karte time mobile par OTP nahi aa raha hai, teen baar try kiya. "
    "AGENT: network check kariye aur resend OTP par click kariye.",
    "AGENT: haan ji. CANDIDATE: mam meri photo upload nahi ho rahi size zyada bata raha hai. AGENT: photo 50 KB se kam "
    "kariye aur JPG format mein upload kariye.",
]


def main() -> None:
    pkl = ROOT / "code" / "models" / "gemini_cls.pkl"
    if not pkl.exists():
        sys.exit("code/models/gemini_cls.pkl is missing: copy it from CallAI_OTR_portal.zip (see MODELS.md)")
    from callai.gemini_cls import GeminiClassifier
    from sentence_transformers import SentenceTransformer

    texts = sys.argv[1:] or SAMPLES
    t0 = time.time()
    gc = GeminiClassifier.load(pkl)
    emb = SentenceTransformer("BAAI/bge-m3", device="cpu")
    print(f"loaded classifier ({len(gc.labels)} sub-categories) + bge-m3 in {time.time() - t0:.0f} s")
    E = emb.encode(texts, normalize_embeddings=True, batch_size=8).astype("float32")
    for t, p in zip(texts, gc.predict(texts, E)):
        print("\n" + t[:110] + ("..." if len(t) > 110 else ""))
        print(f"  -> {p['cat']} / {p['sub']}  (confidence {p['confidence']})")
        for name, conf in p["top"][1:3]:
            print(f"     also: {name} ({conf})")


if __name__ == "__main__":
    main()
