"""Optional spoken answer for OTR: IndicF5 (AI4Bharat, MIT, 0.4B, Hindi) — the light,
open option that handles Hinglish written in Devanagari. It needs a short reference
voice clip plus its transcript; the Kaggle notebook cuts one from a clean agent
utterance in the dataset. Off by default; the batch run never loads it.

Kokoro was rejected for Hinglish; IndicF5's Hinglish fine-tunes (Roman-script input)
are the upgrade path when a licence-clean one is available.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .llm import Messages

TO_DEVANAGARI = """Rewrite the following Hinglish reply in Devanagari script for a text-to-speech
engine. Keep every word and the meaning; write English technical words phonetically in
Devanagari (server → सर्वर, portal → पोर्टल, ticket → टिकट, internet → इंटरनेट). Output the
rewritten text only.

{text}"""


class IndicF5TTS:
    def __init__(self, ref_audio: str | Path, ref_text: str, repo: str = "ai4bharat/IndicF5"):
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(repo, trust_remote_code=True)
        try:
            import torch

            if torch.cuda.is_available():
                self.model = self.model.to("cuda")
        except Exception:
            pass
        self.ref_audio = str(ref_audio)
        self.ref_text = ref_text
        self.sr = 24000

    def synth(self, devanagari_text: str, out_path: str | Path) -> str:
        import soundfile as sf

        audio = self.model(devanagari_text, ref_audio_path=self.ref_audio, ref_text=self.ref_text)
        audio = np.asarray(audio)
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        sf.write(str(out_path), audio.astype(np.float32), self.sr)
        return str(out_path)


def to_devanagari(llm, texts: list[str]) -> list[str]:
    msgs: list[Messages] = [[{"role": "user", "content": TO_DEVANAGARI.format(text=t)}] for t in texts]
    return llm.chat_batch(msgs, max_new_tokens=300)
