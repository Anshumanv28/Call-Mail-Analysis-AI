"""Show a Devanagari transcript as Hinglish, the way helpline staff actually write it.

Our ASR writes Hindi in Devanagari, which is right for the classifier but not what anyone here reads.
`to_hinglish` converts it to Roman script for display only: the stored transcript is never changed.

    from callai.hinglish import to_hinglish
    print(to_hinglish(row["dialogue"]))

The local LLM does the conversion when it is reachable, because it spells the way people do
("nahi aa raha hai", not the transliteration scheme's "nahI.n A rahA hai") and leaves English words
alone. Line count and speaker tags are checked afterwards; any line the model drops or mangles falls
back to a rule-based transliteration, so the output always has the same shape as the input.
"""
from __future__ import annotations

import re

DEVA = re.compile("[" + chr(0x900) + "-" + chr(0x97F) + "]")
TAG = re.compile(r"^\s*([A-Z])\s*:\s*")
CHUNK = 24  # lines per request: keeps the model aligned and the response short

SYSTEM = (
    "You rewrite Hindi written in Devanagari into Roman script (Hinglish), spelled the way Indian "
    "call-centre staff type it: nahi, kya, hai, mera, form, submit, paise. Rules, all mandatory: "
    "output exactly the same number of lines as the input; keep each line's speaker tag (A: or B:) "
    "at the start; transliterate Devanagari words only; leave words already written in English "
    "exactly as they are; do not translate, correct, add, remove or reorder anything. "
    "Output only the rewritten lines, nothing else."
)


def has_devanagari(text: str) -> bool:
    return bool(DEVA.search(text or ""))


def _rule_based(text: str) -> str:
    """Transliteration fallback: correct, just less natural than the model's spelling."""
    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate
    except Exception:  # noqa: BLE001
        return text

    def one(line: str) -> str:
        m = TAG.match(line)
        tag, body = (m.group(0), line[m.end():]) if m else ("", line)
        out = transliterate(body, sanscript.DEVANAGARI, sanscript.ITRANS)
        out = out.replace(".n", "n").replace(".m", "n").replace("~N", "n").replace("~n", "n")
        out = re.sub(r"([aeiouAEIOU])\1+", lambda m2: m2.group(1), out)      # aa -> a, II -> I
        out = re.sub(r"[{}^_]", "", out)                                    # ITRANS control marks
        return tag + out.lower()

    return chr(10).join(one(l) for l in text.split(chr(10)))


def _shape_ok(src: list[str], out: list[str]) -> bool:
    if len(src) != len(out):
        return False
    for a, b in zip(src, out):
        ma, mb = TAG.match(a), TAG.match(b)
        if bool(ma) != bool(mb) or (ma and mb and ma.group(1) != mb.group(1)):
            return False
        if has_devanagari(b):                       # the model left Devanagari behind
            return False
    return True


def to_hinglish(text: str, llm=None, base_url: str = "http://127.0.0.1:8080/v1") -> str:
    """Return the transcript in Roman script. Falls back to transliteration if the LLM cannot help."""
    if not text or not has_devanagari(text):
        return text or ""
    if llm is None:
        try:
            from callai.llm import make_llm

            llm = make_llm("llama", base_url=base_url)
        except Exception:  # noqa: BLE001
            return _rule_based(text)

    lines = text.split(chr(10))
    done: list[str] = []
    for i in range(0, len(lines), CHUNK):
        src = lines[i:i + CHUNK]
        block = chr(10).join(src)
        try:
            reply = llm.chat_batch([[{"role": "system", "content": SYSTEM},
                                     {"role": "user", "content": block}]],
                                   max_new_tokens=max(256, int(len(block) / 1.8)), temperature=0.0)[0]
            out = [l for l in (reply or "").strip().split(chr(10)) if l.strip()]
        except Exception:  # noqa: BLE001
            out = []
        done.extend(out if _shape_ok(src, out) else _rule_based(block).split(chr(10)))
    return chr(10).join(done)
