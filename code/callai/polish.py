"""Transcript polishing: the LLM corrects ASR errors from sentence context (generative error
correction), under tight constraints so it cannot invent content.

    Polisher(llm, glossary).run(dialogues) -> list[str]   # cleaned dialogues, same A:/B: line structure

Rules the prompt enforces and the code checks:
  * edit only words that are ASR mistakes (phonetic confusions, wrong script, split/joined words);
    keep every line, speaker tag, and the order
  * a word the model cannot resolve is kept and marked [?]
  * nothing is added, nothing is removed, no summarising; line count must match or the raw line is kept
The raw transcript is always stored next to the polished one.
"""

from __future__ import annotations

from .llm import Messages, extract_json

SYSTEM = """You clean up automatic speech recognition (ASR) output of phone calls in Hindi / Hinglish.
The ASR makes phonetic mistakes, writes English words in Devanagari, splits or joins words,
and sometimes emits nonsense on noise. Your job is ONLY to correct the words to the most
probable words the speaker actually said, judged from the sentence and from this domain:

{domain}

Glossary of terms that often come out wrong (ASR spelling -> meaning):
{glossary}

Strict rules:
1. Keep every line, its speaker tag (A: / B:) and the order. Same number of lines in, same out.
2. Change a word only when you are confident it is an ASR error and you know the intended word.
   Prefer the domain term. Keep the speaker's language: Hindi stays Hindi (Devanagari),
   English words may be written in Latin script.
3. If a word or phrase is garbled and you cannot tell what was said, keep it as it is and
   append [?] after it. Never replace noise with a guessed sentence.
4. Never add, remove, reorder or summarise anything. Do not answer the call, do not comment.

Work in two steps inside one JSON object. First list every word or phrase that looks like an
ASR error, with the most probable intended word (or "?" if you cannot tell). Then write the
corrected lines applying exactly those fixes. Expect several fixes per call: these transcripts
are noisy and the glossary terms above are usually what was said.

Worked examples of fixes:
- "सिक्सटी परसेंट प्रेजेंट है सेंटर पे" -> "CCTV person present है सेंटर पे"  (sixty percent = CCTV person)
- "आपको रेसिपी नहीं मिली है" -> "आपको receipt नहीं मिली है"
- "सीसीटीवी स्कोर नहीं" -> "CCTV spoc नहीं"
- "अस्टार्ट हुआ है" -> "start हुआ है"
- "हमारे कास्ट में कांग्रेस आंदोलन की नीड़े" -> "हमारे कास्ट में कांग्रेस आंदोलन की नीड़े [?]"  (noise: keep, mark)

Reply with one JSON object and nothing else:
{{"fixes": [{{"heard": "<ASR text>", "meant": "<intended word or ?>"}}, ...],
  "lines": ["A: ...", "B: ...", ...]}}"""

DEFAULT_DOMAIN = "Calls between an exam-monitoring control room and exam centres during a computer-based exam in India."

DEFAULT_GLOSSARY = {
    "सीसीटीवी परसेंट / सिक्सटी परसेंट / सीसीटीवी परसना": "CCTV person (the CCTV technician / spoc)",
    "स्पॉक / स्पोक": "spoc (single point of contact)",
    "वाइप / वाइफ / वॉइस": "VoIP (the VoIP phone)",
    "टीसी कोड": "TC code (centre code)",
    "प्रेजेंट काउंट / प्रेसेंट": "present count (candidates present)",
    "बीएम, टेक वन, टेक टू, एलएस, आरडीएम": "BM, tech-1, tech-2, LS (lab supervisor), RDM (staff roles)",
    "रेसिपी / रेसेप्ट / recept": "receipt",
    "लॉगिन / लौगिन": "login",
    "बायोमेट्रिक": "biometric",
    "इनविजिलेटर / इन्विजिलेटर": "invigilator",
    "फ्रिस्किंग": "frisking",
    "एडमिट कार्ड": "admit card",
    "ओटीआर": "OTR (one time registration)",
    "ओटीपी": "OTP",
}


class Polisher:
    def __init__(self, llm, domain: str | None = None, glossary: dict[str, str] | None = None):
        self.llm = llm
        g = glossary or DEFAULT_GLOSSARY
        self.system = SYSTEM.format(domain=(domain or DEFAULT_DOMAIN).strip(),
                                    glossary="\n".join(f"- {k} -> {v}" for k, v in g.items()))

    def messages(self, dialogue: str) -> Messages:
        return [{"role": "system", "content": self.system},
                {"role": "user", "content": "ASR output:\n" + dialogue}]

    def run(self, dialogues: list[str]) -> list[str]:
        schema = {"type": "object", "properties": {
            "fixes": {"type": "array", "items": {"type": "object", "properties": {"heard": {"type": "string"}, "meant": {"type": "string"}},
                                                  "required": ["heard", "meant"], "additionalProperties": False}},
            "lines": {"type": "array", "items": {"type": "string"}}},
            "required": ["fixes", "lines"], "additionalProperties": False}
        todo = [k for k, d in enumerate(dialogues) if d.strip()]
        replies = self.llm.chat_batch([self.messages(dialogues[k]) for k in todo], max_new_tokens=1200,
                                      json_schema=schema)
        out = list(dialogues)
        self.last_fixes: dict[int, list] = {}
        for k, rep in zip(todo, replies):
            obj = extract_json(rep) or {}
            self.last_fixes[k] = obj.get("fixes") or []
            lines = obj.get("lines")
            raw_lines = dialogues[k].splitlines()
            if not isinstance(lines, list) or len(lines) != len(raw_lines):
                continue  # structure broken: keep the raw transcript
            fixed = []
            for raw, new in zip(raw_lines, lines):
                new = str(new).strip()
                tag = raw.split(":", 1)[0] + ":" if ":" in raw else ""
                if tag and not new.startswith(tag):
                    new = raw  # speaker tag changed: distrust the line
                # a line that grew a lot is a rewrite, not a correction
                if len(new) > 1.6 * len(raw) + 20:
                    new = raw
                fixed.append(new)
            out[k] = "\n".join(fixed)
        return out
