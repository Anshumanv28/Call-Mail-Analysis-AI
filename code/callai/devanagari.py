"""Hinglish (Roman letters) to Devanagari for the Hindi voice, word by word from a fixed dictionary.

Nothing is generated, so no word can change meaning (an LLM was tried and rewrote facts). Words come from
products/otr/kb_devanagari_lexicon.json; acronyms are spelled letter by letter, numbers, dates, money and
percentages are written as spoken Hindi. convert() returns (text, had_placeholder, unknown_words): unknown
words stay in Roman letters, and the voice reads those with English pronunciation.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEX = json.loads((ROOT / "products/otr/kb_devanagari_lexicon.json").read_text(encoding="utf-8"))

LETTER = dict(zip("abcdefghijklmnopqrstuvwxyz",
                  "ए बी सी डी ई एफ़ जी एच आई जे के एल एम एन ओ पी क्यू आर एस टी यू वी डब्ल्यू एक्स वाई ज़ेड".split()))
CAPS_WORDS = {"view", "send", "net", "set", "capture", "photo", "tet", "uptet", "ctet", "slet", "pan",
              "official", "helpline", "email"}
ONES = ("शून्य एक दो तीन चार पाँच छह सात आठ नौ दस ग्यारह बारह तेरह चौदह पंद्रह सोलह सत्रह अठारह उन्नीस बीस "
        "इक्कीस बाईस तेईस चौबीस पच्चीस छब्बीस सत्ताईस अट्ठाईस उनतीस तीस इकतीस बत्तीस तैंतीस चौंतीस पैंतीस छत्तीस "
        "सैंतीस अड़तीस उनतालीस चालीस इकतालीस बयालीस तैंतालीस चवालीस पैंतालीस छियालीस सैंतालीस अड़तालीस उनचास पचास "
        "इक्यावन बावन तिरपन चौवन पचपन छप्पन सत्तावन अट्ठावन उनसठ साठ इकसठ बासठ तिरसठ चौंसठ पैंसठ छियासठ सड़सठ "
        "अड़सठ उनहत्तर सत्तर इकहत्तर बहत्तर तिहत्तर चौहत्तर पचहत्तर छिहत्तर सतहत्तर अठहत्तर उन्यासी अस्सी इक्यासी "
        "बयासी तिरासी चौरासी पचासी छियासी सत्तासी अट्ठासी नवासी नब्बे इक्यानवे बानवे तिरानवे चौरानवे पचानवे "
        "छियानवे सत्तानवे अट्ठानवे निन्यानवे").split()
MONTH = "जनवरी फ़रवरी मार्च अप्रैल मई जून जुलाई अगस्त सितंबर अक्टूबर नवंबर दिसंबर".split()
ORD = {1: "पहली", 2: "दूसरी", 3: "तीसरी", 4: "चौथी", 6: "छठी"}
PLACEHOLDER = "ऑफ़िशियल हेल्पलाइन या ईमेल"


def num(n: int) -> str:
    if n < 100:
        return ONES[n]
    if n < 1000 or 1100 <= n < 2000:            # 1500 -> पंद्रह सौ, 1950 -> उन्नीस सौ पचास
        h, r = divmod(n, 100)
        return f"{num(h)} सौ" + (f" {ONES[r]}" if r else "")
    if n < 100000:
        t, r = divmod(n, 1000)
        return f"{num(t)} हज़ार" + (f" {num(r)}" if r else "")
    return " ".join(ONES[int(d)] for d in str(n))


def digits(s: str) -> str:
    return " ".join(ONES[int(d)] for d in s)


def numbers(t: str) -> str:
    t = re.sub(r"\b(\d{1,2})-(\d{1,2})-(\d{4})\b",
               lambda m: f"{num(int(m[1]))} {MONTH[int(m[2]) - 1]} {num(int(m[3]))}", t)
    t = re.sub(r"\b(\d{1,2})/(\d{4})\b", lambda m: f"{MONTH[int(m[1]) - 1]} {num(int(m[2]))}", t)
    t = re.sub(r"\b(\d{4})-(\d{2})\b", lambda m: f"{num(int(m[1]))} {num(int(m[2]))}", t)
    t = re.sub(r"\b(\d+)(?:st|nd|rd|th)\b", lambda m: ORD.get(int(m[1]), num(int(m[1])) + "वीं"), t)
    t = re.sub(r"\b(\d+)-year\b", lambda m: f"{num(int(m[1]))} ईयर", t)
    t = re.sub(r"₹\s?(\d+)", lambda m: f"{num(int(m[1]))} रुपये", t)
    t = re.sub(r"(\d+)\s?%", lambda m: f"{num(int(m[1]))} परसेंट", t)
    t = re.sub(r"(\d+)\s?×\s?(\d+)", lambda m: f"{num(int(m[1]))} बाई {num(int(m[2]))}", t)
    t = re.sub(r"\b0\d+\b", lambda m: digits(m[0]), t)
    return re.sub(r"\d+", lambda m: num(int(m[0])), t)


def word(w: str) -> str:
    lw = w.lower()
    if len(w) == 1:
        return LETTER.get(lw, w)
    if w.isupper() and lw not in CAPS_WORDS:     # OTR, UP, KB: spoken letter by letter
        return " ".join(LETTER[c] for c in lw)
    return LEX.get(lw, w)


def url(m: re.Match) -> str:
    return " डॉट ".join(LEX[p] if p in ("www", "gov", "in") else " ".join(LETTER[c] for c in p)
                        for p in m[0].lower().split("."))


def convert(t: str) -> tuple[str, bool, list[str]]:
    t = t.replace("â€“", "–").replace("�", ",")
    placeholder = bool(re.search(r"\[[^\]]*\]", t))
    t = re.sub(r"\[[^\]]*\]", f" {PLACEHOLDER} ", t)
    t = re.sub(r"\b(?:www\.)?[a-z]+(?:\.[a-z]+){2,}\b", url, t)
    t = numbers(t)
    t = re.sub(r"(?<=\b[A-Za-z])\.(?=[A-Za-z])|(?<=\b[A-Za-z]{2})\.(?=[A-Za-z])", " ", t)   # D.El.Ed, M.Phil
    t = t.replace("&", " एंड ").replace("+", " प्लस ").replace(">", " फिर ").replace("/", " या ")
    t = re.sub(r"[–—;:()]", ",", t)
    t = re.sub(r"[\"“”]|(?<![A-Za-z])'|'(?![A-Za-z])", "", t)
    t = re.sub(r"(?<=\w)-(?=\w)", " ", t)
    t = t.replace(".", "।")
    unknown = []
    toks = list(re.finditer(r"[A-Za-z]+(?:'[A-Za-z]+)?", t))
    out, last = [], 0
    for i, m in enumerate(toks):
        w = m[0]
        nxt = toks[i + 1][0].lower() if i + 1 < len(toks) else ""
        if w.lower() == "use" and re.match(r"(kar|kij|kiy)", nxt):
            dev = "यूज़"
        else:
            dev = word(w)
        if re.search(r"[A-Za-z]", dev):
            unknown.append(w)
        out.append(t[last:m.start()] + dev)
        last = m.end()
    t = "".join(out) + t[last:]
    t = re.sub(r"\s*,\s*(,\s*)*", ", ", t)
    t = re.sub(r",\s*([।?])", r"\1", t)
    t = re.sub(r"\s+([।?,])", r"\1", t)
    return " ".join(t.split()).strip(" ,"), placeholder, unknown


