"""Devanagari -> Hinglish (Roman script) the way the portal's Gemini transcripts spell it.

Two layers:
  1. a lexicon learned from calls where we hold both our ASR text and the portal's text
     (scripts/build_lexicon.py -> models/romanize_lexicon.json): नहीं -> nahi, नंबर -> number, ओटीआर -> otr;
  2. a Hinglish-style rule transliterator for words the lexicon has never seen: inherent-schwa deletion
     (करना -> karna, not karana), anusvara/chandrabindu -> n, nukta letters (ज़ -> z, फ़ -> f), no ITRANS marks.

    from callai.romanize import Romanizer
    r = Romanizer.load()              # picks up models/romanize_lexicon.json when it exists
    r.text("A: हाँ जी नहीं आ रहा है")    # -> "A: haan ji nahi aa raha hai"

Speaker tags (A:, B:) and line breaks are kept; words already in Latin script pass through unchanged.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEXICON = ROOT / "models" / "romanize_lexicon.json"

_CONS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n", "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "n",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n", "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "ऩ": "n", "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m", "य": "y", "र": "r", "ऱ": "r", "ल": "l",
    "ळ": "l", "ऴ": "l", "व": "v", "श": "sh", "ष": "sh", "स": "s", "ह": "h",
    "क़": "q", "ख़": "kh", "ग़": "g", "ज़": "z", "ड़": "r", "ढ़": "rh", "फ़": "f", "य़": "y",
}
_NUKTA = {"क": "q", "ख": "kh", "ग": "g", "ज": "z", "ड": "r", "ढ": "rh", "फ": "f", "य": "y"}
_INDEP = {"अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo", "ऋ": "ri", "ए": "e", "ऐ": "ai",
          "ओ": "o", "औ": "au", "ऑ": "o", "ऍ": "e", "ऎ": "e", "ऒ": "o"}
_MATRA = {"ा": "a", "ि": "i", "ी": "i", "ु": "u", "ू": "u", "ृ": "ri", "ॅ": "e", "ॆ": "e", "े": "e",
          "ै": "ai", "ॉ": "o", "ॊ": "o", "ो": "o", "ौ": "au"}
_VIRAMA, _NUKTA_SIGN = "्", "़"
_NASAL = {"ं": "n", "ँ": "n"}
_VISARGA = "ः"
_DIGITS = {chr(0x966 + i): str(i) for i in range(10)}
_DEVA = re.compile("[" + chr(0x900) + "-" + chr(0x97F) + "]")
_TAG = re.compile(r"^(\s*[A-Z]\s*:\s*)")
_WORDCH = r"\w" + chr(0x900) + "-" + chr(0x963) + chr(0x966) + "-" + chr(0x97F)   # Devanagari minus the dandas
_EDGE = re.compile(r"^([^" + _WORDCH + r"]*)(.*?)([^" + _WORDCH + r"]*)$", re.S)


def has_devanagari(s: str) -> bool:
    return bool(_DEVA.search(s or ""))


def rule_word(word: str) -> str:
    """Rule transliteration of one Devanagari word into Hinglish spelling."""
    units: list[list[str]] = []          # [consonant, vowel, nasal]; vowel "A" = inherent schwa, "" = none
    i, n = 0, len(word)
    while i < n:
        ch = word[i]
        if ch in _CONS:
            c = _CONS[ch]
            if i + 1 < n and word[i + 1] == _NUKTA_SIGN:
                c = _NUKTA.get(ch, c)
                i += 1
            units.append([c, "A", ""])
        elif ch in _INDEP:
            v = _INDEP[ch]
            units.append(["", v if not units else v.replace("aa", "a") if v == "aa" and units else v, ""])
        elif ch in _MATRA and units:
            units[-1][1] = _MATRA[ch]
        elif ch == _VIRAMA and units:
            units[-1][1] = ""
        elif ch in _NASAL and units:
            units[-1][2] = "n"
        elif ch == _VISARGA and units:
            units[-1][2] = "h"
        elif ch in _DIGITS:
            units.append([_DIGITS[ch], "", ""])
        elif ch == _NUKTA_SIGN:
            pass
        else:
            units.append([ch, "", ""])
        i += 1
    if not units:
        return ""
    # word-initial long aa keeps its length (aap, aaj, aayega); inside a word Hinglish writes a
    if units[0][0] == "" and units[0][1] == "a" and word[0] == "आ":
        units[0][1] = "aa"

    def is_cons(u):
        return u[0] != "" and u[0].isalpha()

    keep = [u[1] != "" for u in units]   # does the unit end in a vowel sound
    # schwa deletion, right to left: final inherent a goes; a medial one goes in V C(a) C V context
    last = len(units) - 1
    if len(units) > 1 and units[last][1] == "A" and not units[last][2]:
        units[last][1] = ""
        keep[last] = False
    for k in range(last - 1, 0, -1):
        u = units[k]
        if u[1] != "A" or u[2]:
            continue
        prev, nxt = units[k - 1], units[k + 1]
        if keep[k - 1] and is_cons(nxt) and keep[k + 1] and is_cons(u):
            u[1] = ""
            keep[k] = False
    out = []
    for c, v, nas in units:
        out.append(c + ("a" if v == "A" else v) + nas)
    return "".join(out)


# English letter names as Hindi speakers say them and the ASR writes them: ओटीआर = OTR, आईडी = ID, पीजीटी = PGT
_LETTERS = {"ए": "A", "बी": "B", "सी": "C", "डी": "D", "ई": "E", "एफ": "F", "जी": "G", "एच": "H", "आई": "I",
            "जे": "J", "के": "K", "एल": "L", "एम": "M", "एन": "N", "ओ": "O", "पी": "P", "क्यू": "Q", "आर": "R",
            "एस": "S", "टी": "T", "यू": "U", "वी": "V", "डब्ल्यू": "W", "एक्स": "X", "वाई": "Y", "ज़ेड": "Z",
            "जेड": "Z"}
_HESITATION = {"आह", "अह", "उह", "उम", "उम्म", "अम्म", "हम्म्म"}
_AMBIGUOUS = {"के", "जी", "ए", "ओ", "ई", "सी", "वी"}   # also everyday Hindi words: a run of only these is not merged
_LONGEST = max(len(k) for k in _LETTERS)


def letters_of(token: str) -> str | None:
    """'ओटीआर' -> 'OTR' when the whole token is a run of letter names (at least two), else None."""
    out, i = [], 0
    while i < len(token):
        for L in range(min(_LONGEST, len(token) - i), 0, -1):
            if token[i:i + L] in _LETTERS:
                out.append(_LETTERS[token[i:i + L]])
                i += L
                break
        else:
            return None
    return "".join(out) if len(out) >= 2 else None


def _letter(tok: str) -> str | None:
    """A token that is exactly one letter: Latin capital (ASR 'O') or a Devanagari letter name ('टी')."""
    if len(tok) == 1 and "A" <= tok <= "Z":
        return tok
    return _LETTERS.get(tok)


def _core(token: str) -> str:
    m = _EDGE.match(token)
    return m.group(2) if m else token


class Romanizer:
    """lexicon: word -> spelling; bigram: 'prev<TAB>word' -> spelling where context changes the portal's choice;
    expand: word -> a word the portal almost always writes right after it and our ASR drops (ठीक -> + hai);
    drop: tokens our ASR writes that the portal almost never has."""

    def __init__(self, lexicon: dict[str, str] | None = None, bigram: dict[str, str] | None = None,
                 expand: dict[str, str] | None = None, drop: set[str] | list[str] | None = None):
        self.lex = lexicon or {}
        self.bigram = bigram or {}
        self.expand = expand or {}
        self.drop = set(drop or ())

    @classmethod
    def load(cls, path: str | Path = LEXICON, parts: str = "all") -> "Romanizer":
        """parts: 'all', or a comma list of map,bigram,expand,drop (for ablations)."""
        p = Path(path)
        d = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        if not isinstance(d, dict):
            return cls()
        use = {"map", "bigram", "expand", "drop"} if parts == "all" else set(parts.split(","))
        return cls(d.get("map", {}) if "map" in use else {}, d.get("bigram", {}) if "bigram" in use else {},
                   d.get("expand", {}) if "expand" in use else {}, d.get("drop", []) if "drop" in use else [])

    def _spell(self, core: str) -> str:
        return self.lex.get(core) or letters_of(core) or rule_word(core)

    def word(self, token: str, prev: str = "") -> str:
        if not has_devanagari(token):
            return token
        m = _EDGE.match(token)
        pre, core, post = (m.group(1), m.group(2), m.group(3)) if m else ("", token, "")
        post = post.replace("।", ".").replace("॥", ".")
        bg = self.bigram.get(prev + chr(9) + core) if prev else None
        if bg:
            return pre + bg + post
        if core in self.lex:
            return pre + self.lex[core] + post
        acro = letters_of(core)
        if acro:
            return pre + acro + post
        # mixed tokens (e.g. digits glued to Hindi) fall back to per-run conversion
        return pre + "".join(rule_word(part) if has_devanagari(part) else part
                             for part in re.split("([" + chr(0x900) + "-" + chr(0x97F) + "]+)", core)) + post

    def line(self, line: str) -> str:
        line = line.replace("।", ". ").replace("॥", ". ")
        m = _TAG.match(line)
        tag, body = (m.group(1), line[m.end():]) if m else ("", line)
        raw = body.split()
        out, i = [], 0
        while i < len(raw):
            j = i
            while j < len(raw) and _letter(raw[j]):
                j += 1
            # 'O T R' / 'पी जी टी' -> one acronym, unless every piece is also a common Hindi word ('जी के')
            if j - i >= 2 and any(raw[k] not in _AMBIGUOUS for k in range(i, j)):
                out.append("".join(_letter(raw[k]) for k in range(i, j)))
                i = j
            elif raw[i] in _HESITATION or _core(raw[i]) in self.drop:   # tokens the portal never writes
                i += 1
            else:
                core = _core(raw[i])
                out.append(self.word(raw[i], _core(raw[i - 1]) if i else ""))
                extra = self.expand.get(core)
                if extra:
                    nxt = _core(raw[i + 1]) if i + 1 < len(raw) else ""
                    nxt_sp = (self._spell(nxt) if has_devanagari(nxt) else nxt).lower()
                    if nxt_sp != extra.lower():
                        out.append(extra)
                i += 1
        return tag + " ".join(out)

    def text(self, text: str) -> str:
        if not text:
            return ""
        return chr(10).join(self.line(l) for l in (text or "").split(chr(10)))
