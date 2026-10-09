"""The prompt of the reply model: one definition, used to build its training data and to run it.

The model sees the KB material for the caller's case (retrieved from what the caller said) and the conversation so
far, and writes the agent's next turn. Training (scripts/build_reply_data.py) and serving (scripts/kb_bot.py) must
build this prompt identically, or the fine-tune learns one format and is asked another.
"""
from __future__ import annotations

import re

SYSTEM = ("You are the helpline agent for the UPESSC myOTR portal, on a phone call with a candidate. "
          "Reply in the caller's language: Hinglish (Hindi in Roman letters) if they speak Hindi or Hinglish, "
          "English if they speak English. Speak like a good agent: polite, short, one or two sentences at a time. "
          "Use ONLY the case material given. Never invent a phone number, email, date, fee, wait time or rule; if "
          "something is not confirmed, say you will escalate or check. Ask the clarification question first if the "
          "caller has not answered it.")

NUM = re.compile(r"\d[\d\s-]{3,}\d")                 # phone numbers, roll numbers, OTR IDs, transaction IDs
EMAIL = re.compile(r"\S+@\S+")


def mask(t: str) -> str:
    """Personal identifiers out of the prompt: the model must never learn or repeat a candidate's numbers."""
    return EMAIL.sub("[EMAIL]", NUM.sub("[NUMBER]", t))


def _txt(v) -> str:
    if isinstance(v, list):
        return " ".join(_txt(x) for x in v)
    if isinstance(v, dict):
        return " ".join(_txt(x) for x in v.values())
    return str(v or "")


def case_block(case: dict | None, row: dict | None) -> str:
    if not case:
        return "CASE: no matching KB case. Ask what the problem is; escalate what you cannot answer from the KB."
    row = row or {}
    parts = [f"CASE: {case.get('sub_category', '')}",
             f"VERIFIED RESOLUTION: {_txt(case.get('verified_resolution'))[:700]}"]
    if row.get("clarify"):
        parts.append(f"ASK FIRST: {row['clarify'][:200]}")
    if row.get("reply"):
        parts.append(f"APPROVED REPLY: {row['reply'][:700]}")
    if _txt(case.get("do_not_say")):
        parts.append(f"DO NOT SAY: {_txt(case.get('do_not_say'))[:300]}")
    if row.get("escalate"):
        parts.append(f"ESCALATE WHEN: {row['escalate'][:200]}")
    return "\n".join(parts)


def messages(case: dict | None, row: dict | None, history: list[tuple[str, str]], max_turns: int = 8) -> list[dict]:
    """history: [(speaker, text)] with speaker 'Caller' or 'Agent', oldest first, ending with the caller's turn."""
    conv = "\n".join(f"{s}: {mask(t)[:300]}" for s, t in history[-max_turns:])
    user = f"{case_block(case, row)}\n\nCONVERSATION SO FAR:\n{conv}\n\nReply as the Agent."
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
