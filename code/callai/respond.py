"""Response head: a playbook-grounded reply for the caller, in Hinglish.

The sub_issue picks the playbook (playbooks.yaml). The model only phrases it — it is
told not to add steps, promises or numbers that are not in the playbook. TTS is a
separate, optional step (tts.py) so the batch run never pays for it.

Who the assistant is differs per product — a control room talking to exam-centre staff is not the
same as a helpline talking to a candidate — so the persona comes from products/<name>/product.yaml
under `respond_persona`, falling back to the ConnectView wording it was written for.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import yaml

from .llm import Messages
from .segregate import Segregation
from .taxonomy import UNRESOLVED

PERSONA_DEFAULT = """You are the control-room assistant for exam centres in India, speaking to a
centre staff member on the phone."""

SYSTEM = """{persona} Reply in natural Hinglish (Hindi in Roman script mixed
with English technical words), polite, calm and short — what a good support agent would
say in 15-25 seconds. Address the caller as "sir/ma'am" naturally.

Ground the reply ONLY on the PLAYBOOK. Do not add steps, timings or promises that are not
in it. Never invent a phone number, a name, a ticket number or a time — if the playbook
does not give one, do not say one. If the playbook asks a question first, ask it. Do not
repeat the caller's words back at length. No bullet points, no headings — spoken
sentences only.

Write in Hinglish, Roman script — spoken sentences, not English prose, and not a written
notice. Keep it to what a good agent says in 15-25 seconds."""

USER = """Caller's issue: {issue_name} → {sub_issue_name}
Summary: {summary}
Transcript excerpt: {evidence}

PLAYBOOK
{playbook}

Write the reply now."""

UNRESOLVED_PLAYBOOK = """The issue could not be placed in a known category.
Ask the caller to describe the problem once more in one or two lines, note their details, and
tell them the team will check and call back. Do not promise a time, and do not give out any
number, date or rule."""


@dataclasses.dataclass
class Response:
    text: str
    playbook_id: str
    audio_path: str | None = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


class Responder:
    def __init__(self, llm, playbooks_path: str | Path, persona: str | None = None):
        self.llm = llm
        raw = yaml.safe_load(Path(playbooks_path).read_text(encoding="utf-8")) or {}
        self.playbooks = {k: str(v).strip() for k, v in raw.items()}
        self.default = self.playbooks.get("default", UNRESOLVED_PLAYBOOK)
        self.system = SYSTEM.format(persona=(persona or PERSONA_DEFAULT).strip())

    def playbook_for(self, seg: Segregation) -> tuple[str, str]:
        if seg.sub_issue == UNRESOLVED:
            return "unresolved", UNRESOLVED_PLAYBOOK
        if seg.sub_issue in self.playbooks:
            return seg.sub_issue, self.playbooks[seg.sub_issue]
        return "default", self.default

    def messages(self, seg: Segregation) -> tuple[str, Messages]:
        pid, pb = self.playbook_for(seg)
        user = USER.format(issue_name=seg.issue_name, sub_issue_name=seg.sub_issue_name,
                           summary=seg.summary or "(none)", evidence=seg.evidence or "(none)", playbook=pb)
        return pid, [{"role": "system", "content": self.system}, {"role": "user", "content": user}]

    def run(self, segs: list[Segregation]) -> list[Response]:
        pids, msgs = zip(*(self.messages(s) for s in segs)) if segs else ((), ())
        replies = self.llm.chat_batch(list(msgs), max_new_tokens=220, temperature=0.3) if msgs else []
        return [Response(text=r.strip(), playbook_id=p) for r, p in zip(replies, pids)]
