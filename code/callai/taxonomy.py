"""The hard segregation, loaded from taxonomy.yaml and enforced in code."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import yaml

UNRESOLVED = "UNRESOLVED"


@dataclasses.dataclass
class SubIssue:
    id: str
    name: str
    description: str
    cues: list[str]


@dataclasses.dataclass
class Issue:
    id: str
    name: str
    description: str
    sub_issues: list[SubIssue]


class Taxonomy:
    def __init__(self, issues: list[Issue]):
        self.issues = issues
        self.by_issue = {i.id: i for i in issues}
        self.by_sub = {s.id: (i, s) for i in issues for s in i.sub_issues}

    @classmethod
    def load(cls, path: str | Path) -> "Taxonomy":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        issues = [
            Issue(
                id=i["id"], name=i["name"], description=i.get("description", ""),
                sub_issues=[
                    SubIssue(s["id"], s["name"], s.get("description", ""), list(s.get("cues", []) or []))
                    for s in i["sub_issues"]
                ],
            )
            for i in raw["issues"]
        ]
        return cls(issues)

    # --- what the model sees -------------------------------------------------
    MAX_BLOCK_CHARS = 48000  # ~12k tokens: the most a 16k slot can take next to a transcript and the answer

    BRIEF_CHARS = 90  # brief tree line length: 197 leaves x ~25 tokens keeps the static prefix near 6k tokens

    @staticmethod
    def _brief(text: str, limit: int = 90) -> str:
        """First sentence of a description (before any 'NOT:' clause), capped at `limit` chars."""
        head = text.split(" NOT:")[0].split(". ")[0].strip()
        return head if len(head) <= limit else head[: limit - 1].rsplit(" ", 1)[0] + "…"

    def prompt_block(self, detail: set[str] | None = None, brief: bool | None = None) -> str:
        """The tree for the prompt. With `detail`, only those sub-categories carry their full
        description and cues; the rest are listed by id and name so the label space stays complete
        while the prompt stays small (196-leaf taxonomies do not fit a 16k slot otherwise).
        `brief` (auto when the full tree exceeds MAX_BLOCK_CHARS) shortens every description to its
        first sentence and drops the cues."""
        if brief is None and detail is None:
            full = self.prompt_block(brief=False)
            return full if len(full) <= self.MAX_BLOCK_CHARS else self.prompt_block(brief=True)
        lines = []
        for i in self.issues:
            lines.append(f"- issue `{i.id}` — {i.name}: {i.description}")
            for s in i.sub_issues:
                if detail is not None and s.id not in detail:
                    lines.append(f"    - sub_issue `{s.id}` — {s.name}")
                    continue
                if brief:
                    lines.append(f"    - sub_issue `{s.id}` — {s.name}: {self._brief(s.description, self.BRIEF_CHARS)}")
                    continue
                cue = f"  e.g. {', '.join(s.cues[:3])}" if s.cues else ""
                lines.append(f"    - sub_issue `{s.id}` — {s.name}: {s.description}{cue}")
        return "\n".join(lines)

    def json_schema(self) -> dict:
        """Closed enums, so a guided decoder (vLLM / SGLang / llama.cpp) cannot leave
        the tree. The in-process backend validates against the same enums."""
        return {
            "type": "object",
            "properties": {
                "issue": {"type": "string", "enum": [i.id for i in self.issues]},
                "sub_issue": {"type": "string", "enum": list(self.by_sub)},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "evidence": {"type": "string"},
                "summary": {"type": "string"},
                "resolved_on_call": {"type": "boolean"},
            },
            "required": ["issue", "sub_issue", "confidence", "evidence", "summary", "resolved_on_call"],
            "additionalProperties": False,
        }

    # --- enforcement -----------------------------------------------------------
    def validate(self, issue: str, sub_issue: str) -> tuple[str, str, str | None]:
        """Return (issue, sub_issue, problem). The pair must exist and belong together;
        if the sub_issue is valid but filed under the wrong issue, we trust the more
        specific label and correct the parent."""
        if sub_issue in self.by_sub:
            parent, _ = self.by_sub[sub_issue]
            if parent.id == issue:
                return issue, sub_issue, None
            return parent.id, sub_issue, f"issue corrected {issue!r} -> {parent.id!r}"
        if issue in self.by_issue:
            return issue, UNRESOLVED, f"unknown sub_issue {sub_issue!r}"
        return UNRESOLVED, UNRESOLVED, f"unknown issue {issue!r} / sub_issue {sub_issue!r}"

    def names(self, issue: str, sub_issue: str) -> tuple[str, str]:
        i = self.by_issue.get(issue)
        s = self.by_sub.get(sub_issue)
        return (i.name if i else issue, s[1].name if s else sub_issue)
