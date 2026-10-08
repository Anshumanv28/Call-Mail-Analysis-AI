"""Issue / sub-issue segregation against the taxonomy, with the enums enforced in code.

The model gets the dialogue and the tree and returns one JSON object. We parse, validate
against the tree, retry once with the validation error in the prompt, and file anything
that still does not fit under UNRESOLVED with a note.

Three modes (product.yaml `classify_mode`, or PipelineConfig.classify_mode):
  single    : one call, the whole tree, label straight away (the original)
  reason    : one call, the whole tree, but the model first writes who is calling, the
              primary purpose and the 2-3 candidate ids, THEN the label — a short, cheap
              chain of thought that the JSON schema forces to come before the decision
  two_stage : call 1 picks the category from 12 with their descriptions and sub-issue names;
              call 2 picks the sub-issue inside that category only, with full descriptions
              and cues, and may bounce ("other_category") back to the reason mode.
"""

from __future__ import annotations

import dataclasses

from .llm import Messages, extract_json
from .taxonomy import UNRESOLVED, Taxonomy

DEFAULT_CONTEXT = "You classify phone calls handled by a support desk in India."

PREAMBLE = """{context}
The calls are in Hindi, English or a mix (Hinglish). The transcript comes from automatic
speech recognition on a phone line, so expect errors, missing words and odd spellings —
read for meaning. Speakers A and B are the two sides of the line; work out from what they
say which one is the control room and which one is the centre."""

RULES = """Choose exactly one issue and one sub_issue from the tree below for the PRIMARY purpose of
the call (what it is mainly about, not every topic touched). Never invent a label. If the
call is understandable but fits nothing, use unclassified / not_classified and describe it
in the summary."""

SYSTEM_SINGLE = PREAMBLE + "\n\n" + RULES + """

TREE
{tree}

Reply with one JSON object and nothing else:
{{"issue": "<issue id>", "sub_issue": "<sub_issue id>", "confidence": <0..1>,
  "evidence": "<the few words from the transcript that decided it, max 20 words>",
  "summary": "<one short sentence, in English, what the caller reports>",
  "resolved_on_call": <true if the issue was resolved during the call>}}"""

SYSTEM_REASON = PREAMBLE + "\n\n" + RULES + """

TREE
{tree}

Think before you label. Reply with one JSON object and nothing else, fields in this order:
{{"caller": "<control_room | centre | unclear>",
  "purpose": "<one sentence: what the call is mainly about and what was asked or reported>",
  "candidates": ["<sub_issue id>", "<sub_issue id>", "<sub_issue id>"],  <- 2 or 3 ids that could fit, most likely first
  "issue": "<issue id>", "sub_issue": "<sub_issue id>", "confidence": <0..1>,
  "evidence": "<the few words from the transcript that decided it, max 20 words>",
  "summary": "<one short sentence, in English, what the caller reports>",
  "resolved_on_call": <true if the issue was resolved during the call>}}"""

SYSTEM_STAGE1 = PREAMBLE + """

First decide only the CATEGORY of the call's primary purpose. Categories, each with the
sub-categories it contains:
{categories}

Reply with one JSON object and nothing else, fields in this order:
{{"caller": "<control_room | centre | unclear>",
  "purpose": "<one sentence: what the call is mainly about and what was asked or reported>",
  "issue": "<issue id>", "confidence": <0..1>}}"""

SYSTEM_STAGE2 = PREAMBLE + """

The call has been placed in the category `{issue_id}` — {issue_name}: {issue_desc}
The caller's purpose was summarised as: {purpose}

Now pick the single sub-category that fits the PRIMARY purpose. Sub-categories:
{subs}
    - sub_issue `other_category` — use only if the call clearly belongs in a different category.

Reply with one JSON object and nothing else:
{{"sub_issue": "<sub_issue id>", "confidence": <0..1>,
  "evidence": "<the few words from the transcript that decided it, max 20 words>",
  "summary": "<one short sentence, in English, what the caller reports>",
  "resolved_on_call": <true if the issue was resolved during the call>}}"""

USER = """Transcript ({duration:.0f} s):
{dialogue}"""

SYSTEM_RERANK = PREAMBLE + """

A first reading of this call shortlisted these sub-categories. Decide between THEM ONLY, using
the full definitions below; pick the one that matches the call's PRIMARY purpose. If none of
them fits at all, you may answer `unclassified` / `not_classified`.

{shortlist}

Reply with one JSON object and nothing else:
{{"why": "<one sentence comparing the candidates against what was actually said>",
  "issue": "<issue id>", "sub_issue": "<sub_issue id>", "confidence": <0..1>,
  "evidence": "<the few words from the transcript that decided it, max 20 words>",
  "summary": "<one short sentence, in English, what the caller reports>",
  "resolved_on_call": <true if the issue was resolved during the call>}}"""

# where per-call material (examples, focused definitions) is inserted: after the static tree, before the
# answer instructions, so the long prefix (context + rules + TREE) is byte-identical across calls and the
# server's prompt cache (--cache-reuse) skips re-processing it
INSTRUCTION_MARK = {"reason": "\n\nThink before you label.", "single": "\n\nReply with one JSON object and nothing else:"}

FOCUS_MAX_IDS = 22  # detailed definitions per call: enough for the candidates, small enough for a 16k slot

EMPTY_DIALOGUE_WORDS = 3
MAX_DIALOGUE_CHARS = 2200
MODES = ("single", "reason", "two_stage", "rerank", "knn_hybrid", "ft")

# prompt the fine-tuned classifier was trained with (kaggle/build_ft_notebook.py): short, no tree, product context only
SYSTEM_FT_PREFIX = ("You classify calls to the OTR (One Time Registration) helpline of the Uttar Pradesh teacher-eligibility exam. "
                    "Read the Hindi / Hinglish / English transcript and decide the caller's PRIMARY purpose. "
                    "Reply with one JSON object: {\"issue\": <category id>, \"sub_issue\": <sub-category id>, \"resolved_on_call\": <true|false>}. "
                    "Use only the official taxonomy ids.\n\n")
FT_CLIP_CHARS = 2200


@dataclasses.dataclass
class Segregation:
    issue: str
    sub_issue: str
    issue_name: str
    sub_issue_name: str
    confidence: float
    evidence: str
    summary: str
    resolved_on_call: bool | None
    needs_review: bool
    note: str | None = None
    raw: str | None = None
    caller: str | None = None
    purpose: str | None = None
    candidates: str | None = None
    knn_label: str | None = None  # nearest-neighbour label from the labelled pool, when one is configured
    knn_unanimous: bool | None = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _clip(dialogue: str) -> str:
    if len(dialogue) > MAX_DIALOGUE_CHARS:  # keep the opening (where the issue is stated) and the end
        head, tail = dialogue[: MAX_DIALOGUE_CHARS * 2 // 3], dialogue[-MAX_DIALOGUE_CHARS // 3 :]
        dialogue = head + chr(10) + "[...]" + chr(10) + tail
    return dialogue


class Segregator:
    def __init__(self, tax: Taxonomy, llm, low_confidence: float = 0.55, context: str | None = None,
                 empty_sub_issue: str | None = None, mode: str = "single",
                 examples: list[dict] | None = None, votes: int = 1, vote_temperature: float = 0.7,
                 retriever=None, knn=None, focus_pool=None, focus_k: int = 0, focus_groups: dict | None = None):
        if mode not in MODES:
            raise ValueError(f"classify_mode must be one of {MODES}, got {mode!r}")
        self.tax = tax
        self.llm = llm
        self.low = low_confidence
        self.mode = mode
        self.votes = max(1, votes)  # >1: sample this many labels and take the majority (self-consistency)
        self.vote_temperature = vote_temperature
        self.examples = [e for e in (examples or []) if e.get("sub_issue") in tax.by_sub]
        self.retriever = retriever  # dynamic few-shot: nearest labelled calls chosen per transcript
        self.knn = knn  # callable(dialogue) -> {"label", "unanimous", ...}; used by knn_hybrid and for the review flag
        self.empty = empty_sub_issue if empty_sub_issue in tax.by_sub else None  # where a no-speech call is filed
        ctx = (context or DEFAULT_CONTEXT).strip()
        self.schema = tax.json_schema()
        # every mode except plain single runs a reason-mode first pass (rerank, knn_hybrid, two_stage fallback),
        # so they need the reason prompt: the schema asks for caller/purpose/candidates
        self.system_tmpl = SYSTEM_SINGLE if mode == "single" else SYSTEM_REASON
        self.system = self.system_tmpl.format(tree=tax.prompt_block(), context=ctx)
        if mode == "ft":  # the fine-tuned model knows the ids; it is served the exact training prompt
            self.system = SYSTEM_FT_PREFIX + (ctx or "").strip()
        self.system_base = self.system
        self.schema_ft = {"type": "object", "properties": {
            "issue": {"type": "string", "enum": [i.id for i in tax.issues]},
            "sub_issue": {"type": "string", "enum": list(tax.by_sub)},
            "resolved_on_call": {"type": "boolean"}},
            "required": ["issue", "sub_issue", "resolved_on_call"], "additionalProperties": False}
        # focus: detailed definitions only for the sub-categories seen among the nearest labelled calls
        self.focus_pool = focus_pool
        self.focus_k = focus_k
        self.focus_groups = focus_groups or {}
        if self.examples:  # few-shot: short excerpts with their correct label, after the tree
            self.system = self._with_examples(self.examples)

        # two-stage prompts
        cats = []
        for i in tax.issues:
            cats.append(f"- issue `{i.id}` — {i.name}: {i.description}\n    contains: " + ", ".join(s.name for s in i.sub_issues))
        self.system_stage1 = SYSTEM_STAGE1.format(context=ctx, categories="\n".join(cats))
        self.ctx = ctx
        self.schema_reason = dict(self.schema)
        self.schema_reason["properties"] = {"caller": {"type": "string", "enum": ["control_room", "centre", "unclear"]},
                                            "purpose": {"type": "string"},
                                            "candidates": {"type": "array", "minItems": 2, "maxItems": 3,
                                                           "items": {"type": "string", "enum": sorted(tax.by_sub)}},
                                            **self.schema["properties"]}
        self.schema_reason["required"] = ["caller", "purpose", "candidates"] + list(self.schema["required"])
        self.schema_stage1 = {"type": "object", "properties": {
            "caller": {"type": "string", "enum": ["control_room", "centre", "unclear"]},
            "purpose": {"type": "string"},
            "issue": {"type": "string", "enum": [i.id for i in tax.issues]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1}},
            "required": ["caller", "purpose", "issue", "confidence"], "additionalProperties": False}

    # ------------------------------------------------------------------ prompts

    def _focused_base(self, dialogue: str) -> str:
        """System prompt whose tree details only the sub-categories plausible for this call."""
        if self.focus_pool is None or not self.focus_k:
            return self.system_base
        near = self.focus_pool.nearest(dialogue, k=self.focus_k, exclude_text=dialogue)
        freq: dict[str, int] = {}
        for e in near:  # nearest first: ties keep retrieval order
            if e["sub_issue"] in self.tax.by_sub:
                freq[e["sub_issue"]] = freq.get(e["sub_issue"], 0) + 1
        ranked = sorted(freq, key=lambda i: -freq[i])
        ids: list[str] = []
        for i in ranked[:FOCUS_MAX_IDS]:
            if i not in ids:
                ids.append(i)
        if self.focus_groups:  # resolution-group mates of the strongest candidates, while there is room
            for i in ranked[:6]:
                for m, g in self.focus_groups.items():
                    if g == self.focus_groups.get(i) and m in self.tax.by_sub and m not in ids and len(ids) < FOCUS_MAX_IDS:
                        ids.append(m)
        if self.empty and self.empty in self.tax.by_sub and self.empty not in ids:
            ids.append(self.empty)
        ids = set(ids)
        lines = []
        for i in self.tax.issues:  # taxonomy order, full definition + cues
            for sub in i.sub_issues:
                if sub.id in ids:
                    cue = f"  e.g. {', '.join(sub.cues[:3])}" if sub.cues else ""
                    lines.append(f"- `{sub.id}` ({i.id}) — {sub.name}: {sub.description}{cue}")
        block = ("DETAILS — full definitions of the sub-categories most likely for this call; the TREE above lists "
                 "every allowed id, these are just the ones to read carefully:\n" + "\n".join(lines))
        base = self._insert_before_instructions(self.system_base, block)
        if self.examples:
            base = self._with_examples(self.examples, base=base)
        return base

    def _insert_before_instructions(self, base: str, block: str) -> str:
        mark = INSTRUCTION_MARK.get(self.mode, INSTRUCTION_MARK["reason"])
        if mark not in base:
            mark = INSTRUCTION_MARK["reason"] if INSTRUCTION_MARK["reason"] in base else None
        if mark is None:
            return base + "\n\n" + block
        return base.replace(mark, "\n\n" + block + mark, 1)

    def _with_examples(self, examples: list[dict], base: str | None = None) -> str:
        base = self.system_base if base is None else base
        ex = []
        for e in examples:
            if e.get("sub_issue") not in self.tax.by_sub:
                continue
            iss = self.tax.by_sub[e["sub_issue"]][0].id
            ex.append(f"- {e['excerpt'].strip()}\n  -> issue `{iss}`, sub_issue `{e['sub_issue']}`"
                      + (f"  ({e['why']})" if e.get("why") else ""))
        if not ex:
            return base
        return self._insert_before_instructions(base, "EXAMPLES of correctly labelled calls similar to this one (transcript excerpt -> label):\n"
                                                + "\n".join(ex))
    def messages(self, dialogue: str, duration: float, error: str | None = None) -> Messages:
        if self.mode == "ft":
            text = dialogue if len(dialogue) <= FT_CLIP_CHARS else dialogue[:FT_CLIP_CHARS] + " ..."
            return [{"role": "system", "content": self.system}, {"role": "user", "content": "Transcript:\n" + text}]
        user = USER.format(dialogue=_clip(dialogue), duration=duration)
        if error:
            user += f"\n\nYour previous answer was rejected: {error}. Use only ids from the tree."
        system = self.system
        if self.focus_pool is not None and self.focus_k:
            system = self._focused_base(dialogue)
            if self.retriever is not None:
                system = self._with_examples(self.retriever(dialogue), base=system)
        elif self.retriever is not None:
            system = self._with_examples(self.retriever(dialogue))
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _stage2_system(self, issue_id: str, purpose: str) -> tuple[str, dict]:
        i = self.tax.by_issue[issue_id]
        subs = "\n".join(f"    - sub_issue `{s.id}` — {s.name}: {s.description}" + (f"  e.g. {', '.join(s.cues[:4])}" if s.cues else "")
                         for s in i.sub_issues)
        system = SYSTEM_STAGE2.format(context=self.ctx, issue_id=i.id, issue_name=i.name, issue_desc=i.description,
                                      purpose=purpose, subs=subs)
        schema = {"type": "object", "properties": {
            "sub_issue": {"type": "string", "enum": [s.id for s in i.sub_issues] + ["other_category"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "evidence": {"type": "string"}, "summary": {"type": "string"}, "resolved_on_call": {"type": "boolean"}},
            "required": ["sub_issue", "confidence", "evidence", "summary", "resolved_on_call"], "additionalProperties": False}
        return system, schema

    # ------------------------------------------------------------------ run
    def run(self, dialogues: list[str], durations: list[float]) -> list[Segregation]:
        out: list[Segregation | None] = [None] * len(dialogues)
        todo = []
        for k, d in enumerate(dialogues):
            if len(d.split()) <= EMPTY_DIALOGUE_WORDS:
                if self.empty:
                    iss = self.tax.by_sub[self.empty][0].id
                    iname, sname = self.tax.names(iss, self.empty)
                    out[k] = Segregation(iss, self.empty, iname, sname, 1.0, "", "No usable speech in the recording.",
                                         None, False, "empty transcript")
                else:
                    out[k] = Segregation(UNRESOLVED, UNRESOLVED, UNRESOLVED, UNRESOLVED, 0.0, "", "No usable speech in the recording.",
                                         None, True, "empty transcript")
            else:
                todo.append(k)
        if not todo:
            return out  # type: ignore[return-value]
        if self.mode == "two_stage":
            self._run_two_stage(todo, dialogues, durations, out)
        elif self.mode == "rerank":
            self._run_rerank(todo, dialogues, durations, out)
        elif self.mode == "knn_hybrid":
            saved = self.mode
            self.mode = "reason"
            try:
                self._run_single(todo, dialogues, durations, out)
            finally:
                self.mode = saved
        else:
            self._run_single(todo, dialogues, durations, out)
        if self.knn is not None:  # attach the neighbour label; disagreement with the LLM is the review signal
            for k in todo:
                seg = out[k]
                r = self.knn(dialogues[k])
                seg.knn_label, seg.knn_unanimous = r["label"], bool(r["unanimous"])
                if self.mode == "knn_hybrid" and r["unanimous"] and r["label"] in self.tax.by_sub and r["label"] != seg.sub_issue:
                    iss = self.tax.by_sub[r["label"]][0].id
                    seg.note = (seg.note + "; " if seg.note else "") + f"knn unanimous overrode {seg.sub_issue}"
                    seg.issue, seg.sub_issue = iss, r["label"]
                    seg.issue_name, seg.sub_issue_name = self.tax.names(iss, r["label"])
                seg.needs_review = seg.needs_review or (seg.knn_label != seg.sub_issue)
        return out  # type: ignore[return-value]

    def _run_single(self, todo, dialogues, durations, out) -> None:
        schema = self.schema_reason if self.mode == "reason" else self.schema_ft if self.mode == "ft" else self.schema
        max_new = 300 if self.mode == "reason" else 60 if self.mode == "ft" else 160
        msgs = [self.messages(dialogues[k], durations[k]) for k in todo]
        replies = self.llm.chat_batch(msgs, max_new_tokens=max_new, json_schema=schema)
        if self.votes > 1:
            replies = self._vote(msgs, replies, max_new, schema)
        retry = []
        for k, rep in zip(todo, replies):
            if self.mode == "ft":  # no confidence field in the trained output: a valid label is trusted, review comes from kNN disagreement
                obj = extract_json(rep)
                if obj and "confidence" not in obj:
                    obj["confidence"] = 0.9
                seg, err = self._parse(rep, parsed=obj) if obj else self._parse(rep)
            else:
                seg, err = self._parse(rep)
            if err:
                retry.append((k, err))
            out[k] = seg
        if retry:
            replies2 = self.llm.chat_batch([self.messages(dialogues[k], durations[k], err) for k, err in retry],
                                           max_new_tokens=max_new, json_schema=schema)
            for (k, _), rep in zip(retry, replies2):
                seg, err = self._parse(rep)
                if not err:
                    out[k] = seg
                else:  # keep the first attempt's (already UNRESOLVED) result, note both errors
                    out[k].note = f"{out[k].note}; retry: {err}"

    def _run_rerank(self, todo, dialogues, durations, out) -> None:
        import re

        saved = self.mode
        self.mode = "reason"
        try:
            self._run_single(todo, dialogues, durations, out)
        finally:
            self.mode = saved
        msgs, schemas, idx = [], [], []
        for k in todo:
            first = out[k]
            ids = [first.sub_issue] + re.findall(r"[a-z][a-z0-9_]+", first.candidates or "")
            if self.retriever is not None:  # labels of the nearest labelled calls widen the shortlist
                ids += [e["sub_issue"] for e in self.retriever(dialogues[k])[:4]]
            seen, cand = set(), []
            for i in ids:
                if i in self.tax.by_sub and i not in seen:
                    seen.add(i)
                    cand.append(i)
            if len(cand) < 2:
                continue  # nothing to decide between
            cand = cand[:5]
            lines = []
            for c in cand:
                issue, sub = self.tax.by_sub[c]
                lines.append(f"- issue `{issue.id}` — {issue.name}: {issue.description}\n    sub_issue `{sub.id}` — {sub.name}: {sub.description}"
                             + (f"\n    e.g. {', '.join(sub.cues[:5])}" if sub.cues else ""))
            system = SYSTEM_RERANK.format(context=self.ctx, shortlist="\n".join(lines))
            schema = {"type": "object", "properties": {
                "why": {"type": "string"},
                "issue": {"type": "string", "enum": sorted({self.tax.by_sub[c][0].id for c in cand} | {"unclassified"})},
                "sub_issue": {"type": "string", "enum": cand + ["not_classified"]},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "evidence": {"type": "string"}, "summary": {"type": "string"}, "resolved_on_call": {"type": "boolean"}},
                "required": ["why", "issue", "sub_issue", "confidence", "evidence", "summary", "resolved_on_call"],
                "additionalProperties": False}
            msgs.append([{"role": "system", "content": system},
                         {"role": "user", "content": USER.format(dialogue=_clip(dialogues[k]), duration=durations[k])}])
            schemas.append(schema)
            idx.append(k)
        if not msgs:
            return
        replies = [self.llm.chat_batch([m], max_new_tokens=220, json_schema=sc)[0] for m, sc in zip(msgs, schemas)]
        for k, rep in zip(idx, replies):
            obj = extract_json(rep) or {}
            sub = str(obj.get("sub_issue", ""))
            first = out[k]
            if sub == UNRESOLVED or sub not in self.tax.by_sub:
                # the re-ranker rejected the whole shortlist: keep the first label but flag it for review
                first.needs_review = True
                first.note = (first.note + "; " if first.note else "") + "rerank unsure"
                continue
            seg, _ = self._parse(rep, parsed={**obj, "issue": self.tax.by_sub[sub][0].id, "caller": first.caller,
                                              "purpose": first.purpose, "candidates": first.candidates})
            if seg.sub_issue != first.sub_issue:
                seg.note = (seg.note + "; " if seg.note else "") + f"reranked from {first.sub_issue}"
            out[k] = seg

    def _run_two_stage(self, todo, dialogues, durations, out) -> None:
        msgs1 = [[{"role": "system", "content": self.system_stage1},
                  {"role": "user", "content": USER.format(dialogue=_clip(dialogues[k]), duration=durations[k])}] for k in todo]
        rep1 = self.llm.chat_batch(msgs1, max_new_tokens=160, json_schema=self.schema_stage1)
        stage2: list[tuple[int, str, dict]] = []
        fallback: list[int] = []
        for k, rep in zip(todo, rep1):
            obj = extract_json(rep) or {}
            issue = str(obj.get("issue", ""))
            if issue not in self.tax.by_issue:
                fallback.append(k)
                continue
            stage2.append((k, issue, obj))
        msgs2, schemas = [], []
        for k, issue, obj in stage2:
            system, schema = self._stage2_system(issue, str(obj.get("purpose", "")))
            msgs2.append([{"role": "system", "content": system},
                          {"role": "user", "content": USER.format(dialogue=_clip(dialogues[k]), duration=durations[k])}])
            schemas.append(schema)
        # schemas differ per call (enum = that category's subs): send one at a time through chat_batch
        rep2 = [self.llm.chat_batch([m], max_new_tokens=160, json_schema=s)[0] for m, s in zip(msgs2, schemas)]
        for (k, issue, obj1), rep in zip(stage2, rep2):
            obj2 = extract_json(rep) or {}
            sub = str(obj2.get("sub_issue", ""))
            if sub == "other_category" or sub not in self.tax.by_sub:
                fallback.append(k)
                continue
            merged = {**obj2, "issue": issue, "caller": obj1.get("caller"), "purpose": obj1.get("purpose"),
                      "candidates": None}
            seg, _ = self._parse(merged if isinstance(merged, dict) else rep, parsed=merged)
            seg.raw = rep
            out[k] = seg
        if fallback:  # the whole-tree reasoning prompt decides the leftovers
            saved = self.mode
            self.mode = "reason"
            try:
                self._run_single(fallback, dialogues, durations, out)
            finally:
                self.mode = saved
            for k in fallback:
                out[k].note = (out[k].note + "; " if out[k].note else "") + "two-stage bounced to whole tree"

    def _vote(self, msgs, greedy, max_new, schema):
        """Self-consistency: the greedy reply plus (votes-1) sampled ones; the majority
        (issue, sub_issue) wins and the first reply carrying it is returned. Ties -> greedy."""
        from collections import Counter

        samples = [greedy]
        for _ in range(self.votes - 1):
            samples.append(self.llm.chat_batch(msgs, max_new_tokens=max_new, json_schema=schema,
                                               temperature=self.vote_temperature))
        out = []
        for i in range(len(msgs)):
            labels = []
            for s in samples:
                o = extract_json(s[i]) or {}
                labels.append((str(o.get("issue", "")), str(o.get("sub_issue", ""))))
            top, n = Counter(labels).most_common(1)[0]
            if n <= 1 or top == labels[0]:
                out.append(greedy[i])
            else:
                out.append(samples[labels.index(top)][i])
        return out

    # ------------------------------------------------------------------ parse
    def _parse(self, reply: str, parsed: dict | None = None) -> tuple[Segregation, str | None]:
        obj = parsed if parsed is not None else extract_json(reply)
        if not obj:
            return (Segregation(UNRESOLVED, UNRESOLVED, UNRESOLVED, UNRESOLVED, 0.0, "", "", None, True,
                                "model did not return JSON", reply), "no JSON object in reply")
        issue, sub, problem = self.tax.validate(str(obj.get("issue", "")), str(obj.get("sub_issue", "")))
        try:
            conf = float(obj.get("confidence", 0))
        except (TypeError, ValueError):
            conf = 0.0
        conf = min(max(conf, 0.0), 1.0)
        iname, sname = self.tax.names(issue, sub)
        seg = Segregation(
            issue=issue, sub_issue=sub, issue_name=iname, sub_issue_name=sname, confidence=conf,
            evidence=str(obj.get("evidence", ""))[:400], summary=str(obj.get("summary", ""))[:400],
            resolved_on_call=obj.get("resolved_on_call") if isinstance(obj.get("resolved_on_call"), bool) else None,
            needs_review=(sub == UNRESOLVED) or conf < self.low, note=problem, raw=reply if parsed is None else None,
            caller=obj.get("caller"), purpose=obj.get("purpose"),
            candidates=(", ".join(map(str, obj["candidates"])) if isinstance(obj.get("candidates"), list)
                        else str(obj["candidates"])) if obj.get("candidates") else None,
        )
        hard_error = problem if sub == UNRESOLVED else None  # a corrected parent is not worth a retry
        return seg, hard_error
