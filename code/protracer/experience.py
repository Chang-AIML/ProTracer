"""Reflective experience for few-shot adaptation (paper Sec. III-B.3).

On a small calibration set, every prediction that misses the ground truth is
shown back to the VLM together with the label; the VLM proposes one edit to a
list of transferable rules (ADD / UPDATE / MERGE / NONE). The edit is kept only
if re-running the diagnosis with it fixes that episode, and the rule list is
compressed whenever it outgrows its word budget. At test time the rules are
appended, unchanged, to the round-2 prompt.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

from . import prompts
from .data import FAILURE_TYPES, Sample, Task, canonical_type, load_episode, load_task, write_atomic
from .llm import message
from .pipeline import ProTracer, ask

ONSET_TOLERANCE = 0.5   # s
WORD_BUDGET = 300       # the rule list is compressed beyond this many words
RULE_WORDS = 44         # longest acceptable rule (asked for <= 40 words)

_LESSON = re.compile(r"```lesson\s*(.*?)\s*```", re.S)
_EXPERIENCE = re.compile(r"```experience\s*(.*?)\s*```", re.S)
_LEAKS = (re.compile(r"\bepisode[ _]?\d+\b", re.I),   # episode identifiers
          re.compile(r"\b\d+\.\d+\s*s\b"))              # episode-specific timestamps


_BULLET = re.compile(r"^\s*(?:[-*•]\s*|\d+[.)]\s+)(\S.*)$")


def parse_rules(experience: str) -> list[str]:
    """Bulleted ('- ', '* ', '• ') or numbered ('1. ') lines -> rules."""
    return [m.group(1).strip() for line in experience.splitlines() if (m := _BULLET.match(line))]


def render_rules(rules: list[str]) -> str:
    return "\n".join(f"- {rule}" for rule in rules)


def edit_rules(rules: list[str], reply: str) -> tuple[list[str], str] | None:
    """Apply the ```lesson``` edit of a reflection reply -> (new rules, edited rule).

    `ADD: r`, `UPDATE i: r`, `MERGE [i, j]: r`, or `NONE`; a bare rule counts as
    ADD. NONE, a malformed UPDATE / MERGE, or a bad index leave the rules unchanged.
    """
    if not (match := _LESSON.search(reply)) or not (op := match.group(1).strip()) or re.match(r"NONE\b", op, re.I):
        return None
    if m := re.match(r"UPDATE\s*#?\s*(\d+)\s*:\s*(.+)$", op, re.S | re.I):
        i, rule = int(m.group(1)) - 1, m.group(2).strip()
        return (rules[:i] + [rule] + rules[i + 1:], rule) if 0 <= i < len(rules) else None
    if m := re.match(r"MERGE\s*\[?([\d,#\s]+?)\]?\s*:\s*(.+)$", op, re.S | re.I):
        merged = {int(x) - 1 for x in re.findall(r"\d+", m.group(1))}
        rule = m.group(2).strip()
        if not merged or not all(0 <= i < len(rules) for i in merged):
            return None
        return [r for i, r in enumerate(rules) if i not in merged] + [rule], rule
    if re.match(r"(UPDATE|MERGE)\b", op, re.I):
        return None
    rule = re.sub(r"^ADD\s*:\s*", "", op, flags=re.I).strip()
    return rules + [rule], rule


def leaks(rule: str) -> bool:
    """A rule must transfer: no episode ids, timestamps, or quoted non-taxonomy names."""
    quoted = re.findall(r"`([^`]+)`", rule)
    return any(p.search(rule) for p in _LEAKS) or \
        any(q.strip() not in (*FAILURE_TYPES, "video_end") for q in quoted)


@dataclass
class Assessment:
    """How a prediction compares with the ground truth of one episode."""
    verdict_ok: bool
    type_ok: bool
    onset_error: float | None

    @property
    def needs_reflection(self) -> bool:
        """Wrong verdict, wrong type, or an onset off by more than the tolerance."""
        onset_off = self.onset_error is not None and self.onset_error > ONSET_TOLERANCE
        return not self.verdict_ok or not self.type_ok or onset_off

    def improved_to(self, after: "Assessment") -> bool:
        """The paper's acceptance rule. An edit is dropped if it breaks a right
        verdict or pushes an onset that was within tolerance outside it.
        Otherwise it is kept if it fixes a wrong verdict, or, with the verdict
        right, fixes the type or brings a wrong onset within tolerance (or
        halves its error, by at least 0.5 s)."""
        tol = ONSET_TOLERANCE
        if self.verdict_ok and not after.verdict_ok:
            return False
        if self.onset_error is not None and self.onset_error <= tol and \
                after.onset_error is not None and after.onset_error > tol:
            return False
        if not self.verdict_ok:
            return after.verdict_ok
        onset_better = (self.onset_error is not None and after.onset_error is not None
                        and self.onset_error > tol
                        and (after.onset_error <= tol or (after.onset_error <= 0.5 * self.onset_error
                                                          and self.onset_error - after.onset_error >= 0.5)))
        return after.verdict_ok and ((not self.type_ok and after.type_ok) or onset_better)


def assess(verdict: dict | None, sample: Sample) -> Assessment:
    if verdict is None:
        return Assessment(False, False, None)
    result = str(verdict.get("result", "")).strip().lower()
    type_ok = not sample.failed or canonical_type(str(verdict.get("type", "")).strip()) in sample.types
    error = None
    if sample.failed and result == "failure":
        try:
            error = round(abs(float(verdict["onset_t"]) - float(sample.onset)), 3)
        except (KeyError, TypeError, ValueError):
            pass
    return Assessment(result == ("failure" if sample.failed else "success"), type_ok, error)


def learn_experience(tracer: ProTracer, samples: list[Sample], tasks_dir: Path, out: Path,
                     epochs: int = 3, seed: int = 42, limit: int | None = None) -> list[str]:
    """Run the reflective loop over a calibration set: the first `limit` episodes
    of a seeded shuffle, in the same order every epoch. The rules are saved to
    `out` after every episode and each step is logged next to it, so running
    the same command again resumes where it stopped."""
    log = out.with_name(out.name + ".log.jsonl")
    rules, done = [], set()
    if log.exists():
        for event in map(json.loads, log.read_text().splitlines()):
            rules, done = event["rules"], done | {(event["epoch"], event["sample"])}
    order = list(samples)
    random.Random(seed).shuffle(order)
    for epoch in range(1, epochs + 1):
        for sample in order[:limit]:
            if (epoch, sample.name) in done:
                continue
            rules, event = _reflect(tracer, sample, load_task(tasks_dir, sample.task), rules)
            write_atomic(out, render_rules(rules) + "\n")
            with open(log, "a") as f:
                f.write(json.dumps({"epoch": epoch, "sample": sample.name, **event, "rules": rules,
                                    "spent": round(tracer.llm.spent, 4)}, ensure_ascii=False) + "\n")
            print(f"[epoch {epoch}] {sample.name}: {event['decision']}  ({len(rules)} rules, "
                  f"${tracer.llm.spent:.2f} this session)", flush=True)
    return rules


def _reflect(tracer: ProTracer, sample: Sample, task: Task, rules: list[str]) -> tuple[list[str], dict]:
    """One calibration episode: diagnose, and on a miss reflect, verify, keep or drop the edit."""
    evidence = tracer.observe(load_episode(sample.path, sample.robot, sample.name))
    captions, _ = tracer.caption(task, evidence)
    messages = tracer.diagnosis_messages(task, evidence, captions, render_rules(rules))
    verdict, response, _ = ask(tracer.llm, messages)
    before = assess(verdict, sample)
    if not before.needs_reflection:
        return rules, {"decision": "correct"}

    reflection = tracer.llm.chat([*messages, message("assistant", response), prompts.reflection(sample.label, rules)])
    edit = edit_rules(rules, reflection.text)
    if edit is None:
        return rules, {"decision": "no lesson"}
    candidate, rule = edit
    if len(rule.split()) > RULE_WORDS or leaks(rule):
        return rules, {"decision": "rejected: not transferable", "rule": rule}

    retry, _, _ = tracer.diagnose(task, evidence, captions, render_rules(candidate))
    if not before.improved_to(assess(retry, sample)):
        return rules, {"decision": "rejected: no fix", "rule": rule}
    if len(render_rules(candidate).split()) > WORD_BUDGET:
        candidate = _compress(tracer, candidate)
    return candidate, {"decision": "accepted", "rule": rule}


def _compress(tracer: ProTracer, rules: list[str]) -> list[str]:
    """Rewrite the rules within the word budget; keep them as they are if the
    reply holds no usable list."""
    reply = tracer.llm.chat(prompts.compression(render_rules(rules)))
    match = _EXPERIENCE.search(reply.text)
    return (parse_rules(match.group(1)) if match else []) or rules
