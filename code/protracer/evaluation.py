"""Metrics of paper Sec. V-A.

* Detection: verdict accuracy over all episodes (a missing prediction is wrong).
* Type: over failed episodes; correct if the predicted type is in the label set.
* Onset: over failed episodes; acc@d counts |error| <= d, MAE averages the error
  over the failures that were also predicted as failures.
* Reason / Avoidance: GPT-5 judge score in [0, 1], summed over failed episodes
  and divided by their number (unanswered failures score 0). The judge follows
  ViFailback's rubrics, JUDGE_RUBRICS below.

Metrics that do not apply (e.g. onset on a split without failures) are None.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from string import Template

from .data import Sample, canonical_type, write_atomic
from .llm import OpenRouter, OpenRouterError, message

TOLERANCES = (0.5, 1.0, 2.0)
JUDGE_FIELDS = ("reason", "avoidance")


def failure_metrics(verdicts: dict[str, dict | None], samples: list[Sample]) -> dict[str, float | None]:
    failures = [s for s in samples if s.failed]
    detected = typed = 0
    errors = []
    for s in samples:
        if not (v := verdicts.get(s.name)):
            continue
        predicted_failure = v.get("result") == "failure"
        detected += predicted_failure == s.failed
        if s.failed and predicted_failure:
            typed += canonical_type(str(v.get("type") or "")) in s.types
            onset = v.get("onset_t")
            if isinstance(onset, (int, float)) and isinstance(s.onset, (int, float)):
                errors.append(abs(onset - s.onset))
    metrics = {"detection": 100 * detected / len(samples) if samples else None}
    if failures:
        metrics["type"] = 100 * typed / len(failures)
        metrics["onset_mae"] = sum(errors) / len(errors) if errors else None
        for d in TOLERANCES:
            metrics[f"onset@{d:g}s"] = 100 * sum(e <= d for e in errors) / len(failures)
    return metrics


def judge_scores(verdicts: dict[str, dict | None], samples: list[Sample], field: str,
                 judge: OpenRouter, cache: Path | None = None, workers: int = 16) -> dict[str, float]:
    """Per-episode judge scores for `field` ("reason" | "avoidance").

    `cache` keeps every score with the judge and the text it scored, so a score
    is reused only for the very same prediction and judge model.
    """
    cached = json.loads(cache.read_text()) if cache and cache.exists() else {}
    scores, todo = {}, []
    for s in samples:
        if not s.failed or not (candidate := _as_text((verdicts.get(s.name) or {}).get(field))):
            continue
        hit = cached.get(s.name)
        if isinstance(hit, dict) and hit.get("judge") == judge.model and hit.get("candidate") == candidate:
            scores[s.name] = hit["score"]
        else:
            todo.append((s, candidate))

    def score(sample: Sample, candidate: str) -> float | None:
        rubric = Template(JUDGE_RUBRICS[field]).substitute(reference=_as_text(sample.label[field]), candidate=candidate)
        reply = judge.chat([message("user", rubric)])
        number = re.search(r"-?\d*\.?\d+", reply.text)
        value = float(number.group()) if number else None
        if value is not None and not 0 <= value <= 1:   # a judge occasionally answers in percent
            value = value / 100 if 0 <= value <= 100 else None
        return value

    failed = 0
    pool = ThreadPoolExecutor(workers)
    try:
        jobs = {pool.submit(score, *job): job for job in todo}
        for job in as_completed(jobs):
            sample, candidate = jobs[job]
            try:
                value = job.result()
            except OpenRouterError as e:
                if e.fatal:
                    raise
                value = None
            if value is None:   # left unscored; the next evaluation retries it
                failed += 1
                continue
            scores[sample.name] = value
            cached[sample.name] = {"score": value, "judge": judge.model, "candidate": candidate}
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        if cache:
            write_atomic(cache, json.dumps(cached, indent=1, ensure_ascii=False))
    if failed:
        print(f"{failed} {field} judgements failed and count as 0; rerun `evaluate` to retry them")
    return scores


def judged_metric(scores: dict[str, float], samples: list[Sample]) -> float | None:
    failures = [s for s in samples if s.failed]
    return 100 * sum(scores.get(s.name, 0.0) for s in failures) / len(failures) if failures else None


def _as_text(value) -> str:
    """Reasons are either the released 'Expected/Observation/Result' string or
    the predicted {expected, observed, result} object."""
    if isinstance(value, dict):
        return "\n".join(f"{label}: {value[key]}" for label, key in
                         (("Expected", "expected"), ("Observed", "observed"), ("Result", "result")) if value.get(key))
    return str(value or "").strip()


# ViFailback's rubrics, as sent to the judge for the paper.
JUDGE_RUBRICS = {
    "reason": """You are an expert evaluator specializing in robotic manipulation tasks, capable of understanding and judging the semantic accuracy of failure reason involving robot actions, objects, and spatial reasoning.

You will rate how semantically consistent Text B is compared to Text A on a continuous scale from 0.0 to 1.0.

**Evaluation criteria:**

- Focus on whether the two descriptions convey the same manipulation **intent, sequence, and outcome**.

- Key aspects if applicapable to consider:

1. **Gripper usage**: left vs. right gripper, open/close actions.

2. **Action correctness**: pick, place, move, align, push, lift, etc.

3. **Object and target consistency**: same object names and corresponding targets.

4. **Order and causality**: whether the sequence of steps is preserved.

5. **Success condition**: whether the goal or outcome matches.

**Scoring guidance:**

- 1.0 → Exactly the same meaning and correct execution description.

- 0.8–0.9 → Minor paraphrasing differences but semantically identical in robot actions and outcomes.

- 0.6–0.7 → Mostly correct but with small errors (e.g., gripper swapped, one step missing).

- 0.3–0.5 → Partially correct but with notable mismatches in object, action, or order.

- 0.1–0.2 → Only slightly related, mostly incorrect.

- 0.0 → Completely unrelated or contradicting actions.

Return **only** the numeric score between 0.0 and 1.0.

Text A (reference): ${reference} Text B (candidate): ${candidate}

ATTENTION: You MUST ONLY give me a score number. The score MUST be from 0.0 to 1.0.""",
    "avoidance": """You are an expert evaluator for robotic manipulation tasks. Rate how semantically consistent Text B (candidate corrective action) is with Text A (reference), on a continuous 0.0–1.0 scale.

Text B should describe a single corrective action taken BEFORE failure onset.

**Score only on dimensions explicitly present in the reference.** Extra detail in the candidate is not penalized unless it directly contradicts the reference.

**Key aspects:**

1. **Gripper**: left vs. right (required if in reference).
2. **Action verb**: close / open / move / rotate / align / etc. — must match in kind.
3. **Object** (only if named in reference): same target.
4. **Direction / magnitude** (only if in reference): spatial direction (left/right/up/down/forward/backward).
5. **Unavoidable case**: if reference says the failure is unavoidable, candidate must also mention that or suggest a practical workaround (e.g., keeping still or waiting for human resolution); 

**Scoring:**

- 1.0 → Same meaning.
- 0.8–0.9 → Minor paraphrasing, semantically identical.
- 0.6–0.7 → Same gripper + verb, but a direction/magnitude *present in reference* is slightly off.
- 0.3–0.5 → Wrong arm, related-but-distinct action, or notable mismatch.
- 0.1–0.2 → Opposite direction or action-kind mismatch (rotate vs shift).
- 0.0 → Unrelated, contradicting, or unavoidable-case mismatch.

Return **only** the numeric score.

Text A (reference): ${reference} Text B (candidate): ${candidate}

ATTENTION: You MUST ONLY give me a score number from 0.0 to 1.0.
""",
}
