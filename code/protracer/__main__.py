"""python -m protracer {diagnose, learn, evaluate} ...

A split is an annotation file with `tasks/` and `episodes/` next to it.
Every model call goes through OpenRouter (set OPENROUTER_API_KEY).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from .data import Sample, load_episode, load_samples, load_task, write_atomic
from .evaluation import JUDGE_FIELDS, failure_metrics, judge_scores, judged_metric
from .experience import learn_experience
from .llm import OpenRouter, OpenRouterError
from .pipeline import ProTracer

MODEL = "google/gemini-3.1-pro-preview"
JUDGE = "openai/gpt-5"


def positive(value: str) -> int:
    if (n := int(value)) < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return n


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="protracer", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    def command(name: str, help: str) -> argparse.ArgumentParser:
        p = commands.add_parser(name, help=help)
        p.add_argument("split", type=Path, help="annotation file of the split")
        p.add_argument("--model", default=MODEL, help="OpenRouter model id")
        p.add_argument("--reasoning", default="high", help="OpenRouter reasoning effort")
        p.add_argument("--limit", type=positive, help="only the first N episodes")
        return p

    p = command("diagnose", "run ProTracer over a split")
    p.add_argument("--out", type=Path, required=True, help="run directory")
    p.add_argument("--experience", type=Path, help="rules from `learn` (ProTracer+Exp.)")
    p.add_argument("--keyframes", choices=("sign-cpd", "continuous-cpd"), default="sign-cpd", help="Table III")
    p.add_argument("--workers", type=positive, default=8)

    p = command("learn", "learn reflective experience on a calibration split")
    p.add_argument("--out", type=Path, required=True, help="rule file; rerunning resumes from <out>.log.jsonl")
    p.add_argument("--epochs", type=positive, default=3)
    p.add_argument("--seed", type=int, default=42, help="shuffles the calibration order")

    p = commands.add_parser("evaluate", help="score a run directory")
    p.add_argument("run", type=Path, help="run directory written by diagnose")
    p.add_argument("splits", type=Path, nargs="+", help="annotation file(s); several are pooled")
    p.add_argument("--judge", default=JUDGE, help="OpenRouter model id of the reason / avoidance judge")
    p.add_argument("--no-judge", action="store_true", help="skip the reason / avoidance scores")

    args = parser.parse_args(argv)
    if getattr(args, "keyframes", None) == "continuous-cpd" and not importlib.util.find_spec("ruptures"):
        parser.error("--keyframes continuous-cpd needs `pip install ruptures`")
    try:
        run_command(args)
    except OpenRouterError as e:
        sys.exit(f"error: {e}")


def run_command(args: argparse.Namespace) -> None:
    if args.command == "evaluate":
        samples = [s for split in args.splits for s in load_samples(split)]
        return evaluate(samples, args.run, None if args.no_judge else args.judge)
    samples = load_samples(args.split)
    tasks = args.split.parent / "tasks"
    llm = OpenRouter(args.model, reasoning_effort=args.reasoning)
    settings = {"command": args.command, "model": args.model, "reasoning": args.reasoning}

    if args.command == "learn":
        rules = learn_experience(ProTracer(llm), samples, tasks, args.out, args.epochs, args.seed, args.limit)
        print(f"{len(rules)} rules -> {args.out}  (${llm.spent:.2f} this session)")
        return

    tracer = ProTracer(llm, keyframes=args.keyframes)
    experience = args.experience.read_text() if args.experience else ""
    settings.update(keyframes=args.keyframes,
                    experience=hashlib.sha1(experience.encode()).hexdigest()[:12] if experience else None)
    run = lambda s: tracer(load_episode(s.path, s.robot, s.name), load_task(tasks, s.task), experience)
    claim_run_dir(args.out, args.split.name, settings)
    run_split(samples[:args.limit], args.out / "predictions", run, args.workers)
    print(f"${llm.spent:.2f} this session")


def claim_run_dir(out: Path, split: str, settings: dict) -> None:
    """Record the settings of a split's run so a resume cannot mix configurations."""
    out.mkdir(parents=True, exist_ok=True)
    path = out / "settings.json"
    saved = json.loads(path.read_text()) if path.exists() else {}
    if saved.get(split, settings) != settings:
        sys.exit(f"error: {out} holds a {split} run with {saved[split]}; use another --out")
    write_atomic(path, json.dumps({**saved, split: settings}, indent=1))


def run_split(samples: list[Sample], out: Path, run: Callable[[Sample], dict], workers: int) -> None:
    """Write run(sample) to out/<sample>.json for every episode without a verdict yet.

    Each worker saves its own result, so an interrupted run keeps whatever
    finished; queued episodes are cancelled."""
    out.mkdir(parents=True, exist_ok=True)
    todo = [s for s in samples if not _has_verdict(out / f"{s.name}.json")]

    def job(sample: Sample) -> dict:
        record = run(sample)
        write_atomic(out / f"{sample.name}.json", json.dumps(record, indent=1, ensure_ascii=False))
        return record

    pool = ThreadPoolExecutor(workers)
    try:
        jobs = {pool.submit(job, s): s for s in todo}
        for i, done in enumerate(as_completed(jobs), start=1):
            name = jobs[done].name
            try:
                v = done.result()["verdict"] or {}
            except OpenRouterError as e:
                if e.fatal:
                    raise
                print(f"[{i}/{len(todo)}] {name}  FAILED  {e}", file=sys.stderr, flush=True)
                continue
            except Exception as e:   # a rerun retries it
                print(f"[{i}/{len(todo)}] {name}  FAILED  {type(e).__name__}: {e}", file=sys.stderr, flush=True)
                continue
            print(f"[{i}/{len(todo)}] {name}  {v.get('result')}  onset={v.get('onset_t')}  type={v.get('type')}",
                  flush=True)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def _has_verdict(path: Path) -> bool:
    return path.exists() and json.loads(path.read_text()).get("verdict") is not None


def evaluate(samples: list[Sample], run: Path, judge_model: str | None) -> None:
    if not (run / "predictions").is_dir():
        sys.exit(f"error: no predictions in {run}")
    verdicts = {}
    for path in (run / "predictions").glob("*.json"):
        record = json.loads(path.read_text())
        verdicts[record["sample"]] = record["verdict"]
    metrics = failure_metrics(verdicts, samples)
    if judge_model:
        judge = OpenRouter(judge_model, reasoning_effort="minimal", max_tokens=512)
        for field in JUDGE_FIELDS:
            scores = judge_scores(verdicts, samples, field, judge, cache=run / f"judge_{field}.json")
            metrics[field] = judged_metric(scores, samples)
    write_atomic(run / "metrics.json", json.dumps(metrics, indent=1))

    missing = sum(s.name not in verdicts for s in samples)
    print(f"{len(samples)} episodes ({sum(s.failed for s in samples)} failures), {missing} without a prediction")
    print("  ".join(f"{k} {'n/a' if v is None else f'{v:.2f}'}" for k, v in metrics.items()))


if __name__ == "__main__":
    main()
