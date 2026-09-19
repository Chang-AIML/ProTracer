"""ProTracer's chat messages. The texts live in templates/ and are the ones used
for the paper; this module only fills their placeholders."""
from __future__ import annotations

import json
import re
from pathlib import Path
from string import Template

from .data import VIEWS, Task
from .llm import image, message, text

_DIR = Path(__file__).parent / "templates"


def _read(name: str) -> str:
    return (_DIR / name).read_text()


def _fill(name: str, **fields) -> str:
    return Template(_read(name)).substitute(**fields)


def _system(task: Task) -> dict:
    """Role description and task context; substages and taxonomy come in round 2."""
    return message("system", _read("system_prompt.xml") + "\n---\n\n" + task.context)


def _keyframes(times: list[float], frames: list[dict[str, bytes]]) -> list[dict]:
    """One <timestamp> block per keyframe holding one tagged image per camera."""
    parts = []
    for t, views in zip(times, frames):
        parts.append(text(f'\n<timestamp t="{t:.2f}s">\n'))
        for view, jpeg in views.items():
            parts += [text(f'  <frame view="{VIEWS[view]}" cam="{view}">\n'), image(jpeg), text("  </frame>\n")]
        parts.append(text("</timestamp>\n"))
    return parts


# ── the two rounds ─────────────────────────────────────────────────────────

def caption(task: Task, times: list[float], frames: list[dict[str, bytes]]) -> list[dict]:
    """Round 1, scene captions: gripper state and object relations per keyframe."""
    head = _fill("round1_scene_caption.xml", n_boundaries=len(times), n_imgs=sum(map(len, frames)),
                 task_desc=task.description)
    return [_system(task), message("user", [text(head), *_keyframes(times, frames)])]


def diagnosis(task: Task, narrative: str, per_arm_frames: bool, experience: str = "") -> dict:
    """Round 2, failure diagnosis: follows round 1 in the same conversation."""
    section = ""
    if experience.strip():
        section = ('\n  <experience note="transferable rules learned from prior val cases — apply when relevant">\n'
                   + experience.strip() + "\n  </experience>")
    guide = "round2_narrative_guide_per_arm_frame.xml" if per_arm_frames else "round2_narrative_guide_world_frame.xml"
    return message("user", _fill("round2_failure_diagnosis.xml",
                                 substages="\n".join(task.substages) or "(no substages defined for this task)",
                                 sensor_semantics=_read(guide).rstrip(), summary=narrative + "\n",
                                 taxonomy=_read("round2_failure_taxonomy.xml").rstrip(), experience_section=section))


# ── reflective experience ──────────────────────────────────────────────────

_FILE_FIELDS = ("sample", "robot", "hdf5")    # where an episode is stored, not what happened in it


def reflection(label: dict, rules: list[str]) -> dict:
    """Asks for one ADD / UPDATE / MERGE / NONE edit after seeing the ground truth."""
    truth = {key: value for key, value in label.items() if key not in _FILE_FIELDS}
    prior = "\n".join(f"{i}. {rule}" for i, rule in enumerate(rules, start=1)) or "(empty)"
    return message("user", _fill("experience_reflection.xml", ground_truth_json=json.dumps(truth, indent=2, ensure_ascii=False),
                                 prior_experience=prior))


def compression(experience: str) -> list[dict]:
    return [message("user", _fill("experience_compression.xml", experience=experience.strip()))]


# ── parsing ────────────────────────────────────────────────────────────────

_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S | re.I)


def parse_verdict(reply: str) -> dict | None:
    """The first ```json {...}``` object of a reply that parses (trailing commas
    tolerated), with `result` lower-cased and a numeric-string `onset_t` read as a number."""
    for block in _JSON_FENCE.findall(reply):
        for candidate in (block, re.sub(r",(\s*[}\]])", r"\1", block)):
            try:
                verdict = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(verdict, dict):
                verdict["result"] = str(verdict.get("result", "")).strip().lower()
                if isinstance(onset := verdict.get("onset_t"), str):
                    try:
                        verdict["onset_t"] = float(onset.strip().rstrip("s"))
                    except ValueError:
                        pass
                return verdict
    return None
