"""ProTracer inference (paper Sec. III-B, Fig. 2).

Proprioception selects the keyframes (Sign-CPD) and narrates the motion
between them; a VLM captions the keyframes (round 1) and then diagnoses the
episode from its captions plus the narrative (round 2), optionally guided by
reflective experience.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import prompts
from .changepoint import continuous_cpd, sign_cpd
from .data import Episode, Task
from .llm import OpenRouter, message
from .narrative import signal_narrative


@dataclass(frozen=True)
class Robot:
    """How the paper processed one robot's recordings."""
    max_phases: int                              # phase budget B of Sign-CPD
    min_phases: int | None = None                # floor of B when it scales with duration
    seconds_per_keyframe: float | None = None    # scale B with episode length
    per_arm_frames: bool = False                 # poses in each arm's own base frame

    def phase_budget(self, duration: float) -> int:
        if self.seconds_per_keyframe is None:
            return self.max_phases
        return max(self.min_phases, min(self.max_phases, round(duration / self.seconds_per_keyframe) - 1))


ROBOTS = {
    "agilex": Robot(max_phases=8),    # FailTime-Short (ViFailback)
    "aloha": Robot(max_phases=14),    # FailTime-Long
    "so101": Robot(max_phases=39, min_phases=14, seconds_per_keyframe=2.2, per_arm_frames=True),   # FailTime-Long
}


@dataclass
class Evidence:
    """The deterministic, proprioception-derived inputs of the VLM rounds."""
    times: list[float]                # keyframe timestamps: start, action boundaries, end [s]
    frames: list[dict[str, bytes]]    # per keyframe: camera view -> JPEG
    narrative: str                    # signal narrative of the phases between keyframes
    per_arm_frames: bool              # the narrative's poses are per-arm, not in a shared frame


def ask(llm: OpenRouter, messages: list[dict], attempts: int = 3) -> tuple[dict | None, str, float]:
    """Query until the reply carries a parsable verdict -> (verdict, reply, cost)."""
    cost = 0.0
    for _ in range(attempts):
        reply = llm.chat(messages)
        cost += reply.cost
        if (verdict := prompts.parse_verdict(reply.text)) is not None:
            break
    return verdict, reply.text, cost


class ProTracer:
    """`keyframes`: "sign-cpd", or "continuous-cpd" for the Table III ablation."""

    def __init__(self, llm: OpenRouter, keyframes: str = "sign-cpd"):
        self.llm, self.keyframes = llm, keyframes

    def observe(self, episode: Episode) -> Evidence:
        robot = ROBOTS[episode.robot]
        cuts = sign_cpd(episode, robot.phase_budget(episode.duration))
        if self.keyframes == "continuous-cpd":
            cuts = continuous_cpd(episode, n_boundaries=len(cuts))
        times = [round(float(t), 4) for t in (0.0, *episode.time[cuts], episode.duration)]
        indices = [0, *cuts, episode.n_frames - 1]
        views = {view: episode.frames(view, indices) for view in episode.views}
        frames = [{view: jpegs[k] for view, jpegs in views.items()} for k in range(len(indices))]
        return Evidence(times, frames, signal_narrative(episode, cuts), robot.per_arm_frames)

    def caption(self, task: Task, evidence: Evidence) -> tuple[str, float]:
        reply = self.llm.chat(prompts.caption(task, evidence.times, evidence.frames))
        return reply.text, reply.cost

    def diagnosis_messages(self, task: Task, evidence: Evidence, captions: str, experience: str = "") -> list[dict]:
        return [*prompts.caption(task, evidence.times, evidence.frames), message("assistant", captions),
                prompts.diagnosis(task, evidence.narrative, evidence.per_arm_frames, experience)]

    def diagnose(self, task: Task, evidence: Evidence, captions: str, experience: str = "") -> tuple[dict | None, str, float]:
        return ask(self.llm, self.diagnosis_messages(task, evidence, captions, experience))

    def __call__(self, episode: Episode, task: Task, experience: str = "") -> dict:
        """Diagnose one episode; returns a JSON-serialisable record."""
        evidence = self.observe(episode)
        captions, caption_cost = self.caption(task, evidence)
        verdict, response, diagnosis_cost = self.diagnose(task, evidence, captions, experience)
        return {"sample": episode.name, "keyframes": evidence.times, "narrative": evidence.narrative,
                "captions": captions, "response": response, "verdict": verdict,
                "cost": round(caption_cost + diagnosis_cost, 6)}
