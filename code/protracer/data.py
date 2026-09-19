"""FailTime data: episodes (proprioception + camera frames), annotations, task specs.

Both recording formats of the benchmark are read into the same `Episode`:

* ViFailback HDF5 (FailTime-Short): joint angles `observations/qpos` and one
  JPEG per frame in `observations/images/<view>`; end-effector poses come from
  `kinematics.piper_fk`.
* LeRobot-style HDF5 (FailTime-Long, ALOHA and SO-101): end-effector poses in
  `observation.state_eef` and one mp4 per camera in
  `observation.images.<view>/mp4_bytes`.

A split is one annotation file next to a `tasks/` directory. Every record
names its episode file (`hdf5`, relative to the annotation file) and the
`robot` that recorded it: "agilex" (ViFailback), "aloha" or "so101".
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Callable

import h5py
import numpy as np
from PIL import Image

from .kinematics import piper_fk, quat_to_matrix

VIEWS = {"cam_left_wrist": "L", "cam_high": "H", "cam_right_wrist": "R"}   # prompt order -> label
_VIEW_ALIASES = {"cam_low": "cam_high", "head": "cam_high",
                 "left_wrist": "cam_left_wrist", "right_wrist": "cam_right_wrist"}
_GRIPPER_TO_CM = 0.08   # FailTime-Long grippers report raw joint units; map them to a cm-like aperture

FAILURE_TYPES = ("wrong_sequence", "wrong_object", "wrong_gripper", "wrong_placement",
                 "closing_error", "open_error", "translation_error", "orientation_error",
                 "avoidable_intervention", "unavoidable_intervention")
_TYPE_ALIASES = {"translation": "translation_error", "orientation": "orientation_error",
                 "rotation": "orientation_error", "rotation_error": "orientation_error",
                 "gripper_close": "closing_error", "gripper_open": "open_error",
                 "open": "open_error", "human_intervention": "avoidable_intervention",
                 "human_intervention_unavoidable": "unavoidable_intervention"}


def canonical_type(name: str | None) -> str:
    """Map a failure-type label from any FailTime release onto the taxonomy name."""
    return _TYPE_ALIASES.get(name or "", name or "")


# ── episodes ───────────────────────────────────────────────────────────────

@dataclass
class Arm:
    """Proprioception of one arm, sampled once per video frame."""
    position: np.ndarray   # (T, 3) end-effector position [cm]
    rotation: np.ndarray   # (T, 3, 3) end-effector orientation
    aperture: np.ndarray   # (T,) gripper aperture [cm]
    fps: float

    @cached_property
    def aperture_velocity(self) -> np.ndarray:
        """Backward-difference gripper velocity [cm/s], 0 at the first frame."""
        v = np.zeros_like(self.aperture)
        v[1:] = np.diff(self.aperture) / (1.0 / self.fps)
        return v

    @cached_property
    def motion(self) -> np.ndarray:
        """Denoised end-effector motion: squared per-frame displacement [cm^2]
        minus its 20th percentile, clipped at 0 so that jitter reads as rest."""
        step = np.zeros(len(self.position))
        step[1:] = (np.diff(self.position, axis=0) ** 2).sum(axis=1)
        return np.maximum(0.0, step - np.percentile(step, 20))


@dataclass
class Episode:
    """One rollout: per-arm proprioception plus lazily decoded camera frames."""
    name: str
    robot: str
    fps: float
    arms: dict[str, Arm]
    views: list[str]                                        # canonical views, prompt order
    read_frames: Callable[[str, list[int]], list[bytes]]    # (view, frame indices) -> JPEGs

    @property
    def n_frames(self) -> int:
        return len(next(iter(self.arms.values())).aperture)

    @property
    def time(self) -> np.ndarray:
        return np.arange(self.n_frames) * (1.0 / self.fps)

    @property
    def duration(self) -> float:
        return self.n_frames * (1.0 / self.fps)

    def frames(self, view: str, indices: list[int]) -> list[bytes]:
        return self.read_frames(view, indices)


def load_episode(path: str | Path, robot: str, name: str | None = None) -> Episode:
    path = Path(path)
    with h5py.File(path, "r") as f:
        if "observations/qpos" in f:
            return _read_vifailback(path, f, name or path.stem, robot)
        if "observation.state_eef" in f:
            return _read_state_eef(path, f, name or path.stem, robot)
    raise ValueError(f"{path}: neither a ViFailback nor a LeRobot-style episode")


def _read_vifailback(path: Path, f: h5py.File, name: str, robot: str) -> Episode:
    qpos, fps = f["observations/qpos"][:], 25.0
    poses = piper_fk(qpos)
    arms = {arm: Arm(*poses[arm], qpos[:, col] * 100.0, fps) for arm, col in (("L", 6), ("R", 13))}
    views = [v for v in VIEWS if f"observations/images/{v}" in f]

    def read_frames(view: str, indices: list[int]) -> list[bytes]:
        with h5py.File(path, "r") as h:
            return [_bgr_jpeg_to_rgb(bytes(h[f"observations/images/{view}"][i])) for i in indices]

    return Episode(name, robot, fps, arms, views, read_frames)


def _read_state_eef(path: Path, f: h5py.File, name: str, robot: str) -> Episode:
    eef = np.asarray(f["observation.state_eef"][:], dtype=np.float64)   # per arm: xyz [m], quat xyzw, gripper
    fps = float(f.attrs.get("fps", 25))
    arms = {arm: Arm(eef[:, s:s + 3] * 100.0, quat_to_matrix(eef[:, s + 3:s + 7]),
                     (eef[:, s + 7] * _GRIPPER_TO_CM).astype(np.float32), fps)
            for arm, s in (("L", 0), ("R", 8))}
    sources = {}   # canonical view -> camera key; a key already named canonically wins
    for key in sorted(k.removeprefix("observation.images.") for k in f if k.startswith("observation.images.")):
        view = _VIEW_ALIASES.get(key, key)
        if view not in sources or key == view:
            sources[view] = key
    views = [v for v in VIEWS if v in sources]

    def read_frames(view: str, indices: list[int]) -> list[bytes]:
        with h5py.File(path, "r") as h:
            group = h[f"observation.images.{sources[view]}"]
            video, last = group["mp4_bytes"][:].tobytes(), len(group["timestamp"]) - 1
        return _decode_mp4(video, [min(i, last) for i in indices])

    return Episode(name, robot, fps, arms, views, read_frames)


def _bgr_jpeg_to_rgb(jpeg: bytes) -> bytes:
    """ViFailback JPEGs hold BGR pixels; swap the channels and re-encode."""
    with Image.open(io.BytesIO(jpeg)) as im:
        r, g, b = im.convert("RGB").split()
        out = io.BytesIO()
        Image.merge("RGB", (b, g, r)).save(out, format="JPEG", quality=92)
    return out.getvalue()


def _decode_mp4(video: bytes, indices: list[int], per_call: int = 64) -> list[bytes]:
    """Frame-exact JPEG extraction of the given frame numbers with ffmpeg.
    A select expression may hold at most 100 terms, hence `per_call`."""
    wanted, jpegs = sorted(set(indices)), {}
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "video.mp4").write_bytes(video)
        for start in range(0, len(wanted), per_call):
            chunk = wanted[start:start + per_call]
            select = "+".join(f"eq(n\\,{i})" for i in chunk)
            subprocess.run(["ffmpeg", "-v", "error", "-i", f"{tmp}/video.mp4", "-vf", f"select='{select}'",
                            "-fps_mode", "passthrough", "-frames:v", str(len(chunk)), "-q:v", "2",
                            f"{tmp}/{start}_%03d.jpg"], check=True)
            jpegs.update({i: (Path(tmp) / f"{start}_{k:03d}.jpg").read_bytes() for k, i in enumerate(chunk, start=1)})
    return [jpegs[i] for i in indices]


def write_atomic(path: Path, content: str) -> None:
    """Write via a temporary file, so an interrupted run never leaves half a file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(content)
    os.replace(tmp, path)


# ── annotations and task specs ─────────────────────────────────────────────

@dataclass
class Sample:
    """One annotated episode of a split; `label` is the released record."""
    name: str
    task: str
    robot: str
    path: Path
    label: dict

    @property
    def failed(self) -> bool:
        return self.label["verdict"] == "fail"

    @property
    def onset(self) -> float | None:
        return self.label.get("onset")

    @property
    def types(self) -> set[str]:
        types = self.label.get("type") or []
        return {canonical_type(t) for t in ([types] if isinstance(types, str) else types)}


def load_samples(annotations: str | Path) -> list[Sample]:
    root = Path(annotations).parent
    return [Sample(r["sample"], r["task_id"], r["robot"], root / r["hdf5"], r)
            for r in json.loads(Path(annotations).read_text())]


@dataclass
class Task:
    """Task spec: instruction, expected substages, and the grounding vocabulary."""
    name: str
    description: str
    substages: list[str]
    context: str   # the spec without substages, shown to the VLM as reference vocabulary


def load_task(tasks_dir: str | Path, name: str) -> Task:
    root = ET.fromstring((Path(tasks_dir) / f"{name}.xml").read_text())
    substages = root.find("substages")
    steps = [s.text.strip() for s in substages.findall("substage") if s.text] if substages is not None else []
    for child in list(root):
        if child.tag == "substages" or child.tag.endswith("_zh"):   # substages come later, in round 2
            root.remove(child)
    ET.indent(root, space="  ")
    return Task(name, root.findtext("description", "").strip(), steps, ET.tostring(root, encoding="unicode"))
