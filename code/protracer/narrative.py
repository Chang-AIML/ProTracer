"""Rule-based signal narratives (paper Sec. III-B.2).

Each phase between consecutive boundaries is summarised per arm in up to three
lines - gripper, translation, wrist rotation - written only in physical
quantities (aperture, displacement, rotation), so the rules transfer unchanged
across embodiments:

    # Phase 4: [3.24, 3.92] s
    L gripper holds at 4.4 cm, then closes 4.1 cm @ 3.36s → 1.3 cm @ 3.92s
    L arm moves 1 cm forward, 1 cm up (path 2 cm)
    L arm wrist: net 5°, total 5°, stability 0.87
"""
from __future__ import annotations

import numpy as np

from .data import Episode
from .kinematics import matrix_to_quat

GRIPPER_SPEED = 2.0        # cm/s; faster aperture change reads as opening / closing
MIN_STAGE_S = 0.12         # shorter gripper stages are merged into the previous one
DRIFT_CM = 0.15            # a still gripper whose aperture changes this much "drifts"
STILL_WRIST_DEG = 5.0      # wrist line omitted when net and total rotation stay below this
MIN_AXIS_DEG = 3.0         # ... or when no world-axis component reaches this
AXES = (("forward", "backward"), ("left", "right"), ("up", "down"))   # +x, +y, +z


def signal_narrative(episode: Episode, boundaries: list[int]) -> str:
    """Narrate every phase delimited by `boundaries` (frame indices)."""
    t = episode.time
    # Phase limits are compared at 0.1 ms resolution, as in the paper runs, so a
    # frame lying on a boundary joins the phase(s) its exact timestamp falls in.
    limits = [round(float(x), 4) for x in (0.0, *t[boundaries], episode.duration)]
    blocks = []
    for k, (start, end) in enumerate(zip(limits, limits[1:]), start=1):
        w = (t >= start) & (t <= end)
        lines = [f"# Phase {k}: [{start:.2f}, {end:.2f}] s"]
        for name, arm in episode.arms.items():
            lines.append(f"{name} gripper {gripper_stages(t[w], arm.aperture[w], arm.aperture_velocity[w])}")
            lines.append(f"{name} arm {translation(arm.position[w])}")
            if wrist := rotation(arm.rotation[w]):
                lines.append(f"{name} arm wrist: {wrist}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def gripper_stages(t: np.ndarray, aperture: np.ndarray, velocity: np.ndarray) -> str:
    """'holds at 5.1 cm, then closes 4.1 cm @ 3.36s → 1.3 cm @ 3.92s, ...'"""
    kind = np.where(velocity > GRIPPER_SPEED, "opens", np.where(velocity < -GRIPPER_SPEED, "closes", "holds"))
    runs = [0, *(np.flatnonzero(kind[1:] != kind[:-1]) + 1), len(kind)]
    stages = []   # [kind, first frame, last frame]
    for i0, i1 in zip(runs, runs[1:]):
        if stages and (t[i1 - 1] - t[i0] < MIN_STAGE_S or stages[-1][0] == kind[i0]):
            stages[-1][2] = i1 - 1   # blips and repeats extend the previous stage
        else:
            stages.append([kind[i0], i0, i1 - 1])

    phrases = []
    for kind, i0, i1 in stages:
        t0, t1 = round(float(t[i0]), 2), round(float(t[i1]), 2)
        a0, a1 = round(float(aperture[i0]), 1), round(float(aperture[i1]), 1)
        if kind != "holds":
            phrases.append(f"{kind} {a0:.1f} cm @ {t0:.2f}s → {a1:.1f} cm @ {t1:.2f}s")
        elif abs(a1 - a0) >= DRIFT_CM:
            phrases.append(f"drifts {a0:.1f} → {a1:.1f} cm")
        else:
            phrases.append(f"holds at {a0:.1f} cm")
    return ", then ".join(phrases)


def translation(position: np.ndarray) -> str:
    """'moves 12 cm right, 6 cm backward (path 32 cm)' in the world frame."""
    delta = [int(round(v)) for v in (position[-1] - position[0]).tolist()]
    path = int(round(float(np.linalg.norm(np.diff(position, axis=0), axis=1).sum())))
    parts = sorted(((abs(v), pos if v > 0 else neg) for v, (pos, neg) in zip(delta, AXES) if abs(v) >= 1),
                   key=lambda p: -p[0])
    if not parts:
        return f"essentially stationary (path {path} cm)"
    return "moves " + ", ".join(f"{m} cm {d}" for m, d in parts) + f" (path {path} cm)"


def rotation(R: np.ndarray) -> str | None:
    """'net 31°, total 54°, stability 0.58', or None for a still wrist.

    net: angle of the start-to-end rotation; total: summed per-frame angles;
    stability: |sum of angle-weighted step axes| / total, 1 for a single-axis
    monotonic turn and near 0 for back-and-forth motion.
    """
    q = matrix_to_quat(R)
    _, net = _axis_angle(_relative(q[:1], q[-1:]))
    axes, steps = _axis_angle(_relative(q[:-1], q[1:]))
    net, total = float(net[0]), float(steps.sum())
    weighted = (axes * steps[:, None]).sum(axis=0)
    norm = float(np.linalg.norm(weighted))
    stability = norm / total if total > 1e-12 else 0.0
    dominant = weighted / norm if norm > 1e-12 else np.zeros(3)
    yaw_pitch_roll = (dominant[2] * net, -dominant[1] * net, -dominant[0] * net)
    if (net < STILL_WRIST_DEG and total < STILL_WRIST_DEG) or \
            all(abs(round(float(c), 2)) < MIN_AXIS_DEG for c in yaw_pitch_roll):
        return None
    return f"net {round(net, 2):.0f}°, total {round(total, 2):.0f}°, stability {round(stability, 3):.2f}"


def _relative(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    """Rotation taking q0 to q1 in the world frame, q1 * q0^-1."""
    x0, y0, z0, w0 = -q0[:, 0], -q0[:, 1], -q0[:, 2], q0[:, 3]
    x1, y1, z1, w1 = q1.T
    q = np.column_stack([w1 * x0 + x1 * w0 + y1 * z0 - z1 * y0,
                         w1 * y0 - x1 * z0 + y1 * w0 + z1 * x0,
                         w1 * z0 + x1 * y0 - y1 * x0 + z1 * w0,
                         w1 * w0 - x1 * x0 - y1 * y0 - z1 * z0])
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def _axis_angle(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit quaternions -> (unit axes, angles in degrees in [0, 180])."""
    q = np.where(q[:, 3:] < 0, -q, q)
    s = np.linalg.norm(q[:, :3], axis=1)
    axes = np.divide(q[:, :3], s[:, None], out=np.zeros((len(q), 3)), where=s[:, None] > 1e-12)
    return axes, np.degrees(2 * np.arctan2(s, q[:, 3]))
