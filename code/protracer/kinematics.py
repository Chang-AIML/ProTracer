"""End-effector kinematics: rotation conversions and the Piper arms' FK.

`piper_fk` is only needed for FailTime-Short, whose ViFailback episodes store
joint angles (FailTime-Long already records end-effector poses). It runs
MuJoCo on the AgileX URDF, as for the paper, so stage-1 inputs are reproduced
bit for bit. Frame: URDF world, shared by both arms (+x forward, +y left, +z up).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

URDF = Path(__file__).parent / "assets" / "piper_bimanual.urdf"
_FINGER_TRAVEL = 0.04   # m, joint limit of one finger


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """(T, 4) quaternions, xyzw -> (T, 3, 3) rotation matrices."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True).clip(1e-12)
    x, y, z, w = q.T
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                     2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                     2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                    axis=-1).reshape(-1, 3, 3)


def matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """(T, 3, 3) rotation matrices -> (T, 4) quaternions, xyzw.

    Solved from the trace when it is positive (then w > 0), otherwise from the
    largest diagonal entry, which keeps the division well conditioned.
    """
    q = np.empty((len(R), 4))
    trace = np.trace(R, axis1=1, axis2=2)
    m = trace > 0
    s = 2.0 * np.sqrt(trace[m] + 1.0)
    q[m] = np.column_stack([(R[m, 2, 1] - R[m, 1, 2]) / s, (R[m, 0, 2] - R[m, 2, 0]) / s,
                            (R[m, 1, 0] - R[m, 0, 1]) / s, 0.25 * s])
    largest = np.argmax(np.diagonal(R, axis1=1, axis2=2), axis=1)
    for i in range(3):
        m = (trace <= 0) & (largest == i)
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + R[m, i, i] - R[m, j, j] - R[m, k, k])
        q[m, i] = 0.25 * s
        q[m, j] = (R[m, i, j] + R[m, j, i]) / s
        q[m, k] = (R[m, i, k] + R[m, k, i]) / s
        q[m, 3] = (R[m, k, j] - R[m, j, k]) / s
    return q


@lru_cache(maxsize=1)
def _model():
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(URDF))
    body = lambda name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    links = {arm: [body(f"{side}_link{i}") for i in (6, 7, 8)] for arm, side in (("L", "fl"), ("R", "fr"))}
    return model, links


def piper_fk(qpos: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Map ViFailback `qpos` (T, 14) to per-arm end-effector poses.

    `qpos` is laid out as [left joints 1-6, left gripper, right joints 1-6,
    right gripper], a gripper value being the finger separation in metres.
    The end effector is the midpoint of the two fingers, its orientation that
    of the wrist (link 6). Returns {"L"|"R": (position (T, 3) [cm], rotation (T, 3, 3))}.
    """
    import mujoco

    model, links = _model()
    data = mujoco.MjData(model)
    # MuJoCo joint order per arm: joints 1-6, then the two mirrored finger slides.
    joints = np.zeros((len(qpos), 16))
    for src, dst in ((0, 0), (7, 8)):
        joints[:, dst:dst + 6] = qpos[:, src:src + 6]
        finger = np.clip(qpos[:, src + 6] / 2.0, 0.0, _FINGER_TRAVEL)
        joints[:, dst + 6], joints[:, dst + 7] = finger, -finger

    poses = {arm: (np.empty((len(qpos), 3)), np.empty((len(qpos), 3, 3))) for arm in links}
    for t, q in enumerate(joints):
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        for arm, (wrist, finger_a, finger_b) in links.items():
            poses[arm][0][t] = 0.5 * (data.xpos[finger_a] + data.xpos[finger_b])
            poses[arm][1][t] = data.xmat[wrist].reshape(3, 3)
    return {arm: (position * 100.0, rotation) for arm, (position, rotation) in poses.items()}
