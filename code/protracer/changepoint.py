"""Sign-CPD: action boundaries from discretised proprioception (paper Sec. III-B.1).

Each arm contributes two symbolic channels, sign(gripper velocity) and
1[end-effector moving]. PELT with the modal-Hamming cost segments the symbol
matrix; every cut is snapped to the nearest gripper standstill and cuts closer
than MIN_PHASE_S are dropped. Boundaries are then pruned greedily, always the
one whose removal raises the cost least: first by cost alone down to
max(B - 1, FIRST_PASS_CUTS), then down to the phase budget B, now preferring
removals that keep phases short.
"""
from __future__ import annotations

import numpy as np

from .data import Episode

PENALTY = 2.0           # PELT cost per change point
MIN_SEGMENT_S = 0.2     # PELT minimum segment length
SNAP_WINDOW_S = 0.2     # a cut moves to the frame of least gripper speed within +-0.2 s
MIN_PHASE_S = 0.5       # minimum spacing between cuts, and between a cut and either end
FIRST_PASS_CUTS = 14    # the first pruning pass stops at max(B - 1, 14) boundaries
MAX_GAP_S = 4.0         # the second pass prefers removals that keep every phase <= 4 s


def symbols(episode: Episode) -> np.ndarray:
    """(T, 2K) symbol matrix: per arm sign(gripper velocity) and 1[moving]."""
    columns = []
    for arm in episode.arms.values():
        columns += [np.sign(arm.aperture_velocity).astype(int), (arm.motion > 0).astype(int)]
    return np.column_stack(columns)


class ModalHammingCost:
    """C(s, e) = sum over channels of the frames in [s, e) that differ from the
    channel's modal symbol there (paper Eq. 1), in O(1) from prefix counts."""

    def __init__(self, X: np.ndarray):
        onehot = np.stack([X == v for v in np.unique(X)], axis=-1)            # (T, C, V)
        self.prefix = np.concatenate([np.zeros((1, *onehot.shape[1:]), int),
                                      onehot.cumsum(axis=0)])                 # (T+1, C, V)

    def __call__(self, s: int, e: int) -> int:
        counts = self.prefix[e] - self.prefix[s]
        return int(counts.sum() - counts.max(axis=1).sum())


def pelt(cost, n: int, penalty: float, min_size: int, jump: int = 2) -> list[int]:
    """Optimal penalised segmentation of [0, n) (Killick et al., 2012).

    Candidate cuts lie on a `jump` grid; returns the interior breakpoints.
    Ties are broken towards the earlier admissible start, as in `ruptures`.
    """
    if n < min_size:
        return []
    best = {0: (0.0, ())}   # end -> (penalised cost of [0, end), its breakpoints)
    starts: list[int] = []
    for end in [k for k in range(0, n, jump) if k >= min_size] + [n]:
        starts.append((end - min_size) // jump * jump)
        options = [(best[s][0] + cost(s, end) + penalty, s) for s in starts if s in best]
        total, start = min(options, key=lambda o: o[0])
        best[end] = (total, best[start][1] + (end,))
        starts = [s for c, s in options if c <= total + penalty]      # PELT pruning
    return list(best[n][1][:-1])


def sign_cpd(episode: Episode, max_phases: int) -> list[int]:
    """Frame indices of at most `max_phases - 1` action boundaries."""
    X, n = symbols(episode), episode.n_frames
    frames = lambda seconds: max(1, int(round(seconds / (1.0 / episode.fps))))
    cost = ModalHammingCost(X)

    cuts = pelt(cost, n, PENALTY, min_size=max(2, frames(MIN_SEGMENT_S)))

    # Snap every cut to the frame where the faster gripper moves least.
    grip_speed = np.max([np.abs(a.aperture_velocity) for a in episode.arms.values()], axis=0)
    snap = frames(SNAP_WINDOW_S)
    snapped = set()
    for cut in cuts:
        c = min(cut - 1, n - 1)
        window = range(max(0, c - snap), min(n, c + snap + 1))
        snapped.add(min(window, key=lambda k: grip_speed[k]))

    # Keep cuts at least MIN_PHASE_S apart from each other and from both ends.
    gap, kept = frames(MIN_PHASE_S), []
    for c in sorted(snapped):
        if gap <= c <= n - gap and (not kept or c - kept[-1] >= gap):
            kept.append(c)

    kept = _prune(kept, max(max_phases - 1, FIRST_PASS_CUTS), cost, n)
    return _prune(kept, max_phases - 1, cost, n, max_gap=frames(MAX_GAP_S))


def _prune(cuts: list[int], max_cuts: int, cost, n: int, max_gap: int | None = None) -> list[int]:
    """Greedily remove the boundary whose removal raises the total cost least,
    S(b_i) = C(b_i-1, b_i+1) - C(b_i-1, b_i) - C(b_i, b_i+1). With `max_gap`,
    removals that keep the merged phase within max_gap frames are preferred."""
    cuts = list(cuts)
    while len(cuts) > max_cuts:
        edges = [0, *cuts, n]
        scores = [cost(edges[i], edges[i + 2]) - cost(edges[i], edges[i + 1]) - cost(edges[i + 1], edges[i + 2])
                  for i in range(len(cuts))]
        pool = [i for i in range(len(cuts)) if max_gap is None or edges[i + 2] - edges[i] <= max_gap]
        cuts.pop(min(pool or range(len(cuts)), key=lambda i: scores[i]))
    return cuts


def continuous_cpd(episode: Episode, n_boundaries: int) -> list[int]:
    """Table III ablation: the same channels left continuous (z-scored gripper
    velocity and end-effector speed, 0.2 s smoothing), segmented by optimal
    L2 dynamic programming into as many boundaries as Sign-CPD found (at least
    one). Pure-Python O(n^2) per boundary: fine for FailTime-Short only."""
    import ruptures

    t, dt = episode.time, 1.0 / episode.fps
    window = max(1, round(MIN_SEGMENT_S / dt))
    smooth = lambda x: np.convolve(x, np.ones(window) / window, mode="same")
    channels = []
    for arm in episode.arms.values():
        channels.append(smooth(np.gradient(arm.aperture.astype(float), t)))
        channels.append(smooth(np.linalg.norm(np.gradient(arm.position, t, axis=0), axis=1)))
    X = np.stack(channels, axis=1)
    std = X.std(axis=0)
    X = (X - X.mean(axis=0)) / np.where(std < 1e-9, 1.0, std)
    min_size = max(2, int(round(MIN_SEGMENT_S / dt)))
    k = max(1, min(n_boundaries, len(X) // min_size - 1))
    cuts = ruptures.Dynp(model="l2", min_size=min_size, jump=1).fit(X).predict(n_bkps=k)[:-1]
    return [c for c in cuts if 0.2 <= t[min(c, len(t) - 1)] <= episode.duration - 0.2]
