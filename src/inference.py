from __future__ import annotations
from dataclasses import dataclass

import numpy as np


def median_filter(probs, kernel=5):
    if probs.size == 0 or kernel <= 1:
        return probs
    half = kernel // 2
    padded = np.pad(probs, (half, half), mode="edge")
    out = np.empty_like(probs)
    for i in range(probs.size):
        out[i] = float(np.median(padded[i:i + kernel]))
    return out


def consecutive_above(probs, threshold):
    above = (probs >= threshold).astype(np.int32)
    if above.size == 0:
        return 0
    best = cur = 0
    for v in above:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


@dataclass
class Verdict:
    decision: str
    smoothed_max: float
    smoothed_median: float
    longest_run: int
    n_above_threshold: int
    n_windows: int
    threshold: float
    consecutive_required: int


def aggregate_verdict(probs, threshold=0.5, median_kernel=5, consecutive_required=3):
    if probs.size == 0:
        return Verdict("uncertain", 0.0, 0.0, 0, 0, 0, threshold, consecutive_required)
    smoothed = median_filter(probs, kernel=median_kernel)
    longest = consecutive_above(smoothed, threshold)
    n_above = int((smoothed >= threshold).sum())
    if longest >= consecutive_required:
        decision = "drone"
    elif smoothed.max() < 0.30:
        decision = "no_drone"
    else:
        decision = "uncertain"
    return Verdict(
        decision=decision,
        smoothed_max=float(smoothed.max()),
        smoothed_median=float(np.median(smoothed)),
        longest_run=longest,
        n_above_threshold=n_above,
        n_windows=int(probs.size),
        threshold=threshold,
        consecutive_required=consecutive_required,
    )
