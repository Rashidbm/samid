from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from src.triangulation import localize


@dataclass
class DecoyConfig:
    window_seconds: float = 1.0
    stride_seconds: float = 0.25
    sample_rate: int = 16_000
    stationary_position_std: float = 0.5
    stationary_smoothness: float = 0.5
    noisy_smoothness: float = 8.0
    min_frames: int = 4


@dataclass
class DecoyVerdict:
    label: str
    position_std: float
    smoothness: float
    linearity: float
    n_frames: int
    positions: list
    explanation: str


def _frame_indices(total, win, stride):
    out = []
    s = 0
    while s + win <= total:
        out.append((s, s + win))
        s += stride
    return out


def analyse(multichannel, mic_positions, cfg=None):
    cfg = cfg or DecoyConfig()
    M, N = multichannel.shape
    win = int(cfg.window_seconds * cfg.sample_rate)
    stride = int(cfg.stride_seconds * cfg.sample_rate)

    positions = []
    for s, e in _frame_indices(N, win, stride):
        try:
            loc = localize(multichannel[:, s:e], mic_positions,
                           fs=cfg.sample_rate, max_tau=0.05)
            positions.append(loc.position)
        except Exception:
            continue

    if len(positions) < cfg.min_frames:
        return DecoyVerdict("insufficient", 0.0, 0.0, 0.0, len(positions), positions,
                            f"need >={cfg.min_frames} frames, got {len(positions)}")

    pos = np.asarray(positions)
    position_std = float(np.linalg.norm(pos.std(axis=0)))
    deltas = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    smoothness = float(np.median(deltas)) if deltas.size else 0.0

    linearity = 0.0
    if pos.shape[0] >= 4:
        centered = pos - pos.mean(axis=0)
        try:
            sv = np.linalg.svd(centered, compute_uv=False)
            linearity = float(sv[0] / (sv.sum() + 1e-8))
        except Exception:
            pass

    is_linear = linearity > 0.85
    stationary = (position_std < cfg.stationary_position_std
                  and smoothness < cfg.stationary_smoothness)
    noisy = smoothness > cfg.noisy_smoothness and not is_linear

    if stationary:
        label = "decoy"
        explanation = (f"position barely moves (std={position_std:.2f} m) — "
                       f"consistent with stationary speaker")
    elif noisy:
        label = "unreliable"
        explanation = (f"position jumps wildly (smoothness={smoothness:.2f} m/frame) — "
                       f"multipath or noise")
    else:
        label = "real_drone"
        explanation = (f"smooth motion: std={position_std:.2f} m, "
                       f"linearity={linearity:.2f}")

    return DecoyVerdict(label=label, position_std=position_std, smoothness=smoothness,
                        linearity=linearity, n_frames=len(positions),
                        positions=positions, explanation=explanation)
