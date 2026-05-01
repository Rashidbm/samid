"""
Spoofing / decoy detection via spatial coherence.

The novel claim: a real flying drone is a MOVING source. A speaker on the
ground playing drone audio is a STATIONARY source. We can tell them apart by
analysing how Time-Difference-of-Arrival (TDoA) across microphone pairs
evolves over time.

  - Real flying drone: TDoAs change SMOOTHLY over time as the drone moves.
    The source position trajectory is consistent with motion (small
    frame-to-frame deltas, but cumulative drift).
  - Stationary speaker: TDoAs are CONSTANT over time. Estimated source
    position has near-zero variance.
  - Random noise / multipath: TDoAs jump randomly. Position estimate is
    inconsistent (very high variance, no smooth trajectory).

We score:
  - position_std: standard deviation of estimated source positions over a
    rolling window. Tiny -> stationary. Large -> noisy/random.
  - smoothness: average distance between consecutive position estimates.
    Stationary or moving smoothly: small. Random: large.
  - tdoa_drift_rate: average change in TDoA per unit time. ~0 = stationary.

Decision (calibrated thresholds):
  - position_std < STATIONARY_STD  AND  smoothness < STATIONARY_SMOOTH ->
        DECOY (stationary speaker)
  - smoothness > NOISY_SMOOTH ->
        UNRELIABLE (probably noise or multipath; do not commit)
  - else -> REAL DRONE

This module has essentially NO published prior art for drone audio
specifically. The principle (moving source vs stationary source acoustic
verification) is well-established in audio forensics and ASVspoof anti-
spoofing research, but applying it as a counter-decoy mechanism in an
acoustic anti-UAV pipeline is novel.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from src.triangulation import estimate_tdoas, localise


# --------------------------- config ----------------------------------------- #


@dataclass
class DecoyConfig:
    window_seconds: float = 1.0           # length of each analysis frame
    stride_seconds: float = 0.25          # how far we slide the frame
    history_seconds: float = 4.0          # rolling buffer to compute stats
    sample_rate: int = 16_000

    # Thresholds (calibrate on data; these are reasonable defaults)
    stationary_position_std: float = 0.5  # metres; below this = stationary
    stationary_smoothness: float = 0.5    # m per frame; below this = stationary
    noisy_smoothness: float = 8.0         # m per frame; above this = unreliable
    min_frames_for_decision: int = 4      # need this many frames before deciding


# --------------------------- core ------------------------------------------- #


@dataclass
class DecoyVerdict:
    label: str                            # "real_drone" / "decoy" / "unreliable" / "insufficient"
    position_std: float
    smoothness: float
    tdoa_drift_rate: float
    n_frames: int
    positions: list                       # per-frame estimated positions for plotting
    explanation: str


def _frame_indices(total_samples: int, win_samples: int, stride_samples: int):
    out = []
    start = 0
    while start + win_samples <= total_samples:
        out.append((start, start + win_samples))
        start += stride_samples
    return out


def analyse(
    multichannel_signal: np.ndarray,      # shape (M, N), M mics, N samples
    mic_positions: np.ndarray,            # shape (M, 3) in metres
    cfg: DecoyConfig | None = None,
) -> DecoyVerdict:
    cfg = cfg or DecoyConfig()
    M, N = multichannel_signal.shape
    win = int(cfg.window_seconds * cfg.sample_rate)
    stride = int(cfg.stride_seconds * cfg.sample_rate)

    frames = _frame_indices(N, win, stride)
    positions = []
    tdoa_history = []

    for start, end in frames:
        chunk = multichannel_signal[:, start:end]
        try:
            loc = localise(chunk, mic_positions, fs=cfg.sample_rate, max_tau=0.05)
        except Exception:
            continue
        positions.append(loc.position)
        tdoa_history.append(loc.tdoas)

    if len(positions) < cfg.min_frames_for_decision:
        return DecoyVerdict(
            label="insufficient",
            position_std=0.0, smoothness=0.0, tdoa_drift_rate=0.0,
            n_frames=len(positions), positions=positions,
            explanation=f"need {cfg.min_frames_for_decision}+ frames, got {len(positions)}",
        )

    pos_arr = np.asarray(positions)                                  # (F, 3)
    tdoa_arr = np.asarray(tdoa_history)                              # (F, M-1)

    # Stats — use MEDIAN frame-to-frame distance for robustness to localization noise
    position_std = float(np.linalg.norm(pos_arr.std(axis=0)))
    deltas = np.linalg.norm(np.diff(pos_arr, axis=0), axis=1)        # (F-1,)
    smoothness = float(np.median(deltas)) if deltas.size else 0.0
    tdoa_diffs = np.diff(tdoa_arr, axis=0)                           # (F-1, M-1)
    tdoa_drift_rate = float(np.linalg.norm(tdoa_diffs, axis=1).mean()) if tdoa_diffs.size else 0.0

    # Trajectory linearity bonus: if the positions lie on a 1-D line in 3D
    # space (consistent with a flying drone moving in a line), pull smoothness
    # toward "smooth motion" even if individual frames are jittery.
    if pos_arr.shape[0] >= 4:
        centered = pos_arr - pos_arr.mean(axis=0)
        # Singular values of the centered trajectory: large 1st, small 2nd/3rd = linear
        try:
            sv = np.linalg.svd(centered, compute_uv=False)
            linearity = float(sv[0] / (sv.sum() + 1e-8))   # 0..1, 1 = perfectly linear
        except Exception:
            linearity = 0.0
    else:
        linearity = 0.0

    # Decision
    stationary = (
        position_std < cfg.stationary_position_std
        and smoothness < cfg.stationary_smoothness
    )
    # Highly linear trajectory + reasonable smoothness = real drone (overrides noise)
    is_linear_motion = linearity > 0.85
    noisy = smoothness > cfg.noisy_smoothness and not is_linear_motion

    if stationary:
        label = "decoy"
        explanation = (
            f"position barely moves (std={position_std:.2f}m) — "
            f"stationary source consistent with speaker decoy"
        )
    elif noisy:
        label = "unreliable"
        explanation = (
            f"position jumps wildly (smoothness={smoothness:.2f}m/frame) — "
            f"likely noise or multipath, defer decision"
        )
    else:
        label = "real_drone"
        explanation = (
            f"motion detected: std={position_std:.2f}m, "
            f"frame-to-frame median={smoothness:.2f}m, "
            f"linearity={linearity:.2f} — consistent with flying drone"
        )

    return DecoyVerdict(
        label=label,
        position_std=position_std,
        smoothness=smoothness,
        tdoa_drift_rate=tdoa_drift_rate,
        n_frames=len(positions),
        positions=positions,
        explanation=explanation,
    )


# --------------------------- self-test -------------------------------------- #


def _simulate_source(
    src_position_fn,                # callable: t (sec) -> (x, y, z)
    mic_positions: np.ndarray,
    duration: float,
    fs: int,
    seed: int = 0,
) -> np.ndarray:
    """Synthesise multi-channel signals from a (possibly moving) source."""
    from src.triangulation import SOUND_SPEED
    rng = np.random.RandomState(seed)
    t = np.arange(int(duration * fs)) / fs
    src_signal = rng.randn(t.size).astype(np.float32)

    M = mic_positions.shape[0]
    out = np.zeros((M, t.size), dtype=np.float32)
    chunk_len = max(1, fs // 10)  # update source position every 0.1s

    for start in range(0, t.size, chunk_len):
        end = min(start + chunk_len, t.size)
        t_mid = (start + end) * 0.5 / fs
        src_pos = np.asarray(src_position_fn(t_mid))
        for i, m in enumerate(mic_positions):
            dist = float(np.linalg.norm(src_pos - m))
            delay = int(round(dist / SOUND_SPEED * fs))
            seg = src_signal[max(0, start - delay): max(0, end - delay)]
            target_len = end - start
            if seg.size < target_len:
                seg = np.concatenate([np.zeros(target_len - seg.size, dtype=np.float32), seg])
            out[i, start:end] += seg[:target_len] / max(dist, 1e-3)
    return out


def _self_test() -> None:
    fs = 16_000
    mics = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 2.0, 0.0],
        [0.0, 2.0, 0.0],
        [1.0, 1.0, 1.5],
    ])

    # 1. Stationary speaker (decoy) — same position the whole time
    sig_decoy = _simulate_source(
        lambda t: (8.0, 7.0, 3.0),  # constant
        mics, duration=4.0, fs=fs, seed=1,
    )
    v = analyse(sig_decoy, mics)
    print(f"[STATIONARY]   {v.label:14s}  std={v.position_std:.2f}  smooth={v.smoothness:.2f}")
    print(f"               {v.explanation}")

    # 2. Moving drone — straight-line trajectory at 5 m/s
    sig_drone = _simulate_source(
        lambda t: (8.0 + 5.0 * t, 7.0, 3.0),
        mics, duration=4.0, fs=fs, seed=2,
    )
    v = analyse(sig_drone, mics)
    print(f"[MOVING DRONE] {v.label:14s}  std={v.position_std:.2f}  smooth={v.smoothness:.2f}")
    print(f"               {v.explanation}")


if __name__ == "__main__":
    _self_test()
