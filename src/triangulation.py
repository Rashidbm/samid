"""
Time-Difference-of-Arrival (TDoA) source localization.

Standard pipeline:
  1. Estimate per-pair TDoA via GCC-PHAT (Generalized Cross-Correlation with
     Phase Transform) — robust to reverberation.
  2. Convert TDoAs into a 2D or 3D source position by solving the hyperbola
     intersection (linearised least-squares, "Chan's method" style).

Inputs:
  - signals: np.ndarray of shape (M, N) for M synchronised mics, N samples
  - mic_positions: np.ndarray of shape (M, 3) in metres (z=0 if planar)
  - sample_rate: int
  - sound_speed: float, default 343 m/s

Outputs:
  - estimated source position (3,)
  - per-pair TDoA estimates in seconds
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np


SOUND_SPEED = 343.0  # m/s, dry air ~20 °C


# ---------------------------- GCC-PHAT -------------------------------------- #


def gcc_phat(
    sig: np.ndarray,
    refsig: np.ndarray,
    fs: int,
    max_tau: float | None = None,
    interp: int = 16,
) -> tuple[float, np.ndarray]:
    """
    Compute the cross-correlation lag (in seconds) between sig and refsig
    using GCC-PHAT.

    Returns (tau, cross_correlation_curve).
    """
    n = sig.shape[0] + refsig.shape[0]
    SIG = np.fft.rfft(sig, n=n)
    REF = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REF)
    R /= np.maximum(np.abs(R), 1e-12)
    cc = np.fft.irfft(R, n=interp * n)
    max_shift = int(interp * n / 2)
    if max_tau is not None:
        max_shift = min(int(interp * fs * max_tau), max_shift)
    cc = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))
    shift = np.argmax(np.abs(cc)) - max_shift
    tau = shift / float(interp * fs)
    return tau, cc


# ---------------------------- localisation ---------------------------------- #


@dataclass
class Localisation:
    position: np.ndarray            # (3,) estimated source position
    tdoas: np.ndarray               # (M-1,) tdoa(mic_i, mic_0) in seconds
    residual: float                 # least-squares residual norm


def estimate_tdoas(
    signals: np.ndarray,
    fs: int,
    max_tau: float | None = None,
) -> np.ndarray:
    """
    Compute TDoA of each mic relative to mic 0.
    """
    M = signals.shape[0]
    tdoas = np.zeros(M - 1, dtype=np.float64)
    for i in range(1, M):
        tau, _ = gcc_phat(signals[i], signals[0], fs=fs, max_tau=max_tau)
        tdoas[i - 1] = tau
    return tdoas


def localise(
    signals: np.ndarray,
    mic_positions: np.ndarray,
    fs: int,
    sound_speed: float = SOUND_SPEED,
    max_tau: float | None = None,
) -> Localisation:
    """
    Closed-form linear least-squares TDoA localisation (no iteration).

    Reference mic = index 0. We solve for source position p such that
        |p - m_i| - |p - m_0| = c * tau_{i,0}
    Expanding both sides squared yields a linear system in (p, |p - m_0|).
    """
    M, dim = mic_positions.shape
    assert signals.shape[0] == M, "signal/mic count mismatch"
    tdoas = estimate_tdoas(signals, fs=fs, max_tau=max_tau)
    d = sound_speed * tdoas                        # (M-1,)

    m0 = mic_positions[0]
    A = np.zeros((M - 1, dim + 1))
    b = np.zeros(M - 1)
    for i in range(1, M):
        mi = mic_positions[i]
        A[i - 1, :dim] = 2 * (mi - m0)
        A[i - 1, dim] = 2 * d[i - 1]
        b[i - 1] = (
            np.dot(mi, mi) - np.dot(m0, m0) - d[i - 1] ** 2
        )

    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    p_rel = sol[:dim]                              # source minus m0 frame
    position = p_rel                               # already absolute since A used (mi-m0) and m0 dot products
    residual = float(np.linalg.norm(A @ sol - b))
    return Localisation(position=position, tdoas=tdoas, residual=residual)


# ---------------------------- self-test ------------------------------------- #


def _self_test() -> None:
    """Synthetic 4-mic square array, source at known position, verify recovery."""
    fs = 16_000
    c = SOUND_SPEED
    duration = 2.0
    t = np.arange(int(duration * fs)) / fs
    src_signal = np.random.RandomState(0).randn(t.size).astype(np.float32)

    # 5 mics, NOT coplanar — required to recover z without ambiguity
    mics = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 2.0, 0.0],
            [0.0, 2.0, 0.0],
            [1.0, 1.0, 1.5],   # elevated 5th mic breaks the planar ambiguity
        ]
    )
    true_source = np.array([5.0, 7.0, 3.0])

    sigs = []
    for m in mics:
        dist = np.linalg.norm(true_source - m)
        delay_samples = int(round(dist / c * fs))
        sig = np.concatenate(
            [np.zeros(delay_samples, dtype=np.float32), src_signal]
        )[: t.size]
        sigs.append(sig / max(dist, 1e-3))   # 1/r attenuation
    signals = np.stack(sigs)

    loc = localise(signals, mics, fs=fs, max_tau=0.05)
    err = np.linalg.norm(loc.position - true_source)
    print(f"true source  : {true_source}")
    print(f"estimated    : {loc.position}")
    print(f"tdoas (ms)   : {(loc.tdoas * 1e3).round(3)}")
    print(f"position err : {err:.3f} m   residual: {loc.residual:.3f}")


if __name__ == "__main__":
    _self_test()
