from __future__ import annotations
from dataclasses import dataclass

import numpy as np


SOUND_SPEED = 343.0


def gcc_phat(sig, refsig, fs, max_tau=None, interp=16):
    n = sig.shape[0] + refsig.shape[0]
    SIG = np.fft.rfft(sig, n=n)
    REF = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REF)
    R /= np.maximum(np.abs(R), 1e-12)
    cc = np.fft.irfft(R, n=interp * n)
    max_shift = int(interp * n / 2)
    if max_tau is not None:
        max_shift = min(int(interp * fs * max_tau), max_shift)
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))
    shift = int(np.argmax(np.abs(cc))) - max_shift
    return shift / float(interp * fs)


def estimate_tdoas(signals, fs, max_tau=None):
    M = signals.shape[0]
    tdoas = np.zeros(M - 1, dtype=np.float64)
    for i in range(1, M):
        tdoas[i - 1] = gcc_phat(signals[i], signals[0], fs=fs, max_tau=max_tau)
    return tdoas


@dataclass
class Localization:
    position: np.ndarray
    tdoas: np.ndarray
    residual: float


def localize(signals, mic_positions, fs, sound_speed=SOUND_SPEED, max_tau=None):
    M, dim = mic_positions.shape
    assert signals.shape[0] == M, "signal count must match mic count"

    tdoas = estimate_tdoas(signals, fs=fs, max_tau=max_tau)
    d = sound_speed * tdoas

    m0 = mic_positions[0]
    A = np.zeros((M - 1, dim + 1))
    b = np.zeros(M - 1)
    for i in range(1, M):
        mi = mic_positions[i]
        A[i - 1, :dim] = 2 * (mi - m0)
        A[i - 1, dim] = 2 * d[i - 1]
        b[i - 1] = np.dot(mi, mi) - np.dot(m0, m0) - d[i - 1] ** 2

    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    position = sol[:dim]
    residual = float(np.linalg.norm(A @ sol - b))
    return Localization(position=position, tdoas=tdoas, residual=residual)


def _self_test():
    fs = 16_000
    duration = 2.0
    rng = np.random.RandomState(0)
    src = rng.randn(int(duration * fs)).astype(np.float32)

    mics = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 2.0, 0.0],
        [0.0, 2.0, 0.0],
        [1.0, 1.0, 1.5],
    ])
    true_position = np.array([5.0, 7.0, 3.0])

    sigs = []
    for m in mics:
        dist = float(np.linalg.norm(true_position - m))
        delay = int(round(dist / SOUND_SPEED * fs))
        sig = np.concatenate([np.zeros(delay, dtype=np.float32), src])[:src.size]
        sigs.append(sig / max(dist, 1e-3))
    signals = np.stack(sigs)

    loc = localize(signals, mics, fs=fs, max_tau=0.05)
    err = float(np.linalg.norm(loc.position - true_position))
    print(f"true     : {true_position}")
    print(f"estimated: {loc.position.round(3)}")
    print(f"tdoa (ms): {(loc.tdoas * 1e3).round(3)}")
    print(f"error    : {err:.3f} m  (residual {loc.residual:.3f})")


if __name__ == "__main__":
    _self_test()
