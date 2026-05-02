"""
Domain-invariance augmentation pipeline for the drone classifier.

Designed in response to a methodological review that flagged shortcut learning
("Clever Hans") on geronimobasso: positive/negative classes are cleanly
segregated by their source datasets, so the model maps recording style instead
of drone presence.

Critical principle from the review: AUGMENTATIONS THAT EXIST IN BOTH CLASSES
MUST BE APPLIED SYMMETRICALLY. If only drone clips get codec round-trip, the
model just learns "has compression artifacts = drone" — a new shortcut.

Asymmetry is allowed only when it MATCHES the deployment distribution:
  - urban-noise overlay onto drone clips (deployment puts drones in urban
    environments; no-drone clips already come from urban environments)

Augmentations implemented (all randomized, applied per-sample at training time):

  Symmetric (applied to both classes):
    - Codec round-trip (mp3/opus encode-decode, lossy)
    - Random EQ filtering (low-pass / high-pass / shelving)
    - Room impulse response convolution (random RIR from a small pool)
    - FilterAugment — random step-like filter on log-mel spectrogram
    - Spectrogram Patchout — drop random rectangular regions of the
      spectrogram (mel x time)
    - SpecAugment — frequency and time masking
    - Mixup at the audio waveform level

  Asymmetric (drone class only):
    - Additive urban-environment noise at random SNR
"""

from __future__ import annotations
import io
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import butter, fftconvolve, filtfilt


# ----------------------------- waveform-level ------------------------------- #


def add_noise(arr: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Add `noise` to `arr` at the requested SNR. Trims/loops noise to length."""
    if noise.size < arr.size:
        reps = int(np.ceil(arr.size / max(1, noise.size)))
        noise = np.tile(noise, reps)
    noise = noise[: arr.size].astype(np.float32, copy=False)
    sig_pow = float(np.mean(arr ** 2)) + 1e-12
    noise_pow = float(np.mean(noise ** 2)) + 1e-12
    target_noise_pow = sig_pow / (10 ** (snr_db / 10))
    scale = float(np.sqrt(target_noise_pow / noise_pow))
    return arr + scale * noise


def random_eq(arr: np.ndarray, sr: int, rng: random.Random) -> np.ndarray:
    """Apply one of {high-pass, low-pass, shelf} with random cutoff."""
    kind = rng.choice(["highpass", "lowpass", "shelf_low", "shelf_high"])
    if kind == "highpass":
        cutoff = rng.uniform(60.0, 250.0)
        b, a = butter(4, cutoff / (sr / 2), btype="high")
    elif kind == "lowpass":
        cutoff = rng.uniform(2_000.0, 6_500.0)
        b, a = butter(4, cutoff / (sr / 2), btype="low")
    elif kind == "shelf_low":
        cutoff = rng.uniform(150.0, 400.0)
        b, a = butter(2, cutoff / (sr / 2), btype="low")
    else:
        cutoff = rng.uniform(3_000.0, 7_000.0)
        b, a = butter(2, cutoff / (sr / 2), btype="high")
    return filtfilt(b, a, arr).astype(np.float32, copy=False)


def codec_roundtrip(arr: np.ndarray, sr: int, rng: random.Random) -> np.ndarray:
    """Encode to ogg/vorbis at random quality then decode. Lossy round-trip."""
    quality = rng.uniform(-0.1, 0.5)  # vorbis quality
    try:
        buf = io.BytesIO()
        sf.write(buf, arr.astype(np.float32, copy=False), sr,
                 format="OGG", subtype="VORBIS")
        buf.seek(0)
        out, _sr = sf.read(buf, dtype="float32", always_2d=False)
        if out.ndim > 1:
            out = out.mean(axis=1)
        # length may differ; pad/truncate to original
        if out.size != arr.size:
            if out.size > arr.size:
                out = out[: arr.size]
            else:
                out = np.concatenate([out, np.zeros(arr.size - out.size, dtype=np.float32)])
        return out
    except Exception:
        return arr


def rir_convolve(arr: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """Convolve audio with a single-channel room impulse response."""
    if rir is None or rir.size == 0:
        return arr
    out = fftconvolve(arr, rir.astype(np.float32, copy=False), mode="full")[: arr.size]
    # normalise to original RMS so loudness is preserved
    rms_in = float(np.sqrt(np.mean(arr ** 2)) + 1e-12)
    rms_out = float(np.sqrt(np.mean(out ** 2)) + 1e-12)
    return (out * (rms_in / rms_out)).astype(np.float32, copy=False)


def mixup_waveforms(a: np.ndarray, b: np.ndarray,
                    label_a: int, label_b: int, alpha: float = 0.2,
                    rng: random.Random | None = None) -> tuple[np.ndarray, float]:
    """Mixup at waveform level. Returns (mixed_audio, soft_label_for_class_1)."""
    rng = rng or random
    lam = float(np.random.default_rng().beta(alpha, alpha))
    mixed = lam * a + (1.0 - lam) * b
    # soft label = lam * one_hot(label_a) + (1-lam) * one_hot(label_b), then
    # we extract the probability of class 1 (drone)
    soft = lam * float(label_a) + (1.0 - lam) * float(label_b)
    return mixed.astype(np.float32, copy=False), soft


# ----------------------------- spectrogram-level ---------------------------- #


def filter_augment(spec: torch.Tensor, n_bands: int = 4,
                   max_db: float = 12.0,
                   rng: random.Random | None = None) -> torch.Tensor:
    """
    FilterAugment (Nam et al., 2022) — apply a random step-like gain across
    frequency bins, simulating different microphone frequency responses.

    spec: (..., F, T) log-mel-style tensor; we add a per-band offset to F.
    """
    rng = rng or random
    if spec.dim() < 2:
        return spec
    F_dim = spec.shape[-2]
    band_edges = sorted(rng.sample(range(1, F_dim), k=min(n_bands - 1, F_dim - 1)))
    band_edges = [0, *band_edges, F_dim]
    gains_db = [rng.uniform(-max_db, max_db) for _ in range(len(band_edges) - 1)]

    gain_vec = spec.new_zeros(F_dim)
    for (lo, hi), g in zip(zip(band_edges[:-1], band_edges[1:]), gains_db):
        gain_vec[lo:hi] = g
    # Convert dB to log-mel offset (mel features are roughly in dB scale)
    gain = gain_vec.view(*([1] * (spec.dim() - 2)), F_dim, 1)
    return spec + gain


def spec_patchout(spec: torch.Tensor, n_patches: int = 3,
                  max_freq_frac: float = 0.15, max_time_frac: float = 0.15,
                  rng: random.Random | None = None) -> torch.Tensor:
    """
    Drop n random rectangular regions of the spectrogram.

    spec: (..., F, T)
    """
    rng = rng or random
    if spec.dim() < 2:
        return spec
    F_dim, T_dim = spec.shape[-2], spec.shape[-1]
    out = spec.clone()
    for _ in range(n_patches):
        f = rng.randint(1, max(1, int(F_dim * max_freq_frac)))
        t = rng.randint(1, max(1, int(T_dim * max_time_frac)))
        f0 = rng.randint(0, F_dim - f)
        t0 = rng.randint(0, T_dim - t)
        out[..., f0:f0 + f, t0:t0 + t] = 0.0
    return out


def spec_augment(spec: torch.Tensor, n_freq_masks: int = 2, n_time_masks: int = 2,
                 freq_mask_param: int = 12, time_mask_param: int = 12,
                 rng: random.Random | None = None) -> torch.Tensor:
    """SpecAugment (Park et al., 2019): F-band zeroing + T-band zeroing."""
    rng = rng or random
    if spec.dim() < 2:
        return spec
    F_dim, T_dim = spec.shape[-2], spec.shape[-1]
    out = spec.clone()
    for _ in range(n_freq_masks):
        f = rng.randint(0, freq_mask_param)
        if f == 0:
            continue
        f0 = rng.randint(0, max(0, F_dim - f))
        out[..., f0:f0 + f, :] = 0.0
    for _ in range(n_time_masks):
        t = rng.randint(0, time_mask_param)
        if t == 0:
            continue
        t0 = rng.randint(0, max(0, T_dim - t))
        out[..., :, t0:t0 + t] = 0.0
    return out


# ----------------------------- the orchestrator ----------------------------- #


class AugConfig:
    """Probabilities for each augmentation. Tune lightly; more is OK here."""
    p_codec = 0.30
    p_eq = 0.40
    p_rir = 0.50
    p_mixup = 0.15
    p_filter_aug = 0.50
    p_patchout = 0.50
    p_specaug = 0.80
    # asymmetric: drone class only
    p_noise_into_drone = 0.70
    snr_drone_low_db = -5.0
    snr_drone_high_db = 20.0


def apply_waveform_augs(arr: np.ndarray, sr: int,
                        is_drone: bool,
                        noise_clips: list[np.ndarray] | None,
                        rir_clips: list[np.ndarray] | None,
                        rng: random.Random,
                        cfg: AugConfig) -> np.ndarray:
    """
    Symmetric augmentations applied to both classes.
    Asymmetric: noise mix-in only for drone clips.
    """
    out = arr.astype(np.float32, copy=False)

    if rir_clips and rng.random() < cfg.p_rir:
        rir = rng.choice(rir_clips)
        out = rir_convolve(out, rir)

    if rng.random() < cfg.p_eq:
        out = random_eq(out, sr, rng)

    if rng.random() < cfg.p_codec:
        out = codec_roundtrip(out, sr, rng)

    if is_drone and noise_clips and rng.random() < cfg.p_noise_into_drone:
        noise = rng.choice(noise_clips)
        snr = rng.uniform(cfg.snr_drone_low_db, cfg.snr_drone_high_db)
        out = add_noise(out, noise, snr)

    return np.clip(out, -1.0, 1.0).astype(np.float32, copy=False)


def apply_spec_augs(spec: torch.Tensor, rng: random.Random,
                    cfg: AugConfig) -> torch.Tensor:
    out = spec
    if rng.random() < cfg.p_filter_aug:
        out = filter_augment(out, rng=rng)
    if rng.random() < cfg.p_patchout:
        out = spec_patchout(out, rng=rng)
    if rng.random() < cfg.p_specaug:
        out = spec_augment(out, rng=rng)
    return out
