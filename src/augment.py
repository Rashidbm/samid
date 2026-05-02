from __future__ import annotations
import io
import random

import numpy as np
import soundfile as sf
import torch
from scipy.signal import butter, fftconvolve, filtfilt


def add_noise(arr, noise, snr_db):
    if noise.size < arr.size:
        reps = int(np.ceil(arr.size / max(1, noise.size)))
        noise = np.tile(noise, reps)
    noise = noise[:arr.size].astype(np.float32, copy=False)
    sig_pow = float(np.mean(arr ** 2)) + 1e-12
    noise_pow = float(np.mean(noise ** 2)) + 1e-12
    target_pow = sig_pow / (10 ** (snr_db / 10))
    scale = float(np.sqrt(target_pow / noise_pow))
    return arr + scale * noise


def random_eq(arr, sr, rng):
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


def codec_roundtrip(arr, sr, rng):
    try:
        buf = io.BytesIO()
        sf.write(buf, arr.astype(np.float32, copy=False), sr,
                 format="OGG", subtype="VORBIS")
        buf.seek(0)
        out, _ = sf.read(buf, dtype="float32", always_2d=False)
        if out.ndim > 1:
            out = out.mean(axis=1)
        if out.size != arr.size:
            if out.size > arr.size:
                out = out[:arr.size]
            else:
                out = np.concatenate([out, np.zeros(arr.size - out.size, dtype=np.float32)])
        return out
    except Exception:
        return arr


def rir_convolve(arr, rir):
    if rir is None or rir.size == 0:
        return arr
    out = fftconvolve(arr, rir.astype(np.float32, copy=False), mode="full")[:arr.size]
    rms_in = float(np.sqrt(np.mean(arr ** 2)) + 1e-12)
    rms_out = float(np.sqrt(np.mean(out ** 2)) + 1e-12)
    return (out * (rms_in / rms_out)).astype(np.float32, copy=False)


def filter_augment(spec, n_bands=4, max_db=12.0, rng=None):
    rng = rng or random
    if spec.dim() < 2:
        return spec
    F_dim = spec.shape[-2]
    edges = sorted(rng.sample(range(1, F_dim), k=min(n_bands - 1, F_dim - 1)))
    edges = [0, *edges, F_dim]
    gains = [rng.uniform(-max_db, max_db) for _ in range(len(edges) - 1)]
    gain_vec = spec.new_zeros(F_dim)
    for (lo, hi), g in zip(zip(edges[:-1], edges[1:]), gains):
        gain_vec[lo:hi] = g
    return spec + gain_vec.view(*([1] * (spec.dim() - 2)), F_dim, 1)


def spec_patchout(spec, n_patches=3, max_freq_frac=0.15, max_time_frac=0.15, rng=None):
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


def spec_augment(spec, n_freq_masks=2, n_time_masks=2,
                 freq_mask_param=12, time_mask_param=12, rng=None):
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


class AugConfig:
    p_codec = 0.30
    p_eq = 0.40
    p_rir = 0.50
    p_mixup = 0.15
    p_filter_aug = 0.50
    p_patchout = 0.50
    p_specaug = 0.80
    p_noise_into_drone = 0.70
    snr_drone_low_db = -5.0
    snr_drone_high_db = 20.0


def apply_waveform_augs(arr, sr, is_drone, noise_clips, rir_clips, rng, cfg):
    out = arr.astype(np.float32, copy=False)
    if rir_clips and rng.random() < cfg.p_rir:
        out = rir_convolve(out, rng.choice(rir_clips))
    if rng.random() < cfg.p_eq:
        out = random_eq(out, sr, rng)
    if rng.random() < cfg.p_codec:
        out = codec_roundtrip(out, sr, rng)
    if is_drone and noise_clips and rng.random() < cfg.p_noise_into_drone:
        snr = rng.uniform(cfg.snr_drone_low_db, cfg.snr_drone_high_db)
        out = add_noise(out, rng.choice(noise_clips), snr)
    return np.clip(out, -1.0, 1.0).astype(np.float32, copy=False)


def apply_spec_augs(spec, rng, cfg):
    out = spec
    if rng.random() < cfg.p_filter_aug:
        out = filter_augment(out, rng=rng)
    if rng.random() < cfg.p_patchout:
        out = spec_patchout(out, rng=rng)
    if rng.random() < cfg.p_specaug:
        out = spec_augment(out, rng=rng)
    return out
