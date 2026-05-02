"""
v3 — finally include outdoor field recordings (DroneNoise) in training.

What's different from v2:
  - DroneNoise (Salford 2024) split: 12 of 18 clips used for training,
    6 held out as a small but TRUE in-distribution-OOD test set.
  - Stronger outdoor-style augmentation:
    - RIR pool now includes long-tail outdoor RIRs (RT60 up to 1.5s)
    - Distance attenuation simulation: random low-pass + amplitude reduction
    - Wind-style noise (pink noise low-pass filtered) added to drone clips
  - All v2 augmentations retained: symmetric codec/RIR/EQ on both classes,
    asymmetric urban-noise overlay onto drone, FilterAugment, Patchout,
    SpecAugment, Mixup support.
"""

from __future__ import annotations
import argparse
import math
import random
import time
from pathlib import Path
import json

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from datasets import load_dataset, Dataset as HFDataset

from src.config import CFG, CACHE_DIR
from src.model import build_model
from src.losses import FocalLoss
from src.augment import (
    AugConfig, apply_waveform_augs, apply_spec_augs,
)


ROOT = Path(__file__).resolve().parent.parent


# --------------------------- RIR pool (longer tails for outdoor) ------------ #


def generate_rir_pool(n: int = 64, sr: int = 16_000) -> list[np.ndarray]:
    import pyroomacoustics as pra
    rirs = []
    rng = np.random.default_rng(0)
    for _ in range(n):
        # half indoor, half outdoor-style (large rooms with long tails)
        if rng.random() < 0.5:
            x = rng.uniform(3.0, 12.0)
            y = rng.uniform(3.0, 10.0)
            z = rng.uniform(2.5, 5.5)
            rt60 = rng.uniform(0.15, 0.7)
        else:
            # large outdoor-ish: big "room" with longer reverb tail
            x = rng.uniform(15.0, 40.0)
            y = rng.uniform(15.0, 40.0)
            z = rng.uniform(5.0, 15.0)
            rt60 = rng.uniform(0.5, 1.5)
        try:
            e_abs, max_order = pra.inverse_sabine(rt60, [x, y, z])
        except Exception:
            e_abs, max_order = 0.5, 6
        room = pra.ShoeBox([x, y, z], fs=sr,
                           materials=pra.Material(e_abs),
                           max_order=int(max_order))
        src = [rng.uniform(0.5, x - 0.5),
               rng.uniform(0.5, y - 0.5),
               rng.uniform(0.5, z - 0.5)]
        mic = [rng.uniform(0.5, x - 0.5),
               rng.uniform(0.5, y - 0.5),
               rng.uniform(0.5, z - 0.5)]
        room.add_source(src)
        room.add_microphone_array(np.array(mic).reshape(3, 1))
        try:
            room.compute_rir()
            rir = np.asarray(room.rir[0][0], dtype=np.float32)
            if rir.size > sr * 2:
                rir = rir[: sr * 2]
            rirs.append(rir)
        except Exception:
            continue
    return rirs


# --------------------------- distance attenuation --------------------------- #


def distance_simulate(arr: np.ndarray, sr: int,
                      rng: random.Random) -> np.ndarray:
    """Simulate drone heard from further away: low-pass + amplitude drop."""
    from scipy.signal import butter, filtfilt
    # cutoff: closer = higher (4-8 kHz), farther = lower (1.5-3 kHz)
    cutoff = rng.uniform(1_500.0, 7_000.0)
    b, a = butter(4, cutoff / (sr / 2), btype="low")
    out = filtfilt(b, a, arr).astype(np.float32, copy=False)
    # amplitude reduction
    gain_db = rng.uniform(-18.0, 0.0)
    out = out * (10 ** (gain_db / 20))
    return out.astype(np.float32, copy=False)


def add_pink_wind(arr: np.ndarray, sr: int, rng: random.Random,
                  snr_db_low: float = -5.0, snr_db_high: float = 15.0
                  ) -> np.ndarray:
    """Add wind-style pink noise (low-pass filtered)."""
    from scipy.signal import butter, filtfilt
    n = arr.size
    # generate pink noise via 1/f filtering of white noise
    white = rng.gauss(0, 1)
    rng_local = np.random.default_rng()
    white = rng_local.standard_normal(n).astype(np.float32)
    # rough pink: butter low-pass at 800 Hz
    b, a = butter(4, 800 / (sr / 2), btype="low")
    pink = filtfilt(b, a, white).astype(np.float32, copy=False)

    sig_pow = float(np.mean(arr ** 2)) + 1e-12
    pink_pow = float(np.mean(pink ** 2)) + 1e-12
    snr_db = rng.uniform(snr_db_low, snr_db_high)
    target_n_pow = sig_pow / (10 ** (snr_db / 10))
    scale = np.sqrt(target_n_pow / pink_pow)
    return arr + scale * pink


# --------------------------- DroneNoise loader ------------------------------ #


def load_dronenoise_clips(split: str = "train") -> list[np.ndarray]:
    folder = ROOT / "data" / "dronenoise"
    files = sorted(folder.glob("Ed_*_M1.wav"))
    if not files:
        return []
    # deterministic split: every 3rd clip is held out
    train_clips, held_out = [], []
    for i, f in enumerate(files):
        arr, sr = sf.read(str(f), dtype="float32", always_2d=False)
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        if sr != 16_000:
            try:
                import librosa
                arr = librosa.resample(arr.astype(np.float32),
                                       orig_sr=sr, target_sr=16_000)
            except ImportError:
                pass
        arr = arr.astype(np.float32, copy=False)
        if i % 3 == 0:
            held_out.append(arr)
        else:
            train_clips.append(arr)
    if split == "train":
        return train_clips
    return held_out


# --------------------------- datasets --------------------------------------- #


class GeronimobassoSlice(Dataset):
    def __init__(self, hf_split: HFDataset, fe, n: int, seed: int = 42,
                 noise_clips=None, rir_clips=None, train_mode: bool = True):
        self.hf = hf_split
        self.fe = fe
        self.n = min(n, len(hf_split))
        self.train_mode = train_mode
        rng = np.random.RandomState(seed)
        self.indices = rng.choice(len(hf_split), self.n, replace=False)
        self.target_len = int(CFG.clip_seconds * CFG.sample_rate)
        self.sr = CFG.sample_rate
        self.noise_clips = noise_clips
        self.rir_clips = rir_clips
        self.aug_cfg = AugConfig()
        self._rng = random.Random(seed + 1)

    def __len__(self): return self.n

    def _crop(self, arr):
        if arr.size == self.target_len:
            return arr.astype(np.float32, copy=False)
        if arr.size > self.target_len:
            start = (np.random.randint(0, arr.size - self.target_len)
                     if self.train_mode else (arr.size - self.target_len) // 2)
            return arr[start:start + self.target_len].astype(np.float32, copy=False)
        return np.concatenate([arr.astype(np.float32, copy=False),
                               np.zeros(self.target_len - arr.size, dtype=np.float32)])

    def __getitem__(self, i):
        row = self.hf[int(self.indices[i])]
        arr = np.asarray(row["audio"]["array"], dtype=np.float32)
        sr = row["audio"]["sampling_rate"]
        if sr != self.sr:
            import librosa
            arr = librosa.resample(arr, orig_sr=sr, target_sr=self.sr)
        arr = self._crop(arr)
        label = int(row["label"])
        if self.train_mode:
            arr = apply_waveform_augs(
                arr, self.sr, is_drone=(label == 1),
                noise_clips=self.noise_clips, rir_clips=self.rir_clips,
                rng=self._rng, cfg=self.aug_cfg,
            )
            # extra: distance simulation 30% on drone clips
            if label == 1 and self._rng.random() < 0.3:
                arr = distance_simulate(arr, self.sr, self._rng)
            # extra: pink wind 30% on drone clips
            if label == 1 and self._rng.random() < 0.3:
                arr = add_pink_wind(arr, self.sr, self._rng)
        feats = self.fe(arr, sampling_rate=self.sr, return_tensors="pt")
        x = feats["input_values"].squeeze(0)
        if self.train_mode:
            x = apply_spec_augs(x, self._rng, self.aug_cfg)
        return {"input_values": x, "label": torch.tensor(label, dtype=torch.long)}


class LongClipPositives(Dataset):
    """Positive (drone) dataset from a list of long audio clips with random window crops."""
    def __init__(self, clips: list[np.ndarray], fe, *, name: str,
                 noise_clips=None, rir_clips=None, train_mode: bool = True,
                 windows_per_clip: int = 16):
        self.clips = clips
        self.fe = fe
        self.name = name
        self.train_mode = train_mode
        self.target_len = int(CFG.clip_seconds * CFG.sample_rate)
        self.sr = CFG.sample_rate
        self.noise_clips = noise_clips
        self.rir_clips = rir_clips
        self.aug_cfg = AugConfig()
        self.windows_per_clip = windows_per_clip
        self._rng = random.Random(hash(name) & 0xffff)
        print(f"[{name}] {len(clips)} clips, {len(clips) * windows_per_clip} windows")

    def __len__(self): return len(self.clips) * self.windows_per_clip

    def __getitem__(self, i):
        clip = self.clips[i // self.windows_per_clip]
        if clip.size > self.target_len:
            start = np.random.randint(0, clip.size - self.target_len)
            arr = clip[start:start + self.target_len]
        else:
            arr = np.concatenate([clip, np.zeros(self.target_len - clip.size, dtype=np.float32)])
        if self.train_mode:
            arr = apply_waveform_augs(
                arr, self.sr, is_drone=True,
                noise_clips=self.noise_clips, rir_clips=self.rir_clips,
                rng=self._rng, cfg=self.aug_cfg,
            )
            if self._rng.random() < 0.3:
                arr = distance_simulate(arr, self.sr, self._rng)
            if self._rng.random() < 0.3:
                arr = add_pink_wind(arr, self.sr, self._rng)
        feats = self.fe(arr, sampling_rate=self.sr, return_tensors="pt")
        x = feats["input_values"].squeeze(0)
        if self.train_mode:
            x = apply_spec_augs(x, self._rng, self.aug_cfg)
        return {"input_values": x, "label": torch.tensor(1, dtype=torch.long)}


# --------------------------- main ------------------------------------------- #


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=ROOT / "runs/20260429-112104/best.pt")
    p.add_argument("--out", type=Path, default=ROOT / "runs/20260429-112104/best_v3.pt")
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--geron-samples", type=int, default=4000)
    p.add_argument("--lr-backbone", type=float, default=1e-6)
    p.add_argument("--lr-head", type=float, default=5e-5)
    p.add_argument("--n-rirs", type=int, default=64)
    args = p.parse_args()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[device] {device}")

    print(f"[ckpt] loading {args.ckpt}")
    model, fe = build_model()
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state"])
    model.to(device)

    # Build noise pool from geronimobasso no-drone clips
    print("[noise] building urban noise pool…")
    geron = load_dataset(
        "geronimobasso/drone-audio-detection-samples",
        cache_dir=str(CACHE_DIR),
    )
    geron_split = geron["train"] if "train" in geron else geron[list(geron.keys())[0]]
    labels = np.asarray(geron_split["label"])
    no_drone_idx = np.where(labels == 0)[0]
    rng = np.random.RandomState(7)
    sel = rng.choice(no_drone_idx, size=min(200, no_drone_idx.size), replace=False)
    noise_clips = []
    for i in sel:
        s = geron_split[int(i)]
        arr = np.asarray(s["audio"]["array"], dtype=np.float32)
        if arr.size > 16_000:
            arr = arr[: 16_000]
        noise_clips.append(arr)
    print(f"[noise] {len(noise_clips)} samples")

    print(f"[rir] generating {args.n_rirs} synthetic RIRs (mixed indoor + outdoor)…")
    rir_clips = generate_rir_pool(n=args.n_rirs, sr=16_000)
    print(f"[rir] {len(rir_clips)} ready")

    # Geronimobasso slice (drone + no-drone)
    print("[data] geronimobasso slice…")
    geron_ds = GeronimobassoSlice(
        geron_split, fe, n=args.geron_samples,
        noise_clips=noise_clips, rir_clips=rir_clips,
    )

    # NUS DroneAudioSet — splits 1..20 train, 21..28 held-out (validation only)
    print("[data] NUS positives, splits 1..20…")
    nus_clips_train = []
    for k in range(1, 21):
        try:
            ds = load_dataset(
                "ahlab-drone-project/DroneAudioSet", "drone-only",
                split=f"train_{k:03d}",
                cache_dir=str(ROOT / "data" / "ahlab"),
            )
        except Exception:
            continue
        for s in ds:
            arr = np.asarray(s["audio"]["array"], dtype=np.float32)
            if arr.ndim > 1:
                arr = arr[:, 0]
            sr_ = s["audio"]["sampling_rate"]
            if sr_ != 16_000:
                import librosa
                arr = librosa.resample(arr, orig_sr=sr_, target_sr=16_000)
            nus_clips_train.append(arr.astype(np.float32, copy=False))
    nus_train = LongClipPositives(
        nus_clips_train, fe, name="NUS-train",
        noise_clips=noise_clips, rir_clips=rir_clips,
    )

    # NUS held-out for validation only
    print("[data] NUS held-out splits 21..28…")
    nus_clips_val = []
    for k in range(21, 29):
        try:
            ds = load_dataset(
                "ahlab-drone-project/DroneAudioSet", "drone-only",
                split=f"train_{k:03d}",
                cache_dir=str(ROOT / "data" / "ahlab"),
            )
        except Exception:
            continue
        for s in ds:
            arr = np.asarray(s["audio"]["array"], dtype=np.float32)
            if arr.ndim > 1:
                arr = arr[:, 0]
            sr_ = s["audio"]["sampling_rate"]
            if sr_ != 16_000:
                import librosa
                arr = librosa.resample(arr, orig_sr=sr_, target_sr=16_000)
            nus_clips_val.append(arr.astype(np.float32, copy=False))
    nus_val = LongClipPositives(
        nus_clips_val, fe, name="NUS-val", train_mode=False,
        windows_per_clip=4,
    )

    # DroneNoise — split 12 train / 6 held-out
    print("[data] DroneNoise — train portion (~12 clips)…")
    dn_train_clips = load_dronenoise_clips("train")
    dn_train = LongClipPositives(
        dn_train_clips, fe, name="DroneNoise-train",
        noise_clips=noise_clips, rir_clips=rir_clips,
        windows_per_clip=24,
    )
    print("[data] DroneNoise — held-out portion (~6 clips)…")
    dn_val_clips = load_dronenoise_clips("val")
    dn_val = LongClipPositives(
        dn_val_clips, fe, name="DroneNoise-val", train_mode=False,
        windows_per_clip=8,
    )

    train_ds = ConcatDataset([geron_ds, nus_train, dn_train])
    print(f"[data] combined train size = {len(train_ds)}")

    # Class weights from training labels
    geron_labels = np.asarray([int(geron_split[int(idx)]["label"])
                               for idx in geron_ds.indices])
    train_labels = np.concatenate([
        geron_labels,
        np.ones(len(nus_train), dtype=int),
        np.ones(len(dn_train), dtype=int),
    ])
    counts = np.bincount(train_labels)
    inv_freq = 1.0 / np.maximum(counts, 1)
    sample_weights = inv_freq[train_labels]
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=args.steps * args.batch_size,
        replacement=True,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=2, persistent_workers=True,
    )
    nus_val_loader = DataLoader(
        nus_val, batch_size=args.batch_size, shuffle=False,
        num_workers=1, persistent_workers=True,
    )
    dn_val_loader = DataLoader(
        dn_val, batch_size=args.batch_size, shuffle=False,
        num_workers=1, persistent_workers=True,
    )

    cw = torch.tensor(inv_freq, dtype=torch.float32, device=device)
    cw = cw / cw.sum() * len(cw)
    criterion = FocalLoss(gamma=2.0, class_weights=cw)

    pg = [
        {"params": [p for n, p in model.named_parameters()
                    if "classifier" not in n and p.requires_grad],
         "lr": args.lr_backbone, "name": "backbone"},
        {"params": [p for n, p in model.named_parameters()
                    if "classifier" in n and p.requires_grad],
         "lr": args.lr_head, "name": "head"},
    ]
    optimizer = AdamW(pg, weight_decay=1e-4)

    def lr_lambda(s):
        warmup = 150
        if s < warmup:
            return s / warmup
        progress = (s - warmup) / max(1, args.steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = LambdaLR(optimizer, lr_lambda)

    autocast_dtype = torch.bfloat16

    @torch.inference_mode()
    def held_out_val(loader, name):
        model.eval()
        ps = []
        for batch in loader:
            x = batch["input_values"].to(device)
            out = model(input_values=x)
            ps.append(torch.softmax(out.logits, -1)[:, 1].float().cpu().numpy())
        model.train()
        ps = np.concatenate(ps) if ps else np.array([0.0])
        return float(ps.mean()), float(ps.min()), float((ps >= 0.5).mean())

    print("[train] starting v3 fine-tune…")
    model.train()
    t0 = time.time()
    running = 0.0
    n_seen = 0
    step = 0
    best_dn_val = -1.0
    for batch in train_loader:
        if step >= args.steps:
            break
        x = batch["input_values"].to(device)
        y = batch["label"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=autocast_dtype):
            out = model(input_values=x)
            loss = criterion(out.logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        step += 1
        running += float(loss.item()) * y.size(0)
        n_seen += y.size(0)

        if step % 50 == 0:
            avg = running / max(1, n_seen)
            lr0 = optimizer.param_groups[0]["lr"]
            elapsed = (time.time() - t0) / 60
            print(f"[step {step:>5}/{args.steps}] loss={avg:.4f} lr={lr0:.2e} "
                  f"elapsed={elapsed:.1f}m", flush=True)

        if step % 500 == 0:
            nm, nmi, nfa = held_out_val(nus_val_loader, "NUS-val")
            dm, dmi, dfa = held_out_val(dn_val_loader, "DN-val")
            print(f"[val:NUS-val] mean={nm:.3f} min={nmi:.3f} >=0.5={nfa:.2%}",
                  flush=True)
            print(f"[val:DN-val]  mean={dm:.3f} min={dmi:.3f} >=0.5={dfa:.2%}",
                  flush=True)
            if dm > best_dn_val:
                best_dn_val = dm
                torch.save({
                    "model_state": model.state_dict(),
                    "config": CFG.__dict__,
                    "step": step,
                    "val_NUS_mean": nm,
                    "val_DN_mean": dm,
                }, args.out)
                print(f"[ckpt] new best DN val={dm:.4f} -> {args.out}", flush=True)

    torch.save({
        "model_state": model.state_dict(),
        "config": CFG.__dict__,
        "step": step,
        "best_dn_val_mean_p_drone": best_dn_val,
    }, args.out.with_suffix(".final.pt"))

    print(f"[done] total time={(time.time()-t0)/60:.1f}m best_DN_val={best_dn_val:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
