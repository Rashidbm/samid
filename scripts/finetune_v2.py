"""
Domain-invariance fine-tune (v2) per reviewer feedback.

Changes from v1:
  - SYMMETRIC augmentation across both classes (RIR, codec, EQ, FilterAugment,
    Patchout, SpecAugment, Mixup applied to both)
  - Asymmetric ONLY: urban-noise overlay onto drone clips
  - FilterAugment + Spectrogram Patchout added (per reviewer)
  - Synthetic RIR pool generated at startup with pyroomacoustics
  - Validation runs on a HELD-OUT 30% of NUS that was NOT used in training
  - DroneNoise (Salford 2024) is held out entirely for the final cross-dataset
    test (not used in training, not used in validation)
"""

from __future__ import annotations
import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler

import soundfile as sf
from datasets import load_dataset, Dataset as HFDataset

from src.config import CFG, CACHE_DIR
from src.model import build_model
from src.losses import FocalLoss
from src.augment import (
    AugConfig, apply_waveform_augs, apply_spec_augs, mixup_waveforms,
)


ROOT = Path(__file__).resolve().parent.parent


# --------------------------- RIR pool --------------------------------------- #


def generate_rir_pool(n: int = 48, sr: int = 16_000) -> list[np.ndarray]:
    """Generate a small pool of synthetic RIRs spanning small/medium/large rooms."""
    import pyroomacoustics as pra
    rirs = []
    rng = np.random.default_rng(0)
    for _ in range(n):
        x = rng.uniform(3.0, 12.0)
        y = rng.uniform(3.0, 10.0)
        z = rng.uniform(2.5, 5.5)
        rt60 = rng.uniform(0.15, 0.7)
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
            if rir.size > sr:        # cap RIR length to 1 sec
                rir = rir[:sr]
            rirs.append(rir)
        except Exception:
            continue
    return rirs


# --------------------------- datasets --------------------------------------- #


class GeronimobassoSlice(Dataset):
    def __init__(self, hf_split: HFDataset, fe, n: int, seed: int = 42,
                 noise_clips: list[np.ndarray] | None = None,
                 rir_clips: list[np.ndarray] | None = None,
                 train_mode: bool = True):
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

    def _crop(self, arr: np.ndarray) -> np.ndarray:
        if arr.size == self.target_len:
            return arr.astype(np.float32, copy=False)
        if arr.size > self.target_len:
            start = (np.random.randint(0, arr.size - self.target_len)
                     if self.train_mode else (arr.size - self.target_len) // 2)
            return arr[start:start + self.target_len].astype(np.float32, copy=False)
        return np.concatenate([arr.astype(np.float32, copy=False),
                               np.zeros(self.target_len - arr.size, dtype=np.float32)])

    def get_raw(self, i):
        row = self.hf[int(self.indices[i])]
        arr = np.asarray(row["audio"]["array"], dtype=np.float32)
        sr = row["audio"]["sampling_rate"]
        if sr != self.sr:
            import librosa
            arr = librosa.resample(arr, orig_sr=sr, target_sr=self.sr)
        return self._crop(arr), int(row["label"])

    def __getitem__(self, i):
        arr, label = self.get_raw(i)
        if self.train_mode:
            arr = apply_waveform_augs(
                arr, self.sr, is_drone=(label == 1),
                noise_clips=self.noise_clips, rir_clips=self.rir_clips,
                rng=self._rng, cfg=self.aug_cfg,
            )
        feats = self.fe(arr, sampling_rate=self.sr, return_tensors="pt")
        x = feats["input_values"].squeeze(0)
        if self.train_mode:
            x = apply_spec_augs(x, self._rng, self.aug_cfg)
        return {"input_values": x, "label": torch.tensor(label, dtype=torch.long)}


class NUSPositives(Dataset):
    """NUS DroneAudioSet drone clips. Splits 1..k_train used for training,
    splits k_train+1..28 reserved for held-out validation."""

    def __init__(self, fe, *, splits_range: tuple[int, int],
                 noise_clips: list[np.ndarray] | None = None,
                 rir_clips: list[np.ndarray] | None = None,
                 train_mode: bool = True,
                 windows_per_clip: int = 8):
        self.fe = fe
        self.train_mode = train_mode
        self.target_len = int(CFG.clip_seconds * CFG.sample_rate)
        self.sr = CFG.sample_rate
        self.noise_clips = noise_clips
        self.rir_clips = rir_clips
        self.aug_cfg = AugConfig()
        self.windows_per_clip = windows_per_clip
        self._rng = random.Random(123 + splits_range[0])

        self.clips: list[np.ndarray] = []
        for k in range(splits_range[0], splits_range[1] + 1):
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
                sr = s["audio"]["sampling_rate"]
                if sr != self.sr:
                    import librosa
                    arr = librosa.resample(arr, orig_sr=sr, target_sr=self.sr)
                self.clips.append(arr.astype(np.float32, copy=False))
        print(f"[NUS] splits {splits_range}: {len(self.clips)} clips, "
              f"{len(self.clips) * self.windows_per_clip} windows")

    def __len__(self): return len(self.clips) * self.windows_per_clip

    def __getitem__(self, i):
        clip = self.clips[i // self.windows_per_clip]
        if clip.size > self.target_len:
            start = np.random.randint(0, clip.size - self.target_len)
            arr = clip[start:start + self.target_len]
        else:
            arr = np.concatenate([clip, np.zeros(self.target_len - clip.size, dtype=np.float32)])

        label = 1
        if self.train_mode:
            arr = apply_waveform_augs(
                arr, self.sr, is_drone=True,
                noise_clips=self.noise_clips, rir_clips=self.rir_clips,
                rng=self._rng, cfg=self.aug_cfg,
            )
        feats = self.fe(arr, sampling_rate=self.sr, return_tensors="pt")
        x = feats["input_values"].squeeze(0)
        if self.train_mode:
            x = apply_spec_augs(x, self._rng, self.aug_cfg)
        return {"input_values": x, "label": torch.tensor(label, dtype=torch.long)}


# --------------------------- main ------------------------------------------- #


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=ROOT / "runs/20260429-112104/best.pt")
    p.add_argument("--out", type=Path, default=ROOT / "runs/20260429-112104/best_v2.pt")
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--geron-samples", type=int, default=4000)
    p.add_argument("--lr-backbone", type=float, default=1e-6)
    p.add_argument("--lr-head", type=float, default=5e-5)
    p.add_argument("--n-rirs", type=int, default=48)
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

    # Build noise pool from geronimobasso no-drone clips (URBAN-style audio)
    print("[noise] building urban noise pool from geronimobasso label=0 clips…")
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
        if arr.size > 16_000:                     # cap each noise sample to 1 sec
            arr = arr[: 16_000]
        noise_clips.append(arr)
    print(f"[noise] {len(noise_clips)} noise samples ready")

    print(f"[rir] generating {args.n_rirs} synthetic RIRs…")
    rir_clips = generate_rir_pool(n=args.n_rirs, sr=16_000)
    print(f"[rir] {len(rir_clips)} RIRs ready")

    print("[data] geronimobasso slice…")
    geron_ds = GeronimobassoSlice(
        geron_split, fe, n=args.geron_samples,
        noise_clips=noise_clips, rir_clips=rir_clips,
    )
    print("[data] NUS positives, splits 1..20 (TRAIN)…")
    nus_train = NUSPositives(
        fe, splits_range=(1, 20),
        noise_clips=noise_clips, rir_clips=rir_clips, train_mode=True,
    )
    print("[data] NUS positives, splits 21..28 (HELD-OUT VAL)…")
    nus_val = NUSPositives(
        fe, splits_range=(21, 28), train_mode=False,
    )

    train_ds = ConcatDataset([geron_ds, nus_train])
    print(f"[data] combined train size = {len(train_ds)}")

    # Compute class weights based on training pool
    geron_labels = np.asarray([int(geron_split[int(idx)]["label"])
                               for idx in geron_ds.indices])
    train_labels = np.concatenate([geron_labels, np.ones(len(nus_train), dtype=int)])
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
    val_loader = DataLoader(
        nus_val, batch_size=args.batch_size, shuffle=False,
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
    def held_out_val():
        model.eval()
        ps = []
        for batch in val_loader:
            x = batch["input_values"].to(device)
            out = model(input_values=x)
            ps.append(torch.softmax(out.logits, -1)[:, 1].float().cpu().numpy())
        model.train()
        ps = np.concatenate(ps)
        return float(ps.mean()), float(ps.min()), float((ps >= 0.5).mean())

    print("[train] starting v2 fine-tune…")
    model.train()
    t0 = time.time()
    running = 0.0
    n_seen = 0
    step = 0
    best_val = -1.0
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
            mean_p, min_p, frac_above = held_out_val()
            print(f"[val:NUS-held-out] mean p(drone)={mean_p:.3f}  "
                  f"min={min_p:.3f}  frac>=0.5={frac_above:.2%}", flush=True)
            if mean_p > best_val:
                best_val = mean_p
                torch.save({
                    "model_state": model.state_dict(),
                    "config": CFG.__dict__,
                    "step": step,
                    "val_mean_p_drone_held_out_NUS": mean_p,
                }, args.out)
                print(f"[ckpt] saved best -> {args.out}", flush=True)

    # final save anyway
    torch.save({
        "model_state": model.state_dict(),
        "config": CFG.__dict__,
        "step": step,
        "best_val_mean_p_drone_held_out_NUS": best_val,
    }, args.out.with_suffix(".final.pt"))

    print(f"[done] total time = {(time.time() - t0)/60:.1f} min  best_val={best_val:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
