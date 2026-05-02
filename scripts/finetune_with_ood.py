"""
Continue fine-tuning the trained model on COMBINED data:
  - geronimobasso (what we trained on) — keeps original capability
  - NUS DroneAudioSet drone clips      — teaches OOD drone recognition
  - light augmentation                 — improves real-world robustness

This is the real fix for the overfit-to-geronimobasso problem.
We keep training short (~1500 steps) to avoid overfitting to NUS instead.
"""

from __future__ import annotations
import argparse
import math
import time
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from datasets import load_dataset, Dataset as HFDataset
from sklearn.model_selection import train_test_split
from rich.console import Console

from src.config import CFG, CACHE_DIR, RUNS_DIR
from src.model import build_model, param_groups
from src.losses import FocalLoss
from src.metrics import report_at_threshold

ROOT = Path(__file__).resolve().parent.parent
console = Console()


# ---------------------- Datasets that handle both sources ------------------- #


class GeronimobassoSlice(Dataset):
    """Random subset of geronimobasso, drone+nodrone."""
    def __init__(self, hf_split: HFDataset, fe, n: int, seed: int = 42, train_mode: bool = True):
        self.hf = hf_split
        self.fe = fe
        self.n = min(n, len(hf_split))
        self.train_mode = train_mode
        rng = np.random.RandomState(seed)
        self.indices = rng.choice(len(hf_split), self.n, replace=False)
        self.target_len = int(CFG.clip_seconds * CFG.sample_rate)
        self.sr = CFG.sample_rate

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

    def __getitem__(self, i):
        row = self.hf[int(self.indices[i])]
        arr = np.asarray(row["audio"]["array"], dtype=np.float32)
        sr = row["audio"]["sampling_rate"]
        if sr != self.sr:
            import librosa
            arr = librosa.resample(arr, orig_sr=sr, target_sr=self.sr)
        arr = self._crop(arr)
        if self.train_mode:
            arr = augment(arr, self.sr)
        feats = self.fe(arr, sampling_rate=self.sr, return_tensors="pt")
        return {
            "input_values": feats["input_values"].squeeze(0),
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
        }


class NUSDronePositives(Dataset):
    """All drone-only clips from ahlab-drone-project/DroneAudioSet, all 28 splits."""
    def __init__(self, fe, train_mode: bool = True):
        self.fe = fe
        self.train_mode = train_mode
        self.target_len = int(CFG.clip_seconds * CFG.sample_rate)
        self.sr = CFG.sample_rate
        # collect all 28 splits
        self.clips: list[np.ndarray] = []
        for k in range(1, 29):
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
                    arr = arr[:, 0]   # NUS layout (samples, channels)
                sr = s["audio"]["sampling_rate"]
                if sr != self.sr:
                    import librosa
                    arr = librosa.resample(arr, orig_sr=sr, target_sr=self.sr)
                # store the long clip; we crop random windows at __getitem__
                self.clips.append(arr.astype(np.float32, copy=False))
        # We expand by sampling many windows from each long clip
        # so the dataset has more training examples per epoch
        self.windows_per_clip = 10
        console.print(f"[NUS] loaded {len(self.clips)} drone recordings; "
                      f"{len(self.clips) * self.windows_per_clip} virtual windows")

    def __len__(self): return len(self.clips) * self.windows_per_clip

    def __getitem__(self, i):
        clip_idx = i // self.windows_per_clip
        arr = self.clips[clip_idx]
        if arr.size > self.target_len:
            start = np.random.randint(0, arr.size - self.target_len)
            arr = arr[start:start + self.target_len]
        else:
            arr = np.concatenate([arr, np.zeros(self.target_len - arr.size, dtype=np.float32)])
        if self.train_mode:
            arr = augment(arr, self.sr)
        feats = self.fe(arr, sampling_rate=self.sr, return_tensors="pt")
        return {
            "input_values": feats["input_values"].squeeze(0),
            "label": torch.tensor(1, dtype=torch.long),  # always positive
        }


# ---------------------- Augmentation ---------------------------------------- #


def augment(arr: np.ndarray, sr: int) -> np.ndarray:
    """Light augmentation. Heavy on noise, light on pitch/time."""
    if np.random.rand() < 0.5:
        # add white noise at 5-25 dB SNR
        snr_db = np.random.uniform(5, 25)
        sig_pow = np.mean(arr ** 2) + 1e-12
        noise_pow = sig_pow / (10 ** (snr_db / 10))
        arr = arr + np.random.randn(arr.size).astype(np.float32) * np.sqrt(noise_pow)
    if np.random.rand() < 0.3:
        # random gain ±6 dB
        gain = 10 ** (np.random.uniform(-6, 6) / 20)
        arr = arr * gain
    if np.random.rand() < 0.2:
        # gentle highpass to simulate different mic types
        from scipy.signal import butter, filtfilt
        cutoff = np.random.uniform(80, 200)
        b, a = butter(2, cutoff / (sr / 2), btype="high")
        arr = filtfilt(b, a, arr).astype(np.float32)
    return np.clip(arr, -1.0, 1.0)


# ---------------------- main ------------------------------------------------ #


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=ROOT / "runs/20260429-112104/best.pt")
    p.add_argument("--out", type=Path, default=ROOT / "runs/20260429-112104/best_finetuned.pt")
    p.add_argument("--steps", type=int, default=1500,
                   help="number of optimizer steps")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--geron-samples", type=int, default=4000,
                   help="how many random geronimobasso samples to mix in")
    p.add_argument("--lr-backbone", type=float, default=1e-6)
    p.add_argument("--lr-head", type=float, default=5e-5)
    args = p.parse_args()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    console.print(f"[bold]Device:[/bold] {device}")

    # Build model and load checkpoint
    console.print(f"[bold]Loading checkpoint:[/bold] {args.ckpt}")
    model, fe = build_model()
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state"])
    model.to(device)

    # Load datasets
    console.print("[bold]Loading geronimobasso slice…[/bold]")
    geron = load_dataset(
        "geronimobasso/drone-audio-detection-samples",
        cache_dir=str(CACHE_DIR),
    )
    geron_split = geron["train"] if "train" in geron else geron[list(geron.keys())[0]]
    geron_ds = GeronimobassoSlice(geron_split, fe, n=args.geron_samples)

    console.print("[bold]Loading NUS positives…[/bold]")
    nus_ds = NUSDronePositives(fe)

    combined = ConcatDataset([geron_ds, nus_ds])
    console.print(f"[bold]Combined dataset size:[/bold] {len(combined)}")

    # Weighted sampler so each batch has balanced labels
    geron_labels = np.asarray([int(geron_split[int(idx)]["label"]) for idx in geron_ds.indices])
    nus_labels = np.ones(len(nus_ds), dtype=int)
    all_labels = np.concatenate([geron_labels, nus_labels])
    counts = np.bincount(all_labels)
    inv_freq = 1.0 / np.maximum(counts, 1)
    sample_weights = inv_freq[all_labels]
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=args.steps * args.batch_size,
        replacement=True,
    )
    loader = DataLoader(
        combined, batch_size=args.batch_size, sampler=sampler,
        num_workers=2, persistent_workers=True,
    )

    # Class weights
    cw = torch.tensor(inv_freq, dtype=torch.float32, device=device)
    cw = cw / cw.sum() * len(cw)
    criterion = FocalLoss(gamma=2.0, class_weights=cw)

    # Param groups with lower LR than initial training
    pg = [
        {"params": [p for n, p in model.named_parameters()
                    if "classifier" not in n and p.requires_grad],
         "lr": args.lr_backbone, "name": "backbone"},
        {"params": [p for n, p in model.named_parameters()
                    if "classifier" in n and p.requires_grad],
         "lr": args.lr_head, "name": "head"},
    ]
    optimizer = AdamW(pg, weight_decay=1e-4)

    # Cosine schedule
    def lr_lambda(s):
        warmup = 100
        if s < warmup:
            return s / warmup
        progress = (s - warmup) / max(1, args.steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = LambdaLR(optimizer, lr_lambda)

    # Training loop
    console.print("[bold]Starting fine-tuning…[/bold]")
    model.train()
    t0 = time.time()
    running = 0.0
    n_seen = 0
    autocast_dtype = torch.bfloat16

    step = 0
    for batch in loader:
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

        if step % 25 == 0:
            avg = running / max(1, n_seen)
            lr0 = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0
            print(f"[step {step:>5}/{args.steps}] loss={avg:.4f} lr={lr0:.2e} "
                  f"elapsed={elapsed/60:.1f}m", flush=True)

    # Save
    out_state = {
        "model_state": model.state_dict(),
        "config": CFG.__dict__,
        "finetune_steps": step,
        "source_ckpt": str(args.ckpt),
    }
    torch.save(out_state, args.out)
    console.print(f"\n[green]Saved -> {args.out}[/green]")
    console.print(f"Total time: {(time.time() - t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
