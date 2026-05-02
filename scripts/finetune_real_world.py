from __future__ import annotations
import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler

from datasets import load_dataset

from src.config import CFG, CACHE_DIR
from src.model import build_model
from src.losses import FocalLoss
from src.augment import AugConfig, apply_waveform_augs, apply_spec_augs


ROOT = Path(__file__).resolve().parent.parent


def load_wav_16k_mono(path):
    import librosa
    arr, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if sr != 16_000:
        arr = librosa.resample(arr, orig_sr=sr, target_sr=16_000)
    return arr.astype(np.float32, copy=False)


def windows_from_clip(arr, win=16_000, hop=8_000):
    if arr.size < win:
        return [np.concatenate([arr, np.zeros(win - arr.size, dtype=np.float32)])]
    return [arr[s:s + win] for s in range(0, arr.size - win + 1, hop)]


def pseudo_label_windows(arr, model, fe, device, threshold):
    win = 16_000
    hop = 8_000
    kept = []
    model.eval()
    with torch.inference_mode():
        for chunk in windows_from_clip(arr, win, hop):
            feats = fe(chunk, sampling_rate=16_000, return_tensors="pt")
            p = float(torch.softmax(
                model(input_values=feats["input_values"].to(device)).logits, -1
            )[0, 1])
            if p >= threshold:
                kept.append(chunk)
    return kept


class PositiveClips(Dataset):
    def __init__(self, clips, fe, reps=50):
        self.clips = clips
        self.fe = fe
        self.reps = reps
        self.target_len = int(CFG.clip_seconds * CFG.sample_rate)
        self.sr = CFG.sample_rate
        self.aug_cfg = AugConfig()
        self._rng = random.Random(20260502)

    def __len__(self):
        return len(self.clips) * self.reps

    def __getitem__(self, i):
        clip = self.clips[i % len(self.clips)]
        if clip.size > self.target_len:
            start = np.random.randint(0, clip.size - self.target_len)
            arr = clip[start:start + self.target_len]
        elif clip.size < self.target_len:
            arr = np.concatenate([clip, np.zeros(self.target_len - clip.size, dtype=np.float32)])
        else:
            arr = clip
        arr = apply_waveform_augs(arr.astype(np.float32, copy=False), self.sr,
                                   is_drone=True, noise_clips=None, rir_clips=None,
                                   rng=self._rng, cfg=self.aug_cfg)
        feats = self.fe(arr, sampling_rate=self.sr, return_tensors="pt")
        x = apply_spec_augs(feats["input_values"].squeeze(0), self._rng, self.aug_cfg)
        return {"input_values": x, "label": torch.tensor(1, dtype=torch.long)}


class GeronimobassoSlice(Dataset):
    def __init__(self, hf_split, fe, n, seed=42):
        self.hf = hf_split
        self.fe = fe
        rng = np.random.RandomState(seed)
        self.indices = rng.choice(len(hf_split), min(n, len(hf_split)), replace=False)
        self.target_len = int(CFG.clip_seconds * CFG.sample_rate)
        self.sr = CFG.sample_rate
        self.aug_cfg = AugConfig()
        self._rng = random.Random(seed + 1)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        row = self.hf[int(self.indices[i])]
        arr = np.asarray(row["audio"]["array"], dtype=np.float32)
        sr = row["audio"]["sampling_rate"]
        if sr != self.sr:
            import librosa
            arr = librosa.resample(arr, orig_sr=sr, target_sr=self.sr)
        if arr.size > self.target_len:
            start = np.random.randint(0, arr.size - self.target_len)
            arr = arr[start:start + self.target_len]
        else:
            arr = np.concatenate([arr, np.zeros(self.target_len - arr.size, dtype=np.float32)])
        label = int(row["label"])
        arr = apply_waveform_augs(arr, self.sr, is_drone=(label == 1),
                                   noise_clips=None, rir_clips=None,
                                   rng=self._rng, cfg=self.aug_cfg)
        feats = self.fe(arr, sampling_rate=self.sr, return_tensors="pt")
        x = apply_spec_augs(feats["input_values"].squeeze(0), self._rng, self.aug_cfg)
        return {"input_values": x, "label": torch.tensor(label, dtype=torch.long)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=ROOT / "runs/20260429-112104/best_abd.pt")
    p.add_argument("--out", type=Path, default=ROOT / "runs/20260429-112104/best_rw.pt")
    p.add_argument("--pos-wav", action="append", default=[], type=Path,
                   help="positive (drone) WAV file. Repeat for multiple.")
    p.add_argument("--force-positive-wav", action="append", default=[], type=Path,
                   help="WAV where every window is treated as drone (no pseudo-labeling)")
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--threshold", type=float, default=0.10,
                   help="pseudo-label threshold for --pos-wav")
    p.add_argument("--lr-backbone", type=float, default=5e-7)
    p.add_argument("--lr-head", type=float, default=2e-5)
    args = p.parse_args()

    if not args.pos_wav and not args.force_positive_wav:
        print("provide at least one --pos-wav or --force-positive-wav")
        return 1

    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu"))
    print(f"[device] {device}")

    model, fe = build_model()
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state"])
    model.to(device)

    all_clips = []
    for path in args.pos_wav:
        arr = load_wav_16k_mono(path)
        clips = pseudo_label_windows(arr, model, fe, device, args.threshold)
        print(f"[pos] {path.name}: {len(clips)} drone-like windows (threshold={args.threshold})")
        all_clips.extend(clips)

    for path in args.force_positive_wav:
        arr = load_wav_16k_mono(path)
        clips = windows_from_clip(arr)
        print(f"[force-pos] {path.name}: {len(clips)} windows (whole clip = drone)")
        all_clips.extend(clips)

    if not all_clips:
        print("no positive windows extracted")
        return 1
    pos_ds = PositiveClips(all_clips, fe)

    geron = load_dataset("geronimobasso/drone-audio-detection-samples",
                         cache_dir=str(CACHE_DIR))
    geron_split = geron["train"] if "train" in geron else geron[list(geron.keys())[0]]
    geron_ds = GeronimobassoSlice(geron_split, fe, n=2000)

    train_ds = ConcatDataset([pos_ds, geron_ds])

    geron_labels = np.asarray([int(geron_split[int(idx)]["label"])
                               for idx in geron_ds.indices])
    labels = np.concatenate([np.ones(len(pos_ds), dtype=int), geron_labels])
    counts = np.bincount(labels)
    inv_freq = 1.0 / np.maximum(counts, 1)
    weights = inv_freq[labels]
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=args.steps * args.batch_size, replacement=True,
    )
    loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                       num_workers=2, persistent_workers=True)

    cw = torch.tensor(inv_freq, dtype=torch.float32, device=device)
    cw = cw / cw.sum() * len(cw)
    criterion = FocalLoss(gamma=2.0, class_weights=cw)

    pg = [
        {"params": [pp for n, pp in model.named_parameters()
                    if "classifier" not in n and pp.requires_grad],
         "lr": args.lr_backbone},
        {"params": [pp for n, pp in model.named_parameters()
                    if "classifier" in n and pp.requires_grad],
         "lr": args.lr_head},
    ]
    optimizer = AdamW(pg, weight_decay=1e-4)

    def lr_lambda(s):
        warmup = 60
        if s < warmup:
            return s / warmup
        progress = (s - warmup) / max(1, args.steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = LambdaLR(optimizer, lr_lambda)

    autocast_dtype = torch.bfloat16
    model.train()
    t0 = time.time()
    running = 0.0
    n_seen = 0
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
            print(f"[step {step:>4}/{args.steps}] loss={running / max(1, n_seen):.4f} "
                  f"elapsed={(time.time() - t0) / 60:.1f}m", flush=True)

    torch.save({
        "model_state": model.state_dict(),
        "config": CFG.__dict__,
        "step": step,
        "source_ckpt": str(args.ckpt),
    }, args.out)
    print(f"[done] saved -> {args.out}, elapsed {(time.time() - t0) / 60:.1f}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
