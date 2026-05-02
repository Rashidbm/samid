from __future__ import annotations
import argparse
import math
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler

from datasets import load_dataset

from src.config import CFG, CACHE_DIR
from src.model import build_model
from src.losses import FocalLoss


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


class CachedFeatureDataset(Dataset):
    """Pre-extracts features once on construction. No augmentation, no per-step CPU work."""
    def __init__(self, features, labels):
        self.features = features  # tensor (N, F, T)
        self.labels = labels      # tensor (N,)

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, i):
        return {"input_values": self.features[i], "label": self.labels[i]}


def precompute_features(clips, labels, fe):
    feats_list = []
    for clip in clips:
        out = fe(clip, sampling_rate=16_000, return_tensors="pt")
        feats_list.append(out["input_values"].squeeze(0))
    return torch.stack(feats_list), torch.tensor(labels, dtype=torch.long)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=ROOT / "runs/20260429-112104/best_rw2.pt")
    p.add_argument("--out", type=Path, default=ROOT / "runs/20260429-112104/best_fast.pt")
    p.add_argument("--pos-wav", action="append", default=[], type=Path)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr-head", type=float, default=5e-4,
                   help="aggressive LR since only training a small head")
    p.add_argument("--neg-samples", type=int, default=400)
    args = p.parse_args()

    if not args.pos_wav:
        print("provide at least one --pos-wav")
        return 1

    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu"))
    print(f"[device] {device}")

    model, fe = build_model()
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state"])
    model.to(device)

    # FREEZE BACKBONE
    head_params = []
    for name, param in model.named_parameters():
        if "classifier" in name:
            param.requires_grad = True
            head_params.append(param)
        else:
            param.requires_grad = False
    n_trainable = sum(p.numel() for p in head_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[frozen] training {n_trainable:,} of {n_total:,} params ({100*n_trainable/n_total:.4f}%)")

    # Build positive set from new wavs (whole-clip-positive)
    pos_clips = []
    for path in args.pos_wav:
        arr = load_wav_16k_mono(path)
        clips = windows_from_clip(arr)
        print(f"[pos] {path.name}: {len(clips)} windows")
        pos_clips.extend(clips)

    # Build small neg set from geronimobasso label=0
    print(f"[neg] sampling {args.neg_samples} no-drone clips...")
    geron = load_dataset("geronimobasso/drone-audio-detection-samples",
                         cache_dir=str(CACHE_DIR))
    geron_split = geron["train"] if "train" in geron else geron[list(geron.keys())[0]]
    labels_arr = np.asarray(geron_split["label"])
    neg_idx = np.where(labels_arr == 0)[0]
    pos_idx = np.where(labels_arr == 1)[0]
    rng = np.random.RandomState(42)
    neg_choice = rng.choice(neg_idx, min(args.neg_samples, neg_idx.size), replace=False)
    pos_choice = rng.choice(pos_idx, min(args.neg_samples, pos_idx.size), replace=False)

    neg_clips, geron_pos_clips = [], []
    target = 16_000
    for i in neg_choice:
        a = np.asarray(geron_split[int(i)]["audio"]["array"], dtype=np.float32)
        sr = geron_split[int(i)]["audio"]["sampling_rate"]
        if sr != 16_000:
            import librosa
            a = librosa.resample(a, orig_sr=sr, target_sr=16_000)
        if a.size > target:
            s = (a.size - target) // 2
            a = a[s:s + target]
        else:
            a = np.concatenate([a, np.zeros(target - a.size, dtype=np.float32)])
        neg_clips.append(a)
    for i in pos_choice:
        a = np.asarray(geron_split[int(i)]["audio"]["array"], dtype=np.float32)
        sr = geron_split[int(i)]["audio"]["sampling_rate"]
        if sr != 16_000:
            import librosa
            a = librosa.resample(a, orig_sr=sr, target_sr=16_000)
        if a.size > target:
            s = (a.size - target) // 2
            a = a[s:s + target]
        else:
            a = np.concatenate([a, np.zeros(target - a.size, dtype=np.float32)])
        geron_pos_clips.append(a)

    print(f"[features] precomputing {len(pos_clips) + len(geron_pos_clips) + len(neg_clips)} feature tensors...")
    t_pre = time.time()
    all_clips = pos_clips + geron_pos_clips + neg_clips
    all_labels = ([1] * len(pos_clips)) + ([1] * len(geron_pos_clips)) + ([0] * len(neg_clips))
    features, labels = precompute_features(all_clips, all_labels, fe)
    print(f"[features] done in {time.time() - t_pre:.1f}s; shape={tuple(features.shape)}")

    train_ds = CachedFeatureDataset(features, labels)

    counts = np.bincount(labels.numpy())
    inv_freq = 1.0 / np.maximum(counts, 1)
    weights = inv_freq[labels.numpy()]
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=args.steps * args.batch_size, replacement=True,
    )
    loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                       num_workers=0)  # features are already in memory

    cw = torch.tensor(inv_freq, dtype=torch.float32, device=device)
    cw = cw / cw.sum() * len(cw)
    criterion = FocalLoss(gamma=2.0, class_weights=cw)

    optimizer = AdamW(head_params, lr=args.lr_head, weight_decay=1e-4)

    def lr_lambda(s):
        warmup = 30
        if s < warmup:
            return s / warmup
        progress = (s - warmup) / max(1, args.steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = LambdaLR(optimizer, lr_lambda)

    autocast_dtype = torch.bfloat16
    model.train()
    # Keep backbone in eval-equivalent mode (no dropout updates)
    for name, mod in model.named_modules():
        if "classifier" not in name:
            mod.eval()

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
        torch.nn.utils.clip_grad_norm_(head_params, 1.0)
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
