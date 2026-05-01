"""
Data loading for geronimobasso/drone-audio-detection-samples.

Key job: kill the duration leak.
Drone clips average 0.6s, no-drone clips average 7.3s. A model can win on
length alone. Fix: every example becomes a 1.0-second window.
  - Longer than 1s -> random crop (training) / center crop (eval)
  - Shorter than 1s -> zero-pad to 1s
"""

from __future__ import annotations
from dataclasses import dataclass
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from datasets import load_dataset, Dataset as HFDataset
from sklearn.model_selection import train_test_split

from src.config import CFG, CACHE_DIR


# ----------------------------- core dataset --------------------------------- #


class CroppedDroneAudio(Dataset):
    """
    Wraps a HuggingFace audio dataset row and emits fixed-length 1-second
    waveforms at 16 kHz.
    """

    def __init__(
        self,
        hf_split: HFDataset,
        feature_extractor,
        train_mode: bool,
        clip_seconds: float = CFG.clip_seconds,
        sample_rate: int = CFG.sample_rate,
        seed: int = CFG.seed,
    ) -> None:
        self.hf = hf_split
        self.fe = feature_extractor
        self.train_mode = train_mode
        self.target_len = int(clip_seconds * sample_rate)
        self.sample_rate = sample_rate
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.hf)

    def _crop_or_pad(self, arr: np.ndarray) -> np.ndarray:
        n = arr.shape[0]
        T = self.target_len
        if n == T:
            return arr.astype(np.float32, copy=False)
        if n > T:
            if self.train_mode:
                start = self.rng.randint(0, n - T)
            else:
                start = (n - T) // 2  # center crop, deterministic
            return arr[start : start + T].astype(np.float32, copy=False)
        # n < T -> zero-pad
        pad = np.zeros(T - n, dtype=np.float32)
        return np.concatenate([arr.astype(np.float32, copy=False), pad])

    def __getitem__(self, idx: int) -> dict:
        row = self.hf[idx]
        audio = row["audio"]
        arr = np.asarray(audio["array"], dtype=np.float32)
        sr = int(audio["sampling_rate"])
        if sr != self.sample_rate:
            # geronimobasso is already 16 kHz; this is just a safety net
            import librosa  # lazy import
            arr = librosa.resample(arr, orig_sr=sr, target_sr=self.sample_rate)

        arr = self._crop_or_pad(arr)

        # AST feature extractor: AST's positional embeddings expect a fixed
        # spectrogram length (~10s = 1024 frames). Letting the feature
        # extractor pad our 1s clip to that default keeps the architecture
        # consistent without us touching positional embeddings.
        feats = self.fe(
            arr,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        )
        input_values = feats["input_values"].squeeze(0)  # (n_mels, T_max)

        return {
            "input_values": input_values,
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
        }


# ----------------------------- splitting ------------------------------------ #


@dataclass
class Splits:
    train: HFDataset
    val: HFDataset
    test: HFDataset


def make_splits(
    val_frac: float = CFG.val_frac,
    test_frac: float = CFG.test_frac,
    seed: int = CFG.seed,
) -> Splits:
    """
    Stratified split by label. Group-by-source would be ideal but the
    dataset doesn't expose source-file metadata in a clean column, so we
    rely on the heavy class imbalance + duration normalization to mitigate
    leakage rather than enforce it structurally.
    """
    raw = load_dataset(
        "geronimobasso/drone-audio-detection-samples",
        cache_dir=str(CACHE_DIR),
    )
    full = raw["train"] if "train" in raw else raw[list(raw.keys())[0]]

    labels = np.asarray(full["label"])
    idx = np.arange(len(full))

    # First carve test
    train_val_idx, test_idx = train_test_split(
        idx, test_size=test_frac, stratify=labels, random_state=seed
    )
    # Then carve val from remaining
    val_size = val_frac / (1.0 - test_frac)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size,
        stratify=labels[train_val_idx],
        random_state=seed,
    )

    return Splits(
        train=full.select(train_idx),
        val=full.select(val_idx),
        test=full.select(test_idx),
    )


# ----------------------------- loaders -------------------------------------- #


def make_weighted_sampler(hf_split: HFDataset) -> WeightedRandomSampler:
    labels = np.asarray(hf_split["label"])
    counts = np.bincount(labels)
    inv_freq = 1.0 / np.maximum(counts, 1)
    sample_weights = inv_freq[labels]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(labels),
        replacement=True,
    )


def make_loaders(splits: Splits, feature_extractor) -> dict[str, DataLoader]:
    train_ds = CroppedDroneAudio(splits.train, feature_extractor, train_mode=True)
    val_ds = CroppedDroneAudio(splits.val, feature_extractor, train_mode=False)
    test_ds = CroppedDroneAudio(splits.test, feature_extractor, train_mode=False)

    train_sampler = (
        make_weighted_sampler(splits.train) if CFG.use_weighted_sampler else None
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=CFG.batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=CFG.num_workers,
        pin_memory=False,            # MPS doesn't benefit
        persistent_workers=CFG.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        persistent_workers=CFG.num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        persistent_workers=CFG.num_workers > 0,
    )
    return {"train": train_loader, "val": val_loader, "test": test_loader}
