from __future__ import annotations
from dataclasses import dataclass
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from datasets import load_dataset, Dataset as HFDataset
from sklearn.model_selection import train_test_split

from src.config import CFG, CACHE_DIR


class CroppedDroneAudio(Dataset):
    def __init__(self, hf_split, feature_extractor, train_mode,
                 clip_seconds=CFG.clip_seconds, sample_rate=CFG.sample_rate,
                 seed=CFG.seed):
        self.hf = hf_split
        self.fe = feature_extractor
        self.train_mode = train_mode
        self.target_len = int(clip_seconds * sample_rate)
        self.sample_rate = sample_rate
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.hf)

    def _crop_or_pad(self, arr):
        n, T = arr.shape[0], self.target_len
        if n == T:
            return arr.astype(np.float32, copy=False)
        if n > T:
            start = self.rng.randint(0, n - T) if self.train_mode else (n - T) // 2
            return arr[start:start + T].astype(np.float32, copy=False)
        return np.concatenate([arr.astype(np.float32, copy=False),
                               np.zeros(T - n, dtype=np.float32)])

    def __getitem__(self, idx):
        row = self.hf[idx]
        arr = np.asarray(row["audio"]["array"], dtype=np.float32)
        sr = int(row["audio"]["sampling_rate"])
        if sr != self.sample_rate:
            import librosa
            arr = librosa.resample(arr, orig_sr=sr, target_sr=self.sample_rate)
        arr = self._crop_or_pad(arr)
        feats = self.fe(arr, sampling_rate=self.sample_rate, return_tensors="pt")
        return {
            "input_values": feats["input_values"].squeeze(0),
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
        }


@dataclass
class Splits:
    train: HFDataset
    val: HFDataset
    test: HFDataset


def make_splits(val_frac=CFG.val_frac, test_frac=CFG.test_frac, seed=CFG.seed):
    raw = load_dataset(
        "geronimobasso/drone-audio-detection-samples",
        cache_dir=str(CACHE_DIR),
    )
    full = raw["train"] if "train" in raw else raw[list(raw.keys())[0]]
    labels = np.asarray(full["label"])
    idx = np.arange(len(full))

    train_val_idx, test_idx = train_test_split(
        idx, test_size=test_frac, stratify=labels, random_state=seed
    )
    val_size = val_frac / (1.0 - test_frac)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_size,
        stratify=labels[train_val_idx], random_state=seed,
    )
    return Splits(
        train=full.select(train_idx),
        val=full.select(val_idx),
        test=full.select(test_idx),
    )


def make_weighted_sampler(hf_split):
    labels = np.asarray(hf_split["label"])
    counts = np.bincount(labels)
    inv_freq = 1.0 / np.maximum(counts, 1)
    weights = inv_freq[labels]
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(labels), replacement=True,
    )


def make_loaders(splits, fe):
    train_ds = CroppedDroneAudio(splits.train, fe, train_mode=True)
    val_ds = CroppedDroneAudio(splits.val, fe, train_mode=False)
    test_ds = CroppedDroneAudio(splits.test, fe, train_mode=False)

    sampler = make_weighted_sampler(splits.train) if CFG.use_weighted_sampler else None

    return {
        "train": DataLoader(train_ds, batch_size=CFG.batch_size, sampler=sampler,
                           shuffle=sampler is None, num_workers=CFG.num_workers,
                           persistent_workers=CFG.num_workers > 0),
        "val": DataLoader(val_ds, batch_size=CFG.batch_size, shuffle=False,
                         num_workers=CFG.num_workers,
                         persistent_workers=CFG.num_workers > 0),
        "test": DataLoader(test_ds, batch_size=CFG.batch_size, shuffle=False,
                          num_workers=CFG.num_workers,
                          persistent_workers=CFG.num_workers > 0),
    }
