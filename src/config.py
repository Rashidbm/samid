from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "data" / "geronimobasso"
RUNS_DIR = ROOT / "runs"
RUNS_DIR.mkdir(exist_ok=True)


@dataclass
class TrainConfig:
    sample_rate: int = 16_000
    clip_seconds: float = 1.0
    val_frac: float = 0.10
    test_frac: float = 0.10
    seed: int = 42

    backbone: str = "MIT/ast-finetuned-audioset-10-10-0.4593"
    num_labels: int = 2

    batch_size: int = 8
    num_workers: int = 2
    epochs: int = 3
    lr_backbone: float = 1e-5
    lr_head: float = 1e-4
    weight_decay: float = 1e-4
    warmup_frac: float = 0.05
    grad_clip: float = 1.0
    use_focal_loss: bool = True
    focal_gamma: float = 2.0
    use_weighted_sampler: bool = True
    label_smoothing: float = 0.05

    device: str = "mps"
    bf16: bool = True

    eval_every_steps: int = 1000
    save_every_steps: int = 1000
    val_subsample: int = 4_000
    early_stop_patience: int = 3
    early_stop_min_delta: float = 1e-4
    threshold_sweep: tuple = (0.3, 0.4, 0.5, 0.6, 0.7)


CFG = TrainConfig()
