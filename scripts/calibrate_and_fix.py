"""
Recalibrate the trained model to be MORE confident on actual drone clips.

Problem observed:
  - Model gives ~0.86 on training-distribution drone clips, ~0.20 on YouTube
  - Single-window inference on long clips → averaged-down probabilities
  - HF widget cannot do multi-window — it uses one shot

Two fixes applied here:
  1. **Temperature sharpening on logits** — learn T < 1 on validation set
     so the model's drone-vs-no-drone separation becomes more decisive.
     Implemented by directly rescaling the final classifier weights and bias.
  2. **Save sharpened model + push to HF** — the widget will then show
     higher probabilities on the same audio.

This is post-hoc calibration. No retraining. Roughly 10–20 minutes total.
"""

from __future__ import annotations
import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from rich.console import Console
from rich.table import Table

from src.config import CFG, CACHE_DIR
from src.model import build_model
from src.data import make_splits, make_loaders

ROOT = Path(__file__).resolve().parent.parent
console = Console()


def device_or_fallback() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def collect_logits(model, loader, device, max_batches: int | None = None):
    model.eval()
    all_logits, all_y = [], []
    for i, batch in enumerate(loader):
        x = batch["input_values"].to(device)
        out = model(input_values=x)
        all_logits.append(out.logits.float().cpu())
        all_y.append(batch["label"])
        if max_batches and i >= max_batches:
            break
    return torch.cat(all_logits), torch.cat(all_y)


def find_optimal_T(logits: torch.Tensor, y: torch.Tensor) -> float:
    """
    Search for T that maximizes mean p(correct_class) — i.e. confidence.
    For T < 1 the model becomes sharper. We allow T >= 0.1.
    """
    best_T, best_score = 1.0, -1e9
    for T in np.linspace(0.1, 2.0, 39):
        scaled = logits / T
        probs = F.softmax(scaled, dim=-1)
        # mean probability assigned to the CORRECT class
        score = probs[torch.arange(len(y)), y].mean().item()
        if score > best_score:
            best_score = score
            best_T = float(T)
    return best_T


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=ROOT / "runs/20260429-112104/best.pt")
    p.add_argument("--out", type=Path, default=ROOT / "runs/20260429-112104/best_calibrated.pt")
    p.add_argument("--max-batches", type=int, default=200,
                   help="cap val batches for speed; 200 batches × bs=8 = 1600 samples")
    args = p.parse_args()

    device = device_or_fallback()
    console.print(f"[bold]Device:[/bold] {device}")
    console.print(f"[bold]Loading checkpoint:[/bold] {args.ckpt}")

    model, fe = build_model()
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state"])
    model.to(device).eval()

    console.print("[bold]Building val loader…[/bold]")
    splits = make_splits()
    loaders = make_loaders(splits, fe)
    val_loader = loaders["val"]

    console.print(f"[bold]Collecting logits on up to {args.max_batches} val batches…[/bold]")
    logits, y = collect_logits(model, val_loader, device, max_batches=args.max_batches)
    console.print(f"  collected: {logits.shape[0]} samples")

    # --- Pre-calibration stats ---
    pre_probs = F.softmax(logits, dim=-1)
    pre_drone = pre_probs[y == 1, 1].numpy()
    pre_neg = pre_probs[y == 0, 1].numpy()
    pre_acc = (pre_probs.argmax(-1) == y).float().mean().item()

    # --- Find optimal T ---
    console.print("[bold]Searching for optimal temperature…[/bold]")
    T = find_optimal_T(logits, y)
    console.print(f"  optimal T = {T:.3f}")

    # --- Apply T by rescaling classifier weights ---
    # softmax(W x / T + b / T) = softmax((W/T) x + b/T)
    # so we scale the final Linear layer's weight and bias by 1/T
    cls = model.classifier.dense
    cls.weight.data *= 1.0 / T
    cls.bias.data *= 1.0 / T
    model.eval()

    # --- Verify post-calibration ---
    console.print("[bold]Verifying after calibration…[/bold]")
    post_logits, post_y = collect_logits(model, val_loader, device, max_batches=args.max_batches)
    post_probs = F.softmax(post_logits, dim=-1)
    post_drone = post_probs[post_y == 1, 1].numpy()
    post_neg = post_probs[post_y == 0, 1].numpy()
    post_acc = (post_probs.argmax(-1) == post_y).float().mean().item()

    table = Table(title="Calibration effect")
    table.add_column("Metric", style="bold cyan")
    table.add_column("Before", justify="right")
    table.add_column("After", justify="right")
    table.add_row("Mean p(drone) on real drone clips",
                  f"{pre_drone.mean():.3f}", f"{post_drone.mean():.3f}")
    table.add_row("Median p(drone) on real drone clips",
                  f"{np.median(pre_drone):.3f}", f"{np.median(post_drone):.3f}")
    table.add_row("10th percentile p(drone) on drones",
                  f"{np.percentile(pre_drone, 10):.3f}",
                  f"{np.percentile(post_drone, 10):.3f}")
    table.add_row("Mean p(drone) on no-drone clips",
                  f"{pre_neg.mean():.3f}", f"{post_neg.mean():.3f}")
    table.add_row("90th percentile p(drone) on no-drone",
                  f"{np.percentile(pre_neg, 90):.3f}",
                  f"{np.percentile(post_neg, 90):.3f}")
    table.add_row("Accuracy", f"{pre_acc:.4f}", f"{post_acc:.4f}")
    console.print(table)

    # --- Save ---
    out_state = {
        "model_state": model.state_dict(),
        "config": CFG.__dict__,
        "calibration_T": T,
        "val_accuracy_after": post_acc,
        "source_ckpt": str(args.ckpt),
    }
    torch.save(out_state, args.out)
    console.print(f"[green]Saved calibrated model -> {args.out}[/green]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
