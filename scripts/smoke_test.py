"""
30-second smoke test. Run this BEFORE the full training to catch setup bugs.

Loads ONE batch through the entire pipeline:
  dataset -> feature extractor -> AST -> focal loss -> backward -> step

If this exits cleanly, the multi-hour training will likely run too.

    uv run python -m scripts.smoke_test
"""

from __future__ import annotations
import time

import torch
from rich.console import Console

from src.config import CFG
from src.data import make_splits, make_loaders
from src.losses import FocalLoss
from src.model import build_model

console = Console()


def main() -> None:
    if CFG.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    console.print(f"[bold]device:[/bold] {device}")

    t0 = time.time()
    console.print("[bold]splits…[/bold]")
    splits = make_splits()
    console.print(f"  train={len(splits.train):,}  val={len(splits.val):,}  test={len(splits.test):,}")

    console.print("[bold]model…[/bold] (this triggers a one-time AST weight download ~340 MB)")
    model, fe = build_model()
    model.to(device).train()

    loaders = make_loaders(splits, fe)
    train_loader = loaders["train"]
    console.print(f"  steps/epoch = {len(train_loader):,}")

    console.print("[bold]one batch…[/bold]")
    it = iter(train_loader)
    batch = next(it)
    x = batch["input_values"].to(device)
    y = batch["label"].to(device)
    console.print(f"  x={tuple(x.shape)}  y={tuple(y.shape)}  y.unique={y.unique().tolist()}")

    out = model(input_values=x)
    console.print(f"  logits={tuple(out.logits.shape)}")

    loss_fn = FocalLoss(gamma=CFG.focal_gamma)
    loss = loss_fn(out.logits, y)
    console.print(f"  loss={float(loss):.4f}")

    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum().item() > 0
                   for p in model.parameters())
    console.print(f"  gradients flowed: {has_grad}")

    console.print(f"\n[green]OK — smoke test passed in {time.time() - t0:.1f}s[/green]")


if __name__ == "__main__":
    main()
