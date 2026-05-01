"""
Train AST binary classifier on geronimobasso, with INTRA-EPOCH validation,
frequent checkpoints, early stopping, and prediction-distribution monitoring.

Best practices implemented:
  - validate every CFG.eval_every_steps (default 1000) — never go blind for hours
  - save latest.pt every CFG.save_every_steps with full optimizer + scheduler state
  - early stop if val F1 doesn't improve for CFG.early_stop_patience validations
  - print prediction class distribution alongside loss to catch majority-class collapse

Run:
    uv run python -m src.train                       # fresh
    uv run python -m src.train --auto-resume         # continue latest run
    uv run python -m src.train --resume runs/<ts>/latest.pt
"""

from __future__ import annotations
import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset

from src.config import CFG, RUNS_DIR
from src.data import make_splits, make_loaders, CroppedDroneAudio
from src.model import build_model, param_groups
from src.losses import FocalLoss
from src.metrics import report_at_threshold, threshold_free


# --------------------------- helpers ---------------------------------------- #


def make_scheduler(optimizer, total_steps: int) -> LambdaLR:
    warmup_steps = max(1, int(total_steps * CFG.warmup_frac))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def device_or_fallback() -> torch.device:
    if CFG.device == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_scores, all_labels = [], []
    for batch in loader:
        x = batch["input_values"].to(device, non_blocking=True)
        y = batch["label"]
        out = model(input_values=x)
        prob_drone = torch.softmax(out.logits, dim=-1)[:, 1].float().cpu().numpy()
        all_scores.append(prob_drone)
        all_labels.append(y.numpy())
    model.train()
    return np.concatenate(all_labels), np.concatenate(all_scores)


def make_fast_val_loader(val_loader: DataLoader, n: int) -> DataLoader:
    """Random subset of the val set for quick intra-epoch checks."""
    full = val_loader.dataset
    if n >= len(full):
        return val_loader
    idx = np.random.RandomState(CFG.seed).choice(len(full), size=n, replace=False)
    sub = Subset(full, idx.tolist())
    return DataLoader(
        sub,
        batch_size=val_loader.batch_size,
        shuffle=False,
        num_workers=val_loader.num_workers,
        persistent_workers=val_loader.num_workers > 0,
    )


def find_latest_run() -> Path | None:
    if not RUNS_DIR.exists():
        return None
    candidates = sorted(
        (d for d in RUNS_DIR.iterdir() if d.is_dir() and (d / "latest.pt").exists()),
        key=lambda p: p.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


# --------------------------- main ------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--auto-resume", action="store_true")
    args = parser.parse_args()

    device = device_or_fallback()
    print(f"[device] {device}", flush=True)

    print("[data]  loading splits…", flush=True)
    splits = make_splits()
    print(f"[data]  train={len(splits.train):,}  val={len(splits.val):,}  "
          f"test={len(splits.test):,}", flush=True)

    print("[model] building AST + 2-way head…", flush=True)
    model, fe = build_model()
    model.to(device)

    loaders = make_loaders(splits, fe)
    fast_val_loader = make_fast_val_loader(loaders["val"], CFG.val_subsample)

    train_labels = np.asarray(splits.train["label"])
    counts = np.bincount(train_labels)
    cw = torch.tensor(1.0 / np.maximum(counts, 1), dtype=torch.float32, device=device)
    cw = cw / cw.sum() * len(cw)

    if CFG.use_focal_loss:
        criterion = FocalLoss(gamma=CFG.focal_gamma, class_weights=cw,
                              label_smoothing=CFG.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=CFG.label_smoothing)

    optimizer = AdamW(param_groups(model), weight_decay=CFG.weight_decay)
    total_steps = len(loaders["train"]) * CFG.epochs
    scheduler = make_scheduler(optimizer, total_steps)

    # ---- resume state ----
    start_epoch = 1
    global_step = 0
    best_f1 = -1.0
    history: list[dict] = []
    no_improve_count = 0

    resume_path: Path | None = args.resume
    if resume_path is None and args.auto_resume:
        latest_run = find_latest_run()
        if latest_run is not None:
            resume_path = latest_run / "latest.pt"
            print(f"[resume] auto-resuming from {resume_path}", flush=True)

    if resume_path is not None and resume_path.exists():
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        scheduler.load_state_dict(ckpt["sched_state"])
        start_epoch = int(ckpt["epoch"])         # restart current epoch from beginning
        global_step = int(ckpt.get("global_step", 0))
        best_f1 = float(ckpt.get("best_f1", -1.0))
        history = list(ckpt.get("history", []))
        no_improve_count = int(ckpt.get("no_improve_count", 0))
        run_dir = resume_path.parent
        print(f"[resume] epoch={start_epoch} global_step={global_step} best_f1={best_f1:.4f}",
              flush=True)
    else:
        run_dir = RUNS_DIR / time.strftime("%Y%m%d-%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(
            json.dumps(CFG.__dict__, indent=2, default=str)
        )

    print(f"[run]   {run_dir}", flush=True)
    autocast_dtype = torch.bfloat16 if CFG.bf16 else torch.float32

    if start_epoch > CFG.epochs:
        print(f"[done]  already past final epoch — nothing to do.", flush=True)
        return 0

    early_stop = False

    def save_latest():
        torch.save({
            "epoch": epoch,
            "global_step": global_step,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "sched_state": scheduler.state_dict(),
            "config": CFG.__dict__,
            "best_f1": best_f1,
            "history": history,
            "no_improve_count": no_improve_count,
        }, run_dir / "latest.pt")

    def save_best(thr: float):
        torch.save({
            "epoch": epoch,
            "global_step": global_step,
            "model_state": model.state_dict(),
            "config": CFG.__dict__,
            "val_f1": best_f1,
            "val_thr": thr,
        }, run_dir / "best.pt")

    def run_validation(label: str, loader_to_use: DataLoader, full: bool) -> dict:
        nonlocal best_f1, no_improve_count, early_stop
        t0 = time.time()
        yv, sv = evaluate(model, loader_to_use, device)
        tf = threshold_free(yv, sv)
        rep = max(
            (report_at_threshold(yv, sv, t) for t in CFG.threshold_sweep),
            key=lambda r: r.f1,
        )
        # majority-class baseline: F1 of always predicting class 1
        majority_f1 = report_at_threshold(yv, np.ones_like(sv), 0.5).f1
        # prediction class distribution at the chosen threshold
        preds = (sv >= rep.threshold).astype(int)
        pred_counts = Counter(preds.tolist())
        true_counts = Counter(yv.tolist())
        eta = time.time() - t0
        print(
            f"[val:{label}] step={global_step} pr_auc={tf['pr_auc']:.4f} "
            f"roc_auc={tf['roc_auc']:.4f} F1={rep.f1:.4f} (best={best_f1:.4f}, "
            f"majority={majority_f1:.4f}) "
            f"thr={rep.threshold:.2f} P={rep.precision:.4f} R={rep.recall:.4f}",
            flush=True,
        )
        print(
            f"[val:{label}] preds: 0={pred_counts.get(0,0):,} 1={pred_counts.get(1,0):,}  "
            f"truth: 0={true_counts.get(0,0):,} 1={true_counts.get(1,0):,}  "
            f"eval_secs={eta:.1f}",
            flush=True,
        )

        improved = rep.f1 > best_f1 + CFG.early_stop_min_delta
        if improved:
            best_f1 = rep.f1
            save_best(rep.threshold)
            print(f"[ckpt]  new best F1={best_f1:.4f} (full_val={full}) -> {run_dir/'best.pt'}",
                  flush=True)
            no_improve_count = 0
        else:
            no_improve_count += 1
            print(f"[earlystop] no_improve={no_improve_count}/{CFG.early_stop_patience}",
                  flush=True)
            if no_improve_count >= CFG.early_stop_patience:
                print(f"[earlystop] STOPPING — F1 plateaued for {no_improve_count} validations",
                      flush=True)
                early_stop = True
        return {"step": global_step, "f1": rep.f1, "pr_auc": tf["pr_auc"],
                "roc_auc": tf["roc_auc"], "majority_f1": majority_f1, "full": full}

    # ---- training loop ----
    running_loss = 0.0
    running_n = 0
    pred_dist = Counter()
    last_log_step = global_step

    for epoch in range(start_epoch, CFG.epochs + 1):
        model.train()
        epoch_t0 = time.time()

        for batch in loaders["train"]:
            x = batch["input_values"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=CFG.bf16):
                out = model(input_values=x)
                loss = criterion(out.logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
            optimizer.step()
            scheduler.step()
            global_step += 1

            running_loss += float(loss.item()) * y.size(0)
            running_n += y.size(0)
            with torch.no_grad():
                preds = out.logits.argmax(-1).cpu().tolist()
            pred_dist.update(preds)

            if global_step % 50 == 0:
                avg = running_loss / running_n
                lr0 = optimizer.param_groups[0]["lr"]
                tot = sum(pred_dist.values())
                pred_share = {k: f"{v/tot:.2%}" for k, v in pred_dist.items()}
                print(
                    f"[train] ep{epoch} step={global_step} loss={avg:.4f} "
                    f"lr={lr0:.2e} pred_share={pred_share}",
                    flush=True,
                )

            # periodic intra-epoch val
            if global_step % CFG.eval_every_steps == 0:
                run_validation(f"fast", fast_val_loader, full=False)
                history.append({"step": global_step, "loss": running_loss / max(1, running_n)})
                (run_dir / "history.json").write_text(json.dumps(history, indent=2))
                running_loss = 0.0
                running_n = 0
                pred_dist = Counter()

            if global_step % CFG.save_every_steps == 0:
                save_latest()

            if early_stop:
                break

        if early_stop:
            print("[earlystop] breaking outer loop", flush=True)
            break

        # full-val at end of epoch
        print(f"[epoch] {epoch} done in {(time.time() - epoch_t0)/3600:.2f} h", flush=True)
        run_validation("full", loaders["val"], full=True)
        save_latest()

    # ---- final test ----
    print("\n[final] running on full test split…", flush=True)
    yt, st = evaluate(model, loaders["test"], device)
    tf = threshold_free(yt, st)
    print(f"[test]  pr_auc={tf['pr_auc']:.4f}  roc_auc={tf['roc_auc']:.4f}", flush=True)
    test_results = {"pr_auc": tf["pr_auc"], "roc_auc": tf["roc_auc"], "thresholds": []}
    for t in CFG.threshold_sweep:
        rep = report_at_threshold(yt, st, t)
        print(rep.pretty(), flush=True)
        test_results["thresholds"].append({
            "threshold": t,
            "f1": rep.f1, "precision": rep.precision, "recall": rep.recall,
            "accuracy": rep.accuracy,
            "confusion": rep.confusion.tolist(),
        })
    (run_dir / "test_results.json").write_text(json.dumps(test_results, indent=2))
    print(f"[done]  test results -> {run_dir/'test_results.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
