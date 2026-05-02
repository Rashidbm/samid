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
from src.data import make_splits, make_loaders
from src.model import build_model, param_groups
from src.losses import FocalLoss
from src.metrics import report_at_threshold, threshold_free


def make_scheduler(optimizer, total_steps):
    warmup = max(1, int(total_steps * CFG.warmup_frac))

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def device_or_fallback():
    if CFG.device == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    scores, labels = [], []
    for batch in loader:
        x = batch["input_values"].to(device, non_blocking=True)
        out = model(input_values=x)
        scores.append(torch.softmax(out.logits, dim=-1)[:, 1].float().cpu().numpy())
        labels.append(batch["label"].numpy())
    model.train()
    return np.concatenate(labels), np.concatenate(scores)


def fast_val_loader(val_loader, n):
    full = val_loader.dataset
    if n >= len(full):
        return val_loader
    idx = np.random.RandomState(CFG.seed).choice(len(full), size=n, replace=False)
    return DataLoader(
        Subset(full, idx.tolist()),
        batch_size=val_loader.batch_size,
        shuffle=False,
        num_workers=val_loader.num_workers,
        persistent_workers=val_loader.num_workers > 0,
    )


def find_latest_run():
    if not RUNS_DIR.exists():
        return None
    candidates = sorted(
        (d for d in RUNS_DIR.iterdir() if d.is_dir() and (d / "latest.pt").exists()),
        key=lambda p: p.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--auto-resume", action="store_true")
    args = parser.parse_args()

    device = device_or_fallback()
    print(f"[device] {device}", flush=True)

    splits = make_splits()
    print(f"[data] train={len(splits.train):,} val={len(splits.val):,} test={len(splits.test):,}", flush=True)

    model, fe = build_model()
    model.to(device)

    loaders = make_loaders(splits, fe)
    val_fast = fast_val_loader(loaders["val"], CFG.val_subsample)

    counts = np.bincount(np.asarray(splits.train["label"]))
    cw = torch.tensor(1.0 / np.maximum(counts, 1), dtype=torch.float32, device=device)
    cw = cw / cw.sum() * len(cw)

    criterion = (FocalLoss(gamma=CFG.focal_gamma, class_weights=cw,
                           label_smoothing=CFG.label_smoothing)
                 if CFG.use_focal_loss
                 else nn.CrossEntropyLoss(weight=cw, label_smoothing=CFG.label_smoothing))

    optimizer = AdamW(param_groups(model), weight_decay=CFG.weight_decay)
    total_steps = len(loaders["train"]) * CFG.epochs
    scheduler = make_scheduler(optimizer, total_steps)

    start_epoch = 1
    global_step = 0
    best_f1 = -1.0
    history = []
    no_improve = 0

    resume_path = args.resume
    if resume_path is None and args.auto_resume:
        latest = find_latest_run()
        if latest is not None:
            resume_path = latest / "latest.pt"

    if resume_path is not None and resume_path.exists():
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        scheduler.load_state_dict(ckpt["sched_state"])
        start_epoch = int(ckpt["epoch"])
        global_step = int(ckpt.get("global_step", 0))
        best_f1 = float(ckpt.get("best_f1", -1.0))
        history = list(ckpt.get("history", []))
        no_improve = int(ckpt.get("no_improve_count", 0))
        run_dir = resume_path.parent
        print(f"[resume] epoch={start_epoch} step={global_step} best_f1={best_f1:.4f}", flush=True)
    else:
        run_dir = RUNS_DIR / time.strftime("%Y%m%d-%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(json.dumps(CFG.__dict__, indent=2, default=str))

    print(f"[run] {run_dir}", flush=True)
    autocast_dtype = torch.bfloat16 if CFG.bf16 else torch.float32

    if start_epoch > CFG.epochs:
        return 0

    early_stop = False

    def save_latest():
        torch.save({
            "epoch": epoch, "global_step": global_step,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "sched_state": scheduler.state_dict(),
            "config": CFG.__dict__,
            "best_f1": best_f1, "history": history,
            "no_improve_count": no_improve,
        }, run_dir / "latest.pt")

    def save_best(thr):
        torch.save({
            "epoch": epoch, "global_step": global_step,
            "model_state": model.state_dict(),
            "config": CFG.__dict__,
            "val_f1": best_f1, "val_thr": thr,
        }, run_dir / "best.pt")

    def run_validation(label, loader_to_use):
        nonlocal best_f1, no_improve, early_stop
        yv, sv = evaluate(model, loader_to_use, device)
        tf = threshold_free(yv, sv)
        rep = max((report_at_threshold(yv, sv, t) for t in CFG.threshold_sweep),
                  key=lambda r: r.f1)
        majority = report_at_threshold(yv, np.ones_like(sv), 0.5).f1
        preds = Counter((sv >= rep.threshold).astype(int).tolist())
        truth = Counter(yv.tolist())
        print(f"[val:{label}] step={global_step} pr_auc={tf['pr_auc']:.4f} "
              f"F1={rep.f1:.4f} (best={best_f1:.4f}, majority={majority:.4f}) "
              f"thr={rep.threshold:.2f} P={rep.precision:.4f} R={rep.recall:.4f}",
              flush=True)
        print(f"[val:{label}] preds 0={preds.get(0,0):,} 1={preds.get(1,0):,}  "
              f"truth 0={truth.get(0,0):,} 1={truth.get(1,0):,}", flush=True)

        if rep.f1 > best_f1 + CFG.early_stop_min_delta:
            best_f1 = rep.f1
            save_best(rep.threshold)
            print(f"[ckpt] new best F1={best_f1:.4f}", flush=True)
            no_improve = 0
        else:
            no_improve += 1
            print(f"[earlystop] no_improve={no_improve}/{CFG.early_stop_patience}", flush=True)
            if no_improve >= CFG.early_stop_patience:
                early_stop = True

    running_loss = 0.0
    running_n = 0
    pred_dist = Counter()

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
                pred_dist.update(out.logits.argmax(-1).cpu().tolist())

            if global_step % 50 == 0:
                avg = running_loss / running_n
                lr0 = optimizer.param_groups[0]["lr"]
                tot = sum(pred_dist.values())
                share = {k: f"{v/tot:.2%}" for k, v in pred_dist.items()}
                print(f"[train] ep{epoch} step={global_step} loss={avg:.4f} "
                      f"lr={lr0:.2e} pred_share={share}", flush=True)

            if global_step % CFG.eval_every_steps == 0:
                run_validation("fast", val_fast)
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
            break

        print(f"[epoch] {epoch} done in {(time.time() - epoch_t0) / 3600:.2f} h", flush=True)
        run_validation("full", loaders["val"])
        save_latest()

    print("\n[final] test split…", flush=True)
    yt, st = evaluate(model, loaders["test"], device)
    tf = threshold_free(yt, st)
    print(f"[test] pr_auc={tf['pr_auc']:.4f} roc_auc={tf['roc_auc']:.4f}", flush=True)
    results = {"pr_auc": tf["pr_auc"], "roc_auc": tf["roc_auc"], "thresholds": []}
    for t in CFG.threshold_sweep:
        rep = report_at_threshold(yt, st, t)
        print(rep.pretty(), flush=True)
        results["thresholds"].append({
            "threshold": t, "f1": rep.f1,
            "precision": rep.precision, "recall": rep.recall,
            "accuracy": rep.accuracy, "confusion": rep.confusion.tolist(),
        })
    (run_dir / "test_results.json").write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
