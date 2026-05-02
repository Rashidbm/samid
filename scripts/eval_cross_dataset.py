"""
Honest cross-dataset evaluation per reviewer protocol.

Evaluates the model on:
  1. NUS DroneAudioSet held-out splits (21..28) — in-domain held-out
  2. DroneNoise Database (Salford 2024) — truly never seen during training

Uses median filter + consecutive-window aggregation per reviewer.
Reports per-dataset metrics independently. NO aggregation across datasets.
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from datasets import load_dataset
from rich.console import Console
from rich.table import Table

from src.config import CFG
from src.model import build_model
from src.inference import aggregate_verdict, median_filter

ROOT = Path(__file__).resolve().parent.parent
console = Console()


def device_or_fallback() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resample_to_16k(arr: np.ndarray, sr: int) -> np.ndarray:
    if sr == 16_000:
        return arr.astype(np.float32, copy=False)
    try:
        import librosa
        return librosa.resample(arr.astype(np.float32), orig_sr=sr, target_sr=16_000)
    except ImportError:
        new_len = int(arr.size * 16_000 / sr)
        return np.interp(
            np.linspace(0, arr.size, new_len, endpoint=False),
            np.arange(arr.size), arr
        ).astype(np.float32)


@torch.no_grad()
def windowed_probs(model, fe, dev, arr: np.ndarray, sr: int = 16_000,
                   hop_sec: float = 0.5, win_sec: float = 1.0) -> np.ndarray:
    win = int(win_sec * sr)
    hop = int(hop_sec * sr)
    if arr.size < win:
        arr = np.concatenate([arr, np.zeros(win - arr.size, dtype=np.float32)])
    probs = []
    for start in range(0, arr.size - win + 1, hop):
        chunk = arr[start:start + win]
        feats = fe(chunk, sampling_rate=sr, return_tensors="pt")
        x = feats["input_values"].to(dev)
        out = model(input_values=x)
        probs.append(float(F.softmax(out.logits, -1)[0, 1]))
    return np.asarray(probs, dtype=np.float32)


def evaluate_clip(model, fe, dev, arr: np.ndarray, sr: int) -> dict:
    probs = windowed_probs(model, fe, dev, arr, sr=sr)
    v = aggregate_verdict(probs, threshold=0.5, median_kernel=5,
                          consecutive_required=3)
    return {
        "n_windows": int(probs.size),
        "raw_max": float(probs.max()) if probs.size else 0.0,
        "smoothed_max": v.smoothed_max,
        "smoothed_median": v.smoothed_median,
        "longest_run": v.longest_run,
        "decision": v.decision,
    }


# --------------------------- evaluation drivers ----------------------------- #


def eval_nus_held_out(model, fe, dev) -> dict:
    """Evaluate on NUS splits 21..28 (not used in training)."""
    console.rule("[bold cyan]NUS DroneAudioSet — held-out splits 21..28")
    results = []
    for k in range(21, 29):
        try:
            ds = load_dataset(
                "ahlab-drone-project/DroneAudioSet", "drone-only",
                split=f"train_{k:03d}",
                cache_dir=str(ROOT / "data" / "ahlab"),
            )
        except Exception as exc:
            console.print(f"[yellow]split {k} failed: {exc}[/yellow]")
            continue
        for s in ds:
            arr = np.asarray(s["audio"]["array"], dtype=np.float32)
            if arr.ndim > 1:
                arr = arr[:, 0]
            sr = s["audio"]["sampling_rate"]
            arr = resample_to_16k(arr, sr)
            r = evaluate_clip(model, fe, dev, arr, 16_000)
            r["file"] = s.get("file_path", f"split{k}")
            results.append(r)

    table = Table(title=f"NUS held-out (n={len(results)})")
    table.add_column("file", overflow="ellipsis", max_width=50)
    table.add_column("decision", justify="center")
    table.add_column("smooth max", justify="right")
    table.add_column("smooth median", justify="right")
    table.add_column("longest run", justify="right")
    n_drone = 0
    for r in results:
        decision_color = {"drone": "green", "uncertain": "yellow",
                          "no_drone": "red"}.get(r["decision"], "white")
        if r["decision"] == "drone":
            n_drone += 1
        table.add_row(
            str(r["file"])[-50:], f"[{decision_color}]{r['decision']}[/]",
            f"{r['smoothed_max']:.3f}", f"{r['smoothed_median']:.3f}",
            str(r["longest_run"]),
        )
    console.print(table)
    n = len(results)
    console.print(f"\n[bold]NUS held-out drone detection rate: {n_drone}/{n} "
                  f"({100*n_drone/max(1,n):.1f}%)[/bold]\n")
    return {"n": n, "detected": n_drone, "results": results}


def eval_dronenoise(model, fe, dev) -> dict:
    """Evaluate on DroneNoise Database — TRUE held-out cross-dataset test."""
    console.rule("[bold cyan]DroneNoise Database (Salford 2024) — TRUE OOD")
    folder = ROOT / "data" / "dronenoise"
    files = sorted(folder.glob("Ed_*_M1.wav"))
    if not files:
        console.print(f"[yellow]No files in {folder}[/yellow]")
        return {"n": 0, "detected": 0, "results": []}

    results = []
    for f in files:
        try:
            arr, sr = sf.read(str(f), dtype="float32", always_2d=False)
            if arr.ndim > 1:
                arr = arr.mean(axis=1)
            arr = resample_to_16k(arr, sr)
            r = evaluate_clip(model, fe, dev, arr, 16_000)
            r["file"] = f.name
            results.append(r)
        except Exception as exc:
            console.print(f"[yellow]{f.name}: {exc}[/yellow]")

    table = Table(title=f"DroneNoise (n={len(results)})")
    table.add_column("file", overflow="ellipsis", max_width=50)
    table.add_column("decision", justify="center")
    table.add_column("smooth max", justify="right")
    table.add_column("smooth median", justify="right")
    table.add_column("longest run", justify="right")
    n_drone = 0
    for r in results:
        decision_color = {"drone": "green", "uncertain": "yellow",
                          "no_drone": "red"}.get(r["decision"], "white")
        if r["decision"] == "drone":
            n_drone += 1
        table.add_row(
            str(r["file"])[-50:], f"[{decision_color}]{r['decision']}[/]",
            f"{r['smoothed_max']:.3f}", f"{r['smoothed_median']:.3f}",
            str(r["longest_run"]),
        )
    console.print(table)
    n = len(results)
    console.print(f"\n[bold]DroneNoise detection rate: {n_drone}/{n} "
                  f"({100*n_drone/max(1,n):.1f}%)[/bold]\n")
    return {"n": n, "detected": n_drone, "results": results}


# --------------------------- main ------------------------------------------- #


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=None)
    args = p.parse_args()

    if args.ckpt is None:
        # default: prefer v2 if it exists, otherwise best.pt
        candidates = [
            ROOT / "runs/20260429-112104/best_v2.pt",
            ROOT / "runs/20260429-112104/best_v2.final.pt",
            ROOT / "runs/20260429-112104/best.pt",
        ]
        for c in candidates:
            if c.exists():
                args.ckpt = c
                break
    console.print(f"[bold]Loading checkpoint:[/bold] {args.ckpt}")

    dev = device_or_fallback()
    console.print(f"[bold]Device:[/bold] {dev}\n")

    model, fe = build_model()
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state"])
    model.to(dev).eval()

    nus_res = eval_nus_held_out(model, fe, dev)
    dn_res = eval_dronenoise(model, fe, dev)

    console.rule("[bold cyan]Summary")
    console.print(f"NUS held-out (in-domain held-out): "
                  f"{nus_res['detected']}/{nus_res['n']} "
                  f"({100*nus_res['detected']/max(1,nus_res['n']):.1f}%)")
    console.print(f"DroneNoise (TRUE OOD):              "
                  f"{dn_res['detected']}/{dn_res['n']} "
                  f"({100*dn_res['detected']/max(1,dn_res['n']):.1f}%)")
    console.print()
    console.print("[dim]Decision protocol: median filter k=5 + 3 consecutive "
                  "windows above threshold 0.5[/dim]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
