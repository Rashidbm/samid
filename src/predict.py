"""
Single-clip inference for the trained AST detector.

Usage:
    uv run python -m src.predict --ckpt runs/<ts>/best.pt --wav path/to/clip.wav
    uv run python -m src.predict --ckpt runs/<ts>/best.pt --wav clip.wav --window-stride 0.5

For clips longer than 1.0 s we slide a 1-second window across the file and
emit per-window probabilities + an aggregate verdict.
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from rich.console import Console
from rich.table import Table

from src.config import CFG
from src.model import build_model

console = Console()


def load_audio(path: Path, target_sr: int = CFG.sample_rate) -> np.ndarray:
    arr, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)  # mono
    if sr != target_sr:
        import librosa
        arr = librosa.resample(arr, orig_sr=sr, target_sr=target_sr)
    return arr.astype(np.float32, copy=False)


def windows(arr: np.ndarray, win_len: int, stride: int) -> list[np.ndarray]:
    if arr.shape[0] <= win_len:
        pad = np.zeros(win_len - arr.shape[0], dtype=np.float32)
        return [np.concatenate([arr, pad])]
    out = []
    for start in range(0, arr.shape[0] - win_len + 1, stride):
        out.append(arr[start : start + win_len])
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--wav", type=Path, required=True)
    p.add_argument("--window-stride", type=float, default=0.5,
                   help="Stride in seconds between 1.0s windows for long clips.")
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args()

    if CFG.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    console.print(f"[bold]Device:[/bold] {device}")
    console.print(f"[bold]Checkpoint:[/bold] {args.ckpt}")
    console.print(f"[bold]Audio:[/bold] {args.wav}")

    model, fe = build_model()
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state"])
    model.eval().to(device)

    arr = load_audio(args.wav)
    win_len = int(CFG.clip_seconds * CFG.sample_rate)
    stride = max(1, int(args.window_stride * CFG.sample_rate))
    chunks = windows(arr, win_len, stride)

    feats = fe(chunks, sampling_rate=CFG.sample_rate, return_tensors="pt", padding=True)
    x = feats["input_values"].to(device)

    with torch.inference_mode():
        out = model(input_values=x)
        probs = torch.softmax(out.logits, dim=-1)[:, 1].cpu().numpy()

    table = Table(title="Per-window drone probability")
    table.add_column("Window", justify="right")
    table.add_column("t_start (s)", justify="right")
    table.add_column("p(drone)", justify="right")
    table.add_column("decision", justify="center")
    for i, p_drone in enumerate(probs):
        t = i * (stride / CFG.sample_rate)
        decision = "[red]DRONE[/red]" if p_drone >= args.threshold else "no drone"
        table.add_row(str(i), f"{t:.2f}", f"{p_drone:.4f}", decision)
    console.print(table)

    console.print(
        f"\n[bold]Aggregate:[/bold] "
        f"max p={probs.max():.4f}  mean p={probs.mean():.4f}  "
        f"frac_above_thr={(probs >= args.threshold).mean():.2%}"
    )
    verdict = "DRONE DETECTED" if probs.max() >= args.threshold else "clear"
    console.print(f"[bold]Verdict at thr={args.threshold}:[/bold] {verdict}")


if __name__ == "__main__":
    main()
