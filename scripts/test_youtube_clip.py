"""
Test a video / audio file with proper multi-window aggregation.

The HF widget and basic inference take a single window. For long noisy clips
(YouTube videos, recordings with narration / wind / music) you need to slide
a window across the whole audio and look at the LOUDEST drone-like moments,
not the average.

Usage:
    uv run python -m scripts.test_youtube_clip --file path/to/video.mp4
    uv run python -m scripts.test_youtube_clip --file recording.wav --window 1.0 --hop 0.25
"""

from __future__ import annotations
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from rich.console import Console
from rich.table import Table

from src.config import CFG
from src.model import build_model

ROOT = Path(__file__).resolve().parent.parent
console = Console()


def load_audio_any(path: Path, target_sr: int = 16_000) -> tuple[np.ndarray, int]:
    """
    Load audio from any common format. Uses ffmpeg under the hood for video.
    Returns mono 16 kHz float32 numpy array.
    """
    suffix = path.suffix.lower()
    if suffix in {".wav", ".flac", ".ogg"}:
        arr, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
    else:
        # use ffmpeg to extract mono 16-kHz audio (handles mp4/mp3/m4a/etc.)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(path),
            "-ac", "1", "-ar", str(target_sr),
            tmp_path,
        ]
        subprocess.run(cmd, check=True)
        arr, sr = sf.read(tmp_path, dtype="float32", always_2d=False)
        if arr.ndim > 1:
            arr = arr.mean(axis=1)

    if sr != target_sr:
        import librosa
        arr = librosa.resample(arr, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return arr.astype(np.float32, copy=False), sr


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--file", type=Path, required=True)
    p.add_argument("--ckpt", type=Path,
                   default=ROOT / "runs/20260429-112104/best.pt")
    p.add_argument("--window", type=float, default=1.0,
                   help="window length in seconds")
    p.add_argument("--hop", type=float, default=0.25,
                   help="stride between consecutive windows in seconds")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--show-top", type=int, default=10,
                   help="show the top N windows by p(drone)")
    args = p.parse_args()

    console.print(f"[bold]Loading audio:[/bold] {args.file}")
    arr, sr = load_audio_any(args.file)
    duration = arr.size / sr
    console.print(f"  duration={duration:.2f}s  sr={sr}Hz")

    console.print(f"[bold]Loading model:[/bold] {args.ckpt}")
    model, fe = build_model()
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state"])
    model.eval()
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)

    win = int(args.window * sr)
    hop = int(args.hop * sr)
    if arr.size < win:
        arr = np.concatenate([arr, np.zeros(win - arr.size, dtype=np.float32)])

    console.print(f"[bold]Sliding {args.window}s windows every {args.hop}s…[/bold]")
    starts = list(range(0, arr.size - win + 1, hop))
    probs = np.zeros(len(starts), dtype=np.float32)
    times = np.zeros(len(starts), dtype=np.float32)

    with torch.inference_mode():
        for i, start in enumerate(starts):
            chunk = arr[start:start + win]
            feats = fe(chunk, sampling_rate=sr, return_tensors="pt")
            x = feats["input_values"].to(device)
            out = model(input_values=x)
            probs[i] = float(torch.softmax(out.logits, dim=-1)[0, 1])
            times[i] = start / sr

    # Stats across the whole clip
    table = Table(title="Multi-window aggregation")
    table.add_column("Statistic", style="bold cyan")
    table.add_column("Value", justify="right")
    table.add_row("Number of windows", str(len(probs)))
    table.add_row("Max p(drone)", f"{probs.max():.4f}")
    table.add_row("99th percentile", f"{np.percentile(probs, 99):.4f}")
    table.add_row("90th percentile", f"{np.percentile(probs, 90):.4f}")
    table.add_row("Mean", f"{probs.mean():.4f}")
    table.add_row("Median", f"{np.median(probs):.4f}")
    table.add_row(f"Frac windows ≥ {args.threshold}",
                  f"{(probs >= args.threshold).mean():.1%}")
    table.add_row("Frac windows ≥ 0.30", f"{(probs >= 0.30).mean():.1%}")
    console.print(table)

    # Top-N moments
    top = np.argsort(probs)[::-1][:args.show_top]
    top_table = Table(title=f"Top {args.show_top} windows by p(drone)")
    top_table.add_column("rank", justify="right")
    top_table.add_column("t_start (s)", justify="right")
    top_table.add_column("p(drone)", justify="right")
    for r, idx in enumerate(top, 1):
        top_table.add_row(str(r), f"{times[idx]:.2f}", f"{probs[idx]:.4f}")
    console.print(top_table)

    # Verdict
    verdict = (
        "DRONE DETECTED at multiple moments"
        if (probs >= args.threshold).sum() >= 3
        else "drone-like signal but below threshold — see top windows"
        if probs.max() >= 0.30
        else "no clear drone signal in this audio"
    )
    console.print(f"\n[bold]Verdict:[/bold] {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
