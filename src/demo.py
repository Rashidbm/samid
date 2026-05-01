"""
End-to-end demo: multichannel audio in -> drone detection per channel -> if any
channel detects a drone, triangulate the source position via TDoA.

Usage:
    uv run python -m src.demo \
        --ckpt runs/<ts>/best.pt \
        --multi-wav path/to/multichannel.wav \
        --mics config/mic_positions.json \
        --threshold 0.5

`mic_positions.json` is a list of [x, y, z] in metres, in the same channel
order as the WAV. Example:
    [[0,0,0], [2,0,0], [2,2,0], [0,2,0], [1,1,1.5]]
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from rich.console import Console
from rich.table import Table

from src.config import CFG
from src.model import build_model
from src.triangulation import localise

console = Console()


def load_multichannel(path: Path, target_sr: int) -> tuple[np.ndarray, int]:
    arr, sr = sf.read(str(path), dtype="float32", always_2d=True)
    # arr shape: (N, M)
    arr = arr.T  # -> (M, N)
    if sr != target_sr:
        import librosa
        arr = np.stack(
            [librosa.resample(c, orig_sr=sr, target_sr=target_sr) for c in arr]
        )
    return arr.astype(np.float32, copy=False), target_sr


def detect_per_channel(model, fe, signals: np.ndarray, device) -> np.ndarray:
    """Returns drone probability per channel for the WHOLE clip (no windowing)."""
    M, N = signals.shape
    target_len = int(CFG.clip_seconds * CFG.sample_rate)
    centered = []
    for ch in signals:
        if N >= target_len:
            start = (N - target_len) // 2
            centered.append(ch[start : start + target_len])
        else:
            pad = np.zeros(target_len - N, dtype=np.float32)
            centered.append(np.concatenate([ch, pad]))
    feats = fe(centered, sampling_rate=CFG.sample_rate, return_tensors="pt", padding=True)
    x = feats["input_values"].to(device)
    with torch.inference_mode():
        out = model(input_values=x)
        return torch.softmax(out.logits, dim=-1)[:, 1].cpu().numpy()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--multi-wav", type=Path, required=True)
    p.add_argument("--mics", type=Path, required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args()

    if CFG.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    mic_positions = np.asarray(json.loads(args.mics.read_text()), dtype=np.float64)
    signals, sr = load_multichannel(args.multi_wav, CFG.sample_rate)
    if signals.shape[0] != mic_positions.shape[0]:
        raise ValueError(
            f"Channel count {signals.shape[0]} != mic count {mic_positions.shape[0]}"
        )

    console.print(
        f"[bold]Loaded[/bold] {signals.shape[0]} ch × {signals.shape[1]/sr:.2f}s @ {sr} Hz"
    )

    model, fe = build_model()
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state"])
    model.eval().to(device)

    probs = detect_per_channel(model, fe, signals, device)
    table = Table(title="Per-channel drone probability")
    table.add_column("Channel")
    table.add_column("Mic position (m)")
    table.add_column("p(drone)", justify="right")
    table.add_column("decision", justify="center")
    for i, (pos, pr) in enumerate(zip(mic_positions, probs)):
        decision = "[red]DRONE[/red]" if pr >= args.threshold else "no drone"
        table.add_row(str(i), str(pos.tolist()), f"{pr:.4f}", decision)
    console.print(table)

    if (probs >= args.threshold).any():
        console.print("\n[bold yellow]At least one channel detects drone — triangulating…[/bold yellow]")
        loc = localise(signals, mic_positions, fs=CFG.sample_rate, max_tau=0.05)
        console.print(f"Estimated source position: {loc.position.round(2).tolist()} m")
        console.print(f"Per-pair TDoAs (ms): {(loc.tdoas * 1e3).round(3).tolist()}")
        console.print(f"LSQ residual: {loc.residual:.4f}")
    else:
        console.print("\n[green]No channel exceeds threshold — no triangulation.[/green]")


if __name__ == "__main__":
    main()
