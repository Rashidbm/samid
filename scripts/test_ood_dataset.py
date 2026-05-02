"""
Out-of-distribution test using HuggingFace datasets the model has NEVER seen.

We use:
  - ahlab-drone-project/DroneAudioSet (NUS, 2025) — drone clips from different
    mics, environments, and recording sessions than geronimobasso.

This is the honest cross-dataset test our pitch needs.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from rich.console import Console
from rich.table import Table

from src.config import CFG
from src.model import build_model

ROOT = Path(__file__).resolve().parent.parent
console = Console()


def device_or_fallback() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def multi_window_max(model, fe, dev, arr: np.ndarray, sr: int = 16_000) -> tuple[float, float, float]:
    """Slide 1-sec windows; return (max, p90, median) of p(drone)."""
    win = sr
    hop = sr // 2
    if arr.size < win:
        arr = np.concatenate([arr, np.zeros(win - arr.size, dtype=np.float32)])
    probs = []
    for start in range(0, arr.size - win + 1, hop):
        chunk = arr[start:start + win]
        feats = fe(chunk, sampling_rate=sr, return_tensors="pt")
        x = feats["input_values"].to(dev)
        out = model(input_values=x)
        probs.append(float(torch.softmax(out.logits, dim=-1)[0, 1]))
    if not probs:
        return 0.0, 0.0, 0.0
    p = np.asarray(probs)
    return float(p.max()), float(np.percentile(p, 90)), float(np.median(p))


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


def main() -> int:
    dev = device_or_fallback()
    console.print(f"[bold]Device:[/bold] {dev}")

    import os
    ckpt = Path(os.environ.get("CKPT_PATH",
                                str(ROOT / "runs/20260429-112104/best_calibrated.pt")))
    console.print(f"[bold]Loading calibrated model:[/bold] {ckpt}")
    model, fe = build_model()
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state"])
    model.to(dev).eval()
    console.print(f"[dim]calibration T = {state.get('calibration_T', 'unknown')}[/dim]\n")

    # ---------------------------- Test 1: NUS DroneAudioSet ----------------- #
    console.print("[bold cyan]Test 1: NUS ahlab-drone-project/DroneAudioSet[/bold cyan]")
    console.print("Different mics, environments, drones than geronimobasso\n")

    try:
        ds = load_dataset(
            "ahlab-drone-project/DroneAudioSet",
            "drone-only",
            split="train_001",
            cache_dir=str(ROOT / "data" / "ahlab"),
        )
        console.print(f"  loaded {len(ds)} samples\n")
    except Exception as exc:
        console.print(f"[red]Failed to load NUS dataset: {exc}[/red]")
        return 1

    table = Table(title="NUS DroneAudioSet (drone-only subset)")
    table.add_column("idx", justify="right")
    table.add_column("max p(drone)", justify="right")
    table.add_column("90th %ile", justify="right")
    table.add_column("median", justify="right")
    table.add_column("verdict")

    n_total = min(6, len(ds))
    n_pass = 0
    max_probs = []
    for i in range(n_total):
        sample = ds[i]
        # NUS dataset stores audio under "audio" with array; we take first channel
        try:
            audio = sample["audio"]
            arr = np.asarray(audio["array"], dtype=np.float32)
            # NUS layout is (samples, channels) — take channel 0 properly
            if arr.ndim > 1:
                arr = arr[:, 0]
            sr = int(audio["sampling_rate"])
            arr = resample_to_16k(arr, sr)
            mx, p90, med = multi_window_max(model, fe, dev, arr)
            verdict = "[green]PASS[/green]" if mx >= 0.5 else "[red]FAIL[/red]"
            if mx >= 0.5:
                n_pass += 1
            max_probs.append(mx)
            table.add_row(str(i), f"{mx:.3f}", f"{p90:.3f}", f"{med:.3f}", verdict)
        except Exception as exc:
            table.add_row(str(i), "—", "—", "—", f"[yellow]error: {exc}[/yellow]")

    console.print(table)
    console.print(f"\n[bold]NUS drone detection rate: {n_pass}/{n_total} "
                  f"({100*n_pass/max(1,n_total):.0f}%)[/bold]")
    if max_probs:
        console.print(f"[bold]Mean max p(drone) on out-of-distribution drone clips: "
                      f"{np.mean(max_probs):.3f}[/bold]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
