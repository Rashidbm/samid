"""
Find WHERE the trained model breaks.

Takes known drone + no-drone clips from the training distribution, applies a
battery of audio perturbations, and reports the model's drone-probability for
each. The pattern of failures tells us which kind of robustness is missing.

Run:
    uv run python -m scripts.diagnose_overfit
"""

from __future__ import annotations
import io
from pathlib import Path

import numpy as np
import torch
import soundfile as sf
from datasets import load_dataset
from rich.console import Console
from rich.table import Table

from src.config import CFG
from src.model import build_model

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "geronimobasso"
console = Console()


# ---------------------------- audio perturbations -------------------------- #


def perturb_noise(arr: np.ndarray, snr_db: float) -> np.ndarray:
    """Add white noise at a target signal-to-noise ratio (dB)."""
    sig_pow = float(np.mean(arr ** 2)) + 1e-12
    noise_pow = sig_pow / (10 ** (snr_db / 10))
    noise = np.random.randn(arr.size).astype(np.float32) * np.sqrt(noise_pow)
    return arr + noise


def perturb_pitch(arr: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    import librosa
    return librosa.effects.pitch_shift(arr, sr=sr, n_steps=semitones)


def perturb_speed(arr: np.ndarray, factor: float) -> np.ndarray:
    """Time-stretch (changes pitch). Simple resample-and-trim."""
    new_len = int(arr.size * factor)
    return np.interp(
        np.linspace(0, arr.size, new_len, endpoint=False),
        np.arange(arr.size), arr
    ).astype(np.float32)


def perturb_volume(arr: np.ndarray, gain_db: float) -> np.ndarray:
    return arr * (10 ** (gain_db / 20))


def perturb_codec_mp3(arr: np.ndarray, sr: int) -> np.ndarray:
    """Encode to mp3 in-memory then decode. Simulates compressed audio."""
    try:
        import soundfile as sf2
        # soundfile may not write mp3; try ogg as a robust proxy
        buf = io.BytesIO()
        sf2.write(buf, arr, sr, format="OGG", subtype="VORBIS")
        buf.seek(0)
        out, _ = sf2.read(buf, dtype="float32")
        return out
    except Exception as exc:
        console.print(f"[yellow]ogg codec test skipped: {exc}[/yellow]")
        return arr


def perturb_resample(arr: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """Resample down then back up. Lossy."""
    import librosa
    down = librosa.resample(arr, orig_sr=sr, target_sr=target_sr)
    return librosa.resample(down, orig_sr=target_sr, target_sr=sr)


def perturb_highpass(arr: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    """Cut frequencies below cutoff (e.g. simulate wind-screened mic)."""
    from scipy.signal import butter, filtfilt
    b, a = butter(4, cutoff_hz / (sr / 2), btype="high")
    return filtfilt(b, a, arr).astype(np.float32)


def perturb_lowpass(arr: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    """Cut frequencies above cutoff (e.g. simulate cheap mic / KY-037)."""
    from scipy.signal import butter, filtfilt
    b, a = butter(4, cutoff_hz / (sr / 2), btype="low")
    return filtfilt(b, a, arr).astype(np.float32)


# ---------------------------- model ---------------------------------------- #


def load_model_and_fe(ckpt_path: Path):
    model, fe = build_model()
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state"])
    model.eval()
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)
    return model, fe, device


@torch.no_grad()
def p_drone(model, fe, device, arr: np.ndarray, sr: int = 16_000) -> float:
    # crop / pad to 1 second
    target = 16_000
    if arr.size >= target:
        start = (arr.size - target) // 2
        arr = arr[start:start + target]
    else:
        arr = np.concatenate([arr, np.zeros(target - arr.size, dtype=np.float32)])
    feats = fe(arr, sampling_rate=sr, return_tensors="pt")
    out = model(input_values=feats["input_values"].to(device))
    return float(torch.softmax(out.logits, dim=-1)[0, 1])


# ---------------------------- diagnostic ----------------------------------- #


def main() -> None:
    ckpt = ROOT / "runs/20260429-112104/best.pt"
    console.print(f"[bold]Loading model from:[/bold] {ckpt}")
    model, fe, device = load_model_and_fe(ckpt)
    console.print(f"[bold]Device:[/bold] {device}\n")

    # Pull a few drone and no-drone clips from training data
    console.print("[bold]Loading sample clips from geronimobasso…[/bold]")
    ds = load_dataset(
        "geronimobasso/drone-audio-detection-samples",
        split="train",
        cache_dir=str(CACHE_DIR),
    )
    np.random.seed(42)
    drone_idx = np.random.choice(np.where(np.asarray(ds["label"]) == 1)[0], 5, replace=False)
    nodrone_idx = np.random.choice(np.where(np.asarray(ds["label"]) == 0)[0], 5, replace=False)

    drone_clips = [np.asarray(ds[int(i)]["audio"]["array"], dtype=np.float32) for i in drone_idx]
    nodrone_clips = [np.asarray(ds[int(i)]["audio"]["array"], dtype=np.float32) for i in nodrone_idx]
    sr = 16_000

    # Define test conditions
    conditions = [
        ("baseline (no perturbation)", lambda x: x),
        ("noise SNR=20dB",  lambda x: perturb_noise(x, 20)),
        ("noise SNR=10dB",  lambda x: perturb_noise(x, 10)),
        ("noise SNR=0dB",   lambda x: perturb_noise(x, 0)),
        ("noise SNR=-5dB",  lambda x: perturb_noise(x, -5)),
        ("pitch +2 semitones", lambda x: perturb_pitch(x, sr, 2)),
        ("pitch -2 semitones", lambda x: perturb_pitch(x, sr, -2)),
        ("pitch +5 semitones", lambda x: perturb_pitch(x, sr, 5)),
        ("speed 0.9x",       lambda x: perturb_speed(x, 0.9)),
        ("speed 1.1x",       lambda x: perturb_speed(x, 1.1)),
        ("volume -10dB",     lambda x: perturb_volume(x, -10)),
        ("volume -20dB",     lambda x: perturb_volume(x, -20)),
        ("volume +10dB",     lambda x: perturb_volume(x, 10)),
        ("ogg codec round-trip", lambda x: perturb_codec_mp3(x, sr)),
        ("resample 8kHz→16kHz",  lambda x: perturb_resample(x, sr, 8_000)),
        ("resample 4kHz→16kHz",  lambda x: perturb_resample(x, sr, 4_000)),
        ("highpass 200Hz",   lambda x: perturb_highpass(x, sr, 200)),
        ("lowpass 4kHz (KY-037-like)", lambda x: perturb_lowpass(x, sr, 4_000)),
        ("lowpass 2kHz (very narrow band)",  lambda x: perturb_lowpass(x, sr, 2_000)),
    ]

    table = Table(title="Drone-probability under perturbations")
    table.add_column("Condition", justify="left")
    table.add_column("avg p(drone) on DRONE clips", justify="right")
    table.add_column("avg p(drone) on NO-DRONE clips", justify="right")
    table.add_column("verdict", justify="left")

    baseline_drone = None
    for name, fn in conditions:
        try:
            ps_drone = [p_drone(model, fe, device, fn(c).astype(np.float32)) for c in drone_clips]
            ps_neg = [p_drone(model, fe, device, fn(c).astype(np.float32)) for c in nodrone_clips]
            avg_d = float(np.mean(ps_drone))
            avg_n = float(np.mean(ps_neg))
        except Exception as exc:
            table.add_row(name, "ERR", "ERR", f"[red]{type(exc).__name__}[/red]")
            continue

        if name.startswith("baseline"):
            baseline_drone = avg_d
            verdict = "[green]reference[/green]"
        else:
            drop = baseline_drone - avg_d
            if drop > 0.30:
                verdict = f"[red]MAJOR drop {drop:.2f}[/red]"
            elif drop > 0.10:
                verdict = f"[yellow]moderate drop {drop:.2f}[/yellow]"
            else:
                verdict = f"[green]robust ({drop:+.2f})[/green]"

        table.add_row(name, f"{avg_d:.3f}", f"{avg_n:.3f}", verdict)

    console.print(table)
    console.print(
        "\n[bold]How to read:[/bold]\n"
        "  • 'baseline' should be near 1.0 for drone, near 0.0 for no-drone\n"
        "  • Big drops on a perturbation mean the model is brittle to that variation\n"
        "  • Robust rows mean augmentation for that perturbation is unnecessary\n"
    )


if __name__ == "__main__":
    main()
