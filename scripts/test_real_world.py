"""
Test the calibrated model on REAL-WORLD audio it has never seen.

We download a curated list of YouTube clips covering:
  - Clearly drone (DJI, Shahed/Geran, hobby quadcopters) — should fire
  - Drone-adjacent (helicopter, motorcycle) — should NOT fire (true negatives)
  - Background (rain, traffic) — should NOT fire

Each clip is run through multi-window aggregation. We report MAX p(drone) over
the whole clip, not single-window.
"""

from __future__ import annotations
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from rich.console import Console
from rich.table import Table

from src.config import CFG
from src.model import build_model

ROOT = Path(__file__).resolve().parent.parent
console = Console()


# --- Curated YouTube test clips (chosen for clear audio of the labeled source) -


# format: (label, expected_class, youtube_url, time_range_seconds)
TEST_CLIPS: list[tuple[str, str, str, tuple[int, int]]] = [
    # DRONE — should give high p(drone)
    ("DJI hover (consumer drone)",                "drone",
     "https://www.youtube.com/watch?v=9qYS4bUyJI8", (0, 30)),
    ("Shahed-136 attack overhead",                "drone",
     "https://www.youtube.com/watch?v=h_8HCpHcpfA", (0, 30)),
    ("Quadcopter close-mic",                      "drone",
     "https://www.youtube.com/watch?v=v_R8aoB8x9k", (0, 30)),
    # NOT DRONE — should give low p(drone)
    ("Helicopter overhead",                       "no_drone",
     "https://www.youtube.com/watch?v=xKgr4V2XXyM", (0, 30)),
    ("Motorcycle revving",                        "no_drone",
     "https://www.youtube.com/watch?v=PKE-vlEGwKw", (0, 30)),
    ("City traffic ambience",                     "no_drone",
     "https://www.youtube.com/watch?v=3a3jYSkS8DE", (0, 30)),
]


def download_clip(url: str, t_start: int, t_end: int, out_path: Path) -> bool:
    """Download a snippet from YouTube as 16-kHz mono wav using yt-dlp + ffmpeg."""
    if out_path.exists():
        return True
    try:
        # download bestaudio segment
        cmd = [
            "uv", "run", "yt-dlp",
            "-f", "bestaudio",
            "--download-sections", f"*{t_start}-{t_end}",
            "--force-keyframes-at-cuts",
            "-o", str(out_path.with_suffix("")) + ".%(ext)s",
            "--no-playlist",
            "-x",                  # extract audio
            "--audio-format", "wav",
            "--postprocessor-args", "-ac 1 -ar 16000",
            "--quiet",
            url,
        ]
        subprocess.run(cmd, check=True, timeout=120)
        # rename to expected path if extension differs
        for ext in [".wav", ".m4a.wav"]:
            cand = out_path.with_suffix("").with_suffix(ext)
            if cand.exists():
                cand.rename(out_path)
                return True
        if out_path.exists():
            return True
        # last resort: any file with the basename
        for f in out_path.parent.glob(f"{out_path.stem}.*"):
            if f.suffix in {".wav", ".m4a", ".webm", ".opus"}:
                # convert to 16kHz mono wav via ffmpeg
                tmp_out = out_path.with_suffix(".converted.wav")
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error",
                                "-i", str(f), "-ac", "1", "-ar", "16000",
                                str(tmp_out)], check=True)
                tmp_out.rename(out_path)
                return True
        return False
    except Exception as exc:
        console.print(f"[yellow]download failed for {url}: {exc}[/yellow]")
        return False


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf
    arr, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if sr != 16_000:
        try:
            import librosa
            arr = librosa.resample(arr, orig_sr=sr, target_sr=16_000)
        except ImportError:
            arr = np.interp(
                np.linspace(0, arr.size, int(arr.size * 16_000 / sr), endpoint=False),
                np.arange(arr.size), arr
            ).astype(np.float32)
        sr = 16_000
    return arr.astype(np.float32, copy=False), sr


@torch.no_grad()
def multi_window_probs(model, fe, dev, arr: np.ndarray, sr: int = 16_000) -> np.ndarray:
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
        p = float(torch.softmax(out.logits, dim=-1)[0, 1])
        probs.append(p)
    return np.asarray(probs, dtype=np.float32)


def main() -> int:
    if shutil.which("ffmpeg") is None:
        console.print("[red]ffmpeg not on PATH; needed for audio extraction.[/red]")
        return 1

    cache = ROOT / "data" / "real_world_test"
    cache.mkdir(parents=True, exist_ok=True)

    # Load model
    if torch.backends.mps.is_available():
        dev = torch.device("mps")
    elif torch.cuda.is_available():
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")
    console.print(f"[bold]Device:[/bold] {dev}")

    ckpt = ROOT / "runs/20260429-112104/best_calibrated.pt"
    console.print(f"[bold]Loading calibrated model:[/bold] {ckpt}")
    model, fe = build_model()
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state"])
    model.to(dev).eval()
    console.print(f"[dim]calibration T = {state.get('calibration_T', 'unknown')}[/dim]\n")

    # Download + test each clip
    table = Table(title="Real-world out-of-dataset evaluation (calibrated model)")
    table.add_column("clip", style="bold")
    table.add_column("expected", justify="center")
    table.add_column("max p(drone)", justify="right")
    table.add_column("90th %ile", justify="right")
    table.add_column("median", justify="right")
    table.add_column("verdict")

    results = []
    for label, expected, url, (t0, t1) in TEST_CLIPS:
        slug = label.lower().replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
        out = cache / f"{slug}.wav"
        console.print(f"[cyan]{label}[/cyan]  ←  {url}")

        ok = download_clip(url, t0, t1, out)
        if not ok or not out.exists():
            table.add_row(label, expected, "—", "—", "—", "[yellow]download failed[/yellow]")
            results.append((label, expected, None))
            continue

        arr, sr = load_audio(out)
        probs = multi_window_probs(model, fe, dev, arr, sr)
        if probs.size == 0:
            table.add_row(label, expected, "—", "—", "—", "[yellow]no windows[/yellow]")
            continue

        mx = float(probs.max())
        p90 = float(np.percentile(probs, 90))
        med = float(np.median(probs))
        n_above = int((probs >= 0.5).sum())

        if expected == "drone":
            ok = mx >= 0.7 or n_above >= 2
            verdict = "[green]PASS[/green]" if ok else "[red]FAIL[/red]"
        else:
            ok = mx < 0.5 and n_above == 0
            verdict = "[green]PASS[/green]" if ok else "[red]FAIL (false positive)[/red]"

        table.add_row(label, expected, f"{mx:.3f}", f"{p90:.3f}", f"{med:.3f}", verdict)
        results.append((label, expected, mx))

    console.print(table)

    # Summary
    console.print("\n[bold]Summary:[/bold]")
    drone_ok = [r for r in results if r[1] == "drone" and r[2] is not None and r[2] >= 0.5]
    drone_total = [r for r in results if r[1] == "drone" and r[2] is not None]
    nodrone_ok = [r for r in results if r[1] == "no_drone" and r[2] is not None and r[2] < 0.5]
    nodrone_total = [r for r in results if r[1] == "no_drone" and r[2] is not None]
    console.print(f"  drone clips correctly detected:    "
                  f"{len(drone_ok)}/{len(drone_total)}")
    console.print(f"  no-drone clips correctly rejected: "
                  f"{len(nodrone_ok)}/{len(nodrone_total)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
