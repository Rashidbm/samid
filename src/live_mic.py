"""
Live microphone drone-detection demo.

Captures audio from the system microphone in real time, runs it through the
trained AST detector, and prints a verdict every second. Cross-platform
(macOS/Linux/Windows) thanks to sounddevice.

Usage:
    uv run python -m src.live_mic --ckpt runs/20260429-112104/best.pt
    uv run python -m src.live_mic --ckpt runs/20260429-112104/best.pt --device 1
    uv run python -m src.live_mic --ckpt ... --threshold 0.5 --smoothing 3

Useful flags:
    --device       int — explicit input device index (run --list-devices first)
    --list-devices print all microphone devices and exit
    --threshold    decision threshold on p(drone), default 0.5
    --smoothing    require N consecutive windows above threshold before alerting
    --window-sec   length of each analysis window in seconds, default 1.0
    --hop-sec      step between consecutive windows, default 0.5
"""

from __future__ import annotations
import argparse
import queue
import signal
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import torch
from rich.console import Console
from rich.live import Live
from rich.table import Table

from src.config import CFG
from src.model import build_model


console = Console()


# ----------------------------- helpers -------------------------------------- #


def list_devices() -> None:
    console.print("[bold]Audio input devices:[/bold]")
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            console.print(
                f"  [{i}] {dev['name']}  "
                f"(channels={dev['max_input_channels']}, "
                f"default sr={int(dev['default_samplerate'])} Hz)"
            )


def device_or_fallback() -> torch.device:
    if CFG.device == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ----------------------------- main loop ------------------------------------ #


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=None,
                   help="Trained checkpoint (default: latest run's best.pt)")
    p.add_argument("--device", type=int, default=None,
                   help="Input device index (see --list-devices)")
    p.add_argument("--list-devices", action="store_true")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--window-sec", type=float, default=1.0)
    p.add_argument("--hop-sec", type=float, default=0.5)
    p.add_argument("--smoothing", type=int, default=2,
                   help="Require N consecutive windows above threshold to alert")
    args = p.parse_args()

    if args.list_devices:
        list_devices()
        return 0

    # Locate checkpoint
    if args.ckpt is None:
        runs = Path(__file__).resolve().parent.parent / "runs"
        candidates = sorted(
            (d for d in runs.iterdir() if d.is_dir() and (d / "best.pt").exists()),
            key=lambda p: p.stat().st_mtime,
        ) if runs.exists() else []
        if not candidates:
            console.print("[red]No best.pt found. Train first or pass --ckpt.[/red]")
            return 1
        args.ckpt = candidates[-1] / "best.pt"
        console.print(f"[dim]using {args.ckpt}[/dim]")

    # Load model
    device = device_or_fallback()
    console.print(f"[bold]Device:[/bold] {device}")
    console.print(f"[bold]Loading model:[/bold] {args.ckpt}")
    model, fe = build_model()
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state"])
    model.eval().to(device)

    sr = CFG.sample_rate
    win_samples = int(args.window_sec * sr)
    hop_samples = int(args.hop_sec * sr)

    audio_q: queue.Queue[np.ndarray] = queue.Queue()
    rolling = np.zeros(win_samples, dtype=np.float32)
    consec_above = 0

    def cb(indata, frames, t, status):
        if status:
            sys.stderr.write(f"[mic] {status}\n")
        # mono-mix if multichannel
        x = indata if indata.ndim == 1 else indata.mean(axis=1)
        audio_q.put(x.copy().astype(np.float32))

    stop = {"flag": False}
    def _sig(*_):
        stop["flag"] = True
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    table_data = {"prob": 0.0, "decision": "—", "since": "0.0s",
                  "consec": 0, "last_alert": "—", "windows_done": 0}

    def render_table() -> Table:
        t = Table(title="🛩️  Live drone detection", show_header=False)
        t.add_column(style="bold cyan", justify="right")
        t.add_column()
        t.add_row("p(drone)", f"{table_data['prob']:.4f}")
        decision_color = {
            "DRONE": "red", "drone (smoothing)": "yellow",
            "no drone": "green", "—": "white",
        }.get(table_data["decision"], "white")
        t.add_row("decision", f"[{decision_color}]{table_data['decision']}[/]")
        t.add_row("consec above", str(table_data["consec"]))
        t.add_row("running for", table_data["since"])
        t.add_row("windows", str(table_data["windows_done"]))
        t.add_row("last alert", table_data["last_alert"])
        return t

    console.print("[bold green]Listening... press Ctrl+C to stop.[/bold green]")
    t0 = time.time()
    n_windows = 0

    try:
        with sd.InputStream(
            samplerate=sr, channels=1, dtype="float32",
            blocksize=hop_samples, device=args.device, callback=cb,
        ):
            with Live(render_table(), refresh_per_second=4, console=console) as live:
                buf = np.zeros(0, dtype=np.float32)
                while not stop["flag"]:
                    try:
                        chunk = audio_q.get(timeout=0.5)
                    except queue.Empty:
                        continue
                    buf = np.concatenate([buf, chunk])
                    while buf.size >= win_samples:
                        window = buf[:win_samples]
                        buf = buf[hop_samples:]
                        # feature extract + classify
                        feats = fe(window, sampling_rate=sr, return_tensors="pt")
                        x = feats["input_values"].to(device)
                        with torch.inference_mode():
                            out = model(input_values=x)
                            p_drone = torch.softmax(out.logits, dim=-1)[0, 1].item()
                        n_windows += 1

                        if p_drone >= args.threshold:
                            consec_above += 1
                        else:
                            consec_above = 0

                        if consec_above >= args.smoothing:
                            decision = "DRONE"
                            table_data["last_alert"] = (
                                f"{time.strftime('%H:%M:%S')} (p={p_drone:.3f})"
                            )
                        elif consec_above > 0:
                            decision = "drone (smoothing)"
                        else:
                            decision = "no drone"

                        table_data["prob"] = p_drone
                        table_data["decision"] = decision
                        table_data["consec"] = consec_above
                        table_data["since"] = f"{time.time() - t0:.1f}s"
                        table_data["windows_done"] = n_windows
                        live.update(render_table())
    except KeyboardInterrupt:
        pass

    console.print("\n[dim]stopped[/dim]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
