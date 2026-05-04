"""CLI entry point for the Saamid backend.

Examples:

    # live mode, default 4-mic square config, default audio input device
    uv run python -m scripts.serve

    # live mode, pick a specific device (use --list-devices first)
    uv run python -m scripts.serve --device 2

    # demo mode without hardware — replay a WAV through the pipeline
    uv run python -m scripts.serve --simulate data/abdulrahman/DroneAbdulrahman.wav

    # custom mic geometry + lower threshold for noisy environments
    uv run python -m scripts.serve --mics configs/mics_4_square.json --threshold 0.20
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import uvicorn


REPO_ROOT = Path(__file__).resolve().parent.parent


def list_audio_devices() -> int:
    import sounddevice as sd
    print("Audio input devices:")
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            print(f"  [{i:>2}] {d['name']}  "
                  f"(channels={d['max_input_channels']}, "
                  f"sr={int(d['default_samplerate'])} Hz)")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="saamid-serve",
                                description="Saamid Early Warning System backend")
    p.add_argument("--host", default="0.0.0.0",
                   help="bind address (default 0.0.0.0 — reachable on the LAN)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--mics", type=Path,
                   default=REPO_ROOT / "configs" / "mics_4_square.json",
                   help="JSON file with N×3 (or N×2) mic positions in metres")
    p.add_argument("--device", type=int, default=None,
                   help="sounddevice input index; --list-devices to discover")
    p.add_argument("--threshold", type=float, default=0.25,
                   help="p(drone) above this counts as a positive window")
    p.add_argument("--site-id", default="RUH-14",
                   help="cueing JSON site identifier")
    p.add_argument("--hub-id", default="Rashidbm/samid-drone-detector",
                   help="HuggingFace model id for the AST drone detector")
    p.add_argument("--simulate", type=Path, default=None,
                   help="WAV file to replay instead of capturing live audio")
    p.add_argument("--list-devices", action="store_true",
                   help="print available audio input devices and exit")
    p.add_argument("--reload", action="store_true",
                   help="dev only — restart on code changes")
    args = p.parse_args(argv)

    if args.list_devices:
        return list_audio_devices()

    # Pass config to the FastAPI factory via env so build_app() stays import-safe.
    os.environ["SAAMID_MICS"] = str(args.mics.resolve())
    os.environ["SAAMID_THRESHOLD"] = str(args.threshold)
    os.environ["SAAMID_SITE_ID"] = args.site_id
    os.environ["SAAMID_HUB_ID"] = args.hub_id
    if args.simulate:
        os.environ["SAAMID_SIMULATE"] = str(args.simulate.resolve())
    if args.device is not None:
        os.environ["SAAMID_DEVICE"] = str(args.device)

    print(f"[serve] http://{args.host}:{args.port}/")
    print(f"[serve] mode    = {'simulate' if args.simulate else 'live'}")
    print(f"[serve] mics    = {args.mics}")
    print(f"[serve] thresh  = {args.threshold}")
    if args.simulate:
        print(f"[serve] wav     = {args.simulate}")

    uvicorn.run(
        "server.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
        ws_ping_interval=20.0,
        ws_ping_timeout=20.0,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
