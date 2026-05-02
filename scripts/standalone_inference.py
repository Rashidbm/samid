"""
Self-contained drone-detection script. ONE FILE, NO PROJECT FILES NEEDED.

Send this single file to a teammate. They install 4 packages and run it.
The model auto-downloads from HuggingFace Hub on first run.

USAGE:
  Install deps (one time):
    pip install torch transformers soundfile sounddevice numpy

  Live microphone:
    python standalone_inference.py --hub-id Rashidbm/samid-drone-detector

  On a wav file (long files use multi-window aggregation, recommended for
  YouTube clips and recordings):
    python standalone_inference.py --hub-id Rashidbm/samid-drone-detector --wav clip.wav

  List available microphones:
    python standalone_inference.py --hub-id Rashidbm/samid-drone-detector --list-devices

NOTES:
  - For long audio (videos with narration / music / silent intros), the script
    slides 1-second windows every 0.5 seconds and reports MAX, not single-window.
    This is essential because real-world audio rarely has the drone present at
    one fixed moment.
  - Model is calibrated to give confident probabilities (>0.95 on training-
    distribution drone clips). If it outputs 0.20 on a clip you believe contains
    a drone, the drone signal is likely faint relative to background — try
    cropping to the loudest moment or using --threshold 0.30.
"""

from __future__ import annotations
import argparse
import queue
import signal
import sys
import time

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification


SR = 16_000


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(hub_id: str):
    print(f"loading model: {hub_id}")
    fe = AutoFeatureExtractor.from_pretrained(hub_id)
    model = AutoModelForAudioClassification.from_pretrained(hub_id)
    return model.eval().to(device()), fe


def predict_window(model, fe, dev, audio: np.ndarray) -> float:
    """Returns drone probability for a 1-second mono 16kHz audio array."""
    feats = fe(audio, sampling_rate=SR, return_tensors="pt")
    x = feats["input_values"].to(dev)
    with torch.inference_mode():
        out = model(input_values=x)
    return float(torch.softmax(out.logits, dim=-1)[0, 1])


# --- modes -----------------------------------------------------------------


def run_wav(model, fe, dev, path: str, threshold: float, hop: float):
    arr, sr = sf.read(path, dtype="float32", always_2d=False)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if sr != SR:
        # quick resample using numpy interp (good enough; install librosa for better)
        arr = np.interp(
            np.linspace(0, arr.size, int(arr.size * SR / sr), endpoint=False),
            np.arange(arr.size), arr
        ).astype(np.float32)
    win = SR
    step = int(hop * SR)
    if arr.size < win:
        arr = np.concatenate([arr, np.zeros(win - arr.size, dtype=np.float32)])
    print(f"file={path}  duration={arr.size/SR:.2f}s")
    print(f"{'t':>8}  {'p(drone)':>10}  decision")
    probs = []
    for start in range(0, arr.size - win + 1, step):
        window = arr[start:start + win]
        p = predict_window(model, fe, dev, window)
        probs.append(p)
        decision = "DRONE" if p >= threshold else "no drone"
        print(f"{start/SR:8.2f}  {p:10.4f}  {decision}")

    if probs:
        p_arr = np.asarray(probs)
        n_above = int((p_arr >= threshold).sum())
        print()
        print("=" * 50)
        print("AGGREGATE RESULT (multi-window):")
        print(f"  windows analyzed:           {len(probs)}")
        print(f"  max p(drone):               {p_arr.max():.4f}")
        print(f"  99th percentile:            {np.percentile(p_arr, 99):.4f}")
        print(f"  90th percentile:            {np.percentile(p_arr, 90):.4f}")
        print(f"  median:                     {np.median(p_arr):.4f}")
        print(f"  windows >= threshold:       {n_above} of {len(probs)}")
        print()
        if n_above >= 3:
            verdict = f"DRONE DETECTED ({n_above} windows above threshold)"
        elif p_arr.max() >= 0.30:
            verdict = (f"drone-like signal detected at peak={p_arr.max():.2f} "
                       f"but below threshold")
        else:
            verdict = "no clear drone signal"
        print(f"  VERDICT: {verdict}")
        print("=" * 50)


def run_mic(model, fe, dev, device_idx, threshold: float, hop: float, smoothing: int):
    win = SR
    step = int(hop * SR)
    audio_q: queue.Queue[np.ndarray] = queue.Queue()
    consec = 0

    def cb(indata, frames, t, status):
        if status:
            sys.stderr.write(f"[mic] {status}\n")
        x = indata if indata.ndim == 1 else indata.mean(axis=1)
        audio_q.put(x.copy().astype(np.float32))

    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))

    print("listening (Ctrl+C to stop)…")
    print(f"{'t':>8}  {'p(drone)':>10}  decision")
    t0 = time.time()
    with sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                        blocksize=step, device=device_idx, callback=cb):
        buf = np.zeros(0, dtype=np.float32)
        while not stop["flag"]:
            try:
                chunk = audio_q.get(timeout=0.5)
            except queue.Empty:
                continue
            buf = np.concatenate([buf, chunk])
            while buf.size >= win:
                window = buf[:win]
                buf = buf[step:]
                p = predict_window(model, fe, dev, window)
                if p >= threshold:
                    consec += 1
                else:
                    consec = 0
                if consec >= smoothing:
                    decision = "DRONE"
                elif consec > 0:
                    decision = "drone (smoothing)"
                else:
                    decision = "no drone"
                print(f"{time.time()-t0:8.1f}  {p:10.4f}  {decision}")


def list_devices() -> None:
    print("Audio input devices:")
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            print(f"  [{i}] {d['name']}  "
                  f"(channels={d['max_input_channels']}, "
                  f"sr={int(d['default_samplerate'])} Hz)")


# --- main ------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--hub-id", required=True,
                   help="HuggingFace repo id, e.g. username/samid-drone-detector")
    p.add_argument("--wav", default=None, help="Optional WAV file (else uses mic)")
    p.add_argument("--device", type=int, default=None, help="Mic device index")
    p.add_argument("--list-devices", action="store_true")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--hop", type=float, default=0.5)
    p.add_argument("--smoothing", type=int, default=2)
    args = p.parse_args()

    if args.list_devices:
        list_devices()
        return 0

    dev = device()
    model, fe = load_model(args.hub_id)
    print(f"device={dev}")

    if args.wav:
        run_wav(model, fe, dev, args.wav, args.threshold, args.hop)
    else:
        run_mic(model, fe, dev, args.device, args.threshold, args.hop, args.smoothing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
