"""
Self-contained drone-detection script. ONE FILE, NO PROJECT FILES NEEDED.

Send this single file to a teammate. They install 5 packages and run it.
The model auto-downloads from HuggingFace on first run.

USAGE
-----
Install deps (one time):
    pip install torch transformers soundfile sounddevice numpy scipy

Test on a file (drone audio from anywhere — Pixabay, YouTube extract, recording):
    python standalone_inference.py --wav clip.mp3
    python standalone_inference.py --wav clip.wav --threshold 0.3

Live microphone:
    python standalone_inference.py
    python standalone_inference.py --device 2     # specific mic, see --list-devices

Use a video file directly (mp4/m4a/etc — needs ffmpeg installed):
    python standalone_inference.py --wav video.mp4

USAGE NOTES
-----------
For the "is there a drone in this audio?" question, the script slides a 1-second
window across the whole file with 0.5s hop and applies a median filter to the
per-window probabilities. A clip is flagged DRONE if at least 3 consecutive
post-filter windows are above the threshold.

If a clip you're sure contains a drone reads as "uncertain" or "no drone":
  - try --threshold 0.3 (more sensitive)
  - try --boost-drone (band-pass filter to drone frequencies before classifying)
  - check that the audio actually contains an audible drone (faint = poor signal)
"""

from __future__ import annotations
import argparse
import queue
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import sounddevice as sd
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification


SR = 16_000
DEFAULT_HUB = "Rashidbm/samid-drone-detector"


# --------------------------- helpers ---------------------------------------- #


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(hub_id: str):
    print(f"[load] {hub_id}")
    fe = AutoFeatureExtractor.from_pretrained(hub_id)
    model = AutoModelForAudioClassification.from_pretrained(hub_id)
    return model.eval().to(device()), fe


def median_filter(x: np.ndarray, k: int = 5) -> np.ndarray:
    if x.size == 0 or k <= 1:
        return x
    half = k // 2
    pad = np.pad(x, (half, half), mode="edge")
    return np.array([np.median(pad[i:i + k]) for i in range(x.size)])


def consec_above(x: np.ndarray, t: float) -> int:
    above = (x >= t).astype(int)
    best = cur = 0
    for v in above:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def boost_drone_band(arr: np.ndarray, sr: int) -> np.ndarray:
    """Band-pass filter to drone-relevant frequencies (~100 Hz to ~8 kHz)."""
    from scipy.signal import butter, filtfilt
    nyq = sr / 2
    low = 100 / nyq
    high = min(8_000, nyq - 100) / nyq
    b, a = butter(4, [low, high], btype="band")
    return filtfilt(b, a, arr).astype(np.float32, copy=False)


def load_audio_any(path: Path, target_sr: int = SR) -> tuple[np.ndarray, int]:
    suffix = path.suffix.lower()
    if suffix in {".wav", ".flac", ".ogg"}:
        try:
            arr, sr = sf.read(str(path), dtype="float32", always_2d=False)
        except Exception:
            arr, sr = _ffmpeg_to_wav(path, target_sr)
    else:
        arr, sr = _ffmpeg_to_wav(path, target_sr)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if sr != target_sr:
        arr = np.interp(
            np.linspace(0, arr.size, int(arr.size * target_sr / sr), endpoint=False),
            np.arange(arr.size), arr,
        ).astype(np.float32)
    return arr.astype(np.float32, copy=False), target_sr


def _ffmpeg_to_wav(path: Path, target_sr: int) -> tuple[np.ndarray, int]:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg is required for non-WAV audio. Install via: brew install ffmpeg "
            "(mac), apt install ffmpeg (linux), or download from ffmpeg.org (windows)"
        )
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
    return arr, sr


# --------------------------- prediction ------------------------------------- #


@torch.inference_mode()
def predict_window(model, fe, dev, audio: np.ndarray) -> float:
    feats = fe(audio, sampling_rate=SR, return_tensors="pt")
    x = feats["input_values"].to(dev)
    out = model(input_values=x)
    return float(torch.softmax(out.logits, dim=-1)[0, 1])


@torch.inference_mode()
def slide_windows(model, fe, dev, arr: np.ndarray,
                  win: int = SR, hop: int = SR // 2) -> np.ndarray:
    if arr.size < win:
        arr = np.concatenate([arr, np.zeros(win - arr.size, dtype=np.float32)])
    probs = []
    for s in range(0, arr.size - win + 1, hop):
        probs.append(predict_window(model, fe, dev, arr[s:s + win]))
    return np.asarray(probs, dtype=np.float32)


# --------------------------- modes ------------------------------------------ #


def run_wav(model, fe, dev, path: str, threshold: float, boost: bool):
    print(f"[file] {path}")
    arr, sr = load_audio_any(Path(path))
    print(f"[audio] duration={arr.size/sr:.2f}s  sr={sr}Hz  rms={np.sqrt(np.mean(arr**2)):.4f}")

    if boost:
        print("[preprocess] band-pass filter 100Hz–8kHz applied")
        arr = boost_drone_band(arr, sr)

    raw = slide_windows(model, fe, dev, arr)
    if raw.size == 0:
        print("[error] no windows produced")
        return

    smoothed = median_filter(raw, k=5)
    longest = consec_above(smoothed, threshold)

    print()
    print("=" * 60)
    print("AGGREGATE RESULT")
    print("=" * 60)
    print(f"  windows analyzed           : {raw.size}")
    print(f"  raw max p(drone)           : {raw.max():.4f}")
    print(f"  smoothed max p(drone)      : {smoothed.max():.4f}")
    print(f"  smoothed median            : {np.median(smoothed):.4f}")
    print(f"  longest consecutive ≥ {threshold}    : {longest}")
    print(f"  windows ≥ {threshold} (smoothed) : {(smoothed >= threshold).sum()} / {raw.size}")
    print()

    # Decision logic per reviewer protocol: median filter + N consecutive
    if longest >= 3 and smoothed.max() >= threshold:
        verdict = f"DRONE DETECTED  (longest run = {longest} consecutive windows above {threshold})"
    elif raw.size <= 2 and raw.max() >= threshold:
        verdict = f"DRONE DETECTED  (single short clip, p={raw.max():.3f})"
    elif smoothed.max() >= 0.30:
        verdict = f"UNCERTAIN  (peak p={smoothed.max():.3f}, signal present but below confident threshold)"
        print(f"  hint: try --threshold 0.3 if you believe a drone is present")
        print(f"  hint: try --boost-drone to band-pass filter drone frequencies")
    else:
        verdict = "NO DRONE"
    print(f"  VERDICT: {verdict}")
    print("=" * 60)


def run_mic(model, fe, dev, device_idx, threshold: float, smoothing: int):
    win = SR
    step = SR // 2
    audio_q: queue.Queue[np.ndarray] = queue.Queue()
    consec = 0

    def cb(indata, frames, t, status):
        if status:
            sys.stderr.write(f"[mic] {status}\n")
        x = indata if indata.ndim == 1 else indata.mean(axis=1)
        audio_q.put(x.copy().astype(np.float32))

    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))

    print("[mic] listening (Ctrl+C to stop)")
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
                print(f"{time.time() - t0:8.1f}  {p:10.4f}  {decision}")


def list_devices() -> None:
    print("Audio input devices:")
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            print(f"  [{i}] {d['name']}  "
                  f"(channels={d['max_input_channels']}, "
                  f"sr={int(d['default_samplerate'])} Hz)")


# --------------------------- main ------------------------------------------- #


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--hub-id", default=DEFAULT_HUB,
                   help=f"HuggingFace repo id (default: {DEFAULT_HUB})")
    p.add_argument("--wav", default=None,
                   help="Audio/video file (else live mic)")
    p.add_argument("--device", type=int, default=None, help="Mic device index")
    p.add_argument("--list-devices", action="store_true")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Decision threshold on p(drone). 0.5 default; try 0.3 for "
                        "harder audio")
    p.add_argument("--smoothing", type=int, default=2,
                   help="Live-mic: require N consecutive windows above threshold")
    p.add_argument("--boost-drone", action="store_true",
                   help="Band-pass filter to drone frequencies before classification")
    args = p.parse_args()

    if args.list_devices:
        list_devices()
        return 0

    dev = device()
    print(f"[device] {dev}")
    model, fe = load_model(args.hub_id)

    if args.wav:
        run_wav(model, fe, dev, args.wav, args.threshold, args.boost_drone)
    else:
        run_mic(model, fe, dev, args.device, args.threshold, args.smoothing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
