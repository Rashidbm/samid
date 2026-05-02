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


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(hub_id):
    print(f"[load] {hub_id}")
    fe = AutoFeatureExtractor.from_pretrained(hub_id)
    model = AutoModelForAudioClassification.from_pretrained(hub_id)
    return model.eval().to(device()), fe


def median_filter(x, k=5):
    if x.size == 0 or k <= 1:
        return x
    half = k // 2
    pad = np.pad(x, (half, half), mode="edge")
    return np.array([np.median(pad[i:i + k]) for i in range(x.size)])


def consec_above(x, t):
    above = (x >= t).astype(int)
    best = cur = 0
    for v in above:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def boost_drone_band(arr, sr):
    from scipy.signal import butter, filtfilt
    nyq = sr / 2
    b, a = butter(4, [100 / nyq, min(8_000, nyq - 100) / nyq], btype="band")
    return filtfilt(b, a, arr).astype(np.float32, copy=False)


def _ffmpeg_to_wav(path, target_sr):
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg required for non-WAV audio")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(path), "-ac", "1", "-ar", str(target_sr), tmp_path,
    ], check=True)
    arr, sr = sf.read(tmp_path, dtype="float32", always_2d=False)
    return arr, sr


def load_audio_any(path, target_sr=SR):
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


@torch.inference_mode()
def predict_window(model, fe, dev, audio):
    feats = fe(audio, sampling_rate=SR, return_tensors="pt")
    out = model(input_values=feats["input_values"].to(dev))
    return float(torch.softmax(out.logits, dim=-1)[0, 1])


@torch.inference_mode()
def slide_windows(model, fe, dev, arr, win=SR, hop=SR // 2):
    if arr.size < win:
        arr = np.concatenate([arr, np.zeros(win - arr.size, dtype=np.float32)])
    return np.asarray([
        predict_window(model, fe, dev, arr[s:s + win])
        for s in range(0, arr.size - win + 1, hop)
    ], dtype=np.float32)


def run_wav(model, fe, dev, path, threshold, boost):
    print(f"[file] {path}")
    arr, sr = load_audio_any(Path(path))
    print(f"[audio] duration={arr.size / sr:.2f}s sr={sr}Hz rms={np.sqrt(np.mean(arr ** 2)):.4f}")

    if boost:
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
    print(f"  longest consecutive >= {threshold}: {longest}")
    print(f"  windows >= {threshold}     : {(smoothed >= threshold).sum()} / {raw.size}")
    print()

    if longest >= 3 and smoothed.max() >= threshold:
        verdict = f"DRONE DETECTED ({longest} consecutive windows above {threshold})"
    elif raw.size <= 2 and raw.max() >= threshold:
        verdict = f"DRONE DETECTED (p={raw.max():.3f})"
    elif smoothed.max() >= 0.30:
        verdict = f"UNCERTAIN (peak p={smoothed.max():.3f})"
    else:
        verdict = "NO DRONE"
    print(f"  VERDICT: {verdict}")
    print("=" * 60)


def run_mic(model, fe, dev, device_idx, threshold, smoothing):
    win = SR
    step = SR // 2
    audio_q = queue.Queue()
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
                consec = consec + 1 if p >= threshold else 0
                if consec >= smoothing:
                    decision = "DRONE"
                elif consec > 0:
                    decision = "drone (smoothing)"
                else:
                    decision = "no drone"
                print(f"{time.time() - t0:8.1f}  {p:10.4f}  {decision}")


def list_devices():
    print("Audio input devices:")
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            print(f"  [{i}] {d['name']} "
                  f"(channels={d['max_input_channels']}, "
                  f"sr={int(d['default_samplerate'])} Hz)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hub-id", default=DEFAULT_HUB)
    p.add_argument("--wav", default=None)
    p.add_argument("--device", type=int, default=None)
    p.add_argument("--list-devices", action="store_true")
    p.add_argument("--threshold", type=float, default=0.25)
    p.add_argument("--smoothing", type=int, default=2)
    p.add_argument("--boost-drone", action="store_true")
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
