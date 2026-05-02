from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

from src.triangulation import localize, estimate_tdoas


SR = 16_000
DEFAULT_HUB = "Rashidbm/samid-drone-detector"


def load_multichannel(path, target_sr=SR):
    arr, sr = sf.read(str(path), dtype="float32", always_2d=True)
    arr = arr.T  # (channels, samples)
    if sr != target_sr:
        try:
            import librosa
            arr = np.stack([
                librosa.resample(c, orig_sr=sr, target_sr=target_sr) for c in arr
            ])
        except ImportError:
            arr = np.stack([
                np.interp(
                    np.linspace(0, c.size, int(c.size * target_sr / sr), endpoint=False),
                    np.arange(c.size), c,
                ).astype(np.float32) for c in arr
            ])
    return arr.astype(np.float32, copy=False), target_sr


def detect_per_window(model, fe, dev, channels, win=SR, hop=SR // 2):
    """Run detection on each channel, average per-window probabilities across channels."""
    n_ch, n_samples = channels.shape
    if n_samples < win:
        pad = np.zeros((n_ch, win - n_samples), dtype=np.float32)
        channels = np.concatenate([channels, pad], axis=1)
        n_samples = channels.shape[1]

    probs_per_window = []
    times = []
    for s in range(0, n_samples - win + 1, hop):
        ch_probs = []
        for c in range(n_ch):
            chunk = channels[c, s:s + win]
            feats = fe(chunk, sampling_rate=SR, return_tensors="pt")
            with torch.inference_mode():
                p = float(F.softmax(
                    model(input_values=feats["input_values"].to(dev)).logits, -1
                )[0, 1])
            ch_probs.append(p)
        probs_per_window.append(np.mean(ch_probs))
        times.append(s / SR)
    return np.asarray(probs_per_window), np.asarray(times)


def median_filter_1d(x, k=5):
    if x.size == 0 or k <= 1:
        return x
    half = k // 2
    pad = np.pad(x, (half, half), mode="edge")
    return np.array([np.median(pad[i:i + k]) for i in range(x.size)])


def localize_window(channels, window_start, window_len, mic_positions, max_tau):
    sigs = channels[:, window_start:window_start + window_len]
    if sigs.shape[1] < window_len:
        return None
    return localize(sigs, mic_positions, fs=SR, max_tau=max_tau)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wav", required=True, type=Path,
                   help="multi-channel WAV (one channel per microphone)")
    p.add_argument("--mics", required=True, type=Path,
                   help="JSON file with list of [x, y, z] mic positions in metres")
    p.add_argument("--threshold", type=float, default=0.25)
    p.add_argument("--max-tau", type=float, default=0.05)
    p.add_argument("--hub-id", default=DEFAULT_HUB)
    p.add_argument("--track-window-sec", type=float, default=2.0,
                   help="window length for per-frame triangulation in trajectory")
    p.add_argument("--track-hop-sec", type=float, default=0.5)
    p.add_argument("--out-json", type=Path, default=None)
    p.add_argument("--no-trajectory", action="store_true",
                   help="skip per-frame triangulation, just report final position")
    args = p.parse_args()

    mic_positions = np.asarray(json.loads(args.mics.read_text()), dtype=np.float64)
    channels, sr = load_multichannel(args.wav)

    if channels.shape[0] != mic_positions.shape[0]:
        raise SystemExit(
            f"channel count {channels.shape[0]} != mic count {mic_positions.shape[0]}"
        )

    print(f"channels: {channels.shape[0]}  duration: {channels.shape[1] / sr:.2f}s  sr: {sr}Hz")
    print(f"mic positions:")
    for i, m in enumerate(mic_positions):
        print(f"  mic {i}: {m.tolist()}")
    print()

    if torch.backends.mps.is_available():
        dev = torch.device("mps")
    elif torch.cuda.is_available():
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")
    print(f"loading detector on {dev}…")
    model = AutoModelForAudioClassification.from_pretrained(args.hub_id).eval().to(dev)
    fe = AutoFeatureExtractor.from_pretrained(args.hub_id)

    print("running detection per window across all channels…")
    raw_probs, times = detect_per_window(model, fe, dev, channels)
    smoothed = median_filter_1d(raw_probs, k=5)

    longest_run = 0
    cur = 0
    for v in smoothed >= args.threshold:
        if v:
            cur += 1
            longest_run = max(longest_run, cur)
        else:
            cur = 0

    print()
    print(f"window-level detection (averaged across {channels.shape[0]} channels):")
    print(f"  raw max p(drone)             : {raw_probs.max():.4f}")
    print(f"  smoothed max                 : {smoothed.max():.4f}")
    print(f"  longest consecutive >= {args.threshold}: {longest_run}")
    print(f"  windows >= {args.threshold}             : {(smoothed >= args.threshold).sum()} / {smoothed.size}")

    drone_detected = longest_run >= 3 and smoothed.max() >= args.threshold
    if not drone_detected:
        print()
        print("VERDICT: NO DRONE — skipping triangulation")
        if args.out_json:
            args.out_json.write_text(json.dumps({
                "drone_detected": False,
                "max_p_drone": float(smoothed.max()),
                "longest_run": int(longest_run),
            }, indent=2))
        return 0

    print()
    print("DRONE DETECTED — triangulating…")
    print()

    # Single-shot localization on the loudest 1-second window
    loud_idx = int(np.argmax(raw_probs))
    loud_start = int(times[loud_idx] * SR)
    loud_loc = localize_window(channels, loud_start, SR, mic_positions, args.max_tau)
    print(f"snapshot at peak (t={times[loud_idx]:.2f}s):")
    print(f"  position : {loud_loc.position.round(3).tolist()} m")
    print(f"  TDoAs    : {(loud_loc.tdoas * 1e3).round(3).tolist()} ms")
    print(f"  residual : {loud_loc.residual:.4f}")

    # Per-frame trajectory
    trajectory = []
    if not args.no_trajectory:
        print()
        print(f"computing trajectory ({args.track_window_sec}s windows, {args.track_hop_sec}s hop)…")
        win_samples = int(args.track_window_sec * SR)
        hop_samples = int(args.track_hop_sec * SR)
        for s in range(0, channels.shape[1] - win_samples + 1, hop_samples):
            t = s / SR
            # only triangulate where detection is confident
            time_idx = int(t / (SR / SR / 2)) if SR > 0 else 0
            time_idx = min(int(t / (1 / (SR / hop_samples))), smoothed.size - 1)
            # simpler: find nearest probability time
            nearest = int(np.argmin(np.abs(times - t)))
            if smoothed[nearest] < args.threshold:
                continue
            try:
                loc = localize_window(channels, s, win_samples, mic_positions, args.max_tau)
                trajectory.append({
                    "t": float(t),
                    "p_drone": float(smoothed[nearest]),
                    "position": loc.position.round(3).tolist(),
                    "residual": float(loc.residual),
                })
            except Exception as exc:
                print(f"  t={t:.1f}s: localization failed ({exc})")

        if trajectory:
            positions = np.asarray([t["position"] for t in trajectory])
            print(f"trajectory: {len(trajectory)} frames")
            print(f"  start position: {positions[0].round(2).tolist()} m")
            print(f"  end position  : {positions[-1].round(2).tolist()} m")
            if len(positions) >= 2:
                total_distance = float(np.linalg.norm(positions[-1] - positions[0]))
                duration = trajectory[-1]["t"] - trajectory[0]["t"]
                avg_speed = total_distance / max(duration, 1e-3)
                print(f"  travelled     : {total_distance:.2f} m over {duration:.1f} s "
                      f"(avg {avg_speed:.2f} m/s)")

    output = {
        "drone_detected": True,
        "max_p_drone": float(smoothed.max()),
        "longest_run_above_threshold": int(longest_run),
        "snapshot_position": loud_loc.position.round(3).tolist(),
        "snapshot_time_s": float(times[loud_idx]),
        "trajectory": trajectory,
    }
    if args.out_json:
        args.out_json.write_text(json.dumps(output, indent=2))
        print(f"\nresults saved to {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
