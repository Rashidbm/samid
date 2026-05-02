from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

from src.triangulation import localize


SR = 16_000
DEFAULT_HUB = "Rashidbm/samid-drone-detector"


def load_multichannel(path, target_sr=SR):
    arr, sr = sf.read(str(path), dtype="float32", always_2d=True)
    arr = arr.T  # (channels, samples)
    if sr != target_sr:
        try:
            import librosa
            arr = np.stack([
                librosa.resample(c, orig_sr=sr, target_sr=target_sr)
                for c in arr
            ])
        except ImportError:
            arr = np.stack([
                np.interp(
                    np.linspace(0, c.size, int(c.size * target_sr / sr), endpoint=False),
                    np.arange(c.size), c,
                ).astype(np.float32)
                for c in arr
            ])
    return arr.astype(np.float32, copy=False), target_sr


def detect_drone(model, fe, dev, signals):
    win = SR
    if signals.shape[1] < win:
        ch = np.concatenate([signals[0], np.zeros(win - signals.shape[1], dtype=np.float32)])
    else:
        ch = signals[0, :win]
    feats = fe(ch, sampling_rate=SR, return_tensors="pt")
    with torch.inference_mode():
        out = model(input_values=feats["input_values"].to(dev))
    return float(torch.softmax(out.logits, dim=-1)[0, 1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wav", required=True, type=Path,
                   help="multi-channel WAV (one channel per microphone)")
    p.add_argument("--mics", required=True, type=Path,
                   help="JSON file with list of [x, y, z] mic positions in metres")
    p.add_argument("--detect", action="store_true",
                   help="also run drone detector before localizing")
    p.add_argument("--hub-id", default=DEFAULT_HUB)
    p.add_argument("--max-tau", type=float, default=0.05,
                   help="max time-of-arrival difference between mic pairs (sec)")
    args = p.parse_args()

    mic_positions = np.asarray(json.loads(args.mics.read_text()), dtype=np.float64)
    signals, sr = load_multichannel(args.wav)

    if signals.shape[0] != mic_positions.shape[0]:
        raise SystemExit(
            f"channel count {signals.shape[0]} != mic count {mic_positions.shape[0]}"
        )

    print(f"channels: {signals.shape[0]}  duration: {signals.shape[1] / sr:.2f}s  sr: {sr}Hz")
    print(f"mic positions:\n{mic_positions}")

    if args.detect:
        if torch.backends.mps.is_available():
            dev = torch.device("mps")
        elif torch.cuda.is_available():
            dev = torch.device("cuda")
        else:
            dev = torch.device("cpu")
        print(f"loading detector: {args.hub_id}")
        model = AutoModelForAudioClassification.from_pretrained(args.hub_id).eval().to(dev)
        fe = AutoFeatureExtractor.from_pretrained(args.hub_id)
        p_drone = detect_drone(model, fe, dev, signals)
        print(f"p(drone) on first second of channel 0: {p_drone:.4f}")
        if p_drone < 0.3:
            print("no drone detected; skipping localization")
            return 0

    loc = localize(signals, mic_positions, fs=sr, max_tau=args.max_tau)
    print()
    print(f"position : {loc.position.round(3).tolist()} m")
    print(f"TDoAs    : {(loc.tdoas * 1e3).round(3).tolist()} ms")
    print(f"residual : {loc.residual:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
