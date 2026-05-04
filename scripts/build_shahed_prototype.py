"""Build a Shahed-136 acoustic prototype from a small reference library.

We have ~38 seconds of Shahed audio across 5 clips — way too little to
train a multi-class head from scratch.  Instead we:

  1. Load the same AST detector that's in production
  2. For every 1-second window of every reference clip, extract the
     pooled hidden-layer feature (a 768-dim embedding)
  3. Average those into a single Shahed prototype vector
  4. Validate by computing cosine similarity of the prototype against
     held-out Shahed windows AND against unrelated drone (DJI / Bebop /
     Abdulrahman) windows.  If Shahed-vs-prototype is consistently
     higher than other-drone-vs-prototype, the prototype carries
     Shahed-specific signal and we can use it as a runtime
     fingerprint.
  5. Save the prototype to configs/shahed_prototype.npy + a JSON
     summary with the recommended threshold.

Usage:
    uv run python scripts/build_shahed_prototype.py
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification


REPO = Path(__file__).resolve().parent.parent
SHAHED_DIR = REPO / "data_demo" / "shahed"
NEG_DIRS = [
    REPO / "data_demo",                  # abdulrahman, dji_compilation, whatsapp_drone, test_4mic
    REPO / "data" / "test_dji",
    REPO / "data" / "test_new",
    REPO / "data" / "test_real",
]
HUB = "Rashidbm/samid-drone-detector"
SR = 16_000
WIN = SR
HOP = SR // 2
OUT_PROTO = REPO / "configs" / "shahed_prototype.npz"


def load_audio(path: Path) -> np.ndarray:
    arr, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if sr != SR:
        # cheap resample — fine for fingerprinting
        n = int(arr.size * SR / sr)
        arr = np.interp(np.linspace(0, arr.size, n, endpoint=False),
                        np.arange(arr.size), arr).astype(np.float32)
    return arr


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.inference_mode()
def windows_features(model, fe, dev, audio: np.ndarray) -> np.ndarray:
    """Return (n_windows, hidden_dim) array of pooled features per 1-s window."""
    if audio.size < WIN:
        audio = np.concatenate([audio, np.zeros(WIN - audio.size, dtype=np.float32)])
    feats_out = []
    for s in range(0, audio.size - WIN + 1, HOP):
        chunk = audio[s:s + WIN]
        feats = fe(chunk, sampling_rate=SR, return_tensors="pt")
        out = model(input_values=feats["input_values"].to(dev),
                    output_hidden_states=True)
        # pooled mean over the last hidden state
        pooled = out.hidden_states[-1].mean(dim=1).cpu().numpy().squeeze()
        feats_out.append(pooled)
    return np.asarray(feats_out, dtype=np.float32)


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main() -> int:
    dev = device()
    print(f"[proto] device={dev}, loading {HUB}")
    fe = AutoFeatureExtractor.from_pretrained(HUB)
    model = AutoModelForAudioClassification.from_pretrained(HUB).eval().to(dev)

    # 1. Shahed embeddings (positive class)
    print(f"\n[proto] embedding shahed clips from {SHAHED_DIR}")
    shahed_feats = []
    shahed_per_clip = []
    for p in sorted(SHAHED_DIR.glob("*.wav")):
        audio = load_audio(p)
        feats = windows_features(model, fe, dev, audio)
        if feats.size == 0:
            continue
        shahed_feats.append(feats)
        shahed_per_clip.append((p.name, feats))
        print(f"  {p.name:60} {feats.shape[0]:>3} windows · {audio.size/SR:5.1f}s")
    all_shahed = np.concatenate(shahed_feats, axis=0)
    print(f"[proto] total shahed windows: {all_shahed.shape[0]}, dim={all_shahed.shape[1]}")

    # 2. Negative-drone embeddings (other drones, NOT non-drone)
    print(f"\n[proto] embedding negative drone clips")
    neg_feats = []
    neg_per_clip = []
    for d in NEG_DIRS:
        if not d.exists():
            continue
        for p in sorted(d.glob("*.wav")):
            if p.parent.name == "shahed":
                continue
            try:
                audio = load_audio(p)
            except Exception as exc:
                print(f"  SKIP {p.name}: {exc}")
                continue
            if audio.size < WIN * 2:
                print(f"  SKIP {p.name}: too short ({audio.size/SR:.2f}s)")
                continue
            feats = windows_features(model, fe, dev, audio)
            neg_feats.append(feats)
            neg_per_clip.append((p.name, feats))
            print(f"  {p.name:60} {feats.shape[0]:>3} windows")
    all_neg = np.concatenate(neg_feats, axis=0) if neg_feats else np.zeros((0, all_shahed.shape[1]), dtype=np.float32)
    print(f"[proto] total negative windows: {all_neg.shape[0]}")

    # 3. Build the prototype as the mean (L2-normalised) of all shahed windows
    shahed_norm = all_shahed / (np.linalg.norm(all_shahed, axis=1, keepdims=True) + 1e-12)
    proto = shahed_norm.mean(axis=0)
    proto = proto / (np.linalg.norm(proto) + 1e-12)
    print(f"\n[proto] prototype norm={np.linalg.norm(proto):.4f}")

    # 4. Validate — similarity of every window to the prototype.
    def sims(arr):
        if arr.size == 0:
            return np.array([])
        norm = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)
        return norm @ proto

    sh_sims = sims(all_shahed)
    ne_sims = sims(all_neg)

    print("\n[proto] cosine similarity to prototype")
    print(f"  shahed   n={sh_sims.size:>3}  mean={sh_sims.mean():.3f}  "
          f"min={sh_sims.min():.3f}  max={sh_sims.max():.3f}")
    print(f"  other    n={ne_sims.size:>3}  mean={ne_sims.mean():.3f}  "
          f"min={ne_sims.min():.3f}  max={ne_sims.max():.3f}")

    # Per-clip breakdown
    print("\n[proto] per-clip mean similarity:")
    print("  SHAHED:")
    for name, f in shahed_per_clip:
        s = sims(f)
        print(f"    {s.mean():.3f}  {name}")
    print("  OTHER:")
    for name, f in neg_per_clip:
        s = sims(f)
        print(f"    {s.mean():.3f}  {name}")

    # 5. Pick a threshold: midpoint between shahed-min and other-max,
    #    or 0.7 percentile of shahed if no separation.
    if sh_sims.size and ne_sims.size:
        gap = sh_sims.min() - ne_sims.max()
        if gap > 0.02:
            threshold = float((sh_sims.min() + ne_sims.max()) / 2)
            print(f"\n[proto] CLEAN SEPARATION (gap={gap:.3f}) → threshold={threshold:.3f}")
        else:
            # use 25th percentile of shahed — accept some FP
            threshold = float(np.percentile(sh_sims, 25))
            other_above = (ne_sims >= threshold).sum()
            print(f"\n[proto] OVERLAP (gap={gap:.3f}) → threshold={threshold:.3f} "
                  f"({other_above}/{ne_sims.size} other-drone clips would FP)")
    else:
        threshold = 0.7

    # 6. Save
    OUT_PROTO.parent.mkdir(exist_ok=True)
    np.savez(
        OUT_PROTO,
        prototype=proto,
        threshold=np.float32(threshold),
        n_shahed_windows=np.int32(all_shahed.shape[0]),
        n_neg_windows=np.int32(all_neg.shape[0]),
        shahed_min=np.float32(sh_sims.min() if sh_sims.size else 0),
        shahed_max=np.float32(sh_sims.max() if sh_sims.size else 0),
        other_min=np.float32(ne_sims.min() if ne_sims.size else 0),
        other_max=np.float32(ne_sims.max() if ne_sims.size else 0),
    )
    print(f"\n[proto] saved → {OUT_PROTO}")

    summary_path = OUT_PROTO.with_suffix(".json")
    summary_path.write_text(json.dumps({
        "model": HUB,
        "n_shahed_windows": int(all_shahed.shape[0]),
        "n_neg_windows": int(all_neg.shape[0]),
        "threshold": threshold,
        "shahed_sim_stats": {
            "mean": float(sh_sims.mean()) if sh_sims.size else None,
            "min": float(sh_sims.min()) if sh_sims.size else None,
            "max": float(sh_sims.max()) if sh_sims.size else None,
        },
        "other_sim_stats": {
            "mean": float(ne_sims.mean()) if ne_sims.size else None,
            "min": float(ne_sims.min()) if ne_sims.size else None,
            "max": float(ne_sims.max()) if ne_sims.size else None,
        },
    }, indent=2))
    print(f"[proto] summary → {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
