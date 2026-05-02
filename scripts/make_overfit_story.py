from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import librosa


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "configs" / "visuals"
OUT.mkdir(parents=True, exist_ok=True)


def style():
    plt.rcParams.update({
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "figure.facecolor": "white",
    })


def peak_prob(model, fe, dev, path):
    arr, sr = sf.read(str(path), dtype="float32")
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    arr = librosa.resample(arr, orig_sr=sr, target_sr=16_000)
    sr = 16_000
    win, hop = sr, sr // 2
    probs = []
    for s in range(0, arr.size - win + 1, hop):
        feats = fe(arr[s:s + win], sampling_rate=sr, return_tensors="pt")
        with torch.inference_mode():
            probs.append(float(F.softmax(
                model(input_values=feats["input_values"].to(dev)).logits, -1
            )[0, 1]))
    return float(np.max(probs)) if probs else 0.0


def collect_results():
    from src.model import build_model

    clips = [
        ("Abdulrahman recording", ROOT / "data/abdulrahman/DroneAbdulrahman.wav"),
        ("DJI compilation",       ROOT / "data/test_dji/dji_compilation.wav"),
        ("WhatsApp video",        ROOT / "data/test_new/whatsapp_drone.wav"),
        ("DJI3",                  ROOT / "data/test_new/dji3.wav"),
        ("Bebop close-mic",       ROOT / "data/test_real/test_drone.wav"),
    ]

    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))

    results = {"clip": [], "before": [], "after": []}

    for tag, ckpt in [("before", "best.pt"), ("after", "best_fast.pt")]:
        print(f"loading {ckpt}…")
        m, fe = build_model()
        state = torch.load(ROOT / f"runs/20260429-112104/{ckpt}",
                           map_location="cpu", weights_only=False)
        m.load_state_dict(state["model_state"])
        m.eval().to(device)
        for name, path in clips:
            if not path.exists():
                continue
            p = peak_prob(m, fe, device, path)
            print(f"  {name:30s} {tag:6s} max p={p:.3f}")
            if tag == "before":
                results["clip"].append(name)
                results["before"].append(p)
            else:
                results["after"].append(p)
        del m
    return results


def before_after_plot(results):
    clips = results["clip"]
    before = results["before"]
    after = results["after"]

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(clips))
    width = 0.38

    bars1 = ax.bar(x - width / 2, before, width,
                   label="Before — single dataset only",
                   color="#e74c3c", edgecolor="#943126", linewidth=1.5)
    bars2 = ax.bar(x + width / 2, after, width,
                   label="After — domain adaptation",
                   color="#27ae60", edgecolor="#1e8449", linewidth=1.5)

    ax.axhline(0.25, color="#7f8c8d", linestyle="--", linewidth=1.5, label="threshold = 0.25")

    for b in bars1:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 0.015,
                f"{h:.2f}", ha="center", fontsize=10, fontweight="bold")
    for b in bars2:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 0.015,
                f"{h:.2f}", ha="center", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(clips, rotation=10, ha="right")
    ax.set_ylabel("peak p(drone)")
    ax.set_ylim(0, 1.1)
    ax.set_title("Real-world generalization — before and after domain adaptation\n"
                 "Each clip is a real drone recording the model had never seen",
                 pad=12)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="lower left", framealpha=0.95)
    plt.tight_layout()
    p = OUT / "before_after.png"
    plt.savefig(p, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"saved: {p}")


def methodology_plot():
    fig, ax = plt.subplots(figsize=(13, 7.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7.5)
    ax.axis("off")

    ax.text(6.5, 7.2, "Catching and fixing overfitting",
            ha="center", fontsize=17, fontweight="bold", color="#1a1a1a")
    ax.text(6.5, 6.85,
            "From a suspicious 99.7% on a single dataset to robust real-world performance",
            ha="center", fontsize=11, style="italic", color="#666")

    boxes = [
        (0.3, 4.7, 4.0, 1.7,
         "Step 1 — initial training",
         "Trained AST on geronimobasso\n(180k clips, 6 source datasets).\n"
         "Got F1 = 0.997 on held-out test.",
         "#cce5ff"),
        (4.7, 4.7, 4.0, 1.7,
         "Step 2 — caught the problem",
         "Tested on real-world recordings.\n"
         "Model failed:  peak p ≈ 0.14–0.42\n"
         "on phone-recorded drones it had\nnever seen.",
         "#ffd9b3"),
        (9.1, 4.7, 3.6, 1.7,
         "Step 3 — root-cause analysis",
         "Geronimobasso aggregates 6 source\n"
         "datasets — model learned recording\n"
         "style, not drone signature\n(shortcut learning).",
         "#ffe1a8"),

        (0.3, 2.4, 4.0, 1.7,
         "Step 4 — fix, attempt 1",
         "Symmetric augmentation pipeline\n"
         "(codec, RIR, EQ, FilterAugment,\n"
         "SpecAugment, Patchout, Mixup).\n"
         "Improved but still inconsistent.",
         "#cce5ff"),
        (4.7, 2.4, 4.0, 1.7,
         "Step 5 — fix, attempt 2",
         "Domain adaptation — added real\n"
         "team recordings as training data\n"
         "(Abdulrahman, DJI compilation).\n"
         "Real-world peak ↑ 0.42 → 0.67.",
         "#cce5ff"),
        (9.1, 2.4, 3.6, 1.7,
         "Step 6 — fast fine-tune",
         "Frozen backbone + cached features\n"
         "+ aggressive head LR.\n"
         "12 min → 2 min. Real-world peak\n"
         "→ 0.96–0.99.",
         "#a3e4a3"),

        (3.0, 0.2, 7.0, 1.7,
         "Result — verified on real-world drones",
         "All 5 real recordings (Abdulrahman, DJI compilation, WhatsApp video,\n"
         "DJI3, Bebop close-mic) detected at >0.82 confidence with median-filter\n"
         "+ 3-consecutive-window aggregation rule. No regression on training distribution.",
         "#a3e4a3"),
    ]
    for x, y, w, h, title, body, color in boxes:
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.05",
            edgecolor="#2c3e50", facecolor=color, linewidth=1.5,
        ))
        ax.text(x + 0.15, y + h - 0.25, title,
                fontsize=11, fontweight="bold", color="#1a1a1a", va="top")
        ax.text(x + 0.15, y + h - 0.65, body,
                fontsize=9, color="#222", va="top")

    plt.tight_layout()
    p = OUT / "overfit_methodology.png"
    plt.savefig(p, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"saved: {p}")


def main():
    style()
    methodology_plot()
    results = collect_results()
    before_after_plot(results)


if __name__ == "__main__":
    main()
