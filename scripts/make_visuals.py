from __future__ import annotations
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F


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


def confusion_matrix_plot():
    data = json.loads((ROOT / "runs/20260429-112104/test_results.json").read_text())
    cm = np.asarray(data["thresholds"][0]["confusion"])
    f1 = data["thresholds"][0]["f1"]
    threshold = data["thresholds"][0]["threshold"]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues", aspect="equal")
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "#1a1a1a"
            ax.text(j, i, f"{cm[i, j]:,}",
                    ha="center", va="center", fontsize=20, fontweight="bold",
                    color=color)
    ax.set_xticks([0, 1], ["no drone", "drone"], fontsize=12)
    ax.set_yticks([0, 1], ["no drone", "drone"], fontsize=12)
    ax.set_xlabel("predicted", fontsize=13)
    ax.set_ylabel("actual", fontsize=13)
    ax.set_title(f"Confusion matrix on held-out test set\n"
                 f"F1 = {f1:.4f}  •  threshold = {threshold}  •  18,032 samples",
                 fontsize=13, pad=15)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)
    plt.tight_layout()
    p = OUT / "confusion_matrix.png"
    plt.savefig(p, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"saved: {p}")


def threshold_sweep_plot():
    data = json.loads((ROOT / "runs/20260429-112104/test_results.json").read_text())
    rows = data["thresholds"]
    thr = [r["threshold"] for r in rows]
    f1 = [r["f1"] for r in rows]
    prec = [r["precision"] for r in rows]
    rec = [r["recall"] for r in rows]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(thr, f1, "o-", label="F1 score", linewidth=2.5, markersize=10, color="#2c3e50")
    ax.plot(thr, prec, "s-", label="Precision", linewidth=2, markersize=8, color="#27ae60")
    ax.plot(thr, rec, "^-", label="Recall", linewidth=2, markersize=8, color="#e74c3c")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_title("Detection performance vs decision threshold\n"
                 "PR-AUC = 1.0000  •  ROC-AUC = 0.9997", pad=12)
    ax.set_ylim(0.985, 1.001)
    ax.set_xticks(thr)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="lower left", framealpha=0.95)
    for x, y in zip(thr, f1):
        ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=9)
    plt.tight_layout()
    p = OUT / "threshold_sweep.png"
    plt.savefig(p, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"saved: {p}")


def loss_curve_plot():
    hist = json.loads((ROOT / "runs/20260429-112104/history.json").read_text())
    steps = [h["step"] for h in hist]
    losses = [h["loss"] for h in hist]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(steps, losses, "o-", linewidth=2.5, markersize=11, color="#2c3e50")
    for x, y in zip(steps, losses):
        ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points",
                    xytext=(8, 8), fontsize=10)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("Training loss trajectory (focal loss, class-weighted)", pad=12)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    plt.tight_layout()
    p = OUT / "loss_curve.png"
    plt.savefig(p, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"saved: {p}")


def abdulrahman_timeline_plot():
    from src.model import build_model
    import librosa

    audio_path = ROOT / "data/abdulrahman/DroneAbdulrahman.wav"
    if not audio_path.exists():
        print(f"skip timeline (file not found: {audio_path})")
        return

    print("loading model for timeline plot…")
    m, fe = build_model()
    state = torch.load(ROOT / "runs/20260429-112104/best_fast.pt",
                       map_location="cpu", weights_only=False)
    m.load_state_dict(state["model_state"])
    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))
    m.eval().to(device)

    arr, sr = sf.read(str(audio_path), dtype="float32")
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    arr = librosa.resample(arr, orig_sr=sr, target_sr=16_000)
    sr = 16_000

    win, hop = sr, sr // 2
    probs, times = [], []
    print("running per-window inference…")
    for s in range(0, arr.size - win + 1, hop):
        feats = fe(arr[s:s + win], sampling_rate=sr, return_tensors="pt")
        with torch.inference_mode():
            p = float(F.softmax(
                m(input_values=feats["input_values"].to(device)).logits, -1
            )[0, 1])
        probs.append(p)
        times.append(s / sr)
    probs = np.asarray(probs)
    times = np.asarray(times)

    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.fill_between(times, 0, probs, alpha=0.4, color="#3498db", linewidth=0)
    ax.plot(times, probs, "-", linewidth=2.5, color="#2c3e50")
    ax.axhline(0.25, color="#e74c3c", linestyle="--", linewidth=1.5,
               label="threshold = 0.25")
    ax.axhline(0.5, color="#27ae60", linestyle=":", linewidth=1.5,
               label="strict = 0.5")

    peak = times[np.argmax(probs)]
    ax.annotate(f"peak {probs.max():.2f}\n@ t={peak:.0f}s",
                xy=(peak, probs.max()),
                xytext=(peak + 3, probs.max() + 0.05),
                fontsize=11, fontweight="bold",
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#1a1a1a"))

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("p(drone)")
    ax.set_title("Drone flyby — per-window probability over time\n"
                 "Real-world phone recording, 41 seconds", pad=12)
    ax.set_xlim(0, times[-1])
    ax.set_ylim(0, 1.02)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="upper right", framealpha=0.95)

    ax.text(2, 0.92, "drone\napproaching", fontsize=10, style="italic", color="#666")
    ax.text(peak - 2, 0.45, "drone overhead", fontsize=10, style="italic", color="#666")
    ax.text(peak + 8, 0.30, "drone leaving", fontsize=10, style="italic", color="#666")

    plt.tight_layout()
    p = OUT / "abdulrahman_timeline.png"
    plt.savefig(p, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"saved: {p}")


def metrics_summary_plot():
    data = json.loads((ROOT / "runs/20260429-112104/test_results.json").read_text())
    best = data["thresholds"][0]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")

    metrics = [
        ("F1 score", f"{best['f1']:.4f}"),
        ("Precision", f"{best['precision']:.4f}"),
        ("Recall", f"{best['recall']:.4f}"),
        ("Accuracy", f"{best['accuracy']:.4f}"),
        ("PR-AUC", f"{data['pr_auc']:.4f}"),
        ("ROC-AUC", f"{data['roc_auc']:.4f}"),
    ]
    for i, (k, v) in enumerate(metrics):
        col, row = i % 3, i // 3
        x = 0.05 + col * 0.32
        y = 0.65 - row * 0.42
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), 0.28, 0.32,
            boxstyle="round,pad=0.02",
            edgecolor="#2c3e50", facecolor="#ecf0f1", linewidth=1.5,
            transform=ax.transAxes,
        ))
        ax.text(x + 0.14, y + 0.22, k, fontsize=12, fontweight="bold",
                ha="center", va="center", color="#34495e",
                transform=ax.transAxes)
        ax.text(x + 0.14, y + 0.09, v, fontsize=22, fontweight="bold",
                ha="center", va="center", color="#2c3e50",
                transform=ax.transAxes)

    ax.text(0.5, 1.0, "Held-out test performance",
            fontsize=16, fontweight="bold", ha="center", transform=ax.transAxes)
    ax.text(0.5, 0.95,
            "18,032 unseen samples  •  threshold 0.3  •  zero false positives",
            fontsize=11, ha="center", color="#666", style="italic",
            transform=ax.transAxes)

    plt.tight_layout()
    p = OUT / "metrics_summary.png"
    plt.savefig(p, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"saved: {p}")


def main():
    style()
    confusion_matrix_plot()
    threshold_sweep_plot()
    loss_curve_plot()
    metrics_summary_plot()
    abdulrahman_timeline_plot()
    print(f"\nall visuals saved to: {OUT}")


if __name__ == "__main__":
    main()
