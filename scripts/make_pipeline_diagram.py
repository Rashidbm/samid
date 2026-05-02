from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def draw_box(ax, x, y, w, h, text, color, status):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.05",
        linewidth=1.5,
        edgecolor="#2c3e50",
        facecolor=color,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2 + 0.05, text,
            ha="center", va="center", fontsize=9, fontweight="bold",
            color="#1a1a1a")
    if status:
        ax.text(x + w / 2, y + h / 2 - 0.18, status,
                ha="center", va="center", fontsize=7, style="italic",
                color="#444")


def arrow(ax, x1, y1, x2, y2, label=None):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="->", mutation_scale=15,
        linewidth=1.3, color="#2c3e50",
    )
    ax.add_patch(a)
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + 0.05, my, label, fontsize=7, color="#666", style="italic")


def main():
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")

    BUILT = "#a3e4a3"        # green
    PARTIAL = "#fff3a0"      # yellow
    PLANNED = "#e8e8e8"      # grey

    # Title
    ax.text(7, 9.55, "Acoustic drone detection pipeline",
            ha="center", fontsize=15, fontweight="bold", color="#1a1a1a")
    ax.text(7, 9.20, "Audio in → drone detection → type ID → localisation → handoff",
            ha="center", fontsize=10, color="#555", style="italic")

    # 1. Microphone input
    draw_box(ax, 0.3, 7.8, 2.0, 0.9, "Microphone input", "#cce5ff", "single mic\nor array")
    # 2. Sliding window
    draw_box(ax, 2.8, 7.8, 2.2, 0.9, "Sliding window\n1.0s, 0.5s hop", BUILT, "BUILT")
    # 3. AST detector
    draw_box(ax, 5.5, 7.8, 2.2, 0.9, "AST detector\n(drone yes/no)", BUILT, "BUILT")
    # 4. Median + N-consec
    draw_box(ax, 8.2, 7.8, 2.4, 0.9, "Median filter +\nN-consecutive rule", BUILT, "BUILT")
    # 5. Decision branch
    draw_box(ax, 11.1, 7.8, 2.4, 0.9, "Drone decision", BUILT, "BUILT")

    arrow(ax, 2.3, 8.25, 2.8, 8.25)
    arrow(ax, 5.0, 8.25, 5.5, 8.25)
    arrow(ax, 7.7, 8.25, 8.2, 8.25)
    arrow(ax, 10.6, 8.25, 11.1, 8.25)

    # if drone detected branch
    arrow(ax, 12.3, 7.8, 12.3, 6.6, "if drone")

    # Type classifier (LoRA)
    draw_box(ax, 10.7, 5.6, 3.2, 0.9, "LoRA type classifier\n(DJI / Shahed / custom)", PLANNED, "PLANNED")
    arrow(ax, 12.3, 6.5, 12.3, 5.6 + 0.9)

    # Open-set check
    draw_box(ax, 10.7, 4.3, 3.2, 0.9,
             "Open-set check\n(unknown UAV → threat)", PLANNED, "PLANNED")
    arrow(ax, 12.3, 5.6, 12.3, 4.3 + 0.9)

    # Multi-mic triangulation
    draw_box(ax, 6.5, 5.6, 3.2, 0.9,
             "Multi-mic triangulation\nGCC-PHAT + LSQ", PARTIAL,
             "math BUILT,\nneeds mic array")
    arrow(ax, 11.1, 6.05, 9.7, 6.05)

    # Decoy detection
    draw_box(ax, 6.5, 4.3, 3.2, 0.9,
             "Decoy detection\n(spatial coherence)", PLANNED, "PLANNED")
    arrow(ax, 8.1, 5.6, 8.1, 4.3 + 0.9)

    # Cueing handoff
    draw_box(ax, 4.0, 2.8, 4.0, 0.9,
             "Cueing handoff\n{class, position, trajectory}", PLANNED, "PLANNED")
    arrow(ax, 8.1, 4.3, 7.0, 3.7)
    arrow(ax, 12.3, 4.3, 7.0, 3.5)

    # Dashboard
    draw_box(ax, 4.0, 1.4, 4.0, 0.9, "Operator dashboard",
             PLANNED, "PLANNED")
    arrow(ax, 6.0, 2.8, 6.0, 2.3)

    # Legend
    legend_y = 0.3
    ax.add_patch(mpatches.Rectangle((0.3, legend_y), 0.4, 0.25, facecolor=BUILT, edgecolor="#2c3e50"))
    ax.text(0.85, legend_y + 0.12, "BUILT", fontsize=9, va="center")
    ax.add_patch(mpatches.Rectangle((1.8, legend_y), 0.4, 0.25, facecolor=PARTIAL, edgecolor="#2c3e50"))
    ax.text(2.35, legend_y + 0.12, "PARTIAL (needs hardware)", fontsize=9, va="center")
    ax.add_patch(mpatches.Rectangle((5.5, legend_y), 0.4, 0.25, facecolor=PLANNED, edgecolor="#2c3e50"))
    ax.text(6.05, legend_y + 0.12, "PLANNED", fontsize=9, va="center")

    out = Path(__file__).resolve().parent.parent / "configs" / "pipeline_diagram.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
