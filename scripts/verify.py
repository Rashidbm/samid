"""
Verify the geronimobasso dataset is loadable and inspect a few samples.

Run after download_geronimobasso.py finishes.
"""

from pathlib import Path
from collections import Counter

import numpy as np
from datasets import load_dataset
from rich.console import Console
from rich.table import Table

console = Console()

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "geronimobasso"


def main() -> None:
    console.rule("[bold cyan]Verifying geronimobasso dataset")
    ds = load_dataset(
        "geronimobasso/drone-audio-detection-samples",
        cache_dir=str(CACHE_DIR),
    )
    split = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]
    console.print(f"Rows: {len(split):,}")
    console.print(f"Columns: {split.column_names}\n")

    # Label distribution
    labels = split["label"]
    counts = Counter(labels)
    table = Table(title="Label distribution")
    table.add_column("Label")
    table.add_column("Count", justify="right")
    table.add_column("Share", justify="right")
    total = sum(counts.values())
    for k in sorted(counts):
        table.add_row(str(k), f"{counts[k]:,}", f"{counts[k]/total:.1%}")
    console.print(table)

    # Duration distribution by label — confirms the leak we worried about
    if "audioduration (s)" in split.column_names:
        durs = np.asarray(split["audioduration (s)"], dtype=np.float32)
        labs = np.asarray(labels)
        leak = Table(title="Mean / median duration by label (THE LEAK CHECK)")
        leak.add_column("Label")
        leak.add_column("Mean (s)", justify="right")
        leak.add_column("Median (s)", justify="right")
        leak.add_column("Min (s)", justify="right")
        leak.add_column("Max (s)", justify="right")
        for k in sorted(counts):
            mask = labs == k
            d = durs[mask]
            leak.add_row(
                str(k),
                f"{d.mean():.2f}",
                f"{float(np.median(d)):.2f}",
                f"{d.min():.2f}",
                f"{d.max():.2f}",
            )
        console.print(leak)
        console.print(
            "[yellow]If means differ wildly across labels → confirmed duration leak. "
            "Mitigation: hard-crop all clips to 1.0s windows before training.[/yellow]"
        )

    # Touch one audio sample to confirm files load
    sample = split[0]
    audio = sample["audio"]
    arr = audio["array"]
    sr = audio["sampling_rate"]
    console.print(
        f"\nSample 0 → label={sample['label']}, "
        f"len={len(arr)/sr:.2f}s, sr={sr}, dtype={arr.dtype}"
    )


if __name__ == "__main__":
    main()
