"""
Download geronimobasso/drone-audio-detection-samples from HuggingFace,
with auto-retry on network errors. Writes a marker file `.geronimobasso.done`
when the dataset is fully present and loadable.

Resilient to:
  - mid-download disconnects (HF datasets resumes from the cache)
  - partial files (HF detects + retries)
  - script restarts (idempotent — does nothing if marker already there)

License: MIT
Size: ~6.81 GB
Samples: 180,320 (163,591 drone / 16,729 no-drone)
Format: 16 kHz, mono, 16-bit WAV
"""

from pathlib import Path
import time
import sys

from datasets import load_dataset
from rich.console import Console

console = Console()

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "geronimobasso"
MARKER = ROOT / "data" / ".geronimobasso.done"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def main() -> int:
    if MARKER.exists():
        console.print(f"[green]marker present, nothing to do: {MARKER}[/green]")
        return 0

    console.rule("[bold cyan]Downloading geronimobasso/drone-audio-detection-samples")
    console.print(f"Cache dir: {CACHE_DIR}")
    console.print("Expected size: ~6.81 GB. Resumable on restart.\n")

    attempts = 0
    max_attempts = 50
    while attempts < max_attempts:
        attempts += 1
        try:
            t0 = time.time()
            ds = load_dataset(
                "geronimobasso/drone-audio-detection-samples",
                cache_dir=str(CACHE_DIR),
            )
            elapsed = time.time() - t0
            console.print(f"\n[green]Loaded in {elapsed/60:.1f} min[/green]")
            console.print(ds)

            # Touch one sample as a final sanity check.
            split = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]
            _ = split[0]["audio"]["array"]
            console.print(f"Total rows: {len(split):,}")

            MARKER.write_text(time.strftime("%Y-%m-%dT%H:%M:%S\n"))
            console.print(f"[bold green]marker written: {MARKER}[/bold green]")
            return 0
        except Exception as exc:
            wait = min(60, 2 ** min(attempts, 6))
            console.print(
                f"[yellow]download error (attempt {attempts}): {exc}\n"
                f"retrying in {wait}s[/yellow]"
            )
            time.sleep(wait)

    console.print("[red]exceeded max retries[/red]")
    return 1


if __name__ == "__main__":
    sys.exit(main())
