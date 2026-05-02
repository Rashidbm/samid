from pathlib import Path
import sys
import time

from datasets import load_dataset


ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "geronimobasso"
MARKER = ROOT / "data" / ".geronimobasso.done"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def main():
    if MARKER.exists():
        print(f"already downloaded: {MARKER}")
        return 0

    for attempt in range(1, 51):
        try:
            t0 = time.time()
            ds = load_dataset(
                "geronimobasso/drone-audio-detection-samples",
                cache_dir=str(CACHE_DIR),
            )
            split = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]
            _ = split[0]["audio"]["array"]
            print(f"loaded {len(split):,} samples in {(time.time() - t0) / 60:.1f}m")
            MARKER.write_text(time.strftime("%Y-%m-%dT%H:%M:%S\n"))
            return 0
        except Exception as exc:
            wait = min(60, 2 ** min(attempt, 6))
            print(f"attempt {attempt}: {exc}; retry in {wait}s")
            time.sleep(wait)

    print("exceeded max retries")
    return 1


if __name__ == "__main__":
    sys.exit(main())
