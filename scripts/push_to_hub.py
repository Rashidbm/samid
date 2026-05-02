"""
Push the trained AST drone-detector to HuggingFace Hub.

One-time setup:
  1. Create a free account at https://huggingface.co/join
  2. Create an access token: https://huggingface.co/settings/tokens (Write scope)
  3. Run:  uv run huggingface-cli login   (paste the token)

Then run this script:
  uv run python scripts/push_to_hub.py --repo <your-username>/samid-drone-detector

Your teammates can then download it with one line:
  AutoModelForAudioClassification.from_pretrained("<your-username>/samid-drone-detector")

Public repos are free. Private repos are also free for individual accounts.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import json

import torch
from huggingface_hub import HfApi, create_repo
from transformers import ASTConfig, ASTFeatureExtractor, ASTForAudioClassification

from src.config import CFG


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True,
                   help="HF repo ID, e.g. 'username/samid-drone-detector'")
    p.add_argument("--ckpt", type=Path,
                   default=Path("runs/20260429-112104/best_calibrated.pt"))
    p.add_argument("--private", action="store_true",
                   help="Make repo private (default: public)")
    args = p.parse_args()

    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)

    # Build model fresh, load weights
    fe = ASTFeatureExtractor.from_pretrained(CFG.backbone)
    cfg = ASTConfig.from_pretrained(CFG.backbone)
    cfg.num_labels = 2
    cfg.id2label = {0: "no_drone", 1: "drone"}
    cfg.label2id = {"no_drone": 0, "drone": 1}
    model = ASTForAudioClassification.from_pretrained(
        CFG.backbone, config=cfg, ignore_mismatched_sizes=True,
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Create repo if not exists
    print(f"Creating repo: {args.repo} (private={args.private})")
    create_repo(args.repo, private=args.private, exist_ok=True)

    # Push model + feature extractor
    print("Pushing model and feature extractor…")
    model.push_to_hub(args.repo)
    fe.push_to_hub(args.repo)

    # Push a README so the page isn't empty
    readme = f"""---
language: en
tags:
  - audio
  - audio-classification
  - drone-detection
  - acoustic
license: mit
---

# Samid Drone Detector

Audio Spectrogram Transformer fine-tuned on geronimobasso/drone-audio-detection-samples
for binary acoustic drone detection.

## Quick use

```python
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import torch, soundfile as sf

model = AutoModelForAudioClassification.from_pretrained("{args.repo}")
fe = AutoFeatureExtractor.from_pretrained("{args.repo}")

audio, sr = sf.read("clip.wav")  # any mono 16 kHz wav
inputs = fe(audio, sampling_rate=16000, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
prob_drone = torch.softmax(logits, dim=-1)[0, 1].item()
print(f"p(drone) = {{prob_drone:.4f}}")
```

## Performance

On the geronimobasso held-out test set (18,032 unseen samples):
- F1 = 0.9974
- Precision = 1.0000
- Recall = 0.9948
- PR-AUC = 1.0000

⚠️ These numbers are within-dataset. Real-world performance depends on
microphone, environment, and distance — expect 10–25 percentage points
of F1 drop in adverse conditions.

## Caveats

- Trained on 1.0-second windows at 16 kHz mono.
- May fire on rotor-like sounds (helicopters, drills, lawnmowers).
- Range and accuracy degrade in wind > 5 m/s.
"""
    HfApi().upload_file(
        path_or_fileobj=readme.encode(),
        path_in_repo="README.md",
        repo_id=args.repo,
    )

    print(f"\nDone. Your teammates can now run:")
    print(f"  from transformers import AutoModelForAudioClassification, AutoFeatureExtractor")
    print(f'  model = AutoModelForAudioClassification.from_pretrained("{args.repo}")')
    print(f'  fe    = AutoFeatureExtractor.from_pretrained("{args.repo}")')
    print(f"\nView it at: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
