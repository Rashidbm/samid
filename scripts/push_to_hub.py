from __future__ import annotations
import argparse
from pathlib import Path

import torch
from huggingface_hub import HfApi, create_repo
from transformers import ASTConfig, ASTFeatureExtractor, ASTForAudioClassification

from src.config import CFG


README = """---
language: en
tags:
  - audio
  - audio-classification
  - drone-detection
  - acoustic
license: mit
---

# Samid drone detector

Audio Spectrogram Transformer fine-tuned for binary acoustic drone detection.

## Quick use

```python
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import torch, soundfile as sf

model = AutoModelForAudioClassification.from_pretrained("{repo}")
fe = AutoFeatureExtractor.from_pretrained("{repo}")

audio, sr = sf.read("clip.wav")
inputs = fe(audio, sampling_rate=16000, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
print(f"p(drone) = {{torch.softmax(logits, dim=-1)[0, 1].item():.4f}}")
```

## Training

- Backbone: `MIT/ast-finetuned-audioset-10-10-0.4593`
- Datasets: geronimobasso/drone-audio-detection-samples,
  ahlab-drone-project/DroneAudioSet (splits 1-20)
- Augmentations applied symmetrically across both classes:
  codec round-trip, synthetic RIR, random EQ, FilterAugment, Patchout,
  SpecAugment, Mixup. Asymmetric: urban-noise overlay onto drone class.

## Performance

| Test | Result |
|---|---|
| NUS DroneAudioSet held-out (48 clips) | 100% detection |
| Geronimobasso sanity (50 random clips) | 24/25 drone, 25/25 no-drone |

## Recommended inference

For long audio, slide a 1-second window with 0.5s hop, apply a median
filter to per-window probabilities, and require N consecutive windows
above threshold. See `scripts/standalone_inference.py` in the repo.

Trained on 1.0s windows at 16 kHz mono. May fire on rotor-like sounds.
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--ckpt", type=Path,
                   default=Path("runs/20260429-112104/best_rw2.pt"))
    p.add_argument("--private", action="store_true")
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)

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

    create_repo(args.repo, private=args.private, exist_ok=True)
    model.push_to_hub(args.repo)
    fe.push_to_hub(args.repo)

    HfApi().upload_file(
        path_or_fileobj=README.format(repo=args.repo).encode(),
        path_in_repo="README.md",
        repo_id=args.repo,
    )

    print(f"Done. https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
