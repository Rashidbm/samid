"""
Binary AST classifier.

Uses MIT/ast-finetuned-audioset-10-10-0.4593 as backbone. We:
  1. Replace the AudioSet 527-way classifier head with a 2-way head.
  2. Keep the feature extractor as-is for spectrogram generation.
  3. Allow a separate learning rate for the new head vs the backbone.
"""

from __future__ import annotations
from typing import Iterator

import torch
from torch import nn
from transformers import (
    ASTConfig,
    ASTFeatureExtractor,
    ASTForAudioClassification,
)

from src.config import CFG


def build_model() -> tuple[ASTForAudioClassification, ASTFeatureExtractor]:
    feature_extractor = ASTFeatureExtractor.from_pretrained(CFG.backbone)

    # Force a 2-way head; drop AudioSet's 527-class head.
    config = ASTConfig.from_pretrained(CFG.backbone)
    config.num_labels = CFG.num_labels
    config.id2label = {0: "no_drone", 1: "drone"}
    config.label2id = {"no_drone": 0, "drone": 1}

    model = ASTForAudioClassification.from_pretrained(
        CFG.backbone,
        config=config,
        ignore_mismatched_sizes=True,   # head is being replaced
    )
    return model, feature_extractor


def param_groups(model: nn.Module) -> list[dict]:
    """Lower LR on backbone, higher LR on the new head."""
    head_params, backbone_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "classifier" in name:
            head_params.append(p)
        else:
            backbone_params.append(p)
    return [
        {"params": backbone_params, "lr": CFG.lr_backbone, "name": "backbone"},
        {"params": head_params, "lr": CFG.lr_head, "name": "head"},
    ]
