from __future__ import annotations

import torch
from torch import nn
from transformers import ASTConfig, ASTFeatureExtractor, ASTForAudioClassification

from src.config import CFG


def build_model():
    fe = ASTFeatureExtractor.from_pretrained(CFG.backbone)
    config = ASTConfig.from_pretrained(CFG.backbone)
    config.num_labels = CFG.num_labels
    config.id2label = {0: "no_drone", 1: "drone"}
    config.label2id = {"no_drone": 0, "drone": 1}
    model = ASTForAudioClassification.from_pretrained(
        CFG.backbone, config=config, ignore_mismatched_sizes=True,
    )
    return model, fe


def param_groups(model):
    head, backbone = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (head if "classifier" in name else backbone).append(p)
    return [
        {"params": backbone, "lr": CFG.lr_backbone},
        {"params": head, "lr": CFG.lr_head},
    ]
