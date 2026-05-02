from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, class_weights=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(
            logits, targets,
            weight=self.class_weights,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()
