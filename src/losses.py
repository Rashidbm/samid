"""
Focal loss for class-imbalanced binary classification.

Geronimobasso is ~91% drone / 9% no-drone after duration normalization, the
class with fewer samples is the harder positive. Focal loss down-weights the
easy negatives so gradient signal stays focused on the hard examples.
"""

from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        class_weights: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.class_weights,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()
