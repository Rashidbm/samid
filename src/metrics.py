"""
Honest evaluation metrics for an imbalanced binary classifier.

We do NOT report bare accuracy — at 91/9 it is a vanity number.
We report: F1 (positive class = drone), precision, recall, PR-AUC, ROC-AUC,
and the full confusion matrix at multiple decision thresholds.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class ClassificationReport:
    threshold: float
    f1: float
    precision: float
    recall: float
    accuracy: float
    confusion: np.ndarray  # 2x2

    def pretty(self) -> str:
        cm = self.confusion
        return (
            f"@thr={self.threshold:.2f}  "
            f"F1={self.f1:.4f}  P={self.precision:.4f}  R={self.recall:.4f}  "
            f"Acc={self.accuracy:.4f}\n"
            f"  TN={cm[0,0]:>6}  FP={cm[0,1]:>6}\n"
            f"  FN={cm[1,0]:>6}  TP={cm[1,1]:>6}"
        )


def report_at_threshold(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float
) -> ClassificationReport:
    y_pred = (y_score >= threshold).astype(np.int64)
    return ClassificationReport(
        threshold=threshold,
        f1=f1_score(y_true, y_pred, zero_division=0),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        accuracy=float((y_true == y_pred).mean()),
        confusion=confusion_matrix(y_true, y_pred, labels=[0, 1]),
    )


def threshold_free(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    return {
        "pr_auc": average_precision_score(y_true, y_score),
        "roc_auc": roc_auc_score(y_true, y_score),
    }
