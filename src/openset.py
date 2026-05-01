"""
Open-set / out-of-distribution (OOD) detection for the trained AST classifier.

Implements three modern OOD scoring methods that work as POST-HOC drop-ins —
no retraining required. All three operate on the trained model's logits and/or
penultimate features.

Methods (in order of pitch quality):
  1. fDBD     — Fast Decision Boundary distance (Liu et al., ICML 2024)
  2. Energy   — log-sum-exp of logits (Liu et al., NeurIPS 2020)
  3. MaxLogit — simple max-of-logits (Hendrycks et al., 2020)

For our binary detector (drone yes/no) plus future multi-class type head,
the fDBD score is the SOTA-defensible choice (confirmed by arxiv 2511.11934
systematic analysis, Nov 2025: 'geometry-aware scores prevail under shift').

Pitch line:
    'Our open-set detection uses fDBD (ICML 2024), confirmed by Nov 2025
     systematic analysis as state-of-the-art for distribution-shifted OOD.
     Drops onto our trained Audio Spectrogram Transformer with no retraining.'

Decision logic at deployment:
    - prob(drone) > 0.5  AND  fdbd > known_threshold  -> "drone confirmed"
    - prob(drone) > 0.5  AND  fdbd < known_threshold  -> "unknown UAV - threat"
    - prob(drone) < 0.5                               -> "no drone"
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


# --------------------------- score functions -------------------------------- #


def maxlogit_score(logits: torch.Tensor) -> torch.Tensor:
    """Higher = more in-distribution."""
    return logits.max(dim=-1).values


def energy_score(logits: torch.Tensor, T: float = 1.0) -> torch.Tensor:
    """
    Energy-based OOD score (Liu et al. 2020). Higher = more in-distribution.
    Energy(x) = -T * logsumexp(logits / T)
    We negate so higher = ID (consistent with fDBD/MaxLogit).
    """
    return T * torch.logsumexp(logits / T, dim=-1)


def fdbd_score(
    features: torch.Tensor,        # (N, D) penultimate features
    classifier: nn.Linear,          # final classifier layer
    train_feature_mean: torch.Tensor,  # (D,) mean of training features
) -> torch.Tensor:
    """
    fDBD score (Liu et al., ICML 2024).
    Distance to the nearest class decision boundary, regularized by deviation
    from training mean. Higher = more in-distribution.

    For predicted class i and competing class j, boundary is:
        (w_i - w_j) . x + (b_i - b_j) = 0
    Distance from x to this boundary:
        |(w_i - w_j) . x + (b_i - b_j)| / ||w_i - w_j||

    fDBD takes the MIN distance over all j != i, then divides by feature norm
    deviation from training mean.
    """
    W = classifier.weight                    # (C, D)
    b = classifier.bias                      # (C,)
    logits = features @ W.T + b              # (N, C)
    pred = logits.argmax(dim=-1)             # (N,)
    N, C = logits.shape

    distances = torch.empty(N, device=features.device)
    for n in range(N):
        i = int(pred[n].item())
        x = features[n]
        # vectorize over j != i
        idx_others = torch.tensor(
            [j for j in range(C) if j != i], device=features.device
        )
        if idx_others.numel() == 0:
            distances[n] = float("inf")
            continue
        Wi = W[i]
        Wj = W[idx_others]                   # (C-1, D)
        bi = b[i]
        bj = b[idx_others]                   # (C-1,)
        diff_w = Wi.unsqueeze(0) - Wj        # (C-1, D)
        diff_b = bi - bj                     # (C-1,)
        numer = (diff_w @ x + diff_b).abs()  # (C-1,)
        denom = diff_w.norm(dim=-1) + 1e-8   # (C-1,)
        d = (numer / denom).min()
        distances[n] = d

    # Normalize by deviation from training mean
    norm_dev = (features - train_feature_mean).norm(dim=-1) + 1e-8
    return distances / norm_dev


# --------------------------- helpers ---------------------------------------- #


@dataclass
class OODConfig:
    """Calibration thresholds learned on validation set."""
    fdbd_threshold: float = 0.0
    energy_threshold: float = 0.0
    maxlogit_threshold: float = 0.0
    train_feature_mean: torch.Tensor | None = None
    drone_prob_threshold: float = 0.5  # Detection threshold (drone vs no-drone)


def compute_features_and_logits(
    model, input_values: torch.Tensor, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Forward pass that returns BOTH penultimate features and logits.

    For HF ASTForAudioClassification, we pass output_hidden_states=True and
    take the [CLS]-equivalent token (mean-pooled hidden state from the last
    layer, which is what the standard classifier head consumes).
    """
    model.eval()
    with torch.no_grad():
        out = model(input_values=input_values.to(device), output_hidden_states=True)
        # Penultimate feature: the input to the classifier head.
        # AST's classifier consumes the mean of the final hidden state.
        last_hidden = out.hidden_states[-1]              # (N, T, D)
        features = last_hidden.mean(dim=1)                # (N, D) — mean-pool
        logits = out.logits                               # (N, num_labels)
    return features, logits


def calibrate_thresholds(
    model, val_loader, device, percentile: float = 5.0,
) -> OODConfig:
    """
    Run val set, compute all three OOD scores, set threshold at the LOW
    percentile of in-distribution scores (so legitimate ID samples pass,
    anything weirder than that is flagged).
    """
    fdbd_scores, energy_scores, maxlogit_scores, all_features = [], [], [], []
    classifier = model.classifier.dense          # final Linear layer in AST head

    # First pass: collect features + logits
    for batch in val_loader:
        x = batch["input_values"]
        features, logits = compute_features_and_logits(model, x, device)
        all_features.append(features.cpu())
        energy_scores.append(energy_score(logits).cpu())
        maxlogit_scores.append(maxlogit_score(logits).cpu())

    all_features_t = torch.cat(all_features)
    train_mean = all_features_t.mean(dim=0).to(device)

    # Second pass: fdbd needs the mean
    for batch in val_loader:
        x = batch["input_values"]
        features, _ = compute_features_and_logits(model, x, device)
        scores = fdbd_score(features, classifier, train_mean)
        fdbd_scores.append(scores.cpu())

    f = torch.cat(fdbd_scores).numpy()
    e = torch.cat(energy_scores).numpy()
    m = torch.cat(maxlogit_scores).numpy()
    return OODConfig(
        fdbd_threshold=float(np.percentile(f, percentile)),
        energy_threshold=float(np.percentile(e, percentile)),
        maxlogit_threshold=float(np.percentile(m, percentile)),
        train_feature_mean=train_mean.cpu(),
    )


def classify_with_openset(
    model, input_values: torch.Tensor, cfg: OODConfig, device: torch.device,
) -> list[dict]:
    """
    Returns one dict per sample:
      {
        'p_drone': float,
        'fdbd': float, 'energy': float, 'maxlogit': float,
        'decision': 'drone' | 'no_drone' | 'unknown_uav',
        'is_ood': bool,
      }
    """
    classifier = model.classifier.dense
    train_mean = cfg.train_feature_mean.to(device) if cfg.train_feature_mean is not None else None

    features, logits = compute_features_and_logits(model, input_values, device)
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    e = energy_score(logits).cpu().numpy()
    m = maxlogit_score(logits).cpu().numpy()
    if train_mean is not None:
        f = fdbd_score(features, classifier, train_mean).cpu().numpy()
    else:
        f = np.zeros(len(probs))

    out = []
    for i in range(len(probs)):
        p_drone = float(probs[i, 1])
        is_ood = bool(f[i] < cfg.fdbd_threshold)  # primary criterion = fDBD
        if p_drone < cfg.drone_prob_threshold:
            decision = "no_drone"
        elif is_ood:
            decision = "unknown_uav"
        else:
            decision = "drone"
        out.append({
            "p_drone": p_drone,
            "fdbd": float(f[i]),
            "energy": float(e[i]),
            "maxlogit": float(m[i]),
            "decision": decision,
            "is_ood": is_ood,
        })
    return out
