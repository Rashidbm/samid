from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class OODConfig:
    fdbd_threshold: float = 0.0
    energy_threshold: float = 0.0
    train_feature_mean: torch.Tensor | None = None
    drone_prob_threshold: float = 0.25


def maxlogit_score(logits):
    return logits.max(dim=-1).values


def energy_score(logits, T=1.0):
    return T * torch.logsumexp(logits / T, dim=-1)


def fdbd_score(features, classifier, train_feature_mean):
    W = classifier.weight
    b = classifier.bias
    logits = features @ W.T + b
    pred = logits.argmax(dim=-1)
    N, C = logits.shape

    distances = torch.empty(N, device=features.device)
    for n in range(N):
        i = int(pred[n].item())
        x = features[n]
        idx_others = torch.tensor(
            [j for j in range(C) if j != i], device=features.device
        )
        if idx_others.numel() == 0:
            distances[n] = float("inf")
            continue
        Wi, Wj = W[i], W[idx_others]
        bi, bj = b[i], b[idx_others]
        diff_w = Wi.unsqueeze(0) - Wj
        diff_b = bi - bj
        numer = (diff_w @ x + diff_b).abs()
        denom = diff_w.norm(dim=-1) + 1e-8
        distances[n] = (numer / denom).min()

    norm_dev = (features - train_feature_mean).norm(dim=-1) + 1e-8
    return distances / norm_dev


def get_features_and_logits(model, input_values, device):
    model.eval()
    with torch.no_grad():
        out = model(input_values=input_values.to(device), output_hidden_states=True)
        features = out.hidden_states[-1].mean(dim=1)
        return features, out.logits


def calibrate(model, val_loader, device, percentile=5.0):
    classifier = model.classifier.dense
    all_features, energy_scores, fdbd_scores = [], [], []

    for batch in val_loader:
        x = batch["input_values"]
        feats, logits = get_features_and_logits(model, x, device)
        all_features.append(feats.cpu())
        energy_scores.append(energy_score(logits).cpu())

    feats_all = torch.cat(all_features)
    train_mean = feats_all.mean(dim=0).to(device)

    for batch in val_loader:
        x = batch["input_values"]
        feats, _ = get_features_and_logits(model, x, device)
        fdbd_scores.append(fdbd_score(feats, classifier, train_mean).cpu())

    f = torch.cat(fdbd_scores).numpy()
    e = torch.cat(energy_scores).numpy()
    return OODConfig(
        fdbd_threshold=float(np.percentile(f, percentile)),
        energy_threshold=float(np.percentile(e, percentile)),
        train_feature_mean=train_mean.cpu(),
    )


def classify(model, input_values, cfg, device):
    classifier = model.classifier.dense
    train_mean = cfg.train_feature_mean.to(device) if cfg.train_feature_mean is not None else None

    features, logits = get_features_and_logits(model, input_values, device)
    probs = F.softmax(logits, dim=-1).cpu().numpy()
    e = energy_score(logits).cpu().numpy()
    f = (fdbd_score(features, classifier, train_mean).cpu().numpy()
         if train_mean is not None else np.zeros(len(probs)))

    out = []
    for i in range(len(probs)):
        p_drone = float(probs[i, 1])
        is_ood = bool(f[i] < cfg.fdbd_threshold)
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
            "decision": decision,
        })
    return out
