#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Training objectives for the unified Track 3 model.

Three primary objectives, selected with `--objective`:

  mse     the official baseline's plain MSE on the per-pair mean score
  corn    CORN ordinal regression (Shi, Cao & Raschka, 2021)
  coral   CORAL ordinal regression (Cao, Mirjalili & Raschka, 2020)

plus one auxiliary that composes with any of them, selected with `--lambda-rnc`:

  rnc     Rank-N-Contrast (Zha et al., NeurIPS 2023) on the interaction vector

The CORN/CORAL code, including the soft-target variants, is taken from
../corn-and-coral/finetune.py. The Rank-N-Contrast implementation is taken from
../rank-n-contrast/loss.py, where it is verified against the authors' reference
implementation.

Ratings are on a 1-5 scale, so num_classes = 5 and the ordinal heads emit 4 logits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from coral_pytorch.dataset import levels_from_labelbatch
from coral_pytorch.losses import coral_loss, corn_loss

NUM_CLASSES = 5
OBJECTIVES = ["mse", "corn", "coral"]


# ---------------------------------------------------------------------------
# Decoding: raw head output -> continuous 1..K rating
# ---------------------------------------------------------------------------
def decode_score(objective, raw, num_classes=NUM_CLASSES):
    """Turn a head's raw output into a continuous rating.

    Shared by training-time evaluation and inference.py so the two cannot drift apart.
    We never threshold to a discrete class: the targets are mean listener ratings, so
    the expected value E[y] = 1 + sum_k P(y > k) is the quantity of interest.

    CORAL's logits are already marginal, so sigmoid gives P(y > k) directly. CORN's are
    *conditional* P(y > k | y > k-1), so they are cumprod'd into marginals first (this
    mirrors coral_pytorch's corn_label_from_logits).
    """
    if objective == "corn":
        return 1.0 + torch.cumprod(torch.sigmoid(raw), dim=1).sum(dim=1)
    if objective == "coral":
        return 1.0 + torch.sigmoid(raw).sum(dim=1)
    return raw


# ---------------------------------------------------------------------------
# Soft ordinal targets
# ---------------------------------------------------------------------------
def soft_survival_targets(targets, num_classes=NUM_CLASSES):
    """Continuous 1..K ratings -> soft marginal survival targets S(k) = P(y > k).

    Assumes a two-point mixture between floor(t) and ceil(t) matching the rating's
    fractional part. By construction sum_k S(k) == t (0-indexed), so this is the
    continuous generalisation of the hard extended-binary encoding used by CORAL
    (`levels_from_labelbatch`), and reduces to it exactly when t is an integer.
    """
    t = torch.clamp(targets - 1, 0, num_classes - 1)
    thresholds = torch.arange(num_classes - 1, device=targets.device, dtype=targets.dtype)
    return torch.clamp(t.unsqueeze(1) - thresholds, 0, 1)


def soft_coral_loss(logits, targets, num_classes=NUM_CLASSES):
    """Soft-target analogue of coral_pytorch's coral_loss.

    CORAL's head already outputs marginal probabilities P(y > k) -- rank-consistent by
    architecture, not by loss -- so it can be trained directly against the soft marginal
    survival targets. Matches coral_loss's reduction (sum over the threshold dim, mean
    over the batch) so the loss scale, and hence the learning rate, stays comparable to
    the hard-label runs.
    """
    soft_targets = soft_survival_targets(targets, num_classes)
    elementwise_bce = F.binary_cross_entropy_with_logits(logits, soft_targets, reduction="none")
    return elementwise_bce.sum(dim=1).mean()


def soft_corn_loss(logits, targets, num_classes=NUM_CLASSES):
    """Soft-target analogue of coral_pytorch's corn_loss.

    CORN's logits are conditional probabilities P(y > k | y > k-1), and corn_loss trains
    each threshold task only on the examples that cleared the previous one. This mirrors
    that: it decomposes the soft marginal survival targets S(k) into conditional targets
    c(k) = S(k) / S(k-1) by the chain rule, and weights each task's BCE by S(k-1) -- a
    soft version of "did this example clear the previous threshold" -- instead of hard
    masking. Reduces to corn_loss exactly when targets are integers.
    """
    survival = soft_survival_targets(targets, num_classes)          # S(k)
    prev_survival = F.pad(survival[:, :-1], (1, 0), value=1.0)      # S(k-1), S(-1) := 1

    # Where prev_survival is 0 the conditional target is undefined (0/0), but its weight
    # is also 0, so it cannot contribute to the loss regardless.
    cond_target = torch.where(
        prev_survival > 0,
        survival / prev_survival.clamp(min=1e-8),
        torch.zeros_like(survival),
    )
    elementwise_bce = F.binary_cross_entropy_with_logits(logits, cond_target, reduction="none")
    return (prev_survival * elementwise_bce).sum() / prev_survival.sum().clamp(min=1e-8)


def ordinal_loss(objective, raw, targets, soft_labels=True, num_classes=NUM_CLASSES):
    """CORN/CORAL loss, soft or hard targets."""
    if soft_labels:
        fn = soft_corn_loss if objective == "corn" else soft_coral_loss
        return fn(raw, targets, num_classes)

    labels = torch.clamp(torch.round(targets - 1), 0, num_classes - 1).to(torch.long)
    if objective == "corn":
        return corn_loss(raw, labels, num_classes=num_classes)
    levels = levels_from_labelbatch(labels, num_classes=num_classes).to(raw.device)
    return coral_loss(raw, levels)


def task_loss(objective, raw, targets, soft_labels=True, num_classes=NUM_CLASSES):
    """The primary loss for whichever objective is active."""
    if objective == "mse":
        return F.mse_loss(raw, targets)
    return ordinal_loss(objective, raw, targets, soft_labels=soft_labels, num_classes=num_classes)


# ---------------------------------------------------------------------------
# Rank-N-Contrast auxiliary
# ---------------------------------------------------------------------------
def label_distances(labels, distance_type="l1"):
    """[n, label_dim] -> [n, n] pairwise label distances."""
    if labels.dim() == 1:
        labels = labels.unsqueeze(1)
    if distance_type == "l1":
        return torch.cdist(labels, labels, p=1)
    if distance_type == "l2":
        return torch.cdist(labels, labels, p=2)
    raise ValueError(f"Unknown label distance: {distance_type}")


def feature_similarities(features, similarity_type="l2"):
    """[n, d] -> [n, n] pairwise similarities (higher = more similar)."""
    if similarity_type == "l2":
        return -torch.cdist(features, features, p=2)
    if similarity_type == "l1":
        return -torch.cdist(features, features, p=1)
    if similarity_type == "cosine":
        normed = F.normalize(features, p=2, dim=-1)
        return normed @ normed.t()
    raise ValueError(f"Unknown feature similarity: {similarity_type}")


def _drop_diagonal(mat):
    """[n, n] -> [n, n-1], removing each row's diagonal entry."""
    n = mat.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=mat.device)
    return mat.masked_select(mask).view(n, n - 1)


def rnc_loss(features, labels, temperature=2.0, label_diff="l1", feature_sim="l2", tie_eps=1e-6):
    """Vectorised Rank-N-Contrast loss.

    For each anchor, every other sample in the batch is a positive exactly once, and its
    negatives are everything at least as far away in label space. Copied from
    ../rank-n-contrast/loss.py, where it is checked against the authors' reference
    implementation.

    features: [n, d]
    labels:   [n] or [n, label_dim]
    """
    if labels.dim() == 1:
        labels = labels.unsqueeze(1)

    n = features.shape[0]
    if n < 2:
        return features.sum() * 0.0

    logits = feature_similarities(features, feature_sim) / temperature
    # Numerical stabilisation; a no-op for negative-distance similarities (whose row max
    # is the zero diagonal) but needed for cosine.
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    diffs = _drop_diagonal(label_distances(labels, label_diff))
    logits = _drop_diagonal(logits)

    # Sort each row by label distance. S(i,j) is then a contiguous suffix, so the
    # denominator for every positive is a suffix sum of exp(logits).
    diffs_sorted, order = torch.sort(diffs, dim=1)
    logits_sorted = logits.gather(1, order)
    exp_sorted = logits_sorted.exp()
    suffix = exp_sorted.flip(1).cumsum(1).flip(1)

    # Ties must share a denominator: S(i,j) starts at the first index holding the same
    # label distance, not at j's own index.
    is_new = torch.ones_like(diffs_sorted, dtype=torch.bool)
    is_new[:, 1:] = (diffs_sorted[:, 1:] - diffs_sorted[:, :-1]) > tie_eps
    idx = torch.arange(n - 1, device=features.device).expand(n, n - 1)
    group_start = torch.cummax(torch.where(is_new, idx, torch.zeros_like(idx)), dim=1).values
    denom = suffix.gather(1, group_start)

    return (torch.log(denom) - logits_sorted).mean()
