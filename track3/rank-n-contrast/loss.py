#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rank-N-Contrast loss (Zha et al., NeurIPS 2023, arXiv:2210.01189).

    S(i,j) = { k != i : d(y_i, y_k) >= d(y_i, y_j) }          (note j in S(i,j))

    L_RNC = 1/n  sum_i  1/(n-1) sum_{j != i}
                -log [ exp(sim(v_i,v_j)/T) / sum_{k in S(i,j)} exp(sim(v_i,v_k)/T) ]

Two implementations are provided and are checked against each other in
`test_loss.py`:

  * `rnc_loss_reference` mirrors the authors' released code line for line
    (papers/Rank-N-Contrast/loss.py). O(n^3) time via a Python loop over
    positives, and it materialises an [n, n, d] tensor for the pairwise
    distances.
  * `rnc_loss` is the default. It sorts each row by label distance, which turns
    every denominator into a suffix sum, so the whole loss is one pass with no
    Python loop and no 3-D intermediate. O(n^2 log n), and it uses `torch.cdist`
    (at n=512, d=1024 the reference's intermediate alone is ~2 GB in fp32).

Defaults follow the paper: negative L2 on *unnormalised* features (their
Table 6b: cosine 6.51 vs negative-L2 6.14 MAE) and temperature 2.0.
"""

import torch
import torch.nn as nn


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
        normed = torch.nn.functional.normalize(features, p=2, dim=-1)
        return normed @ normed.t()
    raise ValueError(f"Unknown feature similarity: {similarity_type}")


def _drop_diagonal(mat):
    """[n, n] -> [n, n-1], removing each row's diagonal entry."""
    n = mat.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=mat.device)
    return mat.masked_select(mask).view(n, n - 1)


def rnc_loss(
    features,
    labels,
    temperature=2.0,
    label_diff="l1",
    feature_sim="l2",
    tie_eps=1e-6,
):
    """Vectorised Rank-N-Contrast loss.

    features: [n, d] or [n, n_views, d] (views are flattened and labels repeated)
    labels:   [n] or [n, label_dim]
    """
    if features.dim() == 3:
        n_views = features.shape[1]
        features = features.transpose(0, 1).reshape(-1, features.shape[-1])
        labels = labels.repeat(n_views, *([1] * (labels.dim() - 1)))
    if labels.dim() == 1:
        labels = labels.unsqueeze(1)

    n = features.shape[0]
    if n < 2:
        return features.sum() * 0.0

    logits = feature_similarities(features, feature_sim) / temperature
    # Numerical stabilisation; a no-op for negative-distance similarities (whose
    # row max is the zero diagonal) but needed for cosine.
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    diffs = _drop_diagonal(label_distances(labels, label_diff))
    logits = _drop_diagonal(logits)

    # Sort each row by label distance. S(i,j) is then a contiguous suffix, so the
    # denominator for every positive is a suffix sum of exp(logits).
    diffs_sorted, order = torch.sort(diffs, dim=1)
    logits_sorted = logits.gather(1, order)
    exp_sorted = logits_sorted.exp()
    suffix = exp_sorted.flip(1).cumsum(1).flip(1)

    # Ties must share a denominator: S(i,j) starts at the first index holding the
    # same label distance, not at j's own index.
    is_new = torch.ones_like(diffs_sorted, dtype=torch.bool)
    is_new[:, 1:] = (diffs_sorted[:, 1:] - diffs_sorted[:, :-1]) > tie_eps
    idx = torch.arange(n - 1, device=features.device).expand(n, n - 1)
    group_start = torch.cummax(torch.where(is_new, idx, torch.zeros_like(idx)), dim=1).values
    denom = suffix.gather(1, group_start)

    return (torch.log(denom) - logits_sorted).mean()


def rnc_loss_reference(
    features, labels, temperature=2.0, label_diff="l1", feature_sim="l2"
):
    """Line-for-line port of the authors' implementation. Used only for testing."""
    if features.dim() == 3:
        n_views = features.shape[1]
        features = features.transpose(0, 1).reshape(-1, features.shape[-1])
        labels = labels.repeat(n_views, *([1] * (labels.dim() - 1)))
    if labels.dim() == 1:
        labels = labels.unsqueeze(1)

    diffs = label_distances(labels, label_diff)
    logits = feature_similarities(features, feature_sim).div(temperature)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = logits.exp()

    n = logits.shape[0]
    logits = _drop_diagonal(logits)
    exp_logits = _drop_diagonal(exp_logits)
    diffs = _drop_diagonal(diffs)

    loss = 0.0
    for k in range(n - 1):
        pos_logits = logits[:, k]
        pos_diffs = diffs[:, k]
        neg_mask = (diffs >= pos_diffs.view(-1, 1)).float()
        pos_log_probs = pos_logits - torch.log((neg_mask * exp_logits).sum(dim=-1))
        loss = loss - (pos_log_probs / (n * (n - 1))).sum()
    return loss


class RnCLoss(nn.Module):
    def __init__(self, temperature=2.0, label_diff="l1", feature_sim="l2", tie_eps=1e-6):
        super().__init__()
        self.temperature = temperature
        self.label_diff = label_diff
        self.feature_sim = feature_sim
        self.tie_eps = tie_eps

    def forward(self, features, labels):
        return rnc_loss(
            features, labels,
            temperature=self.temperature,
            label_diff=self.label_diff,
            feature_sim=self.feature_sim,
            tie_eps=self.tie_eps,
        )


@torch.no_grad()
def rnc_lower_bound(labels, label_diff="l1", tie_eps=1e-6):
    """The paper's tight lower bound (Theorems 1 & 2):

        L* = 1 / (n(n-1)) * sum_i sum_m  n_im * log(n_im)

    where n_im is the number of samples at the m-th distinct label distance from
    i. It is zero only when every label distance within the batch is unique; our
    ratings are averages of ~5 integer scores, so ties are common and L* sits
    well above zero. Report `L_RNC - L*` rather than the raw loss: the raw value
    is not comparable across batches with different tie structure.
    """
    if labels.dim() == 1:
        labels = labels.unsqueeze(1)
    n = labels.shape[0]
    if n < 2:
        return torch.zeros((), device=labels.device)

    diffs = _drop_diagonal(label_distances(labels, label_diff))
    diffs_sorted, _ = torch.sort(diffs, dim=1)
    is_new = torch.ones_like(diffs_sorted, dtype=torch.bool)
    is_new[:, 1:] = (diffs_sorted[:, 1:] - diffs_sorted[:, :-1]) > tie_eps

    group_id = is_new.cumsum(dim=1) - 1               # [n, n-1] group index per entry
    n_groups = int(group_id.max().item()) + 1
    counts = torch.zeros(n, n_groups, device=labels.device)
    counts.scatter_add_(1, group_id, torch.ones_like(group_id, dtype=counts.dtype))
    contrib = counts * torch.log(counts.clamp(min=1))
    return contrib.sum() / (n * (n - 1))


@torch.no_grad()
def feature_label_rank_corr(features, labels, feature_sim="l2", label_diff="l1"):
    """Spearman rho between pairwise feature similarity and pairwise label
    proximity (the paper's Table 1 diagnostic: L1 0.822 -> RNC 0.971).

    Measures representation ordinality directly, independently of any head.
    """
    if labels.dim() == 1:
        labels = labels.unsqueeze(1)
    sims = _drop_diagonal(feature_similarities(features, feature_sim)).flatten()
    prox = -_drop_diagonal(label_distances(labels, label_diff)).flatten()
    if sims.numel() < 2:
        return float("nan")

    def _rank(x):
        order = x.argsort()
        ranks = torch.empty_like(order, dtype=torch.float64)
        ranks[order] = torch.arange(x.numel(), device=x.device, dtype=torch.float64)
        return ranks

    rs, rp = _rank(sims.double()), _rank(prox.double())
    rs = rs - rs.mean()
    rp = rp - rp.mean()
    denom = rs.norm() * rp.norm()
    return float((rs @ rp / denom).item()) if denom > 0 else float("nan")
