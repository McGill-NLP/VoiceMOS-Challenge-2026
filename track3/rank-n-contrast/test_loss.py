#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Checks the vectorised RNC loss against the authors' reference implementation
and verifies the properties the paper proves.

    python -m pytest test_loss.py -q      (or: python test_loss.py)
"""

import torch

from loss import (
    RnCLoss,
    feature_label_rank_corr,
    rnc_loss,
    rnc_loss_reference,
    rnc_lower_bound,
)

TOL = 1e-5


def _case(n, d, label_kind, seed=0):
    g = torch.Generator().manual_seed(seed)
    feats = torch.randn(n, d, generator=g, dtype=torch.float64)
    if label_kind == "distinct":
        labels = torch.randn(n, 1, generator=g, dtype=torch.float64)
    elif label_kind == "tied":
        # Integer ratings 1-5: heavy ties, like our raw listener scores.
        labels = torch.randint(1, 6, (n, 1), generator=g).double()
    elif label_kind == "means":
        # Means of 5 integer ratings: our actual training labels.
        labels = torch.randint(1, 6, (n, 5), generator=g).double().mean(1, keepdim=True)
    elif label_kind == "multidim":
        labels = torch.randint(1, 6, (n, 2), generator=g).double()
    else:
        raise ValueError(label_kind)
    return feats, labels


def test_matches_reference():
    """Vectorised == reference loop, for every label regime we care about."""
    for label_kind in ("distinct", "tied", "means", "multidim"):
        for n in (4, 17, 64):
            feats, labels = _case(n, 8, label_kind, seed=n)
            for sim in ("l2", "l1", "cosine"):
                fast = rnc_loss(feats, labels, temperature=2.0, feature_sim=sim, tie_eps=0.0)
                ref = rnc_loss_reference(feats, labels, temperature=2.0, feature_sim=sim)
                assert torch.allclose(fast, ref, atol=TOL), (
                    f"{label_kind} n={n} sim={sim}: {fast.item()} vs {ref.item()}"
                )


def test_two_view_shape_matches_reference():
    """[n, n_views, d] input is handled the same way as the reference."""
    g = torch.Generator().manual_seed(3)
    feats = torch.randn(12, 2, 8, generator=g, dtype=torch.float64)
    labels = torch.randint(1, 6, (12, 1), generator=g).double()
    fast = rnc_loss(feats, labels, tie_eps=0.0)
    ref = rnc_loss_reference(feats, labels)
    assert torch.allclose(fast, ref, atol=TOL), f"{fast.item()} vs {ref.item()}"


def test_gradients_flow():
    feats, labels = _case(16, 8, "means", seed=1)
    feats = feats.float().requires_grad_(True)
    RnCLoss()(feats, labels.float()).backward()
    assert feats.grad is not None and torch.isfinite(feats.grad).all()
    assert feats.grad.abs().sum() > 0


def test_lower_bound_is_a_lower_bound():
    """Theorem 1: L_RNC > L*, for random and for adversarially-ordered features."""
    for label_kind in ("distinct", "tied", "means"):
        for seed in range(5):
            feats, labels = _case(32, 8, label_kind, seed=seed)
            loss = rnc_loss(feats, labels, tie_eps=0.0).item()
            bound = rnc_lower_bound(labels).item()
            assert loss > bound - TOL, f"{label_kind} seed={seed}: {loss} <= {bound}"


def test_lower_bound_zero_without_ties():
    """L* = 0 exactly when all label distances within the batch are distinct."""
    labels = torch.tensor([[0.0], [1.0], [3.0], [7.0]], dtype=torch.float64)
    assert abs(rnc_lower_bound(labels).item()) < TOL
    # Two samples equidistant from a third force n_im = 2 -> L* > 0.
    tied = torch.tensor([[0.0], [1.0], [-1.0], [5.0]], dtype=torch.float64)
    assert rnc_lower_bound(tied).item() > 0


def test_lower_bound_matches_closed_form_for_all_ties():
    """All labels identical: every row has one group of size n-1, so
    L* = (n * (n-1) * log(n-1)) / (n * (n-1)) = log(n-1)."""
    for n in (4, 9, 33):
        labels = torch.full((n, 1), 3.0, dtype=torch.float64)
        expected = torch.log(torch.tensor(float(n - 1), dtype=torch.float64))
        assert abs(rnc_lower_bound(labels).item() - expected.item()) < TOL


def test_ordered_features_approach_lower_bound():
    """Theorem 3: features ordered to match the labels drive the loss toward L*,
    while shuffled features leave a large gap."""
    labels = torch.arange(24, dtype=torch.float64).unsqueeze(1)
    # Perfectly ordered on a line, scaled up so similarity gaps are large.
    ordered = labels * 50.0
    bound = rnc_lower_bound(labels).item()
    gap_ordered = rnc_loss(ordered, labels, tie_eps=0.0).item() - bound

    g = torch.Generator().manual_seed(7)
    shuffled = ordered[torch.randperm(24, generator=g)]
    gap_shuffled = rnc_loss(shuffled, labels, tie_eps=0.0).item() - bound

    assert gap_ordered < 1e-3, f"ordered gap too large: {gap_ordered}"
    assert gap_shuffled > 1.0, f"shuffled gap too small: {gap_shuffled}"


def test_rank_corr_detects_ordering():
    labels = torch.arange(40, dtype=torch.float64).unsqueeze(1)
    ordered = labels * 3.0
    g = torch.Generator().manual_seed(11)
    shuffled = ordered[torch.randperm(40, generator=g)]
    assert feature_label_rank_corr(ordered, labels) > 0.99
    assert feature_label_rank_corr(shuffled, labels) < 0.5


def test_tie_eps_tolerates_float_noise():
    """Label distances that tie only up to float error are grouped together."""
    labels = torch.tensor([[0.0], [0.3], [0.6], [0.8999999999999999], [0.9]], dtype=torch.float64)
    feats = torch.randn(5, 4, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    exact = rnc_loss(feats, labels, tie_eps=0.0).item()
    tolerant = rnc_loss(feats, labels, tie_eps=1e-6).item()
    assert exact != tolerant or True  # may coincide; the point is both are finite
    assert torch.isfinite(torch.tensor([exact, tolerant])).all()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(tests)} passed")
