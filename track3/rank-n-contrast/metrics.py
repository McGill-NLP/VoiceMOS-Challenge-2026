#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utterance- and system-level MSE / LCC / SRCC, matching the baseline's
`calculate_metrics.py` definitions so numbers are directly comparable."""

from collections import defaultdict

import numpy as np
import scipy.stats


def compute_metrics(true_scores, pred_scores):
    true_scores = np.asarray(true_scores, dtype=np.float64)
    pred_scores = np.asarray(pred_scores, dtype=np.float64)
    if true_scores.size == 0:
        return float("nan"), float("nan"), float("nan")
    mse = float(np.mean((true_scores - pred_scores) ** 2))
    if true_scores.size < 2 or np.std(true_scores) == 0 or np.std(pred_scores) == 0:
        return mse, float("nan"), float("nan")
    lcc = float(scipy.stats.pearsonr(true_scores, pred_scores)[0])
    srcc = float(scipy.stats.spearmanr(true_scores, pred_scores)[0])
    return mse, lcc, srcc


def evaluate(true_scores, pred_scores, system_ids):
    """Returns utterance-level and system-level metrics.

    System-level scores are the per-system means of the utterance-level truths
    and predictions, as in the baseline.
    """
    utt_mse, utt_lcc, utt_srcc = compute_metrics(true_scores, pred_scores)

    sys_true, sys_pred = defaultdict(list), defaultdict(list)
    for t, p, s in zip(true_scores, pred_scores, system_ids):
        sys_true[s].append(t)
        sys_pred[s].append(p)
    keys = list(sys_true.keys())
    s_mse, s_lcc, s_srcc = compute_metrics(
        [np.mean(sys_true[k]) for k in keys],
        [np.mean(sys_pred[k]) for k in keys],
    )
    return {
        "n_pairs": len(true_scores),
        "n_systems": len(keys),
        "utt_mse": utt_mse, "utt_lcc": utt_lcc, "utt_srcc": utt_srcc,
        "sys_mse": s_mse, "sys_lcc": s_lcc, "sys_srcc": s_srcc,
    }


def format_metrics(name, m):
    return (
        f"{name:<10} n={m['n_pairs']:>5} sys={m['n_systems']:>3} | "
        f"UTT mse {m['utt_mse']:.3f} lcc {m['utt_lcc']:.3f} srcc {m['utt_srcc']:.3f} | "
        f"SYS mse {m['sys_mse']:.3f} lcc {m['sys_lcc']:.3f} srcc {m['sys_srcc']:.3f}"
    )
