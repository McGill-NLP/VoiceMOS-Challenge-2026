#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pair feature matrices for the weak learners, assembled from cached frozen embeddings.

The interaction algebra is the same one the deep models use (../unified/interactions.py), so
that a weak learner and a deep model see the same information and any difference in their
errors comes from the function class rather than from the inputs:

    full      [e_a, e_b, |e_a - e_b|, e_a * e_b, cos, ||e_a - e_b||]    4D + 2
    compact   [|e_a - e_b|, e_a * e_b, cos, ||e_a - e_b||]              2D + 2

`compact` drops the two raw embedding blocks. That is not arbitrary: the interaction ablation
found that substituting a WRONG sys019 reference only moved SYS-SRCC 0.932 -> 0.786, so much
of the signal is in e_a alone, and the `no-b` mode (dropping the raw reference block) was one
of the two variants that gained. Here the pressure is stronger -- with 1024-d SSL embeddings
`full` is 4,098 features against 2,800 training pairs.

PCA is fitted on TRAIN EMBEDDINGS ONLY and reused for dev/test. It is applied per side before
the interaction is formed, not to the assembled pair vector, so the algebra above still holds
in the reduced space.
"""

import csv
import os
from collections import defaultdict

import numpy as np

FEATURE_SETS = ["full", "compact"]


def load_embeddings(path):
    """basename -> vector, from an extract_features.py .npz."""
    with np.load(path) as z:
        return {k: z[k].astype(np.float64) for k in z.files}


def read_pairs(csv_path, metrics=("spk_sim", "acc_sim")):
    """Collapse listener-wise rows to unique pairs with mean scores.

    train.csv carries ~5 rating rows per pair; dev/test carry one row per pair (dev already
    averaged, test unlabelled). Averaging here is what makes all three consistent, and it is
    the right target for a classical regressor -- duplicating identical feature rows would
    only reweight pairs by their rater count.
    """
    acc = defaultdict(lambda: {"scores": defaultdict(list), "meta": None})
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = (row["wav_a_path"], row["wav_b_path"])
            rec = acc[key]
            if rec["meta"] is None:
                rec["meta"] = (row.get("system_id", ""), row.get("utterance_id", ""))
            for m in metrics:
                if row.get(m) not in (None, ""):
                    rec["scores"][m].append(float(row[m]))

    pairs = []
    for (wa, wb), rec in acc.items():
        item = {
            "wav_a_path": wa, "wav_b_path": wb,
            "system_id": rec["meta"][0], "utterance_id": rec["meta"][1],
        }
        for m in metrics:
            vals = rec["scores"].get(m)
            item[m] = float(np.mean(vals)) if vals else None
        pairs.append(item)
    return pairs


def read_rating_rows(csv_path, metrics=("spk_sim", "acc_sim")):
    """One entry per LISTENER RATING, without collapsing to pairs.

    The alternative to `read_pairs`: 13,687 rows instead of 2,800 for train.csv. The feature
    vector is identical across a pair's ~5 rows -- it depends only on the two waveforms -- so
    the rows differ solely in their target.

    For a squared-error learner this is not new information. Summing over the k ratings of a
    pair, sum_j (x.b - y_j)^2 = k (x.b - ybar)^2 + const, so fitting the duplicated rows gives
    the same ridge solution as fitting the pair means with sample weight k, and the counts here
    are almost uniform (2,488 pairs with 5 ratings, 311 with 4, 1 with 3). It can matter for
    the non-quadratic losses (LinearSVR, SVR) and for the tree learners, where duplicated rows
    change split counts and leaf statistics.
    """
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            item = {"wav_a_path": row["wav_a_path"], "wav_b_path": row["wav_b_path"],
                    "system_id": row.get("system_id", ""),
                    "utterance_id": row.get("utterance_id", "")}
            keep = False
            for m in metrics:
                v = row.get(m)
                item[m] = float(v) if v not in (None, "") else None
                keep = keep or item[m] is not None
            if keep:
                rows.append(item)
    return rows


def stack_sides(pairs, emb):
    """(A, B) embedding matrices for a list of pairs, in order."""
    A = np.stack([emb[os.path.basename(p["wav_a_path"])] for p in pairs])
    B = np.stack([emb[os.path.basename(p["wav_b_path"])] for p in pairs])
    return A, B


class PairFeaturizer:
    """Builds pair matrices, optionally through a PCA fitted on the training side only."""

    def __init__(self, feature_set="full", pca_dim=0, whiten=False):
        if feature_set not in FEATURE_SETS:
            raise ValueError(f"feature_set must be one of {FEATURE_SETS}")
        self.feature_set = feature_set
        self.pca_dim = pca_dim
        self.whiten = whiten
        self.pca = None
        self.mu = None
        self.sd = None

    def fit(self, pairs, emb):
        A, B = stack_sides(pairs, emb)
        both = np.vstack([A, B])
        if self.pca_dim and self.pca_dim < both.shape[1]:
            from sklearn.decomposition import PCA
            self.pca = PCA(n_components=self.pca_dim, whiten=self.whiten, random_state=0)
            self.pca.fit(both)
        X = self._build(A, B)
        # Standardise on train; SVR and Ridge are both scale-sensitive.
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0)
        self.sd[self.sd < 1e-8] = 1.0
        return self

    def transform(self, pairs, emb):
        A, B = stack_sides(pairs, emb)
        X = self._build(A, B)
        return (X - self.mu) / self.sd

    def fit_transform(self, pairs, emb):
        return self.fit(pairs, emb).transform(pairs, emb)

    def _build(self, A, B):
        if self.pca is not None:
            A = self.pca.transform(A)
            B = self.pca.transform(B)
        # Cosine and distance are computed on the (possibly reduced) vectors actually used.
        an = A / np.clip(np.linalg.norm(A, axis=1, keepdims=True), 1e-8, None)
        bn = B / np.clip(np.linalg.norm(B, axis=1, keepdims=True), 1e-8, None)
        cos = (an * bn).sum(axis=1, keepdims=True)
        l2 = np.linalg.norm(A - B, axis=1, keepdims=True)

        if self.feature_set == "full":
            blocks = [A, B, np.abs(A - B), A * B, cos, l2]
        else:
            blocks = [np.abs(A - B), A * B, cos, l2]
        return np.hstack(blocks)

    def out_dim(self, emb_dim):
        d = self.pca_dim if self.pca else emb_dim
        return (4 * d + 2) if self.feature_set == "full" else (2 * d + 2)
