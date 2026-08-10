#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase 2: combine the deep pool and the frozen-feature weak learners into one submission.

WHY THIS IS NOT THE UTMOS22 STACKER. UTMOS22 fits its level-1 combiner on out-of-fold
predictions from every level-0 model. Our deep models were each trained on all of train, so
they have no out-of-fold predictions -- only dev. Producing them honestly would mean K-fold
retraining 32 models at 47 min to 5 h each, over 200 GPU-hours. That is not available.

So the combiner is fitted on dev, and the number reported for it is the GroupKFold-by-system
out-of-fold estimate on dev, never the in-sample fit. dev holds 23 systems, and grouping
matters because test contains four systems absent from train (sys003, sys004, sys015, sys021):
a combiner tuned across systems it has seen would flatter itself. The submitted weights are
then refitted on all 600 dev pairs, which is standard and is stated plainly rather than hidden.

The methods are ordered by how much they can overfit 600 pairs, and the safest are the
defaults. With 23 groups, fitting 260 free weights is hopeless, so anything beyond `ridge`
operates on POOL MEANS -- a handful of columns -- rather than on individual members.

    mean         unweighted mean of every selected member          0 parameters
    rank-mean    mean of rank-transformed members, remapped        0 parameters
    alpha        a * deep_pool + (1-a) * weak_pool                 1 parameter
    nnls         non-negative, sum-to-one weights over pools       ~n_pools parameters
    ridge        non-negative ridge over individual members        n_members, regularised

`rank-mean` exists because the members are on genuinely different calibrations -- MSE heads,
CORAL cumulative decoding, and ridge outputs do not share a scale -- and SRCC only sees ranks.
Averaging ranks directly would wreck MSE and LCC, which are two of the six reported metrics,
so ranks are mapped back through the empirical quantile function of the deep pool's scores.

    python stack.py --target spk_sim
    python stack.py --target spk_sim --write-test egs/submission
"""

import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np
import scipy.stats
from sklearn.model_selection import GroupKFold

EVAL_ROOT = "../baseline/data/vmc2026_track3_eval_phase_distro_v3_syn"
DEV_LABELS = f"{EVAL_ROOT}/sets/dev_with_labels.csv"
TEST_CSV = f"{EVAL_ROOT}/sets/test.csv"
DEEP_ROOT = "../unified/egs/ensemble_runs"
ENCS = ["ecapa-voxceleb", "commonaccent-ecapa", "eres2netv2", "eres2netv2-w24s4ep4"]
DEEP_CFGS = [f"{e}-{l}-{i}" for e in ENCS for l in ("mse", "coral") for i in ("baseline", "bilinear")]

METHODS = ["mean", "rank-mean", "alpha", "nnls", "ridge"]


# ------------------------------------------------------------------ io

def read_rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def index_of(rows):
    keys = [(r["wav_a_path"], r["wav_b_path"]) for r in rows]
    return keys, {k: i for i, k in enumerate(keys)}


def load_pred(path, metric, kidx, n):
    v = np.full(n, np.nan)
    col = f"pred_{metric}"
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            i = kidx.get((r["wav_a_path"], r["wav_b_path"]))
            if i is not None and col in r:
                v[i] = float(r[col])
    return None if np.isnan(v).any() else v


def collect_members(metric, weak_dir, dev_kidx, test_kidx, n_dev, n_test, require_test):
    """name -> {'dev': vec, 'test': vec or None, 'pool': 'deep'|'weak:<encoder>'}."""
    members = {}

    for cfg in DEEP_CFGS:
        d = f"{DEEP_ROOT}/{cfg}_{metric}/dev_{metric}_best.csv"
        t = f"{DEEP_ROOT}/{cfg}_{metric}/test_{metric}_best.csv"
        if not os.path.exists(d):
            continue
        dv = load_pred(d, metric, dev_kidx, n_dev)
        tv = load_pred(t, metric, test_kidx, n_test) if os.path.exists(t) else None
        if dv is None or (require_test and tv is None):
            continue
        members[f"deep::{cfg}"] = {"dev": dv, "test": tv, "pool": "deep"}

    for d in sorted(glob.glob(f"{weak_dir}/*__{metric}__dev.csv")):
        run = os.path.basename(d)[: -len(f"__{metric}__dev.csv")]
        t = d[: -len("dev.csv")] + "test.csv"
        dv = load_pred(d, metric, dev_kidx, n_dev)
        tv = load_pred(t, metric, test_kidx, n_test) if os.path.exists(t) else None
        if dv is None or (require_test and tv is None):
            continue
        members[f"weak::{run}"] = {"dev": dv, "test": tv,
                                   "pool": "weak:" + run.split("__")[0]}
    return members


# ------------------------------------------------------------------ metrics

def six(t, p, sysid):
    st, sp = defaultdict(list), defaultdict(list)
    for s, a, b in zip(sysid, t, p):
        st[s].append(a); sp[s].append(b)
    x = np.array([np.mean(st[k]) for k in st])
    y = np.array([np.mean(sp[k]) for k in st])
    return (np.mean((t - p) ** 2), scipy.stats.pearsonr(t, p).statistic,
            scipy.stats.spearmanr(t, p).statistic, np.mean((x - y) ** 2),
            scipy.stats.pearsonr(x, y).statistic, scipy.stats.spearmanr(x, y).statistic)


def srcc(a, b):
    return scipy.stats.spearmanr(a, b).statistic


# ------------------------------------------------------------------ combiners

def _ranks(M):
    """Row-wise rank transform of a [n_members, n] matrix, scaled to [0, 1]."""
    return np.array([scipy.stats.rankdata(m) / len(m) for m in M])


def _remap(u, reference):
    """Map values in [0,1] onto the empirical quantiles of `reference`.

    Keeps the rank order the blend produced while restoring a plausible 1-5 scale, so MSE and
    LCC stay meaningful instead of collapsing onto a uniform ramp.
    """
    q = np.clip(u, 0, 1)
    return np.quantile(reference, q)


class Combiner:
    """Fit on a subset of dev, apply to any member matrix."""

    def __init__(self, method, pools, ridge_alpha=100.0):
        self.method = method
        self.pools = pools            # list of pool label per member, aligned with rows of M
        self.ridge_alpha = ridge_alpha
        self.w = None
        self.alpha = None
        self.ref = None

    def _pool_means(self, M):
        labels = sorted(set(self.pools))
        return np.array([M[[i for i, p in enumerate(self.pools) if p == lab]].mean(axis=0)
                         for lab in labels]), labels

    def fit(self, M, y):
        self.ref = M.mean(axis=0)
        if self.method in ("mean", "rank-mean"):
            return self

        if self.method == "alpha":
            deep = [i for i, p in enumerate(self.pools) if p == "deep"]
            weak = [i for i, p in enumerate(self.pools) if p != "deep"]
            if not deep or not weak:
                self.alpha = 1.0 if deep else 0.0
                return self
            d, w = M[deep].mean(axis=0), M[weak].mean(axis=0)
            grid = np.linspace(0, 1, 21)
            self.alpha = float(max(grid, key=lambda a: srcc(y, a * d + (1 - a) * w)))
            return self

        if self.method == "nnls":
            from scipy.optimize import nnls
            P, self.nnls_labels = self._pool_means(M)
            w, _ = nnls(P.T, y)
            self.w = w / w.sum() if w.sum() > 1e-8 else np.ones(len(P)) / len(P)
            return self

        if self.method == "ridge":
            # Non-negative ridge via NNLS on the Tikhonov-augmented system: keeps weights
            # interpretable and stops the combiner cancelling members against each other,
            # which is the classic failure mode of unconstrained stacking at n=600.
            from scipy.optimize import nnls
            k = M.shape[0]
            A = np.vstack([M.T, np.sqrt(self.ridge_alpha) * np.eye(k)])
            b = np.concatenate([y, np.zeros(k)])
            w, _ = nnls(A, b)
            self.w = w / w.sum() if w.sum() > 1e-8 else np.ones(k) / k
            return self

        raise ValueError(self.method)

    def predict(self, M):
        if self.method == "mean":
            return M.mean(axis=0)
        if self.method == "rank-mean":
            return _remap(_ranks(M).mean(axis=0), self.ref)
        if self.method == "alpha":
            deep = [i for i, p in enumerate(self.pools) if p == "deep"]
            weak = [i for i, p in enumerate(self.pools) if p != "deep"]
            if not deep or not weak:
                return M.mean(axis=0)
            return self.alpha * M[deep].mean(axis=0) + (1 - self.alpha) * M[weak].mean(axis=0)
        if self.method == "nnls":
            P, _ = self._pool_means(M)
            return self.w @ P
        if self.method == "ridge":
            return self.w @ M
        raise ValueError(self.method)


# ------------------------------------------------------------------ main

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", default="spk_sim", choices=["spk_sim", "acc_sim"])
    p.add_argument("--weak-dir", default="egs/weak/preds")
    p.add_argument("--methods", nargs="+", default=METHODS)
    p.add_argument("--top-deep", type=int, default=8)
    p.add_argument("--top-weak", type=int, default=8)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--ridge-alpha", type=float, default=100.0)
    p.add_argument("--write-test", default="", help="Directory for the test submission CSV.")
    p.add_argument("--require-test", action="store_true",
                   help="Only admit members that also have test predictions.")
    args = p.parse_args()

    metric = args.target
    dev_rows = read_rows(DEV_LABELS)
    test_rows = read_rows(TEST_CSV)
    _, dev_kidx = index_of(dev_rows)
    _, test_kidx = index_of(test_rows)
    y = np.array([float(r[metric]) for r in dev_rows])
    sysid = np.array([r["system_id"] for r in dev_rows])

    members = collect_members(metric, args.weak_dir, dev_kidx, test_kidx,
                              len(dev_rows), len(test_rows), args.require_test)
    deep = {k: v for k, v in members.items() if v["pool"] == "deep"}
    weak = {k: v for k, v in members.items() if v["pool"] != "deep"}
    if not deep:
        raise SystemExit("no deep members found")

    # Select members on dev by solo UTT-SRCC. This is selection on the same set the combiner
    # is scored on, so the grouped-CV numbers below are optimistic by roughly the amount the
    # earlier split-half analysis measured (~0.00-0.01); it is not a free lunch, just a small
    # and quantified one.
    dsel = sorted(deep, key=lambda k: -srcc(y, deep[k]["dev"]))[: args.top_deep]
    wsel = sorted(weak, key=lambda k: -srcc(y, weak[k]["dev"]))[: args.top_weak]
    sel = dsel + wsel
    pools = [members[k]["pool"] for k in sel]

    n_test_ok = sum(members[k]["test"] is not None for k in sel)
    print(f"\n{'='*96}\n{metric.upper()}  —  phase-2 combination\n{'='*96}")
    print(f"  deep members : {len(dsel)} of {len(deep)} available")
    print(f"  weak members : {len(wsel)} of {len(weak)} available "
          f"({len(set(p for p in pools if p != 'deep'))} encoder pools)")
    print(f"  test preds   : {n_test_ok}/{len(sel)} members have them")

    Mdev = np.array([members[k]["dev"] for k in sel])

    gkf = GroupKFold(n_splits=args.n_splits)
    folds = list(gkf.split(Mdev.T, y, sysid))

    print(f"\n  GroupKFold({args.n_splits}) by system_id on dev — out-of-fold, "
          f"so these are honest estimates of unseen-system behaviour")
    hdr = f"{'method':<14}{'uMSE':>8}{'uLCC':>8}{'uSRCC':>8}{'sMSE':>8}{'sLCC':>8}{'sSRCC':>8}"
    print("  " + hdr + "\n  " + "-" * len(hdr))

    oof_scores = {}
    for method in args.methods:
        oof = np.zeros_like(y)
        for tr, va in folds:
            c = Combiner(method, pools, args.ridge_alpha).fit(Mdev[:, tr], y[tr])
            oof[va] = c.predict(Mdev[:, va])
        oof_scores[method] = oof
        print(f"  {method:<14}" + "".join(f"{q:>8.3f}" for q in six(y, oof, sysid)))

    # Reference rows: the deep pool alone, under the same protocol.
    deep_only = Mdev[[i for i, p in enumerate(pools) if p == "deep"]].mean(axis=0)
    print(f"  {'[deep only]':<14}" + "".join(f"{q:>8.3f}" for q in six(y, deep_only, sysid)))

    best = max(oof_scores, key=lambda m: srcc(y, oof_scores[m]))
    print(f"\n  best by out-of-fold UTT-SRCC: {best} ({srcc(y, oof_scores[best]):.3f})")

    rng = np.random.default_rng(0)
    n = len(y)
    for m in args.methods:
        d = []
        for _ in range(2000):
            s = rng.integers(0, n, n)
            d.append(srcc(y[s], oof_scores[m][s]) - srcc(y[s], deep_only[s]))
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"    {m:<12} minus deep-only: {np.mean(d):+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")

    # ---- refit on all of dev and write the submission -------------------------------
    if args.write_test:
        usable = [k for k in sel if members[k]["test"] is not None]
        if not usable:
            raise SystemExit("no member has test predictions — run the deep test inference job")
        if len(usable) < len(sel):
            print(f"\n  WARNING: {len(sel) - len(usable)} of {len(sel)} members lack test "
                  f"predictions and are dropped from the submission. The combiner is refitted "
                  f"on the survivors, so the dev numbers above no longer describe it exactly.")
        pools_u = [members[k]["pool"] for k in usable]
        Mdev_u = np.array([members[k]["dev"] for k in usable])
        Mtest_u = np.array([members[k]["test"] for k in usable])

        c = Combiner(best, pools_u, args.ridge_alpha).fit(Mdev_u, y)
        pred = np.clip(c.predict(Mtest_u), 1.0, 5.0)

        os.makedirs(args.write_test, exist_ok=True)
        out = os.path.join(args.write_test, f"test_{metric}_stack_{best}.csv")
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["system_id", "utterance_id", "wav_a_path", "wav_b_path", f"pred_{metric}"])
            for r, v in zip(test_rows, pred):
                w.writerow([r["system_id"], r["utterance_id"], r["wav_a_path"], r["wav_b_path"], v])

        man = out.replace(".csv", ".manifest.json")
        with open(man, "w", encoding="utf-8") as fh:
            json.dump({
                "metric": metric, "method": best,
                "selection": "top-K per pool by dev UTT-SRCC",
                "evaluation": f"GroupKFold({args.n_splits}) by system_id on dev, out-of-fold",
                "oof_utt_srcc": round(float(srcc(y, oof_scores[best])), 4),
                "refit": "weights refitted on all 600 dev pairs for the submission",
                "alpha": getattr(c, "alpha", None),
                "weights": (None if c.w is None else
                            dict(zip(getattr(c, "nnls_labels", usable),
                                     [round(float(x), 4) for x in c.w]))),
                "members": usable,
            }, fh, indent=2)
        print(f"\n  wrote {out}\n  wrote {man}")


if __name__ == "__main__":
    main()
