#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 1 of the UTMOS22-style stack: classical regressors on frozen embeddings.

UTMOS22's winning system fits five learner families on mean-pooled frozen SSL features and
only then stacks them. This is that stage, with two changes forced by our data:

  * GROUPED CV BY system_id, not random folds. dev contains two systems absent from train
    (sys003, sys015) and test contains four (sys003, sys004, sys015, sys021), so this is
    partly an unseen-system problem. Random folds would let a model memorise system identity
    and would pick hyperparameters that do not survive the real split.

  * sklearn's HistGradientBoostingRegressor stands in for LightGBM, which is not installed.
    Same algorithm family, no new dependency.

Hyperparameters are chosen by grouped CV on train; dev is never touched during fitting, so
the dev numbers reported by analyze.py are clean. Out-of-fold predictions are written too --
they are what a phase-2 stacker would need.

    python train_weak.py --features egs/features/WAVLM_LARGE_l12.npz --target spk_sim
    python train_weak.py --features egs/features/*.npz --feature-sets full compact
"""

import argparse
import csv
import glob
import json
import logging
import os
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.svm import SVR, LinearSVR

from features import PairFeaturizer, load_embeddings, read_pairs, read_rating_rows

TRAIN_ROOT = "../baseline/data/vmc2026_track3_train_phase_distro_v3_syn"
EVAL_ROOT = "../baseline/data/vmc2026_track3_eval_phase_distro_v3_syn"
TRAIN_CSV = f"{TRAIN_ROOT}/sets/train.csv"
DEV_CSV = f"{EVAL_ROOT}/sets/dev_with_labels.csv"
TEST_CSV = f"{EVAL_ROOT}/sets/test.csv"

SCORE_MIN, SCORE_MAX = 1.0, 5.0

# The five UTMOS22 families. Grids are deliberately small: with 2,800 pairs and a ±0.05
# measurement floor on dev, an exhaustive search would be fitting noise.
LEARNERS = {
    "ridge": (Ridge, [{"alpha": a} for a in (1.0, 10.0, 100.0, 1000.0, 10000.0)]),
    "linsvr": (LinearSVR, [{"C": c, "epsilon": 0.1, "max_iter": 5000, "random_state": 0}
                           for c in (0.01, 0.1, 1.0)]),
    "ksvr": (SVR, [{"kernel": "rbf", "C": c, "gamma": g}
                   for c in (1.0, 10.0) for g in ("scale", 1e-3)]),
    "rf": (RandomForestRegressor, [{"n_estimators": 300, "max_depth": d,
                                    "min_samples_leaf": leaf, "random_state": 0, "n_jobs": -1}
                                   for d in (None, 12) for leaf in (1, 5)]),
    "hgb": (HistGradientBoostingRegressor, [{"max_iter": 300, "learning_rate": lr,
                                             "max_leaf_nodes": n, "random_state": 0}
                                            for lr in (0.05, 0.1) for n in (15, 31)]),
}


def srcc(a, b):
    import scipy.stats
    return scipy.stats.spearmanr(a, b).statistic


def cv_select(cls, grid, X, y, groups, n_splits):
    """Pick hyperparameters by grouped CV, and return OOF predictions for the winner."""
    gkf = GroupKFold(n_splits=n_splits)
    folds = list(gkf.split(X, y, groups))
    best = None
    for params in grid:
        oof = np.zeros_like(y)
        for tr, va in folds:
            m = cls(**params).fit(X[tr], y[tr])
            oof[va] = m.predict(X[va])
        score = srcc(y, oof)
        if not np.isfinite(score):
            continue
        if best is None or score > best[0]:
            best = (score, params, oof)
    if best is None:
        raise RuntimeError("every hyperparameter setting produced a degenerate fit")
    return best


def write_preds(path, pairs, preds, metric):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["system_id", "utterance_id", "wav_a_path", "wav_b_path", f"pred_{metric}"])
        for p, v in zip(pairs, preds):
            w.writerow([p["system_id"], p["utterance_id"], p["wav_a_path"], p["wav_b_path"], v])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", nargs="+", required=True,
                   help=".npz files from extract_features.py (globs allowed).")
    p.add_argument("--train-csv", default=TRAIN_CSV,
                   help="Fitting set. Point at sets/train_plus_dev.csv to train on train+dev "
                        "(allowed by the challenge). When dev is contained in this file the "
                        "reported dev score becomes in-sample and is flagged as such -- there "
                        "is then no held-out set left, so members cannot be re-selected.")
    p.add_argument("--targets", nargs="+", default=["spk_sim", "acc_sim"])
    p.add_argument("--feature-sets", nargs="+", default=["full", "compact"])
    p.add_argument("--learners", nargs="+", default=list(LEARNERS))
    p.add_argument("--pca-threshold", type=int, default=256,
                   help="Embeddings wider than this are PCA-reduced to --pca-dim first.")
    p.add_argument("--pca-dim", type=int, default=128)
    p.add_argument("--per-rating", action="store_true",
                   help="Fit on individual listener ratings (13,687 rows) instead of per-pair "
                        "means (2,800). The feature row is identical within a pair, so for a "
                        "squared-error learner this is equivalent to weighting pairs by rater "
                        "count; it can only really matter for the tree and SVR families.")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--outdir", default="egs/weak")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    feats = sorted({f for pat in args.features for f in glob.glob(pat)})
    if not feats:
        raise SystemExit(f"no feature files matched {args.features}")

    preds_dir = os.path.join(args.outdir, "preds")
    os.makedirs(preds_dir, exist_ok=True)

    train_pairs = (read_rating_rows(args.train_csv) if args.per_rating
                   else read_pairs(args.train_csv))
    dev_pairs = read_pairs(DEV_CSV)
    test_pairs = read_pairs(TEST_CSV, metrics=())
    groups = np.array([p["system_id"] for p in train_pairs])
    unit = "rating rows" if args.per_rating else "pairs"
    logging.info(f"{len(train_pairs)} train {unit} over {len(set(groups))} systems, "
                 f"{len(dev_pairs)} dev, {len(test_pairs)} test  [{args.train_csv}]")

    # Detected from the data rather than from a flag, so a combined CSV under any name is
    # caught. If every dev pair is already in the fitting set, the dev score below is an
    # in-sample fit and must not be compared against the held-out numbers from earlier runs.
    train_keys = {(p["wav_a_path"], p["wav_b_path"]) for p in train_pairs}
    dev_in_train = all((p["wav_a_path"], p["wav_b_path"]) in train_keys for p in dev_pairs)
    if dev_in_train:
        logging.warning("dev is CONTAINED IN the fitting set: reported dev_srcc is IN-SAMPLE, "
                        "not a held-out estimate, and no set remains for member selection.")

    manifest_path = os.path.join(args.outdir, "stage1_manifest.jsonl")
    done = set()
    if os.path.exists(manifest_path) and not args.overwrite:
        with open(manifest_path, encoding="utf-8") as fh:
            for line in fh:
                done.add(json.loads(line)["run"])

    mode = "w" if args.overwrite else "a"
    with open(manifest_path, mode, encoding="utf-8") as manifest:
        for fpath in feats:
            enc = os.path.splitext(os.path.basename(fpath))[0]
            emb = load_embeddings(fpath)
            dim = len(next(iter(emb.values())))
            pca_dim = args.pca_dim if dim > args.pca_threshold else 0

            for fset in args.feature_sets:
                fz = PairFeaturizer(fset, pca_dim=pca_dim)
                Xtr = fz.fit_transform(train_pairs, emb)
                Xdev = fz.transform(dev_pairs, emb)
                Xte = fz.transform(test_pairs, emb)
                logging.info(f"{enc} [{fset}] emb_dim={dim} pca={pca_dim or 'none'} "
                             f"-> {Xtr.shape[1]} features")

                for metric in args.targets:
                    y = np.array([p[metric] for p in train_pairs], dtype=np.float64)
                    for lname in args.learners:
                        run = f"{enc}__{fset}__{lname}__{metric}"
                        if run in done:
                            logging.info(f"  {run}: already done, skipping")
                            continue
                        cls, grid = LEARNERS[lname]
                        t0 = time.time()
                        try:
                            cv_srcc, params, oof = cv_select(
                                cls, grid, Xtr, y, groups, args.n_splits)
                        except Exception as e:  # noqa: BLE001 - one bad family must not stop the sweep
                            logging.warning(f"  {run}: FAILED ({type(e).__name__}: {e})")
                            continue

                        model = cls(**params).fit(Xtr, y)
                        dev_pred = np.clip(model.predict(Xdev), SCORE_MIN, SCORE_MAX)
                        test_pred = np.clip(model.predict(Xte), SCORE_MIN, SCORE_MAX)

                        write_preds(f"{preds_dir}/{run}__oof.csv", train_pairs,
                                    np.clip(oof, SCORE_MIN, SCORE_MAX), metric)
                        write_preds(f"{preds_dir}/{run}__dev.csv", dev_pairs, dev_pred, metric)
                        write_preds(f"{preds_dir}/{run}__test.csv", test_pairs, test_pred, metric)

                        ytrue = np.array([p[metric] for p in dev_pairs])
                        rec = {"run": run, "encoder": enc, "feature_set": fset,
                               "learner": lname, "metric": metric, "params": params,
                               "n_features": int(Xtr.shape[1]), "pca_dim": pca_dim,
                               "train_csv": args.train_csv, "n_train_pairs": len(train_pairs),
                               "per_rating": args.per_rating,
                               "dev_in_train": dev_in_train,
                               "cv_srcc": round(float(cv_srcc), 4),
                               "dev_srcc": round(float(srcc(ytrue, dev_pred)), 4),
                               "seconds": round(time.time() - t0, 1)}
                        manifest.write(json.dumps(rec) + "\n")
                        manifest.flush()
                        logging.info(f"  {run}: cv_srcc={rec['cv_srcc']:.3f} "
                                     f"dev_srcc={rec['dev_srcc']:.3f}"
                                     f"{' [IN-SAMPLE]' if dev_in_train else ''} "
                                     f"({rec['seconds']}s) {params}")

    logging.info(f"manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
