#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Write a submission from an EXPLICIT, frozen member list.

Why this exists separately from stack.py: stack.py selects members by dev UTT-SRCC. Once the
weak learners are refitted on train+dev their dev scores are in-sample fits, so that selection
would silently pick whichever model overfits hardest. The composition must instead stay frozen
at what the held-out runs chose, and this script does exactly that -- it selects nothing,
evaluates nothing, and only averages the members it is given.

The combiner is the unweighted mean, which is what won under the honest GroupKFold protocol on
held-out dev. That it has no parameters is what makes a train+dev refit coherent at all: had
`nnls` or `alpha` won, their weights would need a held-out set that no longer exists.

    python make_submission.py --target spk_sim --weak-dir egs/weak_traindev/preds \
        --out egs/submission_traindev
"""

import argparse
import csv
import json
import os

import numpy as np

EVAL_ROOT = "../baseline/data/vmc2026_track3_eval_phase_distro_v3_syn"
TEST_CSV = f"{EVAL_ROOT}/sets/test.csv"
DEEP_ROOT = "../unified/egs/ensemble_runs"

# Frozen on the held-out runs: top-8 per pool by dev UTT-SRCC, when dev was NOT in training.
# Do not re-derive these from any run whose manifest says dev_in_train.
DEEP_MEMBERS = {
    "spk_sim": ["eres2netv2-w24s4ep4-mse-baseline", "eres2netv2-w24s4ep4-mse-bilinear",
                "eres2netv2-w24s4ep4-coral-baseline", "eres2netv2-coral-baseline",
                "eres2netv2-mse-bilinear", "eres2netv2-w24s4ep4-coral-bilinear",
                "eres2netv2-coral-bilinear", "eres2netv2-mse-baseline"],
    "acc_sim": ["eres2netv2-w24s4ep4-coral-bilinear", "eres2netv2-w24s4ep4-coral-baseline",
                "eres2netv2-w24s4ep4-mse-bilinear", "eres2netv2-w24s4ep4-mse-baseline",
                "ecapa-voxceleb-coral-bilinear", "eres2netv2-coral-baseline",
                "commonaccent-ecapa-coral-bilinear", "ecapa-voxceleb-mse-bilinear"],
}
WEAK_MEMBERS = {
    "spk_sim": ["WAVLM_LARGE_l4__full__ridge", "WAVLM_LARGE_l4__compact__ridge",
                "WAV2VEC2_XLSR_300M_l4__compact__ridge", "WAV2VEC2_XLSR_300M_l4__full__ridge",
                "WAVLM_LARGE_l8__full__ridge", "WAV2VEC2_XLSR_300M_l8__full__ridge",
                "WAVLM_BASE_PLUS_l4__full__ridge", "eres2netv2__full__ksvr"],
    "acc_sim": ["WAVLM_LARGE_l4__compact__ridge", "WAVLM_LARGE_l4__full__ridge",
                "WAVLM_BASE_PLUS_l4__full__ridge", "WAVLM_LARGE_l8__compact__ridge",
                "WAV2VEC2_XLSR_300M_l8__compact__ridge", "WAV2VEC2_XLSR_300M_l8__full__ridge",
                "WAVLM_LARGE_l8__full__ridge", "WAV2VEC2_XLSR_300M_l4__compact__ridge"],
}


def load(path, metric, kidx, n):
    v = np.full(n, np.nan)
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            i = kidx.get((r["wav_a_path"], r["wav_b_path"]))
            if i is not None:
                v[i] = float(r[f"pred_{metric}"])
    if np.isnan(v).any():
        raise SystemExit(f"{path}: missing predictions for {int(np.isnan(v).sum())} test pairs")
    return v


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", required=True, choices=["spk_sim", "acc_sim"])
    p.add_argument("--weak-dir", required=True, help="Directory of weak __test.csv predictions.")
    p.add_argument("--out", required=True)
    p.add_argument("--note", default="", help="Free text recorded in the manifest.")
    args = p.parse_args()

    metric = args.target
    rows = list(csv.DictReader(open(TEST_CSV)))
    kidx = {(r["wav_a_path"], r["wav_b_path"]): i for i, r in enumerate(rows)}
    n = len(rows)

    mats, used = [], []
    for cfg in DEEP_MEMBERS[metric]:
        f = f"{DEEP_ROOT}/{cfg}_{metric}/test_{metric}_best.csv"
        mats.append(load(f, metric, kidx, n)); used.append(f"deep::{cfg}")
    for run in WEAK_MEMBERS[metric]:
        f = f"{args.weak_dir}/{run}__{metric}__test.csv"
        mats.append(load(f, metric, kidx, n)); used.append(f"weak::{run}")

    pred = np.clip(np.mean(mats, axis=0), 1.0, 5.0)

    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, f"test_{metric}.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["system_id", "utterance_id", "wav_a_path", "wav_b_path", f"pred_{metric}"])
        for r, v in zip(rows, pred):
            w.writerow([r["system_id"], r["utterance_id"], r["wav_a_path"], r["wav_b_path"], v])

    with open(out.replace(".csv", ".manifest.json"), "w", encoding="utf-8") as fh:
        json.dump({
            "metric": metric,
            "combiner": "unweighted mean, no fitted parameters",
            "n_members": len(mats),
            "selection": "FROZEN from the held-out runs (dev not in training); not re-derived",
            "weak_dir": args.weak_dir,
            "note": args.note,
            "pred_mean": round(float(pred.mean()), 4),
            "pred_sd": round(float(pred.std()), 4),
            "members": used,
        }, fh, indent=2)

    print(f"  {metric}: {len(mats)} members -> {out}")
    print(f"    mean {pred.mean():.3f}  sd {pred.std():.3f}  range [{pred.min():.3f}, {pred.max():.3f}]")


if __name__ == "__main__":
    main()
