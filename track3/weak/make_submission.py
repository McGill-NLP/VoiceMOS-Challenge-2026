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
DEV_CSV = f"{EVAL_ROOT}/sets/dev_with_labels.csv"
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
# Ranked by held-out dev UTT-SRCC. --weak-k truncates this list; 16 is the measured optimum
# (0.623 / 0.603 combined with the deep top-8, against 0.620 / 0.595 at k=8 and 0.588 / 0.576
# using all 58 candidates). Ranks 9-16 individually score BELOW the top 8 and still help:
# they bring in ksvr, hgb and the speaker-ID encoders, dropping internal residual correlation
# from 0.925 to 0.871 (spk_sim) and 0.955 to 0.877 (acc_sim).
WEAK_MEMBERS = {
    "spk_sim": ["WAVLM_LARGE_l4__full__ridge", "WAVLM_LARGE_l4__compact__ridge",
                "WAV2VEC2_XLSR_300M_l4__compact__ridge", "WAV2VEC2_XLSR_300M_l4__full__ridge",
                "WAVLM_LARGE_l8__full__ridge", "WAV2VEC2_XLSR_300M_l8__full__ridge",
                "WAVLM_BASE_PLUS_l4__full__ridge", "eres2netv2__full__ksvr",
                "WAVLM_LARGE_l8__compact__ridge", "eres2netv2__full__ridge",
                "eres2netv2__full__hgb", "eres2netv2-w24s4ep4__full__ksvr",
                "eres2netv2-w24s4ep4__full__hgb", "ecapa-voxceleb__full__hgb",
                "WAVLM_BASE_PLUS_l4__compact__ridge", "eres2netv2__compact__ksvr"],
    "acc_sim": ["WAVLM_LARGE_l4__compact__ridge", "WAVLM_LARGE_l4__full__ridge",
                "WAVLM_BASE_PLUS_l4__full__ridge", "WAVLM_LARGE_l8__compact__ridge",
                "WAV2VEC2_XLSR_300M_l8__compact__ridge", "WAV2VEC2_XLSR_300M_l8__full__ridge",
                "WAVLM_LARGE_l8__full__ridge", "WAV2VEC2_XLSR_300M_l4__compact__ridge",
                "WAVLM_BASE_PLUS_l4__compact__ridge", "WAV2VEC2_XLSR_300M_l4__full__ridge",
                "ecapa-voxceleb__full__ridge", "commonaccent-ecapa__full__hgb",
                "ecapa-voxceleb__full__ksvr", "WAVLM_LARGE_l24__full__ridge",
                "ecapa-voxceleb__compact__hgb", "ecapa-voxceleb__full__hgb"],
}


def load(path, metric, kidx, n):
    v = np.full(n, np.nan)
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            i = kidx.get((r["wav_a_path"], r["wav_b_path"]))
            if i is not None:
                v[i] = float(r[f"pred_{metric}"])
    if np.isnan(v).any():
        raise SystemExit(f"{path}: missing predictions for {int(np.isnan(v).sum())} pairs")
    return v


def weak_dev_is_in_sample(weak_dir):
    """True when the weak learners in `weak_dir` were fitted on a set containing dev.

    Read from the sibling manifest that train_weak.py writes rather than guessed from the
    directory name, so a pool built under any name is classified correctly. If dev was in the
    fitting set, the dev predictions below are partly memorisation and must not be compared
    against the held-out numbers.
    """
    man = os.path.join(os.path.dirname(weak_dir.rstrip("/")), "stage1_manifest.jsonl")
    if not os.path.exists(man):
        return None
    with open(man, encoding="utf-8") as fh:
        return any(json.loads(line).get("dev_in_train") for line in fh)


def six(t, p, sysid):
    import scipy.stats
    st, sp = {}, {}
    for s, a, b in zip(sysid, t, p):
        st.setdefault(s, []).append(a); sp.setdefault(s, []).append(b)
    x = np.array([np.mean(st[k]) for k in st]); y = np.array([np.mean(sp[k]) for k in st])
    return (np.mean((t - p) ** 2), scipy.stats.pearsonr(t, p).statistic,
            scipy.stats.spearmanr(t, p).statistic, np.mean((x - y) ** 2),
            scipy.stats.pearsonr(x, y).statistic, scipy.stats.spearmanr(x, y).statistic)


def write_csv(path, rows, preds, metric):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["system_id", "utterance_id", "wav_a_path", "wav_b_path", f"pred_{metric}"])
        for r, v in zip(rows, preds):
            w.writerow([r["system_id"], r["utterance_id"], r["wav_a_path"], r["wav_b_path"], v])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", required=True, choices=["spk_sim", "acc_sim"])
    p.add_argument("--weak-dir", required=True, help="Directory of weak __test.csv predictions.")
    p.add_argument("--out", required=True)
    p.add_argument("--weak-k", type=int, default=16,
                   help="How many of the frozen weak members to use (they are stored in "
                        "held-out rank order). 16 is the measured optimum.")
    p.add_argument("--deep-k", type=int, default=8)
    p.add_argument("--note", default="", help="Free text recorded in the manifest.")
    args = p.parse_args()

    metric = args.target
    rows = list(csv.DictReader(open(TEST_CSV)))
    kidx = {(r["wav_a_path"], r["wav_b_path"]): i for i, r in enumerate(rows)}
    n = len(rows)

    dev_rows = list(csv.DictReader(open(DEV_CSV)))
    dev_kidx = {(r["wav_a_path"], r["wav_b_path"]): i for i, r in enumerate(dev_rows)}
    n_dev = len(dev_rows)
    in_sample = weak_dev_is_in_sample(args.weak_dir)

    deep_sel = DEEP_MEMBERS[metric][: args.deep_k]
    weak_sel = WEAK_MEMBERS[metric][: args.weak_k]
    if len(deep_sel) < args.deep_k or len(weak_sel) < args.weak_k:
        raise SystemExit(f"only {len(deep_sel)} deep / {len(weak_sel)} weak members are frozen "
                         f"for {metric}; cannot honour --deep-k {args.deep_k} --weak-k {args.weak_k}")

    mats, dev_mats, used = [], [], []
    for cfg in deep_sel:
        mats.append(load(f"{DEEP_ROOT}/{cfg}_{metric}/test_{metric}_best.csv", metric, kidx, n))
        dev_mats.append(load(f"{DEEP_ROOT}/{cfg}_{metric}/dev_{metric}_best.csv",
                             metric, dev_kidx, n_dev))
        used.append(f"deep::{cfg}")
    for run in weak_sel:
        mats.append(load(f"{args.weak_dir}/{run}__{metric}__test.csv", metric, kidx, n))
        dev_mats.append(load(f"{args.weak_dir}/{run}__{metric}__dev.csv",
                             metric, dev_kidx, n_dev))
        used.append(f"weak::{run}")

    pred = np.clip(np.mean(mats, axis=0), 1.0, 5.0)
    dev_pred = np.clip(np.mean(dev_mats, axis=0), 1.0, 5.0)

    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, f"test_{metric}.csv")
    write_csv(out, rows, pred, metric)

    # Dev predictions of the same ensemble. Held out only when the weak half was fitted
    # without dev; otherwise the filename says so, because the numbers are not comparable.
    suffix = "_IN-SAMPLE" if in_sample else ""
    dev_out = os.path.join(args.out, f"dev_{metric}{suffix}.csv")
    write_csv(dev_out, dev_rows, dev_pred, metric)

    y = np.array([float(r[metric]) for r in dev_rows])
    sysid = np.array([r["system_id"] for r in dev_rows])
    s = six(y, dev_pred, sysid)

    with open(out.replace(".csv", ".manifest.json"), "w", encoding="utf-8") as fh:
        json.dump({
            "metric": metric,
            "combiner": "unweighted mean, no fitted parameters",
            "n_members": len(mats),
            "deep_k": args.deep_k, "weak_k": args.weak_k,
            "selection": "FROZEN from the held-out runs (dev not in training); not re-derived",
            "weak_dir": args.weak_dir,
            "weak_fitted_on_dev": in_sample,
            "dev_scores_are_in_sample": bool(in_sample),
            "dev_metrics": dict(zip(["uMSE", "uLCC", "uSRCC", "sMSE", "sLCC", "sSRCC"],
                                    [round(float(q), 4) for q in s])),
            "note": args.note,
            "pred_mean": round(float(pred.mean()), 4),
            "pred_sd": round(float(pred.std()), 4),
            "members": used,
        }, fh, indent=2)

    tag = "  [IN-SAMPLE, not a held-out estimate]" if in_sample else "  [held out]"
    print(f"  {metric}: {len(mats)} members -> {out}")
    print(f"    test  mean {pred.mean():.3f}  sd {pred.std():.3f}  "
          f"range [{pred.min():.3f}, {pred.max():.3f}]")
    print(f"    dev   -> {dev_out}")
    print(f"      uMSE {s[0]:.3f}  uLCC {s[1]:.3f}  uSRCC {s[2]:.3f}  "
          f"sMSE {s[3]:.3f}  sLCC {s[4]:.3f}  sSRCC {s[5]:.3f}{tag}")


if __name__ == "__main__":
    main()
