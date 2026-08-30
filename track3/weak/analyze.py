#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase-1 verdict: do frozen-feature weak learners make DECORRELATED errors?

This is the question the whole weak-learner track exists to answer, so it is measured
directly rather than inferred from a leaderboard number. The deep pool sits at a mean
pairwise residual correlation of ~0.92, and the relationship to ensemble gain is already
established on this dev set:

    residual corr 0.878 (heterogeneous 16)  ->  +0.062 over mean member
    residual corr 0.921 (factorial 16)      ->  +0.047
    residual corr 0.957 (factorial top-8)   ->  +0.019

So a weak learner earns its place by landing BELOW 0.878 against the deep members, not by
scoring well on its own. Frozen mean-pooled features are expected to lose to fine-tuned
encoders solo -- the heterogeneous pool's best contributors were its weakest members too.

No stacking here, and nothing is fitted on dev: weak learners were trained on train only, so
every number below is clean. A learned combiner is phase 2, and it needs out-of-fold deep
predictions that do not exist yet.

    python analyze.py --target spk_sim
"""

import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np
import scipy.stats

EVAL_ROOT = "../baseline/data/vmc2026_track3_eval_phase_distro_v3_syn"
DEV_LABELS = f"{EVAL_ROOT}/sets/dev_with_labels.csv"
DEEP_ROOT = "../unified/egs/ensemble_runs"
ENCS = ["ecapa-voxceleb", "commonaccent-ecapa", "eres2netv2", "eres2netv2-w24s4ep4"]
DEEP_CFGS = [f"{e}-{l}-{i}" for e in ENCS for l in ("mse", "coral") for i in ("baseline", "bilinear")]


def load_gt():
    rows = list(csv.DictReader(open(DEV_LABELS)))
    keys = [(r["wav_a_path"], r["wav_b_path"]) for r in rows]
    return rows, keys, {k: i for i, k in enumerate(keys)}


def load_pred(path, metric, kidx, n):
    v = np.full(n, np.nan)
    col = f"pred_{metric}"
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            i = kidx.get((r["wav_a_path"], r["wav_b_path"]))
            if i is not None and col in r:
                v[i] = float(r[col])
    return None if np.isnan(v).any() else v


def six(t, p, sysid):
    st, sp = defaultdict(list), defaultdict(list)
    for s, a, b in zip(sysid, t, p):
        st[s].append(a); sp[s].append(b)
    x = np.array([np.mean(st[k]) for k in st])
    y = np.array([np.mean(sp[k]) for k in st])
    return (np.mean((t - p) ** 2), scipy.stats.pearsonr(t, p).statistic,
            scipy.stats.spearmanr(t, p).statistic, np.mean((x - y) ** 2),
            scipy.stats.pearsonr(x, y).statistic, scipy.stats.spearmanr(x, y).statistic)


def mean_resid_corr(residuals):
    if len(residuals) < 2:
        return float("nan")
    C = np.corrcoef(np.array(residuals))
    return float(C[np.triu_indices(len(residuals), 1)].mean())


def cross_resid_corr(a_res, b_res):
    """Mean correlation between every member of one pool and every member of the other."""
    return float(np.mean([[np.corrcoef(x, y)[0, 1] for y in b_res] for x in a_res]))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", default="spk_sim", choices=["spk_sim", "acc_sim"])
    p.add_argument("--weak-dir", default="egs/weak/preds")
    p.add_argument("--top-weak", type=int, default=8, help="Weak members in the weak ensemble.")
    p.add_argument("--top-deep", type=int, default=8, help="Deep members in the deep ensemble.")
    p.add_argument("--show", type=int, default=25, help="Rows of the per-model table to print.")
    args = p.parse_args()

    metric = args.target
    rows, keys, kidx = load_gt()
    n = len(keys)
    t = np.array([float(r[metric]) for r in rows])
    sysid = np.array([r["system_id"] for r in rows])

    # -- deep pool (best checkpoints) --------------------------------------------------
    deep = {}
    for cfg in DEEP_CFGS:
        f = f"{DEEP_ROOT}/{cfg}_{metric}/dev_{metric}_best.csv"
        if os.path.exists(f):
            v = load_pred(f, metric, kidx, n)
            if v is not None:
                deep[cfg] = v
    deep_rank = sorted(deep, key=lambda c: -scipy.stats.spearmanr(t, deep[c]).statistic)

    # -- weak pool ----------------------------------------------------------------------
    weak = {}
    for f in sorted(glob.glob(f"{args.weak_dir}/*__{metric}__dev.csv")):
        run = os.path.basename(f)[: -len(f"__{metric}__dev.csv")]
        v = load_pred(f, metric, kidx, n)
        if v is not None:
            weak[run] = v
    if not weak:
        raise SystemExit(f"no weak predictions in {args.weak_dir} for {metric}")
    weak_rank = sorted(weak, key=lambda r: -scipy.stats.spearmanr(t, weak[r]).statistic)

    W = 52
    print(f"\n{'='*100}\n{metric.upper()}  —  frozen-feature weak learners on dev (600 pairs)\n{'='*100}")
    hdr = (f"{'weak model':<{W}}{'uMSE':>8}{'uLCC':>8}{'uSRCC':>8}"
           f"{'sMSE':>8}{'sLCC':>8}{'sSRCC':>8}{'rho_deep':>10}")
    print(hdr + "\n" + "-" * len(hdr))
    deep_res = [deep[c] - t for c in deep_rank]
    for run in weak_rank[: args.show]:
        s = six(t, weak[run], sysid)
        rho = cross_resid_corr([weak[run] - t], deep_res)
        print(f"{run:<{W}}" + "".join(f"{q:>8.3f}" for q in s) + f"{rho:>10.3f}")
    if len(weak_rank) > args.show:
        print(f"... {len(weak_rank) - args.show} more")

    # -- pools ---------------------------------------------------------------------------
    wsel = weak_rank[: args.top_weak]
    dsel = deep_rank[: args.top_deep]
    w_ens = np.mean([weak[r] for r in wsel], axis=0)
    d_ens = np.mean([deep[c] for c in dsel], axis=0)
    combos = {
        f"deep top-{len(dsel)}": d_ens,
        f"weak top-{len(wsel)}": w_ens,
        "deep + weak (50/50)": 0.5 * d_ens + 0.5 * w_ens,
        "deep + weak (75/25)": 0.75 * d_ens + 0.25 * w_ens,
        "all members pooled": np.mean([deep[c] for c in dsel] + [weak[r] for r in wsel], axis=0),
    }
    print(f"\n{'ensemble':<{W}}{'uMSE':>8}{'uLCC':>8}{'uSRCC':>8}{'sMSE':>8}{'sLCC':>8}{'sSRCC':>8}")
    print("-" * (W + 48))
    for name, v in combos.items():
        print(f"{name:<{W}}" + "".join(f"{q:>8.3f}" for q in six(t, v, sysid)))

    # -- the actual phase-1 verdict --------------------------------------------------------
    w_res = [weak[r] - t for r in wsel]
    d_res = [deep[c] - t for c in dsel]
    print(f"\n  mean pairwise residual correlation")
    print(f"    within deep top-{len(dsel)}          {mean_resid_corr(d_res):.3f}")
    print(f"    within weak top-{len(wsel)}          {mean_resid_corr(w_res):.3f}")
    print(f"    deep vs weak (cross-pool)   {cross_resid_corr(d_res, w_res):.3f}")
    print(f"    reference: heterogeneous deep pool reached 0.878 and gained +0.062")

    rng = np.random.default_rng(0)
    base = combos[f"deep top-{len(dsel)}"]
    for name in ("deep + weak (50/50)", "deep + weak (75/25)", "all members pooled"):
        d = []
        for _ in range(2000):
            s = rng.integers(0, n, n)
            d.append(scipy.stats.spearmanr(t[s], combos[name][s]).statistic
                     - scipy.stats.spearmanr(t[s], base[s]).statistic)
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"  {name} minus deep-only UTT-SRCC: {np.mean(d):+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")

    mpath = "egs/weak/stage1_manifest.jsonl"
    if os.path.exists(mpath):
        recs = [json.loads(l) for l in open(mpath, encoding="utf-8")]
        recs = [r for r in recs if r["metric"] == metric]
        if recs:
            print(f"\n  grouped-CV vs dev agreement over {len(recs)} runs: "
                  f"r={np.corrcoef([r['cv_srcc'] for r in recs], [r['dev_srcc'] for r in recs])[0,1]:.3f}")
            by = defaultdict(list)
            for r in recs:
                by[r["learner"]].append(r["dev_srcc"])
            print("  best dev UTT-SRCC by learner family: " +
                  "  ".join(f"{k}={max(v):.3f}" for k, v in sorted(by.items())))
            by = defaultdict(list)
            for r in recs:
                by[r["encoder"]].append(r["dev_srcc"])
            print("  best dev UTT-SRCC by encoder:")
            for k, v in sorted(by.items(), key=lambda kv: -max(kv[1])):
                print(f"    {max(v):.3f}  {k}")


if __name__ == "__main__":
    main()
