#!/usr/bin/env bash
#SBATCH --job-name=voicemos-track3-weak-outliers
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=david.guzman@mila.quebec

# Does dropping two unreliable listeners improve the weak learners?
#
#   sbatch track3/jobs/weak/voicemos-track3-weak-outliers.sh
#
# WHAT sets/train-without-outliers.csv ACTUALLY IS. Despite the name it does not remove
# outlier ratings scattered through the data: it removes two whole LISTENERS. 25 -> 23
# listeners, 1,097 of 13,687 rating rows (8.0%), and ZERO pairs -- all 2,800 survive, with the
# removals spread evenly across systems (55-65 rows each). Each pair therefore averages ~4.5
# ratings instead of ~4.9, and the per-pair targets shift slightly (spk_sim mean 4.048 ->
# 4.019, sd 1.190 -> 1.205). This is listener-level quality control, in the spirit of the
# judge modelling in MBNet/LDNet, applied by deletion rather than by modelling.
#
# WHY IT IS CHEAP. No feature extraction: every one of the 2,937 referenced wavs is already in
# weak/egs/features/, since no new audio is involved. And because targets are per-pair means,
# the design matrix stays 2,800 x D no matter how many rating rows feed it -- only y changes.
# So this costs exactly what the equivalent train+dev sweep cost: 84 runs, ~28 min of fitting.
# The GPU is requested but unused; the CPU-only queues start slowly on this cluster.
#
# COMPARABILITY. dev is untouched, so it stays genuinely held out and the numbers line up
# directly against the shipped results (deep+weak top-16 reached 0.623 spk_sim / 0.603
# acc_sim; the weak halves alone reached 0.622 / 0.597).
#
# WHAT IS RUN. Exactly the configuration the 16 frozen members need, no more:
#   - ridge on all 13 feature files, both feature sets, both targets   (52 runs)
#   - ksvr + hgb on the four speaker-ID encoders                       (32 runs)
# ksvr and hgb are NOT run on the SSL files: at 4,098 features they cost 10-20 min each and no
# frozen member uses that combination.
#
# Deliberately NOT using `set -e`: one failing family must not kill the sweep.

START_TIME=$SECONDS
echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"
echo "SLURM_NODELIST: $SLURM_NODELIST   cpus: ${SLURM_CPUS_PER_TASK:-?}"

module load miniconda/3
conda activate VoiceMOS

if [ "$CONDA_DEFAULT_ENV" != "VoiceMOS" ]; then
    echo "ERROR: conda env is '${CONDA_DEFAULT_ENV:-none}', expected VoiceMOS"; exit 1
fi
python -c "import numpy, sklearn, scipy" \
    || { echo "ERROR: numpy/sklearn/scipy not importable"; exit 1; }

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS

REPO=${REPO:-/home/mila/g/guzmand/scratch/Repositories/VoiceMOS-Challenge-2026}
cd "$REPO/track3/weak" || exit 1

TRAIN_CSV=${TRAIN_CSV:-../baseline/data/vmc2026_track3_train_phase_distro_v3_syn/sets/train-without-outliers.csv}
FEATDIR=${FEATDIR:-egs/features}
OUTDIR=${OUTDIR:-egs/weak_outliers}

[ -f "$TRAIN_CSV" ] || { echo "ERROR: no training csv at $TRAIN_CSV"; exit 1; }
[ -d "$FEATDIR" ] || { echo "ERROR: no cached features in $FEATDIR"; exit 1; }

echo "=================================================================="
echo "weak learners on the outlier-removed training set"
echo "  train : $TRAIN_CSV"
echo "  feats : $(ls "$FEATDIR"/*.npz | wc -l) cached files, no PCA"
echo "  out   : $OUTDIR      (dev held out, directly comparable)"
echo "=================================================================="

##################################################################
# 1. ridge everywhere (all 13 feature files)
##################################################################
echo ""; echo "### ridge, all feature files ($(date))"
python train_weak.py --train-csv "$TRAIN_CSV" --pca-threshold 99999 \
    --features "$FEATDIR/*.npz" --feature-sets full compact \
    --learners ridge --outdir "$OUTDIR"
[ $? -ne 0 ] && echo "RIDGE SWEEP FAILED"

##################################################################
# 2. ksvr + hgb on the speaker/accent-ID encoders only
##################################################################
echo ""; echo "### ksvr + hgb, SV encoders ($(date))"
python train_weak.py --train-csv "$TRAIN_CSV" --pca-threshold 99999 \
    --features "$FEATDIR/ecapa-voxceleb.npz" "$FEATDIR/commonaccent-ecapa.npz" \
               "$FEATDIR/eres2netv2.npz" "$FEATDIR/eres2netv2-w24s4ep4.npz" \
    --feature-sets full compact --learners ksvr hgb --outdir "$OUTDIR"
[ $? -ne 0 ] && echo "SV SWEEP FAILED"

E=$((SECONDS - START_TIME)); echo "[sweeps done, $((E/3600))h $((E%3600/60))m into the job]"

##################################################################
# 3. Matched per-member comparison, then the k=16 pools
##################################################################
echo ""
echo "=================================================================="
echo "outlier-removed vs original training set, held-out dev"
echo "=================================================================="
python - "$OUTDIR" <<'PY'
import csv, glob, json, os, sys
import numpy as np, scipy.stats
sys.path.insert(0, ".")
from make_submission import DEEP_MEMBERS, WEAK_MEMBERS

out = sys.argv[1]
EV = "../baseline/data/vmc2026_track3_eval_phase_distro_v3_syn/sets/dev_with_labels.csv"
DEEP = "../unified/egs/ensemble_runs"
rows = list(csv.DictReader(open(EV)))
kidx = {(r["wav_a_path"], r["wav_b_path"]): i for i, r in enumerate(rows)}
sysid = np.array([r["system_id"] for r in rows]); n = len(rows)


def load(p, m):
    v = np.full(n, np.nan)
    for r in csv.DictReader(open(p)):
        i = kidx.get((r["wav_a_path"], r["wav_b_path"]))
        if i is not None:
            v[i] = float(r[f"pred_{m}"])
    return None if np.isnan(v).any() else v


def six(t, p):
    st, sp = {}, {}
    for s, a, b in zip(sysid, t, p):
        st.setdefault(s, []).append(a); sp.setdefault(s, []).append(b)
    x = np.array([np.mean(st[k]) for k in st]); y = np.array([np.mean(sp[k]) for k in st])
    return (np.mean((t - p) ** 2), scipy.stats.pearsonr(t, p).statistic,
            scipy.stats.spearmanr(t, p).statistic, np.mean((x - y) ** 2),
            scipy.stats.pearsonr(x, y).statistic, scipy.stats.spearmanr(x, y).statistic)


srcc = lambda a, b: scipy.stats.spearmanr(a, b).statistic
rng = np.random.default_rng(0)

for metric in ("spk_sim", "acc_sim"):
    t = np.array([float(r[metric]) for r in rows])
    print(f"\n### {metric}: the 16 frozen members, refitted")
    print(f"  {'member':<44}{'original':>10}{'no-outliers':>13}{'delta':>8}")
    print("  " + "-" * 75)
    old_v, new_v, missing = [], [], []
    for run in WEAK_MEMBERS[metric]:
        fo = f"egs/weak_nopca/preds/{run}__{metric}__dev.csv"
        fn = f"{out}/preds/{run}__{metric}__dev.csv"
        if not (os.path.exists(fo) and os.path.exists(fn)):
            missing.append(run); continue
        a, b = load(fo, metric), load(fn, metric)
        old_v.append(a); new_v.append(b)
        print(f"  {run:<44}{srcc(t,a):>10.3f}{srcc(t,b):>13.3f}{srcc(t,b)-srcc(t,a):>+8.3f}")
    if missing:
        print(f"  MISSING from one side: {missing}")
    if not new_v:
        continue
    d = [srcc(t, b) - srcc(t, a) for a, b in zip(old_v, new_v)]
    print(f"  mean delta over {len(d)} members: {np.mean(d):+.3f}")

    deep = [load(f"{DEEP}/{c}_{metric}/dev_{metric}_best.csv", metric)
            for c in DEEP_MEMBERS[metric]]
    pools = {
        "deep top-8 (unchanged)":            np.mean(deep, axis=0),
        "weak top-16, original":             np.mean(old_v, axis=0),
        "weak top-16, no-outliers":          np.mean(new_v, axis=0),
        "deep + weak top-16, original":      np.mean(deep + old_v, axis=0),
        "deep + weak top-16, no-outliers":   np.mean(deep + new_v, axis=0),
    }
    print(f"\n  {'pool':<34}{'uMSE':>8}{'uLCC':>8}{'uSRCC':>8}{'sMSE':>8}{'sLCC':>8}{'sSRCC':>8}")
    print("  " + "-" * 82)
    for name, v in pools.items():
        print(f"  {name:<34}" + "".join(f"{q:>8.3f}" for q in six(t, v)))
    a = pools["deep + weak top-16, original"]; b = pools["deep + weak top-16, no-outliers"]
    diff = [srcc(t[s], b[s]) - srcc(t[s], a[s]) for s in (rng.integers(0, n, n) for _ in range(2000))]
    lo, hi = np.percentile(diff, [2.5, 97.5])
    print(f"  no-outliers minus original, deep+weak top-16: "
          f"{np.mean(diff):+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
PY

echo ""
echo "Reference (original training set, held out): deep+weak top-16 reached"
echo "  spk_sim uSRCC 0.623   acc_sim uSRCC 0.603"
echo "A change inside +/-0.05 is inside this dev set's measurement floor; judge by the CI."

ELAPSED=$((SECONDS - START_TIME))
echo ""
echo "Job $SLURM_JOB_ID finished at $(date) after $((ELAPSED/3600))h $((ELAPSED%3600/60))m"
