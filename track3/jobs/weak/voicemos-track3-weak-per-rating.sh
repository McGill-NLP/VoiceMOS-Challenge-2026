#!/usr/bin/env bash
#SBATCH --job-name=voicemos-track3-weak-per-rating
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=david.guzman@mila.quebec

# Does fitting on INDIVIDUAL LISTENER RATINGS beat fitting on per-pair means?
#
#   sbatch track3/jobs/weak/voicemos-track3-weak-per-rating.sh
#
# THE GPU IS REQUESTED BUT NOT USED. The frozen embeddings are already cached in
# weak/egs/features/ and do not depend on the split, so the work here is pure scikit-learn on
# the CPU. It nonetheless asks for --gres=gpu:1 on the `long` partition because the CPU-only
# queues on this cluster start slowly, and waiting minutes for a two-hour job is the worse
# trade. The request is deliberately UNTYPED (gpu:1, not gpu:l40s:1) so any free card will do;
# constraining the type would reintroduce the queueing delay for no benefit.
#
# If the cluster is busy and you would rather not hold a card, `--partition=long-cpu` with the
# --gres line removed runs identically.
#
# THE QUESTION. train.csv holds 13,687 rating rows over 2,800 unique pairs, and everything so
# far -- weak learners and deep models alike -- has trained on the per-pair MEAN, discarding
# the 4.9x row multiplier. This job fits the individual ratings instead.
#
# WHAT TO EXPECT, so the result can be interpreted rather than just read. The feature vector
# is identical across a pair's ~5 rows, since it depends only on the two waveforms. For a
# squared-error learner,
#
#     sum_j (x.b - y_j)^2 = k (x.b - ybar)^2 + const
#
# so per-rating ridge is EXACTLY weighted per-pair ridge with weight k, and the counts are
# nearly uniform (2,488 pairs with 5 ratings, 311 with 4, 1 with 3). Ridge is therefore a
# CONTROL: it should land within about +/-0.01, and a larger move would mean something is
# wrong. `hgb` is the real test -- duplicated rows change split counts and leaf statistics, so
# a tree learner can genuinely fit a different function.
#
# NOT TESTED HERE, and worth stating as a gap: `linsvr` (80 s/fit at 13,687x4,098 -> ~20 min
# per run, ~17 h for the sweep) and `ksvr` (needs a 13,687^2 kernel, infeasible). Both use an
# epsilon-insensitive loss, which is exactly where duplication is NOT equivalent to a weighted
# mean -- so the SVR family remains untested. `rf` is skipped for cost at this row count.
#
# TRAIN ONLY, so dev stays genuinely held out and the numbers are directly comparable to the
# per-pair results (deep+weak reached 0.620 at k=8 and 0.623 at k=16 on spk_sim).
#
# Resumable: the manifest skips completed runs. Three runs from an interactive start are
# already present and will be skipped.
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
echo "python: $(which python)"

# Keep BLAS from oversubscribing: sklearn already parallelises across the grid, and nested
# threading on 16 cores makes the ridge solves slower, not faster.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS

REPO=${REPO:-/home/mila/g/guzmand/scratch/Repositories/VoiceMOS-Challenge-2026}
cd "$REPO/track3/weak" || exit 1

FEATDIR=${FEATDIR:-egs/features}
OUTDIR=${OUTDIR:-egs/weak_perrating}
LEARNERS=${LEARNERS:-"ridge hgb"}
FEATURE_SETS=${FEATURE_SETS:-"full compact"}
TARGETS=${TARGETS:-"spk_sim acc_sim"}

if [ ! -d "$FEATDIR" ] || [ -z "$(ls -A "$FEATDIR" 2>/dev/null)" ]; then
    echo "ERROR: no cached features in $FEATDIR -- run the phase-1 job first"; exit 1
fi

N_FEAT=$(ls "$FEATDIR"/*.npz | wc -l)
echo "=================================================================="
echo "per-rating weak learners (13,687 rating rows, not 2,800 pair means)"
echo "  features : $N_FEAT files in $FEATDIR   (no PCA)"
echo "  learners : $LEARNERS"
echo "  sets     : $FEATURE_SETS      targets: $TARGETS"
echo "  outdir   : $OUTDIR            train only, dev held out"
echo "=================================================================="

##################################################################
# The sweep
##################################################################
python train_weak.py \
    --per-rating \
    --pca-threshold 99999 \
    --features "$FEATDIR/*.npz" \
    --feature-sets $FEATURE_SETS \
    --learners $LEARNERS \
    --targets $TARGETS \
    --outdir "$OUTDIR"
RC=$?
E=$((SECONDS - START_TIME))
echo "[sweep exited $RC, $((E/3600))h $((E%3600/60))m into the job]"

##################################################################
# Matched comparison against the per-pair runs
#
# Only configurations present in BOTH sweeps are compared, so the per-rating vs per-pair
# difference is the only thing varying.
##################################################################
echo ""
echo "=================================================================="
echo "per-rating vs per-pair, matched configurations, held-out dev"
echo "=================================================================="
python - "$OUTDIR" <<'PY'
import json, os, sys
from collections import defaultdict

out = sys.argv[1]
def load(path):
    if not os.path.exists(path):
        return {}
    d = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("dev_in_train"):
                continue
            d[(r["encoder"], r["feature_set"], r["learner"], r["metric"])] = r["dev_srcc"]
    return d

new = load(f"{out}/stage1_manifest.jsonl")
old = {}
for m in ("egs/weak_nopca/stage1_manifest.jsonl", "egs/weak/stage1_manifest.jsonl"):
    for k, v in load(m).items():
        old.setdefault(k, v)          # prefer the no-PCA value where both exist

common = sorted(set(new) & set(old))
if not common:
    print("  no matched configurations yet"); raise SystemExit

for metric in ("spk_sim", "acc_sim"):
    rows = [k for k in common if k[3] == metric]
    if not rows:
        continue
    print(f"\n{metric}")
    print(f"  {'encoder':<26}{'fset':<9}{'learner':<8}{'per-pair':>10}{'per-rating':>12}{'delta':>8}")
    print("  " + "-" * 73)
    for k in sorted(rows, key=lambda k: -new[k]):
        print(f"  {k[0]:<26}{k[1]:<9}{k[2]:<8}{old[k]:>10.3f}{new[k]:>12.3f}{new[k]-old[k]:>+8.3f}")
    by = defaultdict(list)
    for k in rows:
        by[k[2]].append(new[k] - old[k])
    print("  mean delta by learner: " +
          "  ".join(f"{L}={sum(v)/len(v):+.3f} (n={len(v)})" for L, v in sorted(by.items())))
PY

echo ""
echo "Interpretation: ridge is the CONTROL and should be within about +/-0.01 -- it is"
echo "algebraically equivalent to weighting pairs by rater count. hgb is the real test."
echo "A per-rating variant earns a place in the pool by adding a DECORRELATED member, not"
echo "necessarily by scoring higher; run analyze.py against $OUTDIR/preds to check that."

ELAPSED=$((SECONDS - START_TIME))
echo ""
echo "Job $SLURM_JOB_ID finished at $(date) after $((ELAPSED/3600))h $((ELAPSED%3600/60))m"
