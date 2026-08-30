#!/usr/bin/env bash
#SBATCH --job-name=voicemos-track3-weak-ensemble-traindev
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=david.guzman@mila.quebec

# The weak-learner ensemble, end to end, fitted on TRAIN + DEV.
#
#   sbatch track3/jobs/weak/voicemos-track3-weak-ensemble-traindev.sh
#
# Companion to voicemos-track3-weak-ensemble-train.sh. Same encoders, same features, same
# learners, same frozen member list -- the only difference is that dev is folded into the
# fitting set (2,800 -> 3,400 pairs, 21 -> 23 systems), which the challenge permits.
#
# THERE IS NO HELD-OUT SET LEFT, and that has two consequences this job enforces:
#
#   1. NO ANALYSIS STAGE. The train-only job runs analyze.py; this one cannot, because every
#      dev score it could compute would be an in-sample fit. train_weak.py detects the overlap
#      from the pair keys and stamps `dev_in_train: true` into every manifest record, and
#      make_submission.py names the dev CSVs `*_IN-SAMPLE.csv` so they cannot be mistaken for
#      estimates. For scale: the weak-only k=16 pool reads 0.875 in-sample against a true
#      held-out 0.622.
#
#   2. MEMBERS ARE NOT RE-SELECTED. The top-16 lists frozen in make_submission.py were chosen
#      on held-out dev by the train-only run and are reused verbatim. Re-ranking on in-sample
#      scores would pick whichever model memorised hardest -- an RBF-SVR member jumps from
#      0.534 to 0.923 that way. This is why make_submission.py exists instead of stack.py.
#
# WHAT TO EXPECT. The honest estimate for this system is the train-only 0.622 / 0.597
# (weak-only) or 0.623 / 0.603 (with the deep half), plus an unmeasurable gain from 21% more
# data. Test predictions from the two variants correlate at 0.997-0.998, so the gain is small.
#
# THE GPU IS USED ONLY IN STAGE 1, and stage 1 is skipped when the cache is already populated
# -- which it will be if the train-only job ran first, since the feature cache covers all
# 4,160 wavs (train, dev and test) and does not depend on the split.
#
# Deliberately NOT using `set -e`: one failing learner family must not kill the sweep.

START_TIME=$SECONDS
echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"
echo "SLURM_NODELIST: $SLURM_NODELIST   cpus: ${SLURM_CPUS_PER_TASK:-?}"

##################################################################
# Environment
##################################################################
module load miniconda/3
module load gcc/9.3.0
module load cuda/12.3.2

export HF_HOME=$SCRATCH/huggingface
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false

conda activate VoiceMOS
if [ "$CONDA_DEFAULT_ENV" != "VoiceMOS" ]; then
    echo "ERROR: conda env is '${CONDA_DEFAULT_ENV:-none}', expected VoiceMOS"; exit 1
fi
python -c "import torch, torchaudio, sklearn, speechbrain" \
    || { echo "ERROR: torch/torchaudio/sklearn/speechbrain not importable"; exit 1; }

SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH=$SITE_PACKAGES/nvidia/npp/lib:$LD_LIBRARY_PATH

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS

REPO=${REPO:-/home/mila/g/guzmand/scratch/Repositories/VoiceMOS-Challenge-2026}
cd "$REPO/track3/weak" || exit 1

##################################################################
# Configuration
##################################################################
TRAIN_CSV=${TRAIN_CSV:-../baseline/data/vmc2026_track3_train_phase_distro_v3_syn/sets/train_plus_dev.csv}
FEATDIR=${FEATDIR:-egs/features}
OUTDIR=${OUTDIR:-egs/weak_ens_traindev}
SUBDIR=${SUBDIR:-egs/submission_final}
TAG=${TAG:-train_plus_dev}

SV_ENCODERS=${SV_ENCODERS:-"sv:ecapa-voxceleb sv:commonaccent-ecapa sv:eres2netv2 sv:eres2netv2-w24s4ep4"}
SSL_ENCODERS=${SSL_ENCODERS:-"ssl:WAVLM_BASE_PLUS ssl:WAVLM_LARGE ssl:WAV2VEC2_XLSR_300M"}
SSL_LAYERS=${SSL_LAYERS:-"4 8 -1"}

# sets/train_plus_dev.csv concatenates train.csv and dev_with_labels.csv, with an empty
# listener_id on the dev rows. Build it if it is missing; it needs no new audio, and every dev
# wav already resolves under the training distribution's root.
if [ ! -f "$TRAIN_CSV" ]; then
    echo "NOTE: $TRAIN_CSV missing, building it"
    python - "$TRAIN_CSV" <<'PY'
import csv, os, sys
out = sys.argv[1]
TR = "../baseline/data/vmc2026_track3_train_phase_distro_v3_syn"
EV = "../baseline/data/vmc2026_track3_eval_phase_distro_v3_syn"
HDR = ["system_id", "utterance_id", "listener_id", "wav_a_path", "wav_b_path",
       "spk_sim", "acc_sim"]
n_tr = n_dev = 0
with open(out, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=HDR); w.writeheader()
    for r in csv.DictReader(open(f"{TR}/sets/train.csv")):
        w.writerow({k: r.get(k, "") for k in HDR}); n_tr += 1
    for r in csv.DictReader(open(f"{EV}/sets/dev_with_labels.csv")):
        row = {k: r.get(k, "") for k in HDR}; row["listener_id"] = ""
        w.writerow(row); n_dev += 1
print(f"wrote {out}: {n_tr} train rows + {n_dev} dev rows")
PY
    [ -f "$TRAIN_CSV" ] || { echo "ERROR: could not build $TRAIN_CSV"; exit 1; }
fi

echo "=================================================================="
echo "weak-learner ensemble, fitted on TRAIN + DEV"
echo "  train  : $TRAIN_CSV"
echo "  feats  : $FEATDIR   (7 encoders, SSL layers $SSL_LAYERS, no PCA)"
echo "  models : $OUTDIR"
echo "  subs   : $SUBDIR/deep0-weak16_$TAG  and  deep8-weak16_$TAG"
echo "  NOTE   : dev is IN the fitting set -- no held-out estimate is possible"
echo "=================================================================="

##################################################################
# Stage 1 -- cache the frozen encoder features (GPU, ~10 min, skipped if present)
##################################################################
echo ""; echo "### stage 1: feature cache ($(date))"
python extract_features.py \
    --encoders $SV_ENCODERS $SSL_ENCODERS \
    --ssl-layers $SSL_LAYERS \
    --outdir "$FEATDIR"
if [ $? -ne 0 ]; then echo "FEATURE EXTRACTION FAILED"; exit 1; fi
echo "--- cache ---"; ls -la "$FEATDIR"
E=$((SECONDS - START_TIME)); echo "[features ready, $((E/3600))h $((E%3600/60))m in]"

##################################################################
# Stage 2 -- fit the weak learners (CPU, ~30 min)
##################################################################
echo ""; echo "### stage 2a: ridge, all feature files ($(date))"
python train_weak.py --train-csv "$TRAIN_CSV" --pca-threshold 99999 \
    --features "$FEATDIR/*.npz" --feature-sets full compact \
    --learners ridge --outdir "$OUTDIR"
[ $? -ne 0 ] && echo "RIDGE SWEEP FAILED"

echo ""; echo "### stage 2b: ksvr + hgb, speaker/accent-ID encoders ($(date))"
python train_weak.py --train-csv "$TRAIN_CSV" --pca-threshold 99999 \
    --features "$FEATDIR/ecapa-voxceleb.npz" "$FEATDIR/commonaccent-ecapa.npz" \
               "$FEATDIR/eres2netv2.npz" "$FEATDIR/eres2netv2-w24s4ep4.npz" \
    --feature-sets full compact --learners ksvr hgb --outdir "$OUTDIR"
[ $? -ne 0 ] && echo "SV SWEEP FAILED"

E=$((SECONDS - START_TIME)); echo "[fitting done, $((E/3600))h $((E%3600/60))m in]"

# Fail loudly if the overlap detection did not fire -- it is what protects every downstream
# number from being read as a held-out estimate.
if ! grep -q '"dev_in_train": true' "$OUTDIR/stage1_manifest.jsonl" 2>/dev/null; then
    echo "WARNING: no record in $OUTDIR/stage1_manifest.jsonl is flagged dev_in_train."
    echo "         Expected dev to be inside $TRAIN_CSV. Check the fitting set."
fi

##################################################################
# Stage 3 -- submissions
#
# No analysis stage: dev is in the fitting set, so there is nothing honest to report. The dev
# CSVs written below are named *_IN-SAMPLE.csv for the same reason.
##################################################################
echo ""; echo "### stage 3: submissions ($(date))"
for T in spk_sim acc_sim; do
    python make_submission.py --target "$T" --deep-k 0 --weak-k 16 \
        --weak-dir "$OUTDIR/preds" --out "$SUBDIR/deep0-weak16_$TAG" \
        --note "weak top-16 only, fitted on train+dev; members frozen from the held-out run"
done

N_DEEP_TEST=$(ls ../unified/egs/ensemble_runs/*/test_*_best.csv 2>/dev/null | wc -l)
if [ "$N_DEEP_TEST" -ge 32 ]; then
    for T in spk_sim acc_sim; do
        python make_submission.py --target "$T" --deep-k 8 --weak-k 16 \
            --weak-dir "$OUTDIR/preds" --out "$SUBDIR/deep8-weak16_$TAG" \
            --note "deep top-8 (train) + weak top-16 (train+dev); members frozen from the held-out run"
    done
else
    echo "NOTE: only $N_DEEP_TEST deep test predictions found (need 32); skipping the"
    echo "      deep8-weak16 submission. Run voicemos-track3-deep-test-inference.sh first."
fi

echo ""
echo "The dev CSVs written above are IN-SAMPLE and estimate nothing. The honest figure for"
echo "this system is the train-only run's 0.622 / 0.597 (weak-only) or 0.623 / 0.603 (with the"
echo "deep half), plus an unmeasurable gain from 21% more data."

ELAPSED=$((SECONDS - START_TIME))
echo ""
echo "Job $SLURM_JOB_ID finished at $(date) after $((ELAPSED/3600))h $((ELAPSED%3600/60))m"
