#!/usr/bin/env bash
#SBATCH --job-name=voicemos-track3-weak-ensemble-train
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=david.guzman@mila.quebec

# The weak-learner ensemble, end to end, fitted on TRAIN ONLY.
#
#   sbatch track3/jobs/weak/voicemos-track3-weak-ensemble-train.sh
#
# Self-contained: caches the frozen encoder features, fits the weak learners, scores them on
# held-out dev, and writes the submission CSVs. The companion job
# voicemos-track3-weak-ensemble-traindev.sh does the same with dev folded into the fitting set.
#
# WHY TRAIN ONLY. dev stays held out, so every number this job prints is an honest estimate.
# It is the run that produced the shipped figures: weak top-16 alone reaches dev UTT-SRCC 0.622
# (spk_sim) / 0.597 (acc_sim), and deep top-8 + weak top-16 reaches 0.623 / 0.603.
#
# THE GPU IS USED ONLY IN STAGE 1. Feature extraction is a no-grad forward pass; the fitting is
# scikit-learn on CPU. The card is held for the whole job, which is the price of not queueing
# twice. Stage 1 is skipped entirely when the cache is already populated.
#
# OUTPUT DIRECTORIES ARE NEW BY DEFAULT so nothing existing is touched. The shipped results
# live in egs/weak_nopca/; pass OUTDIR=egs/weak_nopca to extend that tree instead (train_weak.py
# is resumable and skips completed runs, so that is safe and idempotent).
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

# torchaudio >= 2.9 dispatches load() to torchcodec, which dies with
# "libnppicc.so.12: cannot open shared object file" unless the NVIDIA libs that ship with
# torch are on the loader path.
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
TRAIN_CSV=${TRAIN_CSV:-../baseline/data/vmc2026_track3_train_phase_distro_v3_syn/sets/train.csv}
FEATDIR=${FEATDIR:-egs/features}
OUTDIR=${OUTDIR:-egs/weak_ens_train}
SUBDIR=${SUBDIR:-egs/submission_final}
TAG=${TAG:-train_only}

SV_ENCODERS=${SV_ENCODERS:-"sv:ecapa-voxceleb sv:commonaccent-ecapa sv:eres2netv2 sv:eres2netv2-w24s4ep4"}
SSL_ENCODERS=${SSL_ENCODERS:-"ssl:WAVLM_BASE_PLUS ssl:WAVLM_LARGE ssl:WAV2VEC2_XLSR_300M"}
SSL_LAYERS=${SSL_LAYERS:-"4 8 -1"}

[ -f "$TRAIN_CSV" ] || { echo "ERROR: no training csv at $TRAIN_CSV"; exit 1; }

echo "=================================================================="
echo "weak-learner ensemble, fitted on TRAIN"
echo "  train  : $TRAIN_CSV"
echo "  feats  : $FEATDIR   (7 encoders, SSL layers $SSL_LAYERS, no PCA)"
echo "  models : $OUTDIR"
echo "  subs   : $SUBDIR/deep0-weak16_$TAG  and  deep8-weak16_$TAG"
echo "=================================================================="

##################################################################
# Stage 1 -- cache the frozen encoder features (GPU, ~10 min, skipped if present)
#
# One no-grad pass over the 4,160 unique wavs referenced by train/dev/test. The cache does not
# depend on the split, so both this job and the train+dev job share it. 24 kHz audio is
# resampled to 16 kHz inside extract_features.py -- skipping that does not error, it silently
# feeds audio at 1.5x speed.
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
#
# Exactly what the 16 frozen members need. ksvr/hgb are NOT run on the SSL files: at 4,098
# features they cost 10-20 min each and no frozen member uses that combination.
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

##################################################################
# Stage 3 -- held-out dev analysis
#
# Valid here because dev was NOT in the fitting set. The companion train+dev job skips this.
##################################################################
for T in spk_sim acc_sim; do
    echo ""; echo "### stage 3: held-out dev analysis, $T ($(date))"
    python analyze.py --target "$T" --weak-dir "$OUTDIR/preds" --show 20
done

##################################################################
# Stage 4 -- submissions
#
# Members are the FROZEN top-16 in make_submission.py, selected on held-out dev. deep0 is
# weak-only; deep8 additionally needs the deep pool's test predictions, which come from
# voicemos-track3-deep-test-inference.sh.
##################################################################
echo ""; echo "### stage 4: submissions ($(date))"
for T in spk_sim acc_sim; do
    python make_submission.py --target "$T" --deep-k 0 --weak-k 16 \
        --weak-dir "$OUTDIR/preds" --out "$SUBDIR/deep0-weak16_$TAG" \
        --note "weak top-16 only, fitted on $TRAIN_CSV, no PCA"
done

N_DEEP_TEST=$(ls ../unified/egs/ensemble_runs/*/test_*_best.csv 2>/dev/null | wc -l)
if [ "$N_DEEP_TEST" -ge 32 ]; then
    for T in spk_sim acc_sim; do
        python make_submission.py --target "$T" --deep-k 8 --weak-k 16 \
            --weak-dir "$OUTDIR/preds" --out "$SUBDIR/deep8-weak16_$TAG" \
            --note "deep top-8 (train) + weak top-16, weak fitted on $TRAIN_CSV, no PCA"
    done
else
    echo "NOTE: only $N_DEEP_TEST deep test predictions found (need 32); skipping the"
    echo "      deep8-weak16 submission. Run voicemos-track3-deep-test-inference.sh first."
fi

echo ""
echo "Reference, held out on dev: weak-only top-16 reached 0.622 (spk_sim) / 0.597 (acc_sim);"
echo "deep top-8 + weak top-16 reached 0.623 / 0.603. A change inside +/-0.05 is inside this"
echo "dev set's measurement floor."

ELAPSED=$((SECONDS - START_TIME))
echo ""
echo "Job $SLURM_JOB_ID finished at $(date) after $((ELAPSED/3600))h $((ELAPSED%3600/60))m"
