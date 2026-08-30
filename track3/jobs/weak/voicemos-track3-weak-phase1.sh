#!/usr/bin/env bash
#SBATCH --job-name=voicemos-track3-weak-phase1
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=david.guzman@mila.quebec

# Phase 1 of the UTMOS22-style weak-learner stack: frozen features -> classical regressors.
#
#   sbatch track3/jobs/weak/voicemos-track3-weak-phase1.sh
#
# SUPERSEDED FOR REPRODUCTION -- KEPT FOR THE RECORD. This is the EXPLORATORY sweep: all five
# learner families on all 13 feature files (260 models, 6 h) to build the candidate pool and
# rank it. It runs WITH PCA-128 on the SSL embeddings, because it predates the finding that
# PCA costs +0.05 to +0.29 per configuration, and its output tree egs/weak/ is the PCA'd pool
# that scored 0.609 / 0.577. Re-running it reproduces those superseded numbers.
#
# To rebuild the SHIPPED system (0.623 / 0.603) use instead:
#   voicemos-track3-weak-ensemble-train.sh       fitted on train, dev held out
#   voicemos-track3-weak-ensemble-traindev.sh    fitted on train+dev
# Those fit only the 84 models the 16 frozen members need, with PCA disabled, and they write
# the submission CSVs. Add `--pca-threshold 99999` below to make this job PCA-free too.
#
# WHAT THIS ANSWERS. The deep pool is stuck at a mean pairwise residual correlation of ~0.92,
# and ensemble gain on this dev set tracks that number almost linearly:
#
#   0.878 (heterogeneous 16)  ->  +0.062 over the mean member
#   0.921 (factorial 16)      ->  +0.047
#   0.957 (factorial top-8)   ->  +0.019
#
# Adding a fourth speaker-ID encoder (ecapa-voxceleb) did not move it, because all four share
# the same inductive bias. This job tests the alternative: keep the encoders frozen and change
# the FUNCTION CLASS on top of them, plus add masked-prediction SSL encoders whose bias is
# genuinely different. A smoke run with one SSL model and ridge alone already measured a
# cross-pool residual correlation of 0.717 and a +0.011 [+0.003, +0.019] gain, so the full
# sweep is worth its four hours.
#
# STAGES.
#   1. GPU  extract frozen embeddings for 4,160 unique wavs x 7 encoders  (~1 h)
#   2. CPU  five learner families x 13 feature files x 2 feature sets x 2 targets  (~3 h)
#   3. CPU  residual-correlation analysis and ensemble comparison         (seconds)
#
# NO LEAKAGE. Learners are fitted on train only, with hyperparameters chosen by GroupKFold on
# system_id -- dev holds two systems absent from train (sys003, sys015) and test holds four,
# so random folds would select hyperparameters that do not survive the real split. dev is
# touched only at scoring time. A learned stacker is phase 2 and needs out-of-fold deep
# predictions that do not exist yet.
#
# Deliberately NOT using `set -e`: a failing learner family must not kill the sweep.

START_TIME=$SECONDS
echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

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
echo "python: $(which python)"

# torchaudio >= 2.9 dispatches load() to torchcodec, which dies with
# "libnppicc.so.12: cannot open shared object file" unless the NVIDIA libs that ship
# with torch are on the loader path.
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH=$SITE_PACKAGES/nvidia/npp/lib:$LD_LIBRARY_PATH

REPO=${REPO:-/home/mila/g/guzmand/scratch/Repositories/VoiceMOS-Challenge-2026}
cd "$REPO/track3/weak" || exit 1

echo "NVIDIA SMI:"; nvidia-smi

##################################################################
# Configuration
##################################################################
# Four speaker/accent-ID encoders (192-d) and three masked-prediction SSL bundles.
# WAVLM_BASE_PLUS/LARGE are the strongest speaker-aware SSL models; XLSR_300M is multilingual,
# so accent variation is signal in its space rather than nuisance -- aimed at acc_sim, which
# sits furthest from its ceiling (0.55 against ~0.77).
SV_ENCODERS=${SV_ENCODERS:-"sv:ecapa-voxceleb sv:commonaccent-ecapa sv:eres2netv2 sv:eres2netv2-w24s4ep4"}
SSL_ENCODERS=${SSL_ENCODERS:-"ssl:WAVLM_BASE_PLUS ssl:WAVLM_LARGE ssl:WAV2VEC2_XLSR_300M"}

# One forward pass returns every layer, so extra layers are free. Lower layers carry speaker
# and channel, middle layers carry phonetic content; UTMOS22 used only the last.
SSL_LAYERS=${SSL_LAYERS:-"4 8 -1"}

FEATURE_SETS=${FEATURE_SETS:-"full compact"}
LEARNERS=${LEARNERS:-"ridge linsvr ksvr rf hgb"}
TARGETS=${TARGETS:-"spk_sim acc_sim"}
FEATDIR=${FEATDIR:-egs/features}
OUTDIR=${OUTDIR:-egs/weak}

echo "=================================================================="
echo "weak-learner phase 1"
echo "  sv encoders : $SV_ENCODERS"
echo "  ssl bundles : $SSL_ENCODERS   layers: $SSL_LAYERS"
echo "  feature sets: $FEATURE_SETS"
echo "  learners    : $LEARNERS"
echo "  targets     : $TARGETS"
echo "=================================================================="

##################################################################
# Stage 1 -- frozen feature extraction (GPU)
##################################################################
echo ""
echo "### stage 1: feature extraction ($(date))"
python extract_features.py \
    --encoders $SV_ENCODERS $SSL_ENCODERS \
    --ssl-layers $SSL_LAYERS \
    --outdir "$FEATDIR"
if [ $? -ne 0 ]; then echo "FEATURE EXTRACTION FAILED"; exit 1; fi

echo "--- cached feature files ---"
ls -la "$FEATDIR"
E=$((SECONDS - START_TIME)); echo "[extraction done, $((E/3600))h $((E%3600/60))m into the job]"

##################################################################
# Stage 2 -- weak learners (CPU)
#
# Resumable: the manifest records completed runs and they are skipped on a rerun, so a
# preemption costs only the run in flight. Pass --overwrite to start clean.
##################################################################
echo ""
echo "### stage 2: weak learners ($(date))"
python train_weak.py \
    --features "$FEATDIR/*.npz" \
    --targets $TARGETS \
    --feature-sets $FEATURE_SETS \
    --learners $LEARNERS \
    --outdir "$OUTDIR"
if [ $? -ne 0 ]; then echo "TRAINING FAILED"; fi

E=$((SECONDS - START_TIME)); echo "[fitting done, $((E/3600))h $((E%3600/60))m into the job]"

##################################################################
# Stage 3 -- the phase-1 verdict
##################################################################
for T in $TARGETS; do
    echo ""
    echo "### stage 3: analysis for $T ($(date))"
    python analyze.py --target "$T" --weak-dir "$OUTDIR/preds" --show 30
done

echo ""
echo "Reference on this dev set (dev held out of training throughout):"
echo "  spk_sim  deep top-8 0.579   deep top-16 0.584   heterogeneous top-16 0.607"
echo "  acc_sim  deep top-8 0.564   deep top-16 0.555   heterogeneous top-8  0.579"
echo "  a weak learner earns its place by LOWERING cross-pool residual correlation,"
echo "  not by scoring well alone -- expect solo scores well below the deep models."

ELAPSED=$((SECONDS - START_TIME))
echo ""
echo "Job $SLURM_JOB_ID finished at $(date) after $((ELAPSED/3600))h $((ELAPSED%3600/60))m"
