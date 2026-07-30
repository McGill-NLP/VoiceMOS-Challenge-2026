#!/usr/bin/env bash
#SBATCH --job-name=voicemos-track3-rank-n-contrast-accent
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=david.guzman@mila.quebec

# Rank-N-Contrast for Track 3 ACCENT similarity (acc_sim).
#
# Sweeps two learning rates. For each: stage 1 (RNC representation learning) ->
# stage 2 (linear probe on the frozen encoder) -> predictions on the official
# unlabelled sets/dev.csv, ready to upload to CodaBench.
#
#   sbatch track3/jobs/voicemos-track3-rank-n-contrast-accent.sh
#
# Deliberately NOT using `set -e`: if one learning rate fails, the other should
# still run rather than wasting the whole allocation.

START_TIME=$SECONDS
echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

##################################################################
# Activate the environment by loading Python and required packages
##################################################################
module load miniconda/3
module load gcc/9.3.0
module load cuda/12.3.2

export HF_HOME=$SCRATCH/huggingface
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

conda activate VoiceMOS

# Fail fast rather than burning the allocation on the wrong interpreter.
if [ "$CONDA_DEFAULT_ENV" != "VoiceMOS" ]; then
    echo "ERROR: conda env is '${CONDA_DEFAULT_ENV:-none}', expected VoiceMOS"; exit 1
fi
python -c "import torch, speechbrain" || { echo "ERROR: torch/speechbrain not importable"; exit 1; }
echo "python: $(which python)"

echo "NVIDIA SMI:"
nvidia-smi
echo "HF_HOME: $HF_HOME"

REPO=/home/mila/g/guzmand/scratch/Repositories/VoiceMOS-Challenge-2026
cd "$REPO/track3/rank-n-contrast" || exit 1

NUM_WORKERS=${SLURM_CPUS_PER_TASK:-8}

##################################################################
# Configuration
##################################################################
METRIC=acc_sim
DR=../baseline/data/vmc2026_track3_train_phase_distro_v3_syn
TRAIN_CSV=$DR/sets/train.csv
DEV_CSV=$DR/sets/dev.csv

# Batch size 128 is the practical ceiling. At 256, ECAPA's attentive-pooling
# F.pad raises "input tensor must fit into 32-bit index math" -- not an OOM, a
# hard indexing limit, hit because repetitive padding stretches every clip in
# the batch to the longest one. Measured at 128: 19.85 GiB peak, 0.56 s/step.
# RNC gains monotonically with in-batch positives (paper Table 6a), so 128 it is.
BATCH=128

# 2,800 unique pairs / 128 = 21.9 steps per epoch. 8750 steps ~= 400 epochs,
# matching the paper's schedule. Checkpoints every 1750 steps (~100 epochs) so
# stage 2 can be re-run against an earlier encoder once dev labels arrive --
# stage 2 costs about two minutes, so that re-run is cheap.
STEPS=8750
SAVE_STEPS=1750

LEARNING_RATES=(1e-3 1e-4)

echo "=================================================================="
echo "metric=$METRIC  batch=$BATCH  steps=$STEPS  lrs=${LEARNING_RATES[*]}"
echo "=================================================================="

##################################################################
# Sweep
##################################################################
FAILED=()
for LR in "${LEARNING_RATES[@]}"; do
    TAG="${METRIC}_lr${LR}"
    OUT="egs/$TAG"
    echo ""
    echo "##################################################################"
    echo "# $TAG  ($(date))"
    echo "##################################################################"

    echo "--- stage 1: RNC representation learning ---"
    python train_rnc.py \
        --data-root "$DR" --train-csv "$TRAIN_CSV" \
        --target-metric "$METRIC" --outdir "$OUT" \
        --batch-size "$BATCH" --train-steps "$STEPS" --save-steps "$SAVE_STEPS" \
        --lr "$LR" --lr-schedule cosine \
        --num-workers "$NUM_WORKERS"
    if [ $? -ne 0 ]; then echo "STAGE 1 FAILED for $TAG"; FAILED+=("$TAG:stage1"); continue; fi

    echo "--- stage 2: linear probe on frozen encoder ---"
    python train_head.py \
        --data-root "$DR" --train-csv "$TRAIN_CSV" \
        --target-metric "$METRIC" --outdir "$OUT/head" \
        --encoder-ckpt "$OUT/encoder_last.pt" --freeze-encoder \
        --head linear --loss l1 \
        --num-workers "$NUM_WORKERS"
    if [ $? -ne 0 ]; then echo "STAGE 2 FAILED for $TAG"; FAILED+=("$TAG:stage2"); continue; fi

    echo "--- inference on the official dev set ---"
    python inference.py \
        --data-root "$DR" --csv-path "$DEV_CSV" \
        --checkpoint "$OUT/head/model_last.pt" --target-metric "$METRIC" \
        --out "$OUT/head/dev_${METRIC}.csv" \
        --num-workers "$NUM_WORKERS"
    if [ $? -ne 0 ]; then echo "INFERENCE FAILED for $TAG"; FAILED+=("$TAG:inference"); fi
done

##################################################################
# Summary
##################################################################
echo ""
echo "=================================================================="
echo "CodaBench submission CSVs:"
for LR in "${LEARNING_RATES[@]}"; do
    F="egs/${METRIC}_lr${LR}/head/dev_${METRIC}.csv"
    if [ -f "$F" ]; then
        echo "  OK      $(pwd)/$F  ($(($(wc -l < "$F") - 1)) rows)"
    else
        echo "  MISSING $F"
    fi
done
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "Failures: ${FAILED[*]}"
fi
echo "=================================================================="

ELAPSED=$(( SECONDS - START_TIME ))
HOURS=$(( ELAPSED / 3600 ))
MINUTES=$(( (ELAPSED % 3600) / 60 ))
SECS=$(( ELAPSED % 60 ))
echo "Job $SLURM_JOB_ID finished on $(hostname) at $(date)"
echo "Total duration: ${HOURS}h ${MINUTES}m ${SECS}s"
