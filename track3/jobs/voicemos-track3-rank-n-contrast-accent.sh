#!/usr/bin/env bash
#SBATCH --job-name=voicemos-track3-rank-n-contrast-accent
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=david.guzman@mila.quebec

# Rank-N-Contrast for Track 3 ACCENT similarity (acc_sim).
#
# Sweeps two learning rates. For each: stage 1 (RNC representation learning) ->
# stage 2 (linear probe of every stage-1 checkpoint, selected on the labelled
# dev set) -> predictions on the official unlabelled sets/dev.csv, ready to
# upload to CodaBench.
#
#   sbatch track3/jobs/voicemos-track3-rank-n-contrast-accent.sh
#
# Stage 1 fine-tunes all 22.15M ECAPA parameters. The first sweep did not: the
# `--unfreeze-ecapa` flag crashed at construction (a batch-of-1 output-dim probe
# hitting BatchNorm in train mode), so RNC could only reshape the 49k projection
# and stage 1 lost just 0.07 nats over 8,750 steps. Batch, learning rates and
# schedule below are all resized for a genuinely trainable backbone.
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

# Labelled dev set, released after the first sweep ran. It gives all three
# stages a real selection signal for the first time: stage 1 monitors the
# feature/label rank correlation and saves encoder_best.pt, stage 2 selects the
# head on dev SYS-SRCC, and the probe loop below picks WHICH stage-1 checkpoint
# to keep instead of taking encoder_last.pt blind.
#
# --data-root stays $DR even though this CSV ships in the eval-phase distro:
# all 748 dev waveforms are present under the train-phase distro, while the
# eval-phase one is missing 160 of them (588/748). Same resolution the encoder
# jobs use.
DEV_LABELS=../baseline/data/vmc2026_track3_eval_phase_distro_v3_syn/sets/dev_with_labels.csv
VAL_NAME=$(basename "$DEV_LABELS" .csv)
SELECT_ON=sys_srcc
EVAL_STEPS=500

# Training ECAPA rather than just the projection costs roughly 5x the memory, so
# the frozen sweep's batch of 128 no longer fits. Measured on a 46 GiB L40S with
# the backbone trainable:
#
#   batch  crop   peak       s/step
#   64     --     36.6 GiB   0.80
#   64     6 s    22.3 GiB   0.58
#   96     6 s    31.6 GiB   0.84
#   128    6 s    39.9 GiB   1.04
#
# Without a crop, peak follows the LONGEST clip in the batch -- repetitive
# padding stretches every clip to it -- and clips run to 14.9 s against a 4.8 s
# mean, so an unlucky batch OOMs hours into the run. --max-audio-sec 6 bounds
# that (p95 duration is 6.9 s, so most clips are untouched and the random crop
# doubles as augmentation over only 2,800 unique pairs). RNC gains monotonically
# with in-batch positives (paper Table 6a), so take the largest batch that still
# leaves real headroom: 96 at 31.6 GiB. 128 fits at 39.9 GiB if you want more
# contrast and can accept 6 GiB of margin.
BATCH=96
CROP_SEC=6

# 2,800 unique pairs / 96 = 29.2 steps per epoch. 11700 steps ~= 400 epochs,
# matching the paper's schedule. Checkpoints every 2340 steps (~80 epochs): dev
# labels are now released, so stage 2 can be re-run against each one and picked
# on real dev SRCC -- stage 2 costs about a minute, so that sweep is cheap.
STEPS=11700
SAVE_STEPS=2340

# Sized for a 22.15M-parameter pretrained backbone. The frozen sweep used
# (1e-3, 1e-4), which was calibrated for a 49k projection; at that rate a
# contrastive objective would destroy the pretrained features.
LEARNING_RATES=(1e-4 1e-5)

echo "=================================================================="
echo "metric=$METRIC  batch=$BATCH  crop=${CROP_SEC}s  steps=$STEPS  lrs=${LEARNING_RATES[*]}"
echo "ECAPA is TRAINABLE in stage 1 (22.15M params)"
echo "selection: $DEV_LABELS on $SELECT_ON"
echo "=================================================================="

if [ ! -f "$DEV_LABELS" ]; then
    echo "ERROR: labelled dev set not found at $DEV_LABELS"; exit 1
fi

##################################################################
# Sweep
##################################################################
FAILED=()
for LR in "${LEARNING_RATES[@]}"; do
    # "ftlr" rather than "lr" so these do not overwrite the frozen-ECAPA sweep
    # already in egs/ (which shares the 1e-4 value), matching the naming used in
    # track3/encoders/egs.
    TAG="${METRIC}_ftlr${LR}"
    OUT="egs/$TAG"
    echo ""
    echo "##################################################################"
    echo "# $TAG  ($(date))"
    echo "##################################################################"

    echo "--- stage 1: RNC representation learning (ECAPA trainable) ---"
    python train_rnc.py \
        --data-root "$DR" --train-csv "$TRAIN_CSV" \
        --target-metric "$METRIC" --outdir "$OUT" \
        --batch-size "$BATCH" --max-audio-sec "$CROP_SEC" \
        --train-steps "$STEPS" --save-steps "$SAVE_STEPS" \
        --lr "$LR" --lr-schedule cosine \
        --val-csv "$DEV_LABELS" --eval-steps "$EVAL_STEPS" \
        --num-workers "$NUM_WORKERS"
    if [ $? -ne 0 ]; then echo "STAGE 1 FAILED for $TAG"; FAILED+=("$TAG:stage1"); continue; fi

    # Stage 2 costs about a minute on cached features, so probe EVERY stage-1
    # checkpoint rather than assuming the last one is the best representation --
    # a contrastive stage has no reason to peak exactly at the final step.
    # encoder_best.pt is stage 1's own pick, by feature/label rank correlation;
    # it competes here against the periodic checkpoints on real dev SYS-SRCC.
    #
    # No --max-audio-sec here or at inference: the crop is a stage-1 training
    # measure only, and features must be extracted from the same full-length
    # audio the eval set is scored on.
    echo "--- stage 2: probing stage-1 checkpoints against $VAL_NAME ---"
    BEST_SCORE=""; BEST_PROBE=""; BEST_NAME=""
    for CKPT in "$OUT"/encoder_step*.pt "$OUT"/encoder_best.pt "$OUT"/encoder_last.pt; do
        [ -f "$CKPT" ] || continue
        # STEPS divides evenly by SAVE_STEPS, so encoder_step$STEPS.pt and
        # encoder_last.pt are the same weights. Probe them once.
        if [ "$CKPT" = "$OUT/encoder_last.pt" ] && [ -f "$OUT/encoder_step${STEPS}.pt" ]; then
            continue
        fi
        NAME=$(basename "$CKPT" .pt)
        PROBE="$OUT/probe_$NAME"

        python train_head.py \
            --data-root "$DR" --train-csv "$TRAIN_CSV" \
            --target-metric "$METRIC" --outdir "$PROBE" \
            --encoder-ckpt "$CKPT" --freeze-encoder \
            --head linear --loss l1 \
            --val-csv "$DEV_LABELS" --select-on "$SELECT_ON" \
            --num-workers "$NUM_WORKERS" > "$PROBE.log" 2>&1
        if [ $? -ne 0 ]; then
            echo "    $NAME -> PROBE FAILED (see $PROBE.log)"
            FAILED+=("$TAG:probe_$NAME"); continue
        fi

        SCORE=$(python - "$PROBE/head_history.json" "$VAL_NAME" "$SELECT_ON" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1]))
vals = [r[sys.argv[2]][sys.argv[3]] for r in rows if sys.argv[2] in r]
vals = [v for v in vals if v == v]          # drop NaNs
print(f"{max(vals):.6f}" if vals else "nan")
PY
)
        echo "    $NAME -> dev $SELECT_ON $SCORE"
        if [ "$SCORE" != "nan" ] && { [ -z "$BEST_SCORE" ] || awk "BEGIN{exit !($SCORE > $BEST_SCORE)}"; }; then
            BEST_SCORE="$SCORE"; BEST_PROBE="$PROBE"; BEST_NAME="$NAME"
        fi
    done

    if [ -z "$BEST_PROBE" ]; then
        echo "STAGE 2 FAILED for $TAG: no probe produced a usable score"
        FAILED+=("$TAG:stage2"); continue
    fi
    echo "    selected $BEST_NAME (dev $SELECT_ON $BEST_SCORE)"

    # Prefer the head checkpoint selected on dev over the last-step one.
    FINAL_CKPT="$BEST_PROBE/model_best.pt"
    [ -f "$FINAL_CKPT" ] || FINAL_CKPT="$BEST_PROBE/model_last.pt"

    echo "--- inference on the official dev set ---"
    python inference.py \
        --data-root "$DR" --csv-path "$DEV_CSV" \
        --checkpoint "$FINAL_CKPT" --target-metric "$METRIC" \
        --out "$OUT/dev_${METRIC}.csv" \
        --num-workers "$NUM_WORKERS"
    if [ $? -ne 0 ]; then echo "INFERENCE FAILED for $TAG"; FAILED+=("$TAG:inference"); continue; fi

    # These numbers are SELECTION-CONTAMINATED: dev picked the stage-1
    # checkpoint and the head step, so they are optimistic. Reported to compare
    # runs against each other; the honest comparison is the CodaBench eval set.
    echo "--- dev metrics (optimistic -- dev was used for selection) ---"
    python calculate_metrics.py \
        --prediction-csv "$OUT/dev_${METRIC}.csv" \
        --ground-truth-csv "$DEV_LABELS"
done

##################################################################
# Summary
##################################################################
echo ""
echo "=================================================================="
echo "CodaBench submission CSVs:"
for LR in "${LEARNING_RATES[@]}"; do
    F="egs/${METRIC}_ftlr${LR}/dev_${METRIC}.csv"
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
