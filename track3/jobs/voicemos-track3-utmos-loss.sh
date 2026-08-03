#!/usr/bin/env bash
#SBATCH --job-name=voicemos-track3-utmos-loss
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=david.guzman@mila.quebec

# UTMOS objective (Saeki et al., Interspeech 2022) on Track 3, for BOTH targets.
#
# Everything except the loss is held fixed at the baseline's configuration -- the same
# SpeechBrain ECAPA encoder, projection, interaction vector and range-clipped head -- so
# that any difference is attributable to the objective. The arms:
#
#   mse          plain MSE. What ../baseline trains on. The control.
#   clipped      clipped MSE alone (tau=0.25), isolating the dead zone.
#   contrastive  UTMOS contrastive alone (margin=0.1). Optimises rank, fixes no scale.
#   utmos        1.0 * clipped + 0.5 * contrastive -- UTMOS Eq. 1 with shipped weights.
#   utmos-g2     1.0 * clipped + 2.0 * contrastive -- see the note on gamma below.
#
#   sbatch track3/jobs/voicemos-track3-utmos-loss.sh
#
# Deliberately NOT using `set -e`: if one arm fails, the others should still run rather
# than wasting the whole allocation.

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

# torchaudio >= 2.9 dispatches load() to torchcodec, which dies with
# "libnppicc.so.12: cannot open shared object file" unless the NVIDIA libs that ship
# with torch are on the loader path. Affects ../baseline identically.
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH=$SITE_PACKAGES/nvidia/npp/lib:$LD_LIBRARY_PATH
python -c "
import torchaudio, glob
f = sorted(glob.glob('$SCRATCH/Repositories/VoiceMOS-Challenge-2026/track3/baseline/data/vmc2026_track3_train_phase_distro_v3_syn/wav/*.wav'))[0]
torchaudio.load(f)
print('torchaudio.load OK')" || { echo "ERROR: torchaudio cannot load wavs"; exit 1; }

echo "NVIDIA SMI:"
nvidia-smi
echo "HF_HOME: $HF_HOME"

REPO=/home/mila/g/guzmand/scratch/Repositories/VoiceMOS-Challenge-2026
cd "$REPO/track3/utmos-approach" || exit 1

NUM_WORKERS=${SLURM_CPUS_PER_TASK:-8}

##################################################################
# Configuration
##################################################################
DR=../baseline/data/vmc2026_track3_train_phase_distro_v3_syn
TRAIN_CSV=$DR/sets/train.csv
DEV_CSV=$DR/sets/dev.csv
DEV_LABELS=${DEV_LABELS:-../baseline/data/vmc2026_track3_eval_phase_distro_v3_syn/sets/dev_with_labels.csv}

# The contrastive term is computed over every ordered pair inside a micro-batch, so batch
# size feeds it quadratically: 16 -> 240 pairs, 32 -> 992, 48 -> 2256. Gradient accumulation
# does NOT substitute, since the loss is per micro-batch. Measured on an L40S (46 GB),
# ECAPA full fine-tuning with the utmos loss:
#     bs 16 ->  9.98 GiB, 0.158 s/step    bs 32 -> 19.69 GiB, 0.314 s/step
#     bs 48 -> 31.77 GiB, 0.514 s/step    bs 64 -> 42.31 GiB, 0.667 s/step
# Held at 16 so this is a clean A/B against the baseline's batch size; the loss is the only
# variable. Re-run with BATCH=32 afterwards to test whether more pairs help.
BATCH=${BATCH:-16}

# Backbone lr 1e-5 with head lr 1e-3. The baseline's 1e-3 everywhere is only survivable
# because its encoder is silently frozen (see ../utmos-approach/README.md).
ENCODER_LR=${ENCODER_LR:-1e-5}
HEAD_LR=${HEAD_LR:-1e-3}

# 4000 steps at batch 16 is ~23 epochs over the 2,800 unique pairs.
TRAIN_STEPS=${TRAIN_STEPS:-4000}
EVAL_STEPS=${EVAL_STEPS:-250}
SAVE_STEPS=${SAVE_STEPS:-1000}
BEST_METRIC=${BEST_METRIC:-srcc_sys}

METRICS=(spk_sim acc_sim)
ARMS=(mse clipped contrastive utmos utmos-g2)

echo "=================================================================="
echo "arms=${ARMS[*]}  metrics=${METRICS[*]}"
echo "batch=$BATCH (contrastive sees $((BATCH*(BATCH-1))) ordered pairs/step)"
echo "steps=$TRAIN_STEPS  encoder_lr=$ENCODER_LR  head_lr=$HEAD_LR"
echo "=================================================================="

mkdir -p egs
if [ ! -f "$DEV_LABELS" ]; then
    echo "ERROR: no labelled dev set at $DEV_LABELS"; exit 1
fi
python make_eval_gt.py --in "$DEV_LABELS" --out egs/dev.mean.csv \
    || { echo "ERROR: could not build dev ground truth"; exit 1; }
DEV_GT=egs/dev.mean.csv

##################################################################
# Sweep
##################################################################
FAILED=()
for METRIC in "${METRICS[@]}"; do
for ARM in "${ARMS[@]}"; do
    TAG="${METRIC}_${ARM}"
    OUT="egs/$TAG"
    echo ""
    echo "##################################################################"
    echo "# $TAG  ($(date))"
    echo "##################################################################"

    # utmos-g2 is the utmos loss with gamma raised. In the smoke test the combined loss at
    # the shipped gamma=0.5 tracked plain MSE closely while contrastive-alone pulled far
    # ahead on rank, which suggests the regression term dominates early on this dataset.
    case "$ARM" in
        utmos-g2) LOSS_ARGS=(--loss utmos --gamma 2.0) ;;
        *)        LOSS_ARGS=(--loss "$ARM") ;;
    esac

    echo "--- training ---"
    python finetune.py \
        --data-root "$DR" --train-csv "$TRAIN_CSV" \
        --target-metric "$METRIC" --outdir "$OUT" \
        "${LOSS_ARGS[@]}" \
        --batch-size "$BATCH" --train-steps "$TRAIN_STEPS" --save-steps "$SAVE_STEPS" \
        --lr "$HEAD_LR" --encoder-lr "$ENCODER_LR" \
        --dev-csv "$DEV_LABELS" --dev-data-root "$DR" \
        --eval-steps "$EVAL_STEPS" --best-metric "$BEST_METRIC" \
        --num-workers "$NUM_WORKERS"
    if [ $? -ne 0 ]; then echo "TRAINING FAILED for $TAG"; FAILED+=("$TAG:train"); continue; fi

    CKPT="$OUT/model_best_${METRIC}.pt"
    if [ ! -f "$CKPT" ]; then
        echo "NOTE: no model_best_${METRIC}.pt, falling back to the final-step checkpoint."
        CKPT="$OUT/finetuned_model_${METRIC}_final.pt"
    fi
    echo "Selected checkpoint: $CKPT"

    echo "--- inference on the official dev set ---"
    rm -f "$OUT/dev_${METRIC}.csv"
    python inference.py \
        --data-root "$DR" --csv-path "$DEV_CSV" \
        --checkpoint "$CKPT" --out "$OUT/dev_${METRIC}.csv"
    if [ $? -ne 0 ]; then
        echo "INFERENCE FAILED for $TAG"; FAILED+=("$TAG:dev")
    else
        echo "--- scoring against the official dev labels ---"
        python calculate_metrics.py \
            --prediction-csv "$OUT/dev_${METRIC}.csv" --ground-truth-csv "$DEV_GT"
    fi
done
done

##################################################################
# Summary
##################################################################
echo ""
echo "=================================================================="
echo "Best dev $BEST_METRIC per arm (step > 0):"
for METRIC in "${METRICS[@]}"; do
for ARM in "${ARMS[@]}"; do
    L="egs/${METRIC}_${ARM}/dev_log_${METRIC}.csv"
    if [ -f "$L" ]; then
        python - "$L" "$BEST_METRIC" "${METRIC}/${ARM}" <<'PY'
import csv, sys, math
path, metric, tag = sys.argv[1], sys.argv[2], sys.argv[3]
rows = [r for r in csv.DictReader(open(path)) if int(r["step"]) > 0]
vals = [(float(r[metric]), int(r["step"])) for r in rows if r[metric] and not math.isnan(float(r[metric]))]
mses = [float(r["mse_utt"]) for r in rows if r["mse_utt"] and not math.isnan(float(r["mse_utt"]))]
if vals:
    best, step = (min if metric.startswith("mse") else max)(vals)
    print(f"  {tag:24s} best {best:+.4f} @ step {step:<6d} final {vals[-1][0]:+.4f}   final mse_utt {mses[-1]:.3f}")
else:
    print(f"  {tag:24s} no evaluations recorded")
PY
    fi
done
done
echo ""
echo "Baseline 2 reference (dev): spk SYS-SRCC 0.860, acc SYS-SRCC 0.861"
echo "Full dev curves: egs/*/dev_log_*.csv"
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
