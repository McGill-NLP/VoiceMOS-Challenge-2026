#!/usr/bin/env bash
#SBATCH --job-name=voicemos-track3-encoders-eres2netv2
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=david.guzman@mila.quebec

# Baseline recipe with the ERes2NetV2 encoder, for BOTH Track 3 targets.
#
# For each of spk_sim and acc_sim, three configurations are trained:
#
#   frozen     encoder frozen, head only -- this is what the official baseline
#              ACTUALLY does (see below), so it is the like-for-like control
#   ftlr1e-5   real fine-tuning, backbone lr 1e-5, head lr 1e-3
#   ftlr1e-4   real fine-tuning, backbone lr 1e-4, head lr 1e-3
#
# Training uses the complete official sets/train.csv. The labelled dev set is scored
# throughout training (every FT_EVAL_STEPS / FROZEN_EVAL_STEPS optimizer steps), so progress
# is visible in the log as it happens rather than only at the end. Each run writes:
#
#   dev_log_<metric>.csv       step, train_mse, and UTT/SYS MSE-LCC-SRCC per evaluation
#   model_best_<metric>.pt     checkpoint with the best dev BEST_METRIC seen
#   dev_<metric>.csv           predictions from that checkpoint, in submission format
#
#   sbatch track3/jobs/voicemos-track3-encoders-eres2netv2.sh
#
# For a held-out read before the dev labels land, train on the local 75% split instead
# and evaluate on dev-ID / dev-OOD:
#
#   sbatch --export=ALL,USE_LOCAL_SPLITS=1 track3/jobs/voicemos-track3-encoders-eres2netv2.sh
#
# Deliberately NOT using `set -e`: if one configuration fails, the others should
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
cd "$REPO/track3/encoders" || exit 1

NUM_WORKERS=${SLURM_CPUS_PER_TASK:-8}

##################################################################
# Configuration
##################################################################
ENCODER=eres2netv2

DR=../baseline/data/vmc2026_track3_train_phase_distro_v3_syn
DEV_CSV=$DR/sets/dev.csv

# Train on the complete official train.csv. Model selection happens against the official
# dev labels once they are released (see DEV_LABELS below), so the local 75% split is not
# needed -- but it stays available for a quick held-out read before then:
#   sbatch --export=ALL,USE_LOCAL_SPLITS=1 ...
USE_LOCAL_SPLITS=${USE_LOCAL_SPLITS:-0}
if [ "$USE_LOCAL_SPLITS" = "1" ]; then
    TRAIN_CSV=../baseline/data/train.csv
    EVAL_SETS=(dev-ID dev-OOD)
    RUN_TAG=split
    echo "USE_LOCAL_SPLITS=1: training on the 75% split, evaluating on dev-ID / dev-OOD."
else
    TRAIN_CSV=$DR/sets/train.csv
    EVAL_SETS=()
    RUN_TAG=full
fi

# Labelled dev set, from the evaluation-phase distribution. Its wav paths resolve against
# the TRAINING distro: the eval distro is missing all 600 sys019 reference wavs (they ship
# separately with VCTK), while the train distro has every dev wav. Hence --dev-data-root=$DR.
DEV_LABELS=${DEV_LABELS:-../baseline/data/vmc2026_track3_eval_phase_distro_v3_syn/sets/dev_with_labels.csv}

# The dev set is scored every EVAL_STEPS optimizer steps during training, so the run can be
# watched rather than only judged at the end. Each evaluation is 600 pairs. Roughly 40
# points across a frozen run and 16 across a fine-tuning run.
FROZEN_EVAL_STEPS=${FROZEN_EVAL_STEPS:-500}
FT_EVAL_STEPS=${FT_EVAL_STEPS:-250}

# Dev metric that decides which checkpoint is kept as model_best_<metric>.pt.
# System-level SRCC is the headline number for the challenge.
BEST_METRIC=${BEST_METRIC:-srcc_sys}

# ERes2NetV2 is a 2D CNN that runs stage 1 at full temporal resolution, so activation
# memory scales with batch x frames. Repetitive padding stretches every clip in a batch
# to the longest one (clips run 2.5-9.0 s), which makes this much heavier than ECAPA.
# Measured on an L40S (46 GB), full fine-tuning:
#     bs 4  -> 13.97 GiB   bs 8  -> 30.70 GiB   bs 12 -> 42.29 GiB   bs 16 -> OOM
# bs 4 x 4 accumulation and bs 8 x 2 have identical throughput (0.588 s per optimizer
# step either way), so take the one with the memory headroom. Effective batch stays 16,
# matching the baseline.
BATCH=4
ACCUM=4

# Frozen runs store no encoder activations: 3.85 GiB at bs 16, 0.245 s/step.
FROZEN_BATCH=16
FROZEN_ACCUM=1

# The baseline's 20000 steps x bs16 is ~114 epochs over the 2,800 unique pairs. That is
# only survivable because its encoder is frozen. For real fine-tuning of a 17.9M-param
# backbone on 2,800 pairs, 4000 steps at effective batch 16 (~23 epochs) is already
# generous. The frozen control keeps 20000 for exact parity with the published baseline.
# Overridable so the script can be smoke-tested cheaply:
#   FROZEN_STEPS=2 FT_STEPS=2 sbatch ...
FROZEN_STEPS=${FROZEN_STEPS:-20000}
FT_STEPS=${FT_STEPS:-4000}
SAVE_STEPS=${SAVE_STEPS:-1000}

HEAD_LR=1e-3
METRICS=(spk_sim acc_sim)
CONFIGS=(frozen ftlr1e-5 ftlr1e-4)

echo "=================================================================="
echo "encoder=$ENCODER  metrics=${METRICS[*]}  configs=${CONFIGS[*]}"
echo "train_csv=$TRAIN_CSV  ($RUN_TAG)"
echo "full-ft: batch=$BATCH x accum=$ACCUM, steps=$FT_STEPS"
echo "frozen : batch=$FROZEN_BATCH, steps=$FROZEN_STEPS"
echo "=================================================================="

##################################################################
# Per-pair ground truth
#
# The rating CSVs are listener-wise, so a pair appears once per listener. Feeding them
# straight to calculate_metrics.py would silently keep whichever listener sorts last.
# Averaging first gives the quantity the challenge actually scores, and cuts the inference
# work by not re-predicting the same pair once per listener.
##################################################################
mkdir -p egs
for SET in "${EVAL_SETS[@]}"; do
    python make_eval_gt.py --in "../baseline/data/$SET.csv" --out "egs/$SET.mean.csv" \
        || { echo "ERROR: could not build ground truth for $SET"; exit 1; }
done

# dev_with_labels.csv is already one row per pair, but run it through make_eval_gt.py
# anyway: the pass-through is a no-op and it keeps working if a listener-wise version is
# ever released.
DEV_GT=""
if [ -f "$DEV_LABELS" ]; then
    if python make_eval_gt.py --in "$DEV_LABELS" --out egs/dev.mean.csv; then
        DEV_GT=egs/dev.mean.csv
        echo "Dev labels: $DEV_LABELS"
        echo "  -> scored every EVAL_STEPS during training, and again after it"
    else
        echo "WARNING: $DEV_LABELS exists but could not be parsed; dev will not be scored."
    fi
else
    echo "ERROR: no labelled dev set at $DEV_LABELS"
    echo "Training would run blind. Fix the path or pass --export=ALL,DEV_LABELS=..."
    exit 1
fi

##################################################################
# Sweep
##################################################################
FAILED=()
for METRIC in "${METRICS[@]}"; do
for CONFIG in "${CONFIGS[@]}"; do
    TAG="${ENCODER}_${METRIC}_${CONFIG}_${RUN_TAG}"
    OUT="egs/$TAG"
    echo ""
    echo "##################################################################"
    echo "# $TAG  ($(date))"
    echo "##################################################################"

    case "$CONFIG" in
        frozen)
            TRAIN_ARGS=(--freeze-encoder --lr "$HEAD_LR"
                        --batch-size "$FROZEN_BATCH" --accumulate-steps "$FROZEN_ACCUM"
                        --train-steps "$FROZEN_STEPS" --eval-steps "$FROZEN_EVAL_STEPS")
            ;;
        ftlr1e-5)
            TRAIN_ARGS=(--encoder-lr 1e-5 --lr "$HEAD_LR"
                        --batch-size "$BATCH" --accumulate-steps "$ACCUM"
                        --train-steps "$FT_STEPS" --eval-steps "$FT_EVAL_STEPS")
            ;;
        ftlr1e-4)
            TRAIN_ARGS=(--encoder-lr 1e-4 --lr "$HEAD_LR"
                        --batch-size "$BATCH" --accumulate-steps "$ACCUM"
                        --train-steps "$FT_STEPS" --eval-steps "$FT_EVAL_STEPS")
            ;;
        *)
            echo "Unknown config $CONFIG"; FAILED+=("$TAG:config"); continue ;;
    esac

    echo "--- fine-tuning (dev scored during training) ---"
    python finetune.py \
        --data-root "$DR" --train-csv "$TRAIN_CSV" \
        --target-metric "$METRIC" --encoder "$ENCODER" --outdir "$OUT" \
        "${TRAIN_ARGS[@]}" --save-steps "$SAVE_STEPS" \
        --dev-csv "$DEV_LABELS" --dev-data-root "$DR" \
        --eval-batch-size "$FROZEN_BATCH" --best-metric "$BEST_METRIC" \
        --num-workers "$NUM_WORKERS"
    if [ $? -ne 0 ]; then echo "TRAINING FAILED for $TAG"; FAILED+=("$TAG:train"); continue; fi

    # Prefer the checkpoint selected on dev over the last-step one.
    CKPT="$OUT/model_best_${METRIC}.pt"
    if [ ! -f "$CKPT" ]; then
        echo "NOTE: no model_best_${METRIC}.pt, falling back to the final-step checkpoint."
        CKPT="$OUT/finetuned_model_${METRIC}_final.pt"
    fi
    echo "Selected checkpoint: $CKPT"

    # Local held-out evaluation. Encoder and target metric are read from the checkpoint.
    for SET in "${EVAL_SETS[@]}"; do
        echo "--- evaluating on $SET ---"
        python inference.py \
            --data-root "$DR" --csv-path "egs/$SET.mean.csv" \
            --checkpoint "$CKPT" --out "$OUT/${SET}_${METRIC}.csv"
        if [ $? -ne 0 ]; then echo "INFERENCE FAILED for $TAG on $SET"; FAILED+=("$TAG:$SET"); continue; fi

        python calculate_metrics.py \
            --prediction-csv "$OUT/${SET}_${METRIC}.csv" \
            --ground-truth-csv "egs/$SET.mean.csv"
    done

    echo "--- inference on the official dev set with the selected checkpoint ---"
    # Clear any output from a previous run first, so a failure here cannot leave a stale
    # CSV that the summary would then report as OK.
    rm -f "$OUT/dev_${METRIC}.csv"
    python inference.py \
        --data-root "$DR" --csv-path "$DEV_CSV" \
        --checkpoint "$CKPT" --out "$OUT/dev_${METRIC}.csv"
    if [ $? -ne 0 ]; then
        echo "INFERENCE FAILED for $TAG"; FAILED+=("$TAG:dev")
    elif [ -n "$DEV_GT" ]; then
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
echo "CodaBench submission CSVs:"
for METRIC in "${METRICS[@]}"; do
for CONFIG in "${CONFIGS[@]}"; do
    F="egs/${ENCODER}_${METRIC}_${CONFIG}_${RUN_TAG}/dev_${METRIC}.csv"
    if [ -f "$F" ]; then
        echo "  OK      $(pwd)/$F  ($(($(wc -l < "$F") - 1)) rows)"
    else
        echo "  MISSING $F"
    fi
done
done
echo ""
echo "Best dev $BEST_METRIC per run:"
for METRIC in "${METRICS[@]}"; do
for CONFIG in "${CONFIGS[@]}"; do
    L="egs/${ENCODER}_${METRIC}_${CONFIG}_${RUN_TAG}/dev_log_${METRIC}.csv"
    if [ -f "$L" ]; then
        python - "$L" "$BEST_METRIC" "${METRIC}/${CONFIG}" <<'PY'
import csv, sys, math
path, metric, tag = sys.argv[1], sys.argv[2], sys.argv[3]
rows = [r for r in csv.DictReader(open(path)) if int(r["step"]) > 0]
vals = [(float(r[metric]), int(r["step"])) for r in rows if r[metric] and not math.isnan(float(r[metric]))]
if vals:
    best, step = (min if metric.startswith("mse") else max)(vals)
    last = vals[-1][0]
    print(f"  {tag:24s} best {best:+.4f} @ step {step:<6d} final {last:+.4f}")
else:
    print(f"  {tag:24s} no evaluations recorded")
PY
    fi
done
done
echo ""
echo "Full dev curves: egs/*/dev_log_*.csv   (also grep this log for '[dev @')"
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
