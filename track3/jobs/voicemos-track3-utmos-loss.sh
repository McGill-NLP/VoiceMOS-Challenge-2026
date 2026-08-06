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
#   utmos-g2     1.0 * clipped + 2.0 * contrastive
#   utmos-g5     1.0 * clipped + 5.0 * contrastive -- see the note on gamma below.
#
#   sbatch track3/jobs/voicemos-track3-utmos-loss.sh
#
# Two things changed after the first sweep:
#
#  1. EVERY ARM IS ALSO SCORED RECALIBRATED. The contrastive term constrains only score
#     differences, so it fixes no absolute scale and its predictions settle near the
#     range-clipped head's tanh centre (3.0) against labels averaging 4.0. That leaves
#     SRCC/LCC untouched but wrecks MSE, which made the contrastive arm look far worse
#     than it is. recalibrate.py fits y = a*pred + b on the TRAINING pairs and applies it
#     to dev: measured a~0.97, b~+1.0, i.e. almost pure intercept. On the first sweep's
#     contrastive arm that took spk_sim utt MSE 0.885 -> 0.458 and sys MSE 0.609 -> 0.067
#     with SRCC bit-identical, making it the best arm on every spk_sim metric.
#
#  2. GAMMA IS SWEPT FURTHER. The per-term logging added to finetune.py shows the
#     regression term outweighing the contrastive one about 4.5x at the shipped gamma=0.5
#     (clipped 0.92 vs 0.5*0.41=0.21 at step 20), so the ranking signal is largely
#     switched off. gamma=2 is roughly balanced and gamma=5 is contrastive-dominant;
#     together with contrastive-alone this sweeps the ratio from pure regression to pure
#     ranking.
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
# Selection is on utterance-level SRCC. srcc_sys peaks early and stays flat: on the CORAL
# runs it picked checkpoints as early as step 1000 while utterance metrics kept improving
# to step 16000-19000, costing up to 0.08 UTT-SRCC on the checkpoint that was kept.
BEST_METRIC=${BEST_METRIC:-srcc_utt}

METRICS=(spk_sim acc_sim)
ARMS=(mse clipped contrastive utmos utmos-g2 utmos-g5)

echo "=================================================================="
echo "arms=${ARMS[*]}  metrics=${METRICS[*]}"
echo "batch=$BATCH (contrastive sees $((BATCH*(BATCH-1))) ordered pairs/step)"
echo "steps=$TRAIN_STEPS  encoder_lr=$ENCODER_LR  head_lr=$HEAD_LR"
echo "dev labels : $DEV_LABELS"
echo "             -> scored during training, selects model_best on $BEST_METRIC,"
echo "                and is the ground truth for the final raw/recalibrated scoring"
echo "=================================================================="

mkdir -p egs
if [ ! -f "$DEV_LABELS" ]; then
    echo "ERROR: no labelled dev set at $DEV_LABELS"; exit 1
fi
# Scored directly against the released labels -- no make_eval_gt.py round-trip. That script
# collapses listener-wise rows to per-pair means, and dev_with_labels.csv is already one row
# per pair (600 rows, both metrics populated), so running it produced a byte-identical copy.
DEV_GT=$DEV_LABELS

# train.csv DOES need the round-trip: it is listener-wise, 13,687 rows over 2,800 unique
# pairs at ~4.9 ratings each. Used only to FIT the recalibration -- fitting on dev and then
# reporting dev would be contamination, only two parameters but still reading the answer.
python make_eval_gt.py --in "$TRAIN_CSV" --out egs/train.mean.csv \
    || { echo "ERROR: could not build train ground truth"; exit 1; }
TRAIN_GT=egs/train.mean.csv

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

    # The utmos-g* arms raise gamma against the fixed beta=1.0. See the gamma note in the
    # header: the per-term dev-log columns show the regression term outweighing the
    # contrastive one ~4.5x at the shipped gamma=0.5.
    case "$ARM" in
        utmos-g2) LOSS_ARGS=(--loss utmos --gamma 2.0) ;;
        utmos-g5) LOSS_ARGS=(--loss utmos --gamma 5.0) ;;
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
    rm -f "$OUT/dev_${METRIC}.csv" "$OUT/dev_${METRIC}.recal.csv"
    python inference.py \
        --data-root "$DR" --csv-path "$DEV_CSV" \
        --checkpoint "$CKPT" --out "$OUT/dev_${METRIC}.csv"
    if [ $? -ne 0 ]; then
        echo "INFERENCE FAILED for $TAG"; FAILED+=("$TAG:dev"); continue
    fi

    echo "--- scoring against the official dev labels (raw) ---"
    python calculate_metrics.py \
        --prediction-csv "$OUT/dev_${METRIC}.csv" --ground-truth-csv "$DEV_GT"

    # Recalibration: predict the TRAINING pairs, fit y = a*pred + b there, apply to dev.
    # ~2,800 pairs, about a minute. Rank metrics are invariant to this, so only MSE moves.
    echo "--- recalibration (affine fitted on train, applied to dev) ---"
    python inference.py \
        --data-root "$DR" --csv-path "$TRAIN_GT" \
        --checkpoint "$CKPT" --out "$OUT/train_${METRIC}.csv"
    if [ $? -ne 0 ]; then
        echo "TRAIN INFERENCE FAILED for $TAG"; FAILED+=("$TAG:train_pred"); continue
    fi

    python recalibrate.py \
        --fit-csv "$OUT/train_${METRIC}.csv" \
        --apply-csv "$OUT/dev_${METRIC}.csv" \
        --out "$OUT/dev_${METRIC}.recal.csv"
    if [ $? -ne 0 ]; then
        echo "RECALIBRATION FAILED for $TAG"; FAILED+=("$TAG:recal"); continue
    fi

    echo "--- scoring against the official dev labels (recalibrated) ---"
    python calculate_metrics.py \
        --prediction-csv "$OUT/dev_${METRIC}.recal.csv" --ground-truth-csv "$DEV_GT"
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
echo "Final dev metrics per arm, raw vs recalibrated (rank metrics are affine-invariant,"
echo "so only MSE moves; the affine was fitted on train, never on dev):"
python - "$DEV_GT" "${METRICS[*]}" "${ARMS[*]}" <<'PY'
import os, sys, csv
sys.path.insert(0, os.getcwd())
from calculate_metrics import compute_metrics
from collections import defaultdict
import numpy as np

gt_path, metrics, arms = sys.argv[1], sys.argv[2].split(), sys.argv[3].split()
gt = {}
for r in csv.DictReader(open(gt_path)):
    gt[(r["wav_a_path"], r["wav_b_path"])] = r

def score(path, metric):
    if not os.path.exists(path):
        return None
    y, p, sysid = [], [], []
    for r in csv.DictReader(open(path)):
        key = (r["wav_a_path"], r["wav_b_path"])
        g = gt.get(key)
        if not g or not g.get(metric, "").strip():
            continue
        y.append(float(g[metric])); p.append(float(r[f"pred_{metric}"]))
        sysid.append(g.get("system_id", ""))
    if not y:
        return None
    y, p = np.array(y), np.array(p)
    ut = compute_metrics(y, p)
    ty, tp = defaultdict(list), defaultdict(list)
    for s, a, b in zip(sysid, y, p):
        ty[s].append(a); tp[s].append(b)
    sy = compute_metrics(np.array([np.mean(ty[s]) for s in ty]),
                         np.array([np.mean(tp[s]) for s in ty]))
    return ut, sy

hdr = f"  {'arm':<22}{'utt MSE':>9}{'utt LCC':>9}{'utt SRCC':>10}{'sys MSE':>9}{'sys LCC':>9}{'sys SRCC':>10}"
for metric in metrics:
    print(f"\n  [{metric}]")
    print(hdr)
    for arm in arms:
        d = f"egs/{metric}_{arm}"
        for label, path in ((arm, f"{d}/dev_{metric}.csv"),
                            (f"{arm} +recal", f"{d}/dev_{metric}.recal.csv")):
            r = score(path, metric)
            if r is None:
                continue
            (um, ul, us), (sm, sl, ss) = r
            print(f"  {label:<22}{um:>9.4f}{ul:>9.4f}{us:>10.4f}{sm:>9.4f}{sl:>9.4f}{ss:>10.4f}")
PY
echo ""
echo "Baseline 2 reference (dev): spk SYS-SRCC 0.860, acc SYS-SRCC 0.861"
echo "Full dev curves (incl. per-term loss columns): egs/*/dev_log_*.csv"
echo ""
echo "NOTE: model_best is selected on dev $BEST_METRIC and scored on the same 600 pairs,"
echo "      so these numbers are optimistic. CodaBench eval is the honest comparison."
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
