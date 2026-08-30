#!/usr/bin/env bash
#SBATCH --job-name=voicemos-track3-unified-sweep
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=david.guzman@mila.quebec

# Sweep over stacked configurations of ../unified/finetune.py, for BOTH targets.
#
# The unified script makes four axes independent, each taken from whichever branch it
# helped on (see ../../BRANCHES.md):
#
#   --encoder                      pluggable backbone            dev.dg/contrastive
#   --head mlp|moe                 mixture-of-experts head       dev.yj/empirical
#   --freeze-steps N               two-phase freeze schedule     dev.yj/empirical
#   --objective mse|corn|coral     ordinal regression            dev.ap/CORN
#   --lambda-rnc F                 Rank-N-Contrast auxiliary     dev.dg/contrastive
#
# This job runs a LADDER, not a grid: a reference arm, then one ingredient at a time, then
# the stack. That is what makes a gain attributable. A full grid is 2 heads x 3 objectives
# x 2 schedules x 2 rnc x 3 encoders x 2 metrics = 288 runs; do not try to run it.
#
# ONE JOB = len(CONFIGS) x len(METRICS) runs. By default that is 6 arms x 2 targets = 12,
# each one training + inference + scoring. Nine arms are defined below, so the most a
# single job can run is 9 x 2 = 18.
#
# Measured cost per arm on an L40S at the default 8,000 steps (ECAPA, dev evaluated every
# 250 steps at 3.2 s each, plus a final 600-pair inference):
#
#     base / moe / corn / stack   0.192 s/step, 8000 steps ->  29 min
#     freeze                      0.161 s/step, 8000 steps ->  25 min
#     stack-rnc (batch 32)        0.375 s/step, 4000 steps ->  28 min
#
# so the default 12 runs are about 5.6 h for ECAPA -- inside the 12 h wall. ERes2NetV2 is
# roughly 3x slower per optimizer step (batch 4 x accum 4) and the same ladder will NOT
# fit; split it, e.g. --export=ALL,ENCODER=eres2netv2,METRICS=spk_sim.
#
#   sbatch track3/jobs/voicemos-track3-unified-sweep.sh
#
# One encoder per job, so the encoder axis is swept by launching several:
#
#   for E in ecapa-voxceleb eres2netv2 commonaccent-ecapa; do
#       sbatch --export=ALL,ENCODER=$E track3/jobs/voicemos-track3-unified-sweep.sh
#   done
#
# Other useful overrides:
#
#   --export=ALL,CONFIGS="base corn coral"        pick the arms
#   --export=ALL,METRICS=acc_sim                  one target only
#   --export=ALL,TRAIN_STEPS=20000,FREEZE_STEPS=5000    the yj step budget
#   --export=ALL,TRAIN_STEPS=4,EVAL_STEPS=2,SAVE_STEPS=4    smoke test
#
# Deliberately NOT using `set -e`: if one arm fails the rest should still run.

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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

conda activate VoiceMOS

if [ "$CONDA_DEFAULT_ENV" != "VoiceMOS" ]; then
    echo "ERROR: conda env is '${CONDA_DEFAULT_ENV:-none}', expected VoiceMOS"; exit 1
fi
python -c "import torch, speechbrain, coral_pytorch" \
    || { echo "ERROR: torch/speechbrain/coral_pytorch not importable"; exit 1; }
echo "python: $(which python)"

# torchaudio >= 2.9 dispatches load() to torchcodec, which dies with
# "libnppicc.so.12: cannot open shared object file" unless the NVIDIA libs that ship
# with torch are on the loader path.
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH=$SITE_PACKAGES/nvidia/npp/lib:$LD_LIBRARY_PATH

REPO=/home/mila/g/guzmand/scratch/Repositories/VoiceMOS-Challenge-2026
cd "$REPO/track3/unified" || exit 1

python -c "
import torchaudio, glob
f = sorted(glob.glob('../baseline/data/vmc2026_track3_train_phase_distro_v3_syn/wav/*.wav'))[0]
torchaudio.load(f)
print('torchaudio.load OK')" || { echo "ERROR: torchaudio cannot load wavs"; exit 1; }

echo "NVIDIA SMI:"; nvidia-smi
NUM_WORKERS=${SLURM_CPUS_PER_TASK:-8}

##################################################################
# Configuration
##################################################################
# The TRAINING distribution is the right data root for train and dev alike: the eval
# distro is missing all 600 sys019 reference wavs, so inference against it drops
# every row. The labelled dev CSV is metadata only, so its paths resolve against $DR.
DR=../baseline/data/vmc2026_track3_train_phase_distro_v3_syn
TRAIN_CSV=$DR/sets/train.csv
DEV_CSV=${DEV_CSV:-$DR/sets/dev.csv}
DEV_LABELS=${DEV_LABELS:-../baseline/data/vmc2026_track3_eval_phase_distro_v3_syn/sets/dev_with_labels.csv}

ENCODER=${ENCODER:-ecapa-voxceleb}
METRICS=(${METRICS:-spk_sim acc_sim})
# `corn` is in the default set because `stack` contains it: without the single-ingredient
# corn arm, a difference between `stack` and `base` cannot be attributed to the head, the
# schedule or the objective.
CONFIGS=(${CONFIGS:-base moe freeze corn stack stack-rnc})

# Step budget. dev.yj used freeze 5,000 of 20,000 (25%); this keeps the ratio at a
# quarter of the budget while costing a third as much, so the ladder fits one allocation.
TRAIN_STEPS=${TRAIN_STEPS:-8000}
FREEZE_STEPS=${FREEZE_STEPS:-2000}
SAVE_STEPS=${SAVE_STEPS:-2000}
EVAL_STEPS=${EVAL_STEPS:-250}

BATCH=${BATCH:-16}
ACCUM=${ACCUM:-1}
LR=${LR:-1e-3}

# Rank-N-Contrast ranks each sample against the others IN ITS BATCH, so accumulation does
# not help it and a batch of 16 gives it very little to work with. ../rank-n-contrast used
# 96, but with a 6 s crop that this script does not have. Measured here on a 46 GiB L40S,
# ECAPA full fine-tuning with the RnC term:
#
#     batch 32 -> 21.2 GiB, 0.375 s/step
#     batch 48 -> 31.7 GiB, 0.571 s/step
#     batch 64 -> 42.2 GiB, 0.805 s/step   <- only ~4 GiB spare, and peak follows the
#                                             longest clip in the batch, so this OOMs
#                                             as soon as a long one turns up
#
# Hence 32 by default. ERes2NetV2 is far heavier and OOMs at plain batch 16 in full
# fine-tuning, so the override below drops it to 8.
RNC_BATCH=${RNC_BATCH:-32}
LAMBDA_RNC=${LAMBDA_RNC:-0.5}

# ERes2NetV2 is a 2D CNN at full temporal resolution and needs a smaller micro-batch with
# accumulation to reach an effective batch of 16. Measured on a 46 GB L40S.
if [[ "$ENCODER" == *eres2netv2* ]]; then
    BATCH=${BATCH_OVERRIDE:-4}
    ACCUM=${ACCUM_OVERRIDE:-4}
    RNC_BATCH=${RNC_BATCH_OVERRIDE:-8}
    echo "NOTE: $ENCODER detected -> batch $BATCH x accum $ACCUM, rnc batch $RNC_BATCH."
fi

# Computed AFTER the encoder override above, so it reflects the batch sizes actually used.
#
# The RnC arms must run at a larger batch for the loss to have anything to rank, which
# would otherwise hand them twice the data exposure of every other arm at the same step
# count -- 8000 x 32 = 256k sample presentations against 8000 x 16 = 128k. A win would then
# be unattributable: more RnC, or just more epochs? So the RnC arms are matched on samples
# seen rather than on optimizer steps. Pass RNC_STEPS=$TRAIN_STEPS to compare at equal
# steps instead.
EFF_BATCH=$((BATCH * ACCUM))
RNC_EFF_BATCH=$((RNC_BATCH * ACCUM))
if [ "$RNC_EFF_BATCH" -gt 0 ]; then
    RNC_STEPS=${RNC_STEPS:-$((TRAIN_STEPS * EFF_BATCH / RNC_EFF_BATCH))}
else
    RNC_STEPS=${RNC_STEPS:-$TRAIN_STEPS}
fi
[ "$RNC_STEPS" -lt 1 ] && RNC_STEPS=1

# Selection is on utterance-level SRCC. srcc_sys peaks early and stays flat: on the CORAL
# runs it picked checkpoints as early as step 1000 while utterance metrics kept improving
# to step 16000-19000, costing up to 0.08 UTT-SRCC on the checkpoint that was kept.
BEST_METRIC=${BEST_METRIC:-srcc_utt}

if [ ! -f "$DEV_LABELS" ]; then
    echo "ERROR: no labelled dev set at $DEV_LABELS"; exit 1
fi

echo "=================================================================="
echo "encoder=$ENCODER   metrics=${METRICS[*]}"
echo "configs=${CONFIGS[*]}"
echo "steps=$TRAIN_STEPS  freeze=$FREEZE_STEPS  batch=$BATCH x $ACCUM  lr=$LR"
echo "rnc arms: batch=$RNC_BATCH x $ACCUM  steps=$RNC_STEPS  lambda=$LAMBDA_RNC"
echo "  (both see $((TRAIN_STEPS * EFF_BATCH)) sample presentations, so arms are epoch-matched)"
echo "selection: best $BEST_METRIC on $DEV_LABELS"
echo "=================================================================="

mkdir -p egs

##################################################################
# The ladder
#
# Each arm changes ONE thing relative to `base`, except the two `stack` arms, which are
# the point of the exercise. Keep it that way when adding arms: an arm that changes two
# things at once cannot be attributed.
##################################################################
config_args() {
    case "$1" in
        base)       echo "--head mlp --objective mse" ;;
        moe)        echo "--head moe --objective mse" ;;
        freeze)     echo "--head mlp --objective mse --freeze-steps $FREEZE_STEPS --backbone-lr-mult 0.1" ;;
        corn)       echo "--head mlp --objective corn" ;;
        coral)      echo "--head mlp --objective coral" ;;
        rnc)        echo "--head mlp --objective mse --lambda-rnc $LAMBDA_RNC --batch-size $RNC_BATCH --train-steps $RNC_STEPS" ;;
        moe-freeze) echo "--head moe --objective mse --freeze-steps $FREEZE_STEPS --backbone-lr-mult 0.1" ;;
        stack)      echo "--head moe --objective corn --freeze-steps $FREEZE_STEPS --backbone-lr-mult 0.1" ;;
        stack-rnc)  echo "--head moe --objective corn --freeze-steps $FREEZE_STEPS --backbone-lr-mult 0.1 --lambda-rnc $LAMBDA_RNC --batch-size $RNC_BATCH --train-steps $RNC_STEPS" ;;
        *)          echo "" ;;
    esac
}

FAILED=()
for METRIC in "${METRICS[@]}"; do
for CONFIG in "${CONFIGS[@]}"; do
    EXTRA=$(config_args "$CONFIG")
    if [ -z "$EXTRA" ]; then
        echo "Unknown config '$CONFIG' -- skipping."; FAILED+=("$CONFIG:unknown"); continue
    fi

    TAG="${ENCODER}_${METRIC}_${CONFIG}"
    OUT="egs/$TAG"
    echo ""
    echo "##################################################################"
    echo "# $TAG   ($(date))"
    echo "#   $EXTRA"
    echo "##################################################################"

    # --batch-size appears in $EXTRA for the rnc arms and must win, so the default goes first.
    python finetune.py \
        --data-root "$DR" --train-csv "$TRAIN_CSV" \
        --target-metric "$METRIC" --outdir "$OUT" \
        --encoder "$ENCODER" \
        --batch-size "$BATCH" --accumulate-steps "$ACCUM" --lr "$LR" \
        --train-steps "$TRAIN_STEPS" --save-steps "$SAVE_STEPS" \
        $EXTRA \
        --dev-csv "$DEV_LABELS" --dev-data-root "$DR" \
        --eval-steps "$EVAL_STEPS" --best-metric "$BEST_METRIC" \
        --num-workers "$NUM_WORKERS"
    if [ $? -ne 0 ]; then echo "TRAINING FAILED for $TAG"; FAILED+=("$TAG:train"); continue; fi

    CKPT="$OUT/model_best_${METRIC}.pt"
    if [ ! -f "$CKPT" ]; then
        echo "NOTE: no model_best, falling back to the final-step checkpoint."
        CKPT="$OUT/finetuned_model_${METRIC}_final.pt"
    fi
    echo "Selected checkpoint: $CKPT"

    # Clear any previous output first, so a failure here cannot leave a stale CSV that
    # the summary would then report as OK.
    rm -f "$OUT/dev_${METRIC}.csv"
    python inference.py \
        --data-root "$DR" --csv-path "$DEV_CSV" \
        --checkpoint "$CKPT" --out "$OUT/dev_${METRIC}.csv"
    if [ $? -ne 0 ]; then echo "INFERENCE FAILED for $TAG"; FAILED+=("$TAG:dev"); continue; fi

    echo "--- scoring against the official dev labels ---"
    python calculate_metrics.py \
        --prediction-csv "$OUT/dev_${METRIC}.csv" --ground-truth-csv "$DEV_LABELS"
done
done

##################################################################
# Summary
##################################################################
echo ""
echo "=================================================================="
echo "All six metrics per arm (the final ranking combines several of them,"
echo "so do not read a single column):"
echo ""
python - "$DEV_LABELS" "$ENCODER" "${METRICS[*]}" "${CONFIGS[*]}" <<'PY'
import csv, sys, os
import numpy as np, scipy.stats
from collections import defaultdict

labels, encoder, metrics, configs = sys.argv[1], sys.argv[2], sys.argv[3].split(), sys.argv[4].split()
gt = {}
for r in csv.DictReader(open(labels)):
    gt[(r["wav_a_path"], r["wav_b_path"])] = r

hdr = f"{'arm':<24}{'n':>6}{'uMSE':>8}{'uLCC':>8}{'uSRCC':>8}{'sMSE':>8}{'sLCC':>8}{'sSRCC':>8}"
for m in metrics:
    print(f"\n{m}\n{hdr}\n{'-' * len(hdr)}")
    for c in configs:
        f = f"egs/{encoder}_{m}_{c}/dev_{m}.csv"
        if not os.path.exists(f):
            print(f"{c:<24}{'MISSING':>6}"); continue
        ut, up = [], []
        st, sp = defaultdict(list), defaultdict(list)
        for r in csv.DictReader(open(f)):
            k = (r["wav_a_path"], r["wav_b_path"])
            if k not in gt or f"pred_{m}" not in r:
                continue
            t, p, s = float(gt[k][m]), float(r[f"pred_{m}"]), gt[k]["system_id"]
            ut.append(t); up.append(p); st[s].append(t); sp[s].append(p)
        # n is printed rather than assumed: a short row count means inference dropped
        # pairs, which is the signature of the wrong --data-root.
        if len(ut) < 3:
            print(f"{c:<24}{len(ut):>6}   too few rows to score"); continue
        ut, up = np.array(ut), np.array(up)
        a = np.array([np.mean(st[s]) for s in st]); b = np.array([np.mean(sp[s]) for s in sp])
        print(f"{c:<24}{len(ut):>6}{np.mean((ut-up)**2):>8.3f}{scipy.stats.pearsonr(ut,up).statistic:>8.3f}"
              f"{scipy.stats.spearmanr(ut,up).statistic:>8.3f}{np.mean((a-b)**2):>8.3f}"
              f"{scipy.stats.pearsonr(a,b).statistic:>8.3f}{scipy.stats.spearmanr(a,b).statistic:>8.3f}")
PY

echo ""
echo ""
echo "TensorBoard (every arm appears as its own run):"
echo "  tensorboard --logdir $(pwd)/egs"
echo ""
echo "Reference, official Baseline 2 (frozen encoder), dev set:"
echo "  spk_sim  uMSE 0.438  uLCC 0.511  uSRCC 0.451  sMSE 0.069  sLCC 0.916  sSRCC 0.860"
echo "  acc_sim  uMSE 0.418  uLCC 0.465  uSRCC 0.440  sMSE 0.060  sLCC 0.902  sSRCC 0.861"

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""; echo "FAILURES: ${FAILED[*]}"
fi

ELAPSED=$((SECONDS - START_TIME))
echo ""
echo "Job $SLURM_JOB_ID finished at $(date) after $((ELAPSED / 3600))h $((ELAPSED % 3600 / 60))m"
