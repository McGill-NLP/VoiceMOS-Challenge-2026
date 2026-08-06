#!/usr/bin/env bash
#SBATCH --job-name=voicemos-track3-unified-coral-speaker
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=david.guzman@mila.quebec

# CORAL + two-phase freeze, across encoder x head, for SPEAKER similarity (spk_sim).
#
#   sbatch track3/jobs/voicemos-track3-unified-coral-speaker.sh
#
# Four arms, every one at --objective coral --freeze-steps 5000 --backbone-lr-mult 0.1
# --train-steps 20000 --eval-steps 1000. Each fine-tuning command is written out in full
# below rather than generated, so what runs is exactly what you read.
#
#   commonaccent-mlp   --encoder commonaccent-ecapa --head mlp   ~1.5 h
#   commonaccent-moe   --encoder commonaccent-ecapa --head moe   ~1.5 h
#   eres2netv2-mlp     --encoder eres2netv2         --head mlp   ~7.9 h
#   eres2netv2-moe     --encoder eres2netv2         --head moe   ~7.9 h
#
# MEASURED COST (steady state on an idle L40S, CORAL objective, warm-up excluded):
#
#   commonaccent-ecapa  bs16x1   frozen 0.232 s/step, full-ft 0.287 s/step, 11.2 GiB
#   eres2netv2          bs4x4    frozen 0.580 s/step, full-ft 1.683 s/step, 12.7 GiB
#   dev evaluation, 600 pairs    commonaccent 11.1 s, eres2netv2 15.0 s
#
# At 20,000 steps with 5,000 frozen that is ~1.5 h per commonaccent arm and ~7.8 h per
# eres2netv2 arm, so ALL FOUR ARMS IS ABOUT 19 HOURS. Hence --time=24:00:00.
#
# ERes2NetV2 is ~6x the cost of ECAPA per optimizer step, not the ~3x implied by the older
# note in ../unified/README.md's predecessor: its fbank is computed one utterance at a time
# in Python inside the forward pass, so batch 4 with 4 accumulation steps means 32 serial
# fbank calls per optimizer step. It also OOMs at batch 16 in full fine-tuning, which is
# why it runs 4 x 4 for the same effective batch of 16 as the other arms.
#
# The arms run CHEAPEST FIRST on purpose: if the job is preempted or hits the wall, the two
# commonaccent results are already written and scored rather than being lost behind eight
# hours of ERes2NetV2. To split across jobs instead (recommended over one 19 h job):
#
#   sbatch --export=ALL,ARMS="commonaccent-mlp commonaccent-moe" --time=06:00:00 \
#       track3/jobs/voicemos-track3-unified-coral-speaker.sh
#   sbatch --export=ALL,ARMS=eres2netv2-mlp --time=12:00:00 \
#       track3/jobs/voicemos-track3-unified-coral-speaker.sh
#   sbatch --export=ALL,ARMS=eres2netv2-moe --time=12:00:00 \
#       track3/jobs/voicemos-track3-unified-coral-speaker.sh
#
# Best spk_sim so far on this dev set, for comparison (see ../../BRANCHES.md and the ecapa
# ladder): corn/ecapa uSRCC 0.472, sSRCC 0.974; base/ecapa uSRCC 0.501, sSRCC 0.946. The
# utterance-level record is still eres2netv2 + mse + no schedule at uSRCC 0.521.
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

REPO=${REPO:-/home/mila/g/guzmand/scratch/Repositories/VoiceMOS-Challenge-2026}
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
METRIC=spk_sim

# The TRAINING distribution is the right data root for train and dev alike: the eval
# distro is missing all 600 sys019 reference wavs, so inference against it drops every row.
# The labelled dev CSV is metadata only, so its paths resolve against $DR too.
DR=../baseline/data/vmc2026_track3_train_phase_distro_v3_syn
TRAIN_CSV=$DR/sets/train.csv
DEV_CSV=${DEV_CSV:-$DR/sets/dev.csv}
DEV_LABELS=${DEV_LABELS:-../baseline/data/vmc2026_track3_eval_phase_distro_v3_syn/sets/dev_with_labels.csv}

# Overridable so a single arm can be sent to its own job, and so the whole thing can be
# smoke-tested cheaply:
#   TRAIN_STEPS=4 FREEZE_STEPS=2 EVAL_STEPS=2 SAVE_STEPS=4 sbatch ...
ARMS=${ARMS:-"commonaccent-mlp commonaccent-moe eres2netv2-mlp eres2netv2-moe"}
TRAIN_STEPS=${TRAIN_STEPS:-20000}
FREEZE_STEPS=${FREEZE_STEPS:-5000}
EVAL_STEPS=${EVAL_STEPS:-1000}
SAVE_STEPS=${SAVE_STEPS:-5000}

if [ ! -f "$DEV_LABELS" ]; then
    echo "ERROR: no labelled dev set at $DEV_LABELS"; exit 1
fi

echo "=================================================================="
echo "CORAL + two-phase freeze   metric=$METRIC"
echo "arms: $ARMS"
echo "steps=$TRAIN_STEPS  freeze=$FREEZE_STEPS  eval every $EVAL_STEPS"
echo "=================================================================="

mkdir -p egs
FAILED=()

should_run () { [[ " $ARMS " == *" $1 "* ]]; }

# Shared post-training steps: pick the dev-selected checkpoint, predict the official dev
# set with it, score that prediction. Only the fine-tuning commands differ per arm.
finish_arm () {
    local TAG=$1
    local OUT="egs/$TAG"

    local CKPT="$OUT/model_best_${METRIC}.pt"
    if [ ! -f "$CKPT" ]; then
        echo "NOTE: no model_best, falling back to the final-step checkpoint."
        CKPT="$OUT/finetuned_model_${METRIC}_final.pt"
    fi
    echo "Selected checkpoint: $CKPT"

    # Clear any previous output first, so a failure here cannot leave a stale CSV that the
    # summary would then report as OK.
    rm -f "$OUT/dev_${METRIC}.csv"
    python inference.py \
        --data-root "$DR" \
        --csv-path "$DEV_CSV" \
        --checkpoint "$CKPT" \
        --out "$OUT/dev_${METRIC}.csv"
    if [ $? -ne 0 ]; then echo "INFERENCE FAILED for $TAG"; FAILED+=("$TAG:dev"); return 1; fi

    echo "--- scoring against the official dev labels ---"
    python calculate_metrics.py \
        --prediction-csv "$OUT/dev_${METRIC}.csv" \
        --ground-truth-csv "$DEV_LABELS"

    local E=$((SECONDS - START_TIME))
    echo "[$TAG done, $((E / 3600))h $((E % 3600 / 60))m into the job]"
}

##################################################################
# Arm 1/4  commonaccent-ecapa + mlp        ~1.5 h
##################################################################
if should_run commonaccent-mlp; then
TAG=coral_commonaccent-mlp_${METRIC}
echo ""
echo "##################################################################"
echo "# $TAG   ($(date))"
echo "##################################################################"

python finetune.py \
    --data-root "$DR" \
    --train-csv "$TRAIN_CSV" \
    --target-metric "$METRIC" \
    --outdir "egs/$TAG" \
    --encoder commonaccent-ecapa \
    --head mlp \
    --objective coral \
    --batch-size 16 \
    --accumulate-steps 1 \
    --lr 1e-3 \
    --backbone-lr-mult 0.1 \
    --freeze-steps "$FREEZE_STEPS" \
    --train-steps "$TRAIN_STEPS" \
    --save-steps "$SAVE_STEPS" \
    --eval-steps "$EVAL_STEPS" \
    --eval-batch-size 16 \
    --best-metric srcc_utt \
    --dev-csv "$DEV_LABELS" \
    --dev-data-root "$DR" \
    --num-workers "$NUM_WORKERS"

if [ $? -ne 0 ]; then echo "TRAINING FAILED for $TAG"; FAILED+=("$TAG:train"); else finish_arm "$TAG"; fi
fi

##################################################################
# Arm 2/4  commonaccent-ecapa + moe        ~1.5 h
##################################################################
if should_run commonaccent-moe; then
TAG=coral_commonaccent-moe_${METRIC}
echo ""
echo "##################################################################"
echo "# $TAG   ($(date))"
echo "##################################################################"

python finetune.py \
    --data-root "$DR" \
    --train-csv "$TRAIN_CSV" \
    --target-metric "$METRIC" \
    --outdir "egs/$TAG" \
    --encoder commonaccent-ecapa \
    --head moe \
    --objective coral \
    --batch-size 16 \
    --accumulate-steps 1 \
    --lr 1e-3 \
    --backbone-lr-mult 0.1 \
    --freeze-steps "$FREEZE_STEPS" \
    --train-steps "$TRAIN_STEPS" \
    --save-steps "$SAVE_STEPS" \
    --eval-steps "$EVAL_STEPS" \
    --eval-batch-size 16 \
    --best-metric srcc_utt \
    --dev-csv "$DEV_LABELS" \
    --dev-data-root "$DR" \
    --num-workers "$NUM_WORKERS"

if [ $? -ne 0 ]; then echo "TRAINING FAILED for $TAG"; FAILED+=("$TAG:train"); else finish_arm "$TAG"; fi
fi

##################################################################
# Arm 3/4  eres2netv2 + mlp                ~7.9 h
#
# batch 4 x 4 accumulation: ERes2NetV2 is a 2D CNN at full temporal resolution and OOMs at
# batch 16 in full fine-tuning. Effective batch stays 16, matching the other arms.
##################################################################
if should_run eres2netv2-mlp; then
TAG=coral_eres2netv2-mlp_${METRIC}
echo ""
echo "##################################################################"
echo "# $TAG   ($(date))"
echo "##################################################################"

python finetune.py \
    --data-root "$DR" \
    --train-csv "$TRAIN_CSV" \
    --target-metric "$METRIC" \
    --outdir "egs/$TAG" \
    --encoder eres2netv2 \
    --head mlp \
    --objective coral \
    --batch-size 4 \
    --accumulate-steps 4 \
    --lr 1e-3 \
    --backbone-lr-mult 0.1 \
    --freeze-steps "$FREEZE_STEPS" \
    --train-steps "$TRAIN_STEPS" \
    --save-steps "$SAVE_STEPS" \
    --eval-steps "$EVAL_STEPS" \
    --eval-batch-size 16 \
    --best-metric srcc_utt \
    --dev-csv "$DEV_LABELS" \
    --dev-data-root "$DR" \
    --num-workers "$NUM_WORKERS"

if [ $? -ne 0 ]; then echo "TRAINING FAILED for $TAG"; FAILED+=("$TAG:train"); else finish_arm "$TAG"; fi
fi

##################################################################
# Arm 4/4  eres2netv2 + moe                ~7.9 h
##################################################################
if should_run eres2netv2-moe; then
TAG=coral_eres2netv2-moe_${METRIC}
echo ""
echo "##################################################################"
echo "# $TAG   ($(date))"
echo "##################################################################"

python finetune.py \
    --data-root "$DR" \
    --train-csv "$TRAIN_CSV" \
    --target-metric "$METRIC" \
    --outdir "egs/$TAG" \
    --encoder eres2netv2 \
    --head moe \
    --objective coral \
    --batch-size 4 \
    --accumulate-steps 4 \
    --lr 1e-3 \
    --backbone-lr-mult 0.1 \
    --freeze-steps "$FREEZE_STEPS" \
    --train-steps "$TRAIN_STEPS" \
    --save-steps "$SAVE_STEPS" \
    --eval-steps "$EVAL_STEPS" \
    --eval-batch-size 16 \
    --best-metric srcc_utt \
    --dev-csv "$DEV_LABELS" \
    --dev-data-root "$DR" \
    --num-workers "$NUM_WORKERS"

if [ $? -ne 0 ]; then echo "TRAINING FAILED for $TAG"; FAILED+=("$TAG:train"); else finish_arm "$TAG"; fi
fi

##################################################################
# Summary
##################################################################
echo ""
echo "=================================================================="
echo "All six metrics per arm (the final ranking combines several of them,"
echo "so do not read a single column):"
echo ""
python - "$DEV_LABELS" "$METRIC" "$ARMS" <<'PY'
import csv, sys, os
import numpy as np, scipy.stats
from collections import defaultdict

labels, metric, arms = sys.argv[1], sys.argv[2], sys.argv[3].split()
gt = {}
for r in csv.DictReader(open(labels)):
    gt[(r["wav_a_path"], r["wav_b_path"])] = r

hdr = f"{'arm':<20}{'n':>6}{'uMSE':>8}{'uLCC':>8}{'uSRCC':>8}{'sMSE':>8}{'sLCC':>8}{'sSRCC':>8}"
print(f"\n{metric}\n{hdr}\n{'-' * len(hdr)}")
for a in arms:
    f = f"egs/coral_{a}_{metric}/dev_{metric}.csv"
    if not os.path.exists(f):
        print(f"{a:<20}{'MISSING':>6}"); continue
    ut, up = [], []
    st, sp = defaultdict(list), defaultdict(list)
    for r in csv.DictReader(open(f)):
        k = (r["wav_a_path"], r["wav_b_path"])
        if k not in gt or f"pred_{metric}" not in r:
            continue
        t, p, s = float(gt[k][metric]), float(r[f"pred_{metric}"]), gt[k]["system_id"]
        ut.append(t); up.append(p); st[s].append(t); sp[s].append(p)
    # n is printed rather than assumed: a short row count means inference dropped pairs,
    # which is the signature of the wrong --data-root.
    if len(ut) < 3:
        print(f"{a:<20}{len(ut):>6}   too few rows to score"); continue
    ut, up = np.array(ut), np.array(up)
    x = np.array([np.mean(st[s]) for s in st]); y = np.array([np.mean(sp[s]) for s in sp])
    print(f"{a:<20}{len(ut):>6}{np.mean((ut-up)**2):>8.3f}{scipy.stats.pearsonr(ut,up).statistic:>8.3f}"
          f"{scipy.stats.spearmanr(ut,up).statistic:>8.3f}{np.mean((x-y)**2):>8.3f}"
          f"{scipy.stats.pearsonr(x,y).statistic:>8.3f}{scipy.stats.spearmanr(x,y).statistic:>8.3f}")
PY

echo ""
echo "For reference on the same dev set:"
echo "  spk_sim  Baseline 2 published     uMSE 0.438  uLCC 0.511  uSRCC 0.451  sMSE 0.069  sLCC 0.916  sSRCC 0.860"
echo "  spk_sim  best so far (corn/ecapa) uMSE 0.451  uLCC 0.518  uSRCC 0.472  sMSE 0.039  sLCC 0.958  sSRCC 0.974"
echo ""
echo "TensorBoard (every arm appears as its own run):"
echo "  tensorboard --logdir $(pwd)/egs"

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""; echo "FAILURES: ${FAILED[*]}"
fi

ELAPSED=$((SECONDS - START_TIME))
echo ""
echo "Job $SLURM_JOB_ID finished at $(date) after $((ELAPSED / 3600))h $((ELAPSED % 3600 / 60))m"
