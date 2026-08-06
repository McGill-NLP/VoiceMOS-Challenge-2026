#!/usr/bin/env bash
#SBATCH --job-name=voicemos-track3-unified-mse-speaker
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=david.guzman@mila.quebec

# Plain MSE + two-phase freeze, across encoder x head, for SPEAKER similarity (spk_sim).
#
#   sbatch track3/jobs/voicemos-track3-unified-mse-speaker.sh
#
# The MSE counterpart of voicemos-track3-unified-coral-speaker.sh. Everything except
# --objective is held identical to that job -- same encoders, heads, batching, learning
# rates, schedule, step budget, checkpoint selection -- so the objective is the only
# variable and the two sets of results are directly comparable.
#
# Four arms, every one at --objective mse --freeze-steps 5000 --backbone-lr-mult 0.1
# --train-steps 20000 --eval-steps 1000. Each fine-tuning command is written out in full
# below rather than generated, so what runs is exactly what you read.
#
#   commonaccent-mlp   --encoder commonaccent-ecapa --head mlp   ~50 min
#   commonaccent-moe   --encoder commonaccent-ecapa --head moe   ~50 min
#   eres2netv2-mlp     --encoder eres2netv2         --head mlp   ~2.3 h
#   eres2netv2-moe     --encoder eres2netv2         --head moe   ~2.3 h
#
# COST, measured from the completed 20,000-step CORAL jobs (10292464 / 10292465) rather
# than from an isolated benchmark, which overestimated ERes2NetV2 by 3.4x:
#
#   commonaccent-ecapa  bs16x1   0.14-0.16 s/step   ->  ~50 min per arm
#   eres2netv2          bs4x4    0.415-0.418 s/step ->  ~2.3 h per arm
#
# All four arms came to 6h15m and 6h25m for the two CORAL jobs. MSE is the same shape of
# computation, so 12 h is ample; the CORAL scripts ask for 24 h only because they were
# written before those measurements existed.
#
# ERes2NetV2 runs batch 4 x 4 accumulation because it OOMs at batch 16 in full fine-tuning.
# Effective batch stays 16, matching the other arms.
#
# The arms run CHEAPEST FIRST on purpose: if the job is preempted or hits the wall, the two
# commonaccent results are already written and scored rather than being lost behind the
# ERes2NetV2 arms. To split across jobs instead:
#
#   sbatch --export=ALL,ARMS="commonaccent-mlp commonaccent-moe" --time=03:00:00 \
#       track3/jobs/voicemos-track3-unified-mse-speaker.sh
#   sbatch --export=ALL,ARMS=eres2netv2-mlp --time=04:00:00 \
#       track3/jobs/voicemos-track3-unified-mse-speaker.sh
#   sbatch --export=ALL,ARMS=eres2netv2-moe --time=04:00:00 \
#       track3/jobs/voicemos-track3-unified-mse-speaker.sh
#
# NOTE ON CHECKPOINT SELECTION. BEST_METRIC is srcc_utt, where the completed CORAL runs
# used srcc_sys: that picked checkpoints as early as step 1000 while utterance metrics kept
# improving to step 16000-19000. The two sets are therefore NOT selection-matched, but the
# CORAL dev logs record every evaluation, so their srcc_utt-optimal numbers can be read
# back for a like-for-like comparison. Both knobs are overridable:
#
#   sbatch --export=ALL,BEST_METRIC=srcc_sys,SAVE_STEPS=1000 \
#       track3/jobs/voicemos-track3-unified-mse-speaker.sh
#
# Best spk_sim so far on this dev set, at the checkpoint each run selected (see
# ../../BRANCHES.md): corn/ecapa uSRCC 0.472, sSRCC 0.974; base/ecapa uSRCC 0.501,
# sSRCC 0.946; CORAL/eres2netv2-moe uSRCC 0.479, sSRCC 0.935. Selected instead on
# UTT-SRCC, CORAL/eres2netv2-moe reached 0.562 at step 16000.
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
# Selection is on utterance-level SRCC. srcc_sys peaks early and stays flat: on the CORAL
# runs it picked checkpoints as early as step 1000 while utterance metrics kept improving
# to step 16000-19000, costing up to 0.08 UTT-SRCC on the checkpoint that was kept.
BEST_METRIC=${BEST_METRIC:-srcc_utt}

if [ ! -f "$DEV_LABELS" ]; then
    echo "ERROR: no labelled dev set at $DEV_LABELS"; exit 1
fi

echo "=================================================================="
echo "MSE + two-phase freeze   metric=$METRIC"
echo "arms: $ARMS"
echo "steps=$TRAIN_STEPS  freeze=$FREEZE_STEPS  eval every $EVAL_STEPS  save every $SAVE_STEPS  select on $BEST_METRIC"
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
TAG=mse_commonaccent-mlp_${METRIC}
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
    --objective mse \
    --batch-size 16 \
    --accumulate-steps 1 \
    --lr 1e-3 \
    --backbone-lr-mult 0.1 \
    --freeze-steps "$FREEZE_STEPS" \
    --train-steps "$TRAIN_STEPS" \
    --save-steps "$SAVE_STEPS" \
    --eval-steps "$EVAL_STEPS" \
    --eval-batch-size 16 \
    --best-metric "$BEST_METRIC" \
    --dev-csv "$DEV_LABELS" \
    --dev-data-root "$DR" \
    --num-workers "$NUM_WORKERS"

if [ $? -ne 0 ]; then echo "TRAINING FAILED for $TAG"; FAILED+=("$TAG:train"); else finish_arm "$TAG"; fi
fi

##################################################################
# Arm 2/4  commonaccent-ecapa + moe        ~1.5 h
##################################################################
if should_run commonaccent-moe; then
TAG=mse_commonaccent-moe_${METRIC}
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
    --objective mse \
    --batch-size 16 \
    --accumulate-steps 1 \
    --lr 1e-3 \
    --backbone-lr-mult 0.1 \
    --freeze-steps "$FREEZE_STEPS" \
    --train-steps "$TRAIN_STEPS" \
    --save-steps "$SAVE_STEPS" \
    --eval-steps "$EVAL_STEPS" \
    --eval-batch-size 16 \
    --best-metric "$BEST_METRIC" \
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
TAG=mse_eres2netv2-mlp_${METRIC}
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
    --objective mse \
    --batch-size 4 \
    --accumulate-steps 4 \
    --lr 1e-3 \
    --backbone-lr-mult 0.1 \
    --freeze-steps "$FREEZE_STEPS" \
    --train-steps "$TRAIN_STEPS" \
    --save-steps "$SAVE_STEPS" \
    --eval-steps "$EVAL_STEPS" \
    --eval-batch-size 16 \
    --best-metric "$BEST_METRIC" \
    --dev-csv "$DEV_LABELS" \
    --dev-data-root "$DR" \
    --num-workers "$NUM_WORKERS"

if [ $? -ne 0 ]; then echo "TRAINING FAILED for $TAG"; FAILED+=("$TAG:train"); else finish_arm "$TAG"; fi
fi

##################################################################
# Arm 4/4  eres2netv2 + moe                ~7.9 h
##################################################################
if should_run eres2netv2-moe; then
TAG=mse_eres2netv2-moe_${METRIC}
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
    --objective mse \
    --batch-size 4 \
    --accumulate-steps 4 \
    --lr 1e-3 \
    --backbone-lr-mult 0.1 \
    --freeze-steps "$FREEZE_STEPS" \
    --train-steps "$TRAIN_STEPS" \
    --save-steps "$SAVE_STEPS" \
    --eval-steps "$EVAL_STEPS" \
    --eval-batch-size 16 \
    --best-metric "$BEST_METRIC" \
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
    f = f"egs/mse_{a}_{metric}/dev_{metric}.csv"
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
echo "Same arms under CORAL (job 10292464/10292465), at each run's selected checkpoint:"
echo "                    uMSE   uLCC  uSRCC    sMSE   sLCC  sSRCC"
echo "  commonaccent-mlp  0.405  0.532  0.513   0.110  0.959  0.934"
echo "  commonaccent-moe  0.406  0.537  0.512   0.106  0.941  0.934"
echo "  eres2netv2-mlp    0.427  0.547  0.498   0.075  0.943  0.935"
echo "  eres2netv2-moe    0.452  0.540  0.479   0.068  0.929  0.935"
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
