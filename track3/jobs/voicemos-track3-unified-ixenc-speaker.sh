#!/usr/bin/env bash
#SBATCH --job-name=voicemos-track3-unified-ixenc-speaker
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=18:00:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=david.guzman@mila.quebec

# no-b and bilinear interactions on the two ERes2NetV2 encoders, SPEAKER similarity.
#
#   sbatch track3/jobs/voicemos-track3-unified-ixenc-speaker.sh
#
# Four arms: {eres2netv2, eres2netv2-w24s4ep4} x {no-b, bilinear}. Everything else is held
# at the configuration that produced the current records -- MSE, MoE head, freeze
# 5000/20000, --backbone-lr-mult 0.1, 20,000 steps -- so the interaction vector is the only
# variable and the `baseline` runs already on disk are the controls:
#
#   eres2netv2  + moe + mse + baseline   spk uSRCC 0.563  (job 10298520)
#   w24s4ep4    + moe + mse + baseline   spk uSRCC 0.572  (job 10305056)
#
# Both are quoted in the summary at the end, so no re-run is needed to compare.
#
# MEASURED COST, from the completed 20,000-step runs rather than an isolated benchmark:
#
#   eres2netv2  + moe   0.42 s/step  ->  ~2.3 h per arm   (jobs 10298519 / 10298520)
#   w24s4ep4    + moe   0.89 s/step  ->  ~5.0 h per arm   (jobs 10305054 / 10305056)
#
# So all four arms is ~14.6 h. Hence --time=18:00:00. Cheapest first, so a preemption or a
# wall-clock overrun costs the w24s4ep4 arms rather than everything. To split instead:
#
#   sbatch --export=ALL,ARMS="eres2netv2-no-b eres2netv2-bilinear" --time=06:00:00 ...
#   sbatch --export=ALL,ARMS="w24s4ep4-no-b w24s4ep4-bilinear"     --time=12:00:00 ...
#
# WHY THESE TWO INTERACTIONS. In the ECAPA ablation (job 10298522) no-b was the largest
# gain on spk_sim (+0.030 UTT-SRCC over baseline) and bilinear the most consistent across
# targets (+0.017 spk, +0.030 acc). Both caveats are worth carrying: that ablation ran on
# ECAPA with NO freeze schedule and a single LR group, where within-run noise was sd 0.024
# and most differences did not clear it. Under this recipe the same runs are 2-3x quieter
# (sd 0.007-0.015), so effects of this size should be measurable here even though they were
# marginal there. no-b-bilinear is deliberately excluded: it lost to both components.
#
# ERes2NetV2 needs batch 4 x 4 accumulation for an effective batch of 16 -- it OOMs at
# batch 16 outright, and w24s4ep4 (53.5M params, 3x the plain model) peaks at 29.0 GiB of a
# 46 GiB card even at batch 4. If an arm OOMs, halve the micro-batch and double accumulation:
#
#   sbatch --export=ALL,BATCH=2,ACCUM=8 ...
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

# Both checkpoints come from ModelScope; w24s4ep4 has no Hugging Face mirror. Fetch them
# before the clock starts, so a network problem fails in seconds rather than hours in.
python -c "
from encoders import build_encoder
for name in ('eres2netv2', 'eres2netv2-w24s4ep4'):
    enc = build_encoder(name)
    print(f'{name} ready: {sum(p.numel() for p in enc.parameters())/1e6:.2f}M params')" \
    || { echo "ERROR: could not build the ERes2NetV2 encoders"; exit 1; }

echo "NVIDIA SMI:"; nvidia-smi
NUM_WORKERS=${SLURM_CPUS_PER_TASK:-8}

##################################################################
# Configuration
##################################################################
METRIC=spk_sim

# The TRAINING distribution is the right data root for train and dev alike: the eval
# distro is missing all 600 sys019 reference wavs, so inference against it drops every row.
DR=../baseline/data/vmc2026_track3_train_phase_distro_v3_syn
TRAIN_CSV=$DR/sets/train.csv
DEV_CSV=${DEV_CSV:-$DR/sets/dev.csv}
DEV_LABELS=${DEV_LABELS:-../baseline/data/vmc2026_track3_eval_phase_distro_v3_syn/sets/dev_with_labels.csv}

ARMS=${ARMS:-"eres2netv2-no-b eres2netv2-bilinear w24s4ep4-no-b w24s4ep4-bilinear"}

BATCH=${BATCH:-4}
ACCUM=${ACCUM:-4}
LR=${LR:-1e-3}
BILINEAR_RANK=${BILINEAR_RANK:-64}
TRAIN_STEPS=${TRAIN_STEPS:-20000}
FREEZE_STEPS=${FREEZE_STEPS:-5000}
EVAL_STEPS=${EVAL_STEPS:-1000}
SAVE_STEPS=${SAVE_STEPS:-2000}
BEST_METRIC=${BEST_METRIC:-srcc_utt}

if [ ! -f "$DEV_LABELS" ]; then
    echo "ERROR: no labelled dev set at $DEV_LABELS"; exit 1
fi

echo "=================================================================="
echo "interaction x ERes2NetV2 encoder   metric=$METRIC"
echo "arms: $ARMS"
echo "held fixed: mse + moe, freeze $FREEZE_STEPS/$TRAIN_STEPS, backbone-lr-mult 0.1"
echo "batch=${BATCH}x${ACCUM}  lr=$LR  eval every $EVAL_STEPS  save every $SAVE_STEPS"
echo "select on $BEST_METRIC"
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
# Arm 1/4  eres2netv2 + no-b            ~2.3 h
##################################################################
if should_run eres2netv2-no-b; then
TAG=ixenc_eres2netv2-no-b_${METRIC}
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
    --interaction no-b \
    --batch-size "$BATCH" \
    --accumulate-steps "$ACCUM" \
    --lr "$LR" \
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
# Arm 2/4  eres2netv2 + bilinear        ~2.3 h
##################################################################
if should_run eres2netv2-bilinear; then
TAG=ixenc_eres2netv2-bilinear_${METRIC}
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
    --interaction bilinear \
    --bilinear-rank "$BILINEAR_RANK" \
    --batch-size "$BATCH" \
    --accumulate-steps "$ACCUM" \
    --lr "$LR" \
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
# Arm 3/4  eres2netv2-w24s4ep4 + no-b        ~5.0 h
##################################################################
if should_run w24s4ep4-no-b; then
TAG=ixenc_w24s4ep4-no-b_${METRIC}
echo ""
echo "##################################################################"
echo "# $TAG   ($(date))"
echo "##################################################################"

python finetune.py \
    --data-root "$DR" \
    --train-csv "$TRAIN_CSV" \
    --target-metric "$METRIC" \
    --outdir "egs/$TAG" \
    --encoder eres2netv2-w24s4ep4 \
    --head moe \
    --objective mse \
    --interaction no-b \
    --batch-size "$BATCH" \
    --accumulate-steps "$ACCUM" \
    --lr "$LR" \
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
# Arm 4/4  eres2netv2-w24s4ep4 + bilinear    ~5.0 h
##################################################################
if should_run w24s4ep4-bilinear; then
TAG=ixenc_w24s4ep4-bilinear_${METRIC}
echo ""
echo "##################################################################"
echo "# $TAG   ($(date))"
echo "##################################################################"

python finetune.py \
    --data-root "$DR" \
    --train-csv "$TRAIN_CSV" \
    --target-metric "$METRIC" \
    --outdir "egs/$TAG" \
    --encoder eres2netv2-w24s4ep4 \
    --head moe \
    --objective mse \
    --interaction bilinear \
    --bilinear-rank "$BILINEAR_RANK" \
    --batch-size "$BATCH" \
    --accumulate-steps "$ACCUM" \
    --lr "$LR" \
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
echo "All six metrics (the final ranking combines several of them, so do"
echo "not read a single column):"
echo ""
python - "$DEV_LABELS" "$METRIC" "$ARMS" <<'PY'
import csv, sys, os
import numpy as np, scipy.stats
from collections import defaultdict

labels, metric, arms = sys.argv[1], sys.argv[2], sys.argv[3].split()
gt = {}
for r in csv.DictReader(open(labels)):
    gt[(r["wav_a_path"], r["wav_b_path"])] = r

def score(path):
    ut, up = [], []
    st, sp = defaultdict(list), defaultdict(list)
    for r in csv.DictReader(open(path)):
        k = (r["wav_a_path"], r["wav_b_path"])
        if k not in gt or f"pred_{metric}" not in r:
            continue
        t, p, s = float(gt[k][metric]), float(r[f"pred_{metric}"]), gt[k]["system_id"]
        ut.append(t); up.append(p); st[s].append(t); sp[s].append(p)
    if len(ut) < 3:
        return None
    ut, up = np.array(ut), np.array(up)
    x = np.array([np.mean(st[s]) for s in st]); y = np.array([np.mean(sp[s]) for s in sp])
    return (len(ut), np.mean((ut-up)**2), scipy.stats.pearsonr(ut,up).statistic,
            scipy.stats.spearmanr(ut,up).statistic, np.mean((x-y)**2),
            scipy.stats.pearsonr(x,y).statistic, scipy.stats.spearmanr(x,y).statistic)

hdr = f"{'arm':<24}{'n':>6}{'uMSE':>8}{'uLCC':>8}{'uSRCC':>8}{'sMSE':>8}{'sLCC':>8}{'sSRCC':>8}"
print(f"A) selected checkpoint, scored by inference.py\n\n{metric}\n{hdr}\n{'-'*len(hdr)}")
for a in arms:
    f = f"egs/ixenc_{a}_{metric}/dev_{metric}.csv"
    if not os.path.exists(f):
        print(f"{a:<24}{'MISSING':>6}"); continue
    s = score(f)
    if s is None:
        print(f"{a:<24}{'too few rows':>6}"); continue
    print(f"{a:<24}{s[0]:>6}" + "".join(f"{v:>8.3f}" for v in s[1:]))

# The peak is the max of a noisy sequence, so the mean over the second half is reported
# beside it. On the ECAPA interaction ablation the two disagreed and the mean was right;
# under this recipe (freeze + backbone-lr-mult) runs are 2-3x quieter, sd 0.007-0.015.
hdr2 = f"{'arm':<24}{'peak@':>7}{'peak uSRCC':>12}{'mean uSRCC':>12}{'sd':>8}"
print(f"\n\nB) dev-log trajectory, steps >= half of training\n\n{metric}\n{hdr2}\n{'-'*len(hdr2)}")
for a in arms:
    log = f"egs/ixenc_{a}_{metric}/dev_log_{metric}.csv"
    if not os.path.exists(log):
        print(f"{a:<24}{'MISSING':>7}"); continue
    rows = [r for r in csv.DictReader(open(log)) if int(r["step"]) > 0
            and r["srcc_utt"] not in ("", "nan")]
    if not rows:
        print(f"{a:<24}{'no evals':>7}"); continue
    half = max(int(r["step"]) for r in rows) // 2
    v = np.array([float(r["srcc_utt"]) for r in rows if int(r["step"]) >= half])
    b = max(rows, key=lambda r: float(r["srcc_utt"]))
    print(f"{a:<24}{int(b['step']):>7}{float(b['srcc_utt']):>12.3f}{v.mean():>12.3f}{v.std():>8.3f}")
PY

echo ""
echo "CONTROLS -- same recipe, --interaction baseline, already on disk:"
echo "                         uMSE   uLCC  uSRCC    sMSE   sLCC  sSRCC   mean uSRCC"
echo "  eres2netv2  baseline  0.361  0.617  0.563   0.056  0.934  0.931       0.544"
echo "  w24s4ep4    baseline  0.355  0.604  0.572   0.048  0.940  0.924       0.557"
echo ""
echo "Best spk_sim on this dev set before this job: uSRCC 0.572, uMSE 0.355 (w24s4ep4"
echo "baseline); uLCC 0.617 (eres2netv2 baseline); sSRCC 0.974 (corn/ecapa ladder)."
echo ""
echo "TensorBoard (every arm appears as its own run):"
echo "  tensorboard --logdir $(pwd)/egs"

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""; echo "FAILURES: ${FAILED[*]}"
fi

ELAPSED=$((SECONDS - START_TIME))
echo ""
echo "Job $SLURM_JOB_ID finished at $(date) after $((ELAPSED / 3600))h $((ELAPSED % 3600 / 60))m"
