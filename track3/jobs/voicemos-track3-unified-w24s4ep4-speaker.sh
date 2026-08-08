#!/usr/bin/env bash
#SBATCH --job-name=voicemos-track3-unified-w24s4ep4-speaker
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=14:00:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=david.guzman@mila.quebec

# ERes2NetV2-w24s4ep4 against the plain ERes2NetV2 results, for SPEAKER similarity.
#
#   sbatch track3/jobs/voicemos-track3-unified-w24s4ep4-speaker.sh
#
# Two arms, identical except for the objective. Everything else matches the existing
# eres2netv2 + moe runs so the encoder is the only thing that changed:
#
#   coral   --encoder eres2netv2-w24s4ep4 --objective coral --head moe --interaction baseline
#   mse     --encoder eres2netv2-w24s4ep4 --objective mse   --head moe --interaction baseline
#
#   both at --freeze-steps 5000 --backbone-lr-mult 0.1 --train-steps 20000 --eval-steps 1000
#
# THE ENCODER IS 53.51M PARAMETERS, not the "~24M" the registry description claims -- that
# entry is wrong and this is the first job to use it. For scale, plain eres2netv2 is 17.86M.
# Measured here on an idle L40S with the MoE head and CORAL, warm-up excluded:
#
#   bs4x4  full-ft   1.065 s/step, peak 29.0 GiB
#   bs4x4  frozen    0.325 s/step, peak  1.4 GiB
#   dev evaluation, 600 pairs at batch 16: 17.0 s
#
# So 20,000 steps with 5,000 frozen is ~5.1 h per arm and ~10.1 h for both. Hence 14 h.
#
# MEMORY IS THE RISK HERE, not time. 29.0 GiB of a 46 GiB card leaves ~17 GiB spare, and
# peak follows the longest clip in the batch because repetitive padding stretches every
# clip up to it (clips run 2.5-9.0 s, median 4.5 s). The benchmark sampled 416 pairs so it
# very likely saw a long one, but if this OOMs, halve the micro-batch and double the
# accumulation to keep the effective batch at 16:
#
#   sbatch --export=ALL,BATCH=2,ACCUM=8 track3/jobs/voicemos-track3-unified-w24s4ep4-speaker.sh
#
# To split the two arms across jobs:
#
#   sbatch --export=ALL,ARMS=coral --time=07:00:00 ...
#   sbatch --export=ALL,ARMS=mse   --time=07:00:00 ...
#
# Selection is on srcc_utt, and --save-steps is 2000 rather than 5000: the CORAL runs
# peaked at steps 16000-19000 and those checkpoints were not kept.
#
# Deliberately NOT using `set -e`: if one arm fails the other should still run.

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

# The w24s4ep4 checkpoint comes from ModelScope with NO Hugging Face mirror in the
# registry, unlike plain eres2netv2. Fetch it before the clock starts on training, so a
# network problem fails in seconds rather than after the first arm.
python -c "
from encoders import build_encoder
enc = build_encoder('eres2netv2-w24s4ep4')
n = sum(p.numel() for p in enc.parameters())
print(f'eres2netv2-w24s4ep4 ready: {n/1e6:.2f}M params, output_dim {enc.output_dim}')" \
    || { echo "ERROR: could not build eres2netv2-w24s4ep4"; exit 1; }

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

ARMS=${ARMS:-"coral mse"}

# batch 4 x 4 accumulation for an effective batch of 16, matching every other run. See the
# memory note in the header before changing these.
BATCH=${BATCH:-4}
ACCUM=${ACCUM:-4}
LR=${LR:-1e-3}
TRAIN_STEPS=${TRAIN_STEPS:-20000}
FREEZE_STEPS=${FREEZE_STEPS:-5000}
EVAL_STEPS=${EVAL_STEPS:-1000}
SAVE_STEPS=${SAVE_STEPS:-2000}
BEST_METRIC=${BEST_METRIC:-srcc_utt}

if [ ! -f "$DEV_LABELS" ]; then
    echo "ERROR: no labelled dev set at $DEV_LABELS"; exit 1
fi

echo "=================================================================="
echo "eres2netv2-w24s4ep4 + MoE + baseline interaction   metric=$METRIC"
echo "arms: $ARMS"
echo "steps=$TRAIN_STEPS  freeze=$FREEZE_STEPS  batch=${BATCH}x${ACCUM}  lr=$LR"
echo "eval every $EVAL_STEPS  save every $SAVE_STEPS  select on $BEST_METRIC"
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
# Arm 1/2  CORAL      ~5.1 h
##################################################################
if should_run coral; then
TAG=w24s4ep4_coral_${METRIC}
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
    --objective coral \
    --interaction baseline \
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
# Arm 2/2  MSE        ~5.1 h
##################################################################
if should_run mse; then
TAG=w24s4ep4_mse_${METRIC}
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
    --interaction baseline \
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

hdr = f"{'arm':<10}{'n':>6}{'uMSE':>8}{'uLCC':>8}{'uSRCC':>8}{'sMSE':>8}{'sLCC':>8}{'sSRCC':>8}"
print(f"A) selected checkpoint, scored by inference.py\n\n{metric}\n{hdr}\n{'-'*len(hdr)}")
for a in arms:
    f = f"egs/w24s4ep4_{a}_{metric}/dev_{metric}.csv"
    if not os.path.exists(f):
        print(f"{a:<10}{'MISSING':>6}"); continue
    s = score(f)
    print(f"{a:<10}{s[0]:>6}" + "".join(f"{v:>8.3f}" for v in s[1:]) if s
          else f"{a:<10}{'too few rows':>6}")

# The peak is the max of a noisy sequence -- successive dev evaluations of the SAME run
# swing by ~0.025 sd on this task -- so the mean over the second half is reported next to
# it. Read the mean when the two disagree.
hdr2 = f"{'arm':<10}{'peak@':>7}{'peak uSRCC':>12}{'mean uSRCC':>12}{'sd':>8}"
print(f"\n\nB) dev-log trajectory, steps >= half of training\n\n{metric}\n{hdr2}\n{'-'*len(hdr2)}")
for a in arms:
    log = f"egs/w24s4ep4_{a}_{metric}/dev_log_{metric}.csv"
    if not os.path.exists(log):
        print(f"{a:<10}{'MISSING':>7}"); continue
    rows = [r for r in csv.DictReader(open(log)) if int(r["step"]) > 0
            and r["srcc_utt"] not in ("", "nan")]
    if not rows:
        print(f"{a:<10}{'no evals':>7}"); continue
    half = max(int(r["step"]) for r in rows) // 2
    tail = [r for r in rows if int(r["step"]) >= half]
    v = np.array([float(r["srcc_utt"]) for r in tail])
    b = max(rows, key=lambda r: float(r["srcc_utt"]))
    print(f"{a:<10}{int(b['step']):>7}{float(b['srcc_utt']):>12.3f}{v.mean():>12.3f}{v.std():>8.3f}")
PY

echo ""
echo "Plain eres2netv2 + moe, same head/schedule/steps, selected checkpoint:"
echo "            uMSE   uLCC  uSRCC    sMSE   sLCC  sSRCC"
echo "  coral    0.452  0.540  0.479   0.068  0.929  0.935   (selected on srcc_sys)"
echo "  mse      0.361  0.617  0.563   0.056  0.934  0.931"
echo "The CORAL row is not selection-matched; its srcc_utt-optimal dev-log point was"
echo "0.562 at step 16000, whose checkpoint was not saved."
echo ""
echo "TensorBoard (every arm appears as its own run):"
echo "  tensorboard --logdir $(pwd)/egs"

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""; echo "FAILURES: ${FAILED[*]}"
fi

ELAPSED=$((SECONDS - START_TIME))
echo ""
echo "Job $SLURM_JOB_ID finished at $(date) after $((ELAPSED / 3600))h $((ELAPSED % 3600 / 60))m"
