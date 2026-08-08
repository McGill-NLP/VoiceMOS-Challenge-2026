#!/usr/bin/env bash
#SBATCH --job-name=voicemos-track3-unified-listener-accent
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=david.guzman@mila.quebec

# Listener-bias modelling for ACCENT similarity (acc_sim).
#
#   sbatch track3/jobs/voicemos-track3-unified-listener-accent.sh
#
# Two arms. Everything is held at the configuration that produced the current records --
# eres2netv2, MSE, MoE, baseline interaction, freeze 5000/20000, --backbone-lr-mult 0.1,
# 20,000 steps -- and the ONLY difference is --use-listener-bias. The `control` arm is a
# rerun of that exact configuration rather than a quote from an old log, so the comparison
# is paired and unaffected by anything that has changed in the code since.
#
#   control    per-pair mean targets, one head          ~2.3 h
#   listener   listener-wise targets, consensus + bias  ~2.3 h
#
# WHY THIS AND NOT ANOTHER HEAD TWEAK. Measured on the labelled training ratings: two
# listeners agree with two others at only SRCC 0.369 on this target -- lower than the 0.454
# for speaker -- so the reliability of a 5-listener mean is ~0.594 and a PERFECT model would
# score UTT-SRCC ~0.770 against it. The current best is 0.547, about 71% of that. Accent is
# the noisier of the two targets, so it has the most to gain here. Every head, interaction and objective
# variant tried so far has moved the number by less than the dev set can resolve
# (bootstrap 95% CI on 600 pairs is +-0.05). Listener modelling is the one remaining idea
# that attacks the label noise itself rather than working underneath it: the bias head
# absorbs "this rater runs high", so the consensus head can fit signal instead of
# disagreement. It was UTMOS's single largest ablation effect.
#
# NO LISTENER COLUMN IS NEEDED AT INFERENCE. The model has two heads on the same
# interaction vector: a listener-independent consensus head, and a bias head that takes
# [interaction, listener embedding]. Training supervises consensus+bias against the
# individual rating and the consensus alone against the pair mean. inference.py passes no
# listener, so only the consensus head runs -- which is what makes this usable on test.csv.
# --listener-dropout 0.5 zeroes the listener identity half the time during training, so
# the consensus head is regularly forced to work standalone, exactly as at inference.
#
# A STEP MEANS SOMETHING DIFFERENT IN THE LISTENER ARM. It trains on the 13,687
# listener-wise rows rather than 2,800 averaged pairs, so 20,000 steps at batch 4 x 4 is
# ~23 epochs against the control's ~114. That is a confound in this comparison and it
# cannot be removed without changing one of the two things being compared -- if the
# listener arm wins, more-epochs-of-noisier-targets is an alternative explanation worth
# ruling out with a second control at matched epochs.
#
# Cost from the completed 20,000-step runs: eres2netv2 + moe is 0.42 s/step, ~2.3 h per
# arm, ~4.7 h for both. To run one arm alone:
#
#   sbatch --export=ALL,ARMS=listener --time=03:30:00 ...
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

echo "NVIDIA SMI:"; nvidia-smi
NUM_WORKERS=${SLURM_CPUS_PER_TASK:-8}

##################################################################
# Configuration
##################################################################
METRIC=acc_sim

# The TRAINING distribution is the right data root for train and dev alike: the eval
# distro is missing all 600 sys019 reference wavs, so inference against it drops every row.
DR=../baseline/data/vmc2026_track3_train_phase_distro_v3_syn
# sets/train.csv is the LISTENER-WISE file -- one row per listener per pair. The listener
# arm needs it; a pre-averaged split would make --use-listener-bias fail loudly.
TRAIN_CSV=$DR/sets/train.csv
DEV_CSV=${DEV_CSV:-$DR/sets/dev.csv}
DEV_LABELS=${DEV_LABELS:-../baseline/data/vmc2026_track3_eval_phase_distro_v3_syn/sets/dev_with_labels.csv}

ARMS=${ARMS:-"control listener"}

BATCH=${BATCH:-4}
ACCUM=${ACCUM:-4}
LR=${LR:-1e-3}
TRAIN_STEPS=${TRAIN_STEPS:-20000}
FREEZE_STEPS=${FREEZE_STEPS:-5000}
EVAL_STEPS=${EVAL_STEPS:-1000}
SAVE_STEPS=${SAVE_STEPS:-2000}
BEST_METRIC=${BEST_METRIC:-srcc_utt}
LISTENER_DROPOUT=${LISTENER_DROPOUT:-0.5}
LAMBDA_MEAN=${LAMBDA_MEAN:-1.0}

if [ ! -f "$DEV_LABELS" ]; then
    echo "ERROR: no labelled dev set at $DEV_LABELS"; exit 1
fi

echo "=================================================================="
echo "listener-bias vs control   metric=$METRIC"
echo "arms: $ARMS"
echo "held fixed: eres2netv2 + mse + moe + baseline interaction,"
echo "            freeze $FREEZE_STEPS/$TRAIN_STEPS, backbone-lr-mult 0.1, batch ${BATCH}x${ACCUM}"
echo "listener arm: dropout=$LISTENER_DROPOUT  lambda_mean=$LAMBDA_MEAN"
echo "select on $BEST_METRIC"
echo "=================================================================="

mkdir -p egs
FAILED=()

should_run () { [[ " $ARMS " == *" $1 "* ]]; }

# Shared post-training steps. Note inference.py passes NO listener, so the listener arm is
# scored through its consensus head -- the same path the test set would use.
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
# Arm 1/2  control -- per-pair mean targets, no listener head     ~2.3 h
##################################################################
if should_run control; then
TAG=lb_control_${METRIC}
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
# Arm 2/2  listener bias -- listener-wise targets, consensus + bias  ~2.3 h
##################################################################
if should_run listener; then
TAG=lb_listener_${METRIC}
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
    --interaction baseline \
    --use-listener-bias \
    --listener-dropout "$LISTENER_DROPOUT" \
    --lambda-mean "$LAMBDA_MEAN" \
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
echo "All six metrics, plus a bootstrap CI on the paired difference:"
echo ""
python - "$DEV_LABELS" "$METRIC" "$ARMS" <<'PY'
import csv, sys, os
import numpy as np, scipy.stats
from collections import defaultdict

labels, metric, arms = sys.argv[1], sys.argv[2], sys.argv[3].split()
gt = {}
for r in csv.DictReader(open(labels)):
    gt[(r["wav_a_path"], r["wav_b_path"])] = r

def load(path):
    t, p, s = [], [], []
    for r in csv.DictReader(open(path)):
        k = (r["wav_a_path"], r["wav_b_path"])
        if k not in gt or f"pred_{metric}" not in r:
            continue
        t.append(float(gt[k][metric])); p.append(float(r[f"pred_{metric}"]))
        s.append(gt[k]["system_id"])
    return np.array(t), np.array(p), np.array(s)

def six(t, p, s):
    st, sp = defaultdict(list), defaultdict(list)
    for k, a, b in zip(s, t, p):
        st[k].append(a); sp[k].append(b)
    x = np.array([np.mean(st[k]) for k in st]); y = np.array([np.mean(sp[k]) for k in st])
    return (np.mean((t-p)**2), scipy.stats.pearsonr(t,p).statistic,
            scipy.stats.spearmanr(t,p).statistic, np.mean((x-y)**2),
            scipy.stats.pearsonr(x,y).statistic, scipy.stats.spearmanr(x,y).statistic)

hdr = f"{'arm':<12}{'n':>6}{'uMSE':>8}{'uLCC':>8}{'uSRCC':>8}{'sMSE':>8}{'sLCC':>8}{'sSRCC':>8}"
print(f"A) selected checkpoint, scored by inference.py\n\n{metric}\n{hdr}\n{'-'*len(hdr)}")
preds = {}
for a in arms:
    f = f"egs/lb_{a}_{metric}/dev_{metric}.csv"
    if not os.path.exists(f):
        print(f"{a:<12}{'MISSING':>6}"); continue
    t, p, s = load(f)
    preds[a] = (t, p, s)
    print(f"{a:<12}{len(t):>6}" + "".join(f"{v:>8.3f}" for v in six(t, p, s)))

# 600 dev pairs give a 95% CI of about +-0.05 on a single UTT-SRCC, so an unpaired
# comparison of two numbers cannot resolve anything smaller. The paired bootstrap is
# tighter and is what decides whether the listener arm actually helped.
if "control" in preds and "listener" in preds:
    t, pc, _ = preds["control"]; _, pl, _ = preds["listener"]
    rng = np.random.default_rng(0); n = len(t); d = []
    for _ in range(4000):
        i = rng.integers(0, n, n)
        d.append(scipy.stats.spearmanr(t[i], pl[i]).statistic
                 - scipy.stats.spearmanr(t[i], pc[i]).statistic)
    d = np.array(d); lo, hi = np.percentile(d, [2.5, 97.5])
    verdict = "SIGNIFICANT" if lo * hi > 0 else "not significant"
    print(f"\nlistener - control, UTT-SRCC: {d.mean():+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  -> {verdict}")

hdr2 = f"{'arm':<12}{'peak@':>7}{'peak uSRCC':>12}{'mean uSRCC':>12}{'sd':>8}"
print(f"\n\nB) dev-log trajectory, steps >= half of training\n\n{metric}\n{hdr2}\n{'-'*len(hdr2)}")
for a in arms:
    log = f"egs/lb_{a}_{metric}/dev_log_{metric}.csv"
    if not os.path.exists(log):
        print(f"{a:<12}{'MISSING':>7}"); continue
    rows = [r for r in csv.DictReader(open(log)) if int(r["step"]) > 0
            and r["srcc_utt"] not in ("", "nan")]
    half = max(int(r["step"]) for r in rows) // 2
    v = np.array([float(r["srcc_utt"]) for r in rows if int(r["step"]) >= half])
    b = max(rows, key=lambda r: float(r["srcc_utt"]))
    print(f"{a:<12}{int(b['step']):>7}{float(b['srcc_utt']):>12.3f}{v.mean():>12.3f}{v.std():>8.3f}")
PY

echo ""
echo "Context for acc_sim on this dev set:"
echo "  Baseline 2 published                       uSRCC 0.440"
echo "  best so far (w24s4ep4 + coral + moe)       uSRCC 0.547"
echo "  label-noise ceiling for a perfect model    uSRCC ~0.770"
echo ""
echo "TensorBoard (every arm appears as its own run):"
echo "  tensorboard --logdir $(pwd)/egs"

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""; echo "FAILURES: ${FAILED[*]}"
fi

ELAPSED=$((SECONDS - START_TIME))
echo ""
echo "Job $SLURM_JOB_ID finished at $(date) after $((ELAPSED / 3600))h $((ELAPSED % 3600 / 60))m"
