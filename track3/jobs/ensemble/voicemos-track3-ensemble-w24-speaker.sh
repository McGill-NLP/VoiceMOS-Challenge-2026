#!/usr/bin/env bash
#SBATCH --job-name=voicemos-track3-ensemble-w24-speaker
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=david.guzman@mila.quebec

# Ensemble-candidate grid, SPEAKER similarity, the EXPENSIVE encoder.
#
#   sbatch track3/jobs/ensemble/voicemos-track3-ensemble-w24-speaker.sh
#
# Part of a 24-model grid split across four jobs so the expensive encoder does not block
# the cheap ones:
#
#   ensemble-small-speaker    commonaccent-ecapa + eres2netv2   x {mse,coral} x {baseline,bilinear}   8 arms
#   ensemble-w24-speaker      eres2netv2-w24s4ep4               x {mse,coral} x {baseline,bilinear}   4 arms
#   ensemble-small-accent     (same as small-speaker, acc_sim)                                        8 arms
#   ensemble-w24-accent       (same as w24-speaker, acc_sim)                                          4 arms
#
# Held fixed everywhere: MoE head, freeze 5000/20000, --backbone-lr-mult 0.1, 20,000 steps,
# lr 1e-3, effective batch 16, selection on srcc_utt.
#
# WHY THIS GRID. These 12 configurations per target are the ensemble candidate pool. The
# members will be chosen on the dev set -- which is honest here because dev is held out of
# training in this job -- and only the winners will then be retrained on train+dev. That
# ordering is what lets you see dev metrics before committing dev to training.
#
# CHECKPOINTS. Written under egs/ensemble_runs/, a new tree, so nothing already on disk is
# touched. Two per run, as requested:
#
#   model_best_<metric>.pt              best dev UTT-SRCC seen
#   finetuned_model_<metric>_final.pt   the last step (20,000)
#
# SAVE_STEPS is deliberately set beyond TRAIN_STEPS so no periodic checkpoints are written
# and exactly those two survive. If you would rather have crash-recovery points on the long
# runs, pass SAVE_STEPS=10000.
#
# MEASURED COST, from completed 20,000-step runs:
#   eres2netv2-w24s4ep4  bs4x4  0.89 s/step  ->  ~5.0 h per arm  (53.5M params, 29 GiB peak)
# So 4 x 5.0 = ~20 h plus inference. Hence 24 h -- the longest job of the four. Splitting
# it in two is reasonable if the queue is busy:
#
#   sbatch --export=ALL,ARMS="eres2netv2-w24s4ep4:mse:baseline eres2netv2-w24s4ep4:mse:bilinear" --time=12:00:00 ...
#   sbatch --export=ALL,ARMS="eres2netv2-w24s4ep4:coral:baseline eres2netv2-w24s4ep4:coral:bilinear" --time=12:00:00 ...
#
# If an arm OOMs (29.0 GiB of a 46 GiB card, peak follows the longest clip in the batch),
# halve the micro-batch:  --export=ALL,BATCH_OVERRIDE=2,ACCUM_OVERRIDE=8
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

# The TRAINING distribution is a valid data root for train and dev alike. (The eval distro
# also works now that the eval VCTK wavs have been merged into it, but this keeps the
# grid identical to every earlier run.)
DR=../baseline/data/vmc2026_track3_train_phase_distro_v3_syn
TRAIN_CSV=$DR/sets/train.csv
DEV_CSV=${DEV_CSV:-$DR/sets/dev.csv}
DEV_LABELS=${DEV_LABELS:-../baseline/data/vmc2026_track3_eval_phase_distro_v3_syn/sets/dev_with_labels.csv}

# encoder:objective:interaction, cheapest encoder first.
ALL_ARMS="eres2netv2-w24s4ep4:mse:baseline eres2netv2-w24s4ep4:mse:bilinear \
eres2netv2-w24s4ep4:coral:baseline eres2netv2-w24s4ep4:coral:bilinear"
ARMS=${ARMS:-"$ALL_ARMS"}

LR=${LR:-1e-3}
BILINEAR_RANK=${BILINEAR_RANK:-64}
TRAIN_STEPS=${TRAIN_STEPS:-20000}
FREEZE_STEPS=${FREEZE_STEPS:-5000}
EVAL_STEPS=${EVAL_STEPS:-1000}
# Above TRAIN_STEPS on purpose: keep only model_best and the final-step checkpoint.
SAVE_STEPS=${SAVE_STEPS:-25000}
BEST_METRIC=${BEST_METRIC:-srcc_utt}

OUTROOT=egs/ensemble_runs
mkdir -p "$OUTROOT"

if [ ! -f "$DEV_LABELS" ]; then
    echo "ERROR: no labelled dev set at $DEV_LABELS"; exit 1
fi

echo "=================================================================="
echo "ensemble candidate grid   metric=$METRIC"
echo "arms: $ARMS"
echo "held fixed: moe head, freeze $FREEZE_STEPS/$TRAIN_STEPS, backbone-lr-mult 0.1, lr=$LR"
echo "checkpoints -> $OUTROOT/<encoder>-<loss>-<interaction>_$METRIC/"
echo "keeping model_best_$METRIC.pt (best $BEST_METRIC) and finetuned_model_${METRIC}_final.pt"
echo "=================================================================="

FAILED=()

##################################################################
# Grid
#
# One loop over an explicit arm table rather than eight copies: the arms differ only in
# encoder / objective / interaction, so the command below IS what runs, with those three
# values substituted. Batch size is per-encoder because ERes2NetV2 OOMs at batch 16.
##################################################################
for ARM in $ARMS; do
    IFS=':' read -r ENC LOSS IX <<< "$ARM"
    case "$ENC" in
        eres2netv2|eres2netv2-w24s4ep4) BATCH=${BATCH_OVERRIDE:-4}; ACCUM=${ACCUM_OVERRIDE:-4} ;;
        *)                              BATCH=16; ACCUM=1 ;;
    esac

    TAG="${ENC}-${LOSS}-${IX}_${METRIC}"
    OUT="$OUTROOT/$TAG"
    echo ""
    echo "##################################################################"
    echo "# $TAG   ($(date))   batch ${BATCH}x${ACCUM}"
    echo "##################################################################"

    python finetune.py \
        --data-root "$DR" \
        --train-csv "$TRAIN_CSV" \
        --target-metric "$METRIC" \
        --outdir "$OUT" \
        --encoder "$ENC" \
        --head moe \
        --objective "$LOSS" \
        --interaction "$IX" \
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

    if [ $? -ne 0 ]; then echo "TRAINING FAILED for $TAG"; FAILED+=("$TAG:train"); continue; fi

    # Score BOTH kept checkpoints. model_best is the ensemble candidate; the final-step one
    # is kept so a later selection can prefer it, and because the two can differ by more
    # than the dev set can resolve.
    for KIND in best final; do
        case "$KIND" in
            best)  CKPT="$OUT/model_best_${METRIC}.pt" ;;
            final) CKPT="$OUT/finetuned_model_${METRIC}_final.pt" ;;
        esac
        if [ ! -f "$CKPT" ]; then
            echo "NOTE: $CKPT missing, skipping."; FAILED+=("$TAG:$KIND:missing"); continue
        fi
        PRED="$OUT/dev_${METRIC}_${KIND}.csv"
        rm -f "$PRED"
        python inference.py \
            --data-root "$DR" \
            --csv-path "$DEV_CSV" \
            --checkpoint "$CKPT" \
            --out "$PRED"
        if [ $? -ne 0 ]; then echo "INFERENCE FAILED for $TAG/$KIND"; FAILED+=("$TAG:$KIND"); continue; fi
        echo "--- $TAG [$KIND] against the official dev labels ---"
        python calculate_metrics.py --prediction-csv "$PRED" --ground-truth-csv "$DEV_LABELS"
    done

    E=$((SECONDS - START_TIME))
    echo "[$TAG done, $((E / 3600))h $((E % 3600 / 60))m into the job]"
done

##################################################################
# Summary
##################################################################
echo ""
echo "=================================================================="
echo "All six metrics per arm and checkpoint (the final ranking combines"
echo "several of them, so do not read a single column):"
echo ""
python - "$DEV_LABELS" "$METRIC" "$OUTROOT" "$ARMS" <<'PY'
import csv, sys, os
import numpy as np, scipy.stats
from collections import defaultdict

labels, metric, outroot, arms = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split()
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

hdr = f"{'arm':<44}{'ckpt':>6}{'n':>6}{'uMSE':>8}{'uLCC':>8}{'uSRCC':>8}{'sMSE':>8}{'sLCC':>8}{'sSRCC':>8}"
print(f"{metric}\n{hdr}\n{'-'*len(hdr)}")
rows = []
for a in arms:
    enc, loss, ix = a.split(":")
    tag = f"{enc}-{loss}-{ix}_{metric}"
    for kind in ("best", "final"):
        f = f"{outroot}/{tag}/dev_{metric}_{kind}.csv"
        if not os.path.exists(f):
            print(f"{enc+'-'+loss+'-'+ix:<44}{kind:>6}{'MISSING':>6}"); continue
        s = score(f)
        if s is None:
            print(f"{enc+'-'+loss+'-'+ix:<44}{kind:>6}{'no rows':>6}"); continue
        rows.append((f"{enc}-{loss}-{ix}", kind, s))
        print(f"{enc+'-'+loss+'-'+ix:<44}{kind:>6}{s[0]:>6}" + "".join(f"{v:>8.3f}" for v in s[1:]))

if rows:
    print(f"\nranked by UTT-SRCC:")
    for name, kind, s in sorted(rows, key=lambda r: -r[2][3])[:10]:
        print(f"  {s[3]:.3f}  {name} [{kind}]")
PY

echo ""
echo "Reference on this dev set (dev held out of training, as here):"
echo "  spk_sim  best single so far   uSRCC 0.572 (w24s4ep4 + mse + moe + baseline)"
echo "  spk_sim  top-16 ensemble      uSRCC 0.607"
echo ""
echo "TensorBoard:  tensorboard --logdir $(pwd)/$OUTROOT"

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""; echo "FAILURES: ${FAILED[*]}"
fi

ELAPSED=$((SECONDS - START_TIME))
echo ""
echo "Job $SLURM_JOB_ID finished at $(date) after $((ELAPSED / 3600))h $((ELAPSED % 3600 / 60))m"
