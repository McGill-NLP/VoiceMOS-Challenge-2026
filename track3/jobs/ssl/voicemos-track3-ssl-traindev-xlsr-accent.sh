#!/usr/bin/env bash
#SBATCH --job-name=voicemos-track3-ssl-traindev-xlsr-accent
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=david.guzman@mila.quebec

# SSL encoder in the DEEP pipeline, ACCENT similarity, XLS-R 300M read at layer 4.
#
#   sbatch track3/jobs/ssl/voicemos-track3-ssl-traindev-xlsr-accent.sh
#
# WHY THIS EXISTS. On frozen features the SSL bundles were the strongest weak learners by a
# margin -- `WAVLM_LARGE_l4__full__ridge` scores 0.602 dev UTT-SRCC on spk_sim, beating every
# fine-tuned speaker-ID model in ../../unified/ (best 0.569) and nearly matching the entire deep
# top-8 ensemble (0.579). Those numbers came from a ridge regression with no gradient path into
# the encoder. This grid asks the obvious follow-up: what do the same representations do when
# the backbone is allowed to move.
#
# The recipe is the train+dev factorial from ../strong/, with the encoder axis swapped:
#
#   encoder      xlsr-300m-l4   (fixed in this job)
#   objective    mse | coral
#   interaction  baseline | bilinear
#
# 4 cells here; ../strong/ covers the four speaker/accent-ID encoders under identical settings,
# so the two grids concatenate into one 7-encoder pool for ensembling.
#
# Held fixed, matching ../strong/ exactly: MoE head, freeze 5000/20000, --backbone-lr-mult 0.1,
# 20,000 steps, lr 1e-3, effective batch 16, --best-metric srcc_utt, fitted on
# sets/train_plus_dev.csv (3,400 pairs, 23 systems).
#
# THE STACK IS TRUNCATED AT LAYER 4. encoders.py deletes layers 5+ rather than computing and
# discarding them, which makes this a 63.5M-parameter backbone instead of the full bundle
# -- comparable to eres2netv2-w24s4ep4's 53.5M, and the reason a 315M-parameter model is
# affordable here at all. Outputs are identical to the untruncated model at atol 1e-5. Layer 4
# is the weak sweep's winner for every bundle and both targets; -l8 and the last layer are
# registered in ENCODER_REGISTRY if the ablation is ever worth extending.
#
# DEV IS IN THE FITTING SET, so --best-metric srcc_utt selects on an IN-SAMPLE score, dev
# predictions are written as dev_<metric>_<kind>_IN-SAMPLE.csv, and both checkpoints are kept
# with the final-step one the safer default downstream. Same three consequences spelled out in
# ../strong/; the member lists frozen in ../../weak/make_submission.py must not be re-derived
# from anything this job writes.
#
# EVALUATION IS ON THE LABELLED TEST SPLIT, which is the point of the run: test predictions are
# written inline for both checkpoints and scored in the summary against
# sets/vmc2026_track3_test_with_labels.csv. Scoring only -- nothing in the training loop reads
# those labels, and no member selection may either.
#
# MEASURED COST on an L40S at batch 16 x 1 (see voicemos-track3-ssl-sizing.sh):
#
#   0.383 s/step and 13.2 GiB at batch 16 x 1; 63.5M backbone, layer 4 of 24, dim 1024
#   training               4 arms x ~1.8h = ~7.1h
#   inference              4 arms x 4 passes x ~2 min = ~35 min
#   total                  ~7.8h, hence 10 h with slack.
#
# NOTHING IS OVERWRITTEN: checkpoints go to egs/ssl_runs_traindev/, a sibling of
# egs/ensemble_runs_traindev/, with the same <encoder>-<loss>-<interaction>_<metric> tags.
#
# RESUMABLE: an arm whose final checkpoint exists is skipped, as is any prediction file already
# on disk. Resubmit the same script after a timeout or preemption.
#
# To run one cell, or to split across shorter allocations:
#   sbatch --export=ALL,ARMS="xlsr-300m-l4:coral:bilinear" --time=03:00:00 \
#       track3/jobs/ssl/voicemos-track3-ssl-traindev-xlsr-accent.sh
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
python -c "import torch, torchaudio, speechbrain, coral_pytorch" \
    || { echo "ERROR: torch/torchaudio/speechbrain/coral_pytorch not importable"; exit 1; }
echo "python: $(which python)"

SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH=$SITE_PACKAGES/nvidia/npp/lib:$LD_LIBRARY_PATH

REPO=${REPO:-/home/mila/g/guzmand/scratch/Repositories/VoiceMOS-Challenge-2026}
cd "$REPO/track3/unified" || exit 1

python -c "
import torchaudio, glob
f = sorted(glob.glob('../baseline/data/vmc2026_track3_train_phase_distro_v3_syn/wav/*.wav'))[0]
torchaudio.load(f)
print('torchaudio.load OK')" || { echo "ERROR: torchaudio cannot load wavs"; exit 1; }

# The SSL weights come from torchaudio's own cache, not the encoder cache the sv: models use.
python -c "
from encoders import build_encoder
e = build_encoder('xlsr-300m-l4')
print('xlsr-300m-l4 ready, output_dim', e.output_dim)" \
    || { echo "ERROR: cannot build xlsr-300m-l4"; exit 1; }

echo "NVIDIA SMI:"; nvidia-smi
NUM_WORKERS=${SLURM_CPUS_PER_TASK:-8}

##################################################################
# Configuration
##################################################################
METRIC=acc_sim
ENC=xlsr-300m-l4

DR=../baseline/data/vmc2026_track3_train_phase_distro_v3_syn
EV=../baseline/data/vmc2026_track3_eval_phase_distro_v3_syn
TRAIN_CSV=${TRAIN_CSV:-$DR/sets/train_plus_dev.csv}
DEV_CSV=${DEV_CSV:-$DR/sets/dev.csv}
DEV_LABELS=${DEV_LABELS:-$EV/sets/dev_with_labels.csv}
TEST_CSV=${TEST_CSV:-$EV/sets/test.csv}
TEST_LABELS=${TEST_LABELS:-$EV/sets/vmc2026_track3_test_with_labels.csv}

ALL_ARMS="$ENC:mse:baseline $ENC:mse:bilinear $ENC:coral:baseline $ENC:coral:bilinear"
ARMS=${ARMS:-"$ALL_ARMS"}

BATCH=${BATCH:-16}
ACCUM=${ACCUM:-1}
LR=${LR:-1e-3}
BILINEAR_RANK=${BILINEAR_RANK:-64}
TRAIN_STEPS=${TRAIN_STEPS:-20000}
FREEZE_STEPS=${FREEZE_STEPS:-5000}
EVAL_STEPS=${EVAL_STEPS:-1000}
SAVE_STEPS=${SAVE_STEPS:-25000}
BEST_METRIC=${BEST_METRIC:-srcc_utt}

OUTROOT=${OUTROOT:-egs/ssl_runs_traindev}
mkdir -p "$OUTROOT"

for F in "$DEV_LABELS" "$TEST_CSV"; do
    [ -f "$F" ] || { echo "ERROR: missing $F"; exit 1; }
done

if [ ! -f "$TRAIN_CSV" ]; then
    echo "ERROR: $TRAIN_CSV missing. Build it with track3/jobs/strong/ or the weak train+dev job."
    exit 1
fi

# This job is the TRAIN+DEV variant; the file naming below would lie on a held-out fitting set.
python - "$TRAIN_CSV" "$DEV_LABELS" <<'PY' || exit 1
import csv, sys
tr = {(r["wav_a_path"], r["wav_b_path"]) for r in csv.DictReader(open(sys.argv[1]))}
dv = {(r["wav_a_path"], r["wav_b_path"]) for r in csv.DictReader(open(sys.argv[2]))}
n = len(tr & dv)
print(f"fitting set: {len(tr)} unique pairs, {n}/{len(dv)} dev pairs inside it")
if n != len(dv):
    raise SystemExit("ERROR: dev is not fully inside the fitting set.")
PY

echo "=================================================================="
echo "SSL deep grid, fitted on TRAIN + DEV   metric=$METRIC   encoder=$ENC"
echo "arms: $ARMS"
echo "batch ${BATCH}x${ACCUM} (effective 16), moe head, freeze $FREEZE_STEPS/$TRAIN_STEPS, lr=$LR"
echo "select: best $BEST_METRIC on dev -- IN-SAMPLE, dev is in the fitting set"
echo "score : $TEST_LABELS (held out, report only)"
echo "out   : $OUTROOT/<encoder>-<loss>-<interaction>_$METRIC/"
echo "=================================================================="

FAILED=()

##################################################################
# Grid
##################################################################
for ARM in $ARMS; do
    IFS=':' read -r E LOSS IX <<< "$ARM"
    TAG="${E}-${LOSS}-${IX}_${METRIC}"
    OUT="$OUTROOT/$TAG"
    echo ""
    echo "##################################################################"
    echo "# $TAG   ($(date))   batch ${BATCH}x${ACCUM}"
    echo "##################################################################"

    if [ -f "$OUT/finetuned_model_${METRIC}_final.pt" ]; then
        echo "already trained, skipping training (delete $OUT to force a rerun)"
    else
        python finetune.py \
            --data-root "$DR" \
            --train-csv "$TRAIN_CSV" \
            --target-metric "$METRIC" \
            --outdir "$OUT" \
            --encoder "$E" \
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
            --eval-batch-size "$BATCH" \
            --best-metric "$BEST_METRIC" \
            --dev-csv "$DEV_LABELS" \
            --dev-data-root "$DR" \
            --num-workers "$NUM_WORKERS"

        if [ $? -ne 0 ]; then echo "TRAINING FAILED for $TAG"; FAILED+=("$TAG:train"); continue; fi
    fi

    for KIND in best final; do
        case "$KIND" in
            best)  CKPT="$OUT/model_best_${METRIC}.pt" ;;
            final) CKPT="$OUT/finetuned_model_${METRIC}_final.pt" ;;
        esac
        [ -f "$CKPT" ] || { echo "NOTE: $CKPT missing"; FAILED+=("$TAG:$KIND:missing"); continue; }

        PRED="$OUT/dev_${METRIC}_${KIND}_IN-SAMPLE.csv"
        if [ -f "$PRED" ]; then echo "dev [$KIND] already predicted"; else
            python inference.py --data-root "$DR" --csv-path "$DEV_CSV" \
                --checkpoint "$CKPT" --out "$PRED"
            [ $? -ne 0 ] && { echo "DEV INFERENCE FAILED for $TAG/$KIND"; FAILED+=("$TAG:$KIND:dev"); rm -f "$PRED"; }
        fi

        TPRED="$OUT/test_${METRIC}_${KIND}.csv"
        if [ -f "$TPRED" ]; then echo "test [$KIND] already predicted"; else
            python inference.py --data-root "$EV" --csv-path "$TEST_CSV" \
                --checkpoint "$CKPT" --out "$TPRED"
            [ $? -ne 0 ] && { echo "TEST INFERENCE FAILED for $TAG/$KIND"; FAILED+=("$TAG:$KIND:test"); rm -f "$TPRED"; }
        fi
    done

    E2=$((SECONDS - START_TIME))
    echo "[$TAG done, $((E2 / 3600))h $((E2 % 3600 / 60))m into the job]"
done

##################################################################
# Summary
##################################################################
echo ""
echo "=================================================================="
echo "All six metrics per arm and checkpoint:"
echo ""
python - "$METRIC" "$OUTROOT" "$ARMS" \
    "dev|$DEV_LABELS|dev_{metric}_{kind}_IN-SAMPLE.csv|IN-SAMPLE -- dev is in the fitting set, this estimates nothing" \
    "test|$TEST_LABELS|test_{metric}_{kind}.csv|held out -- the evaluation this grid is for" <<'PY'
import csv, sys, os
import numpy as np, scipy.stats
from collections import defaultdict

metric, outroot, arms = sys.argv[1], sys.argv[2], sys.argv[3].split()

def score(path, gt):
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

for spec in sys.argv[4:]:
    name, labels, pattern, caveat = spec.split("|", 3)
    print(f"\n### {name}   ({caveat})")
    if not os.path.exists(labels):
        print(f"    no labels at {labels}, skipping"); continue
    gt = {(r["wav_a_path"], r["wav_b_path"]): r for r in csv.DictReader(open(labels))}
    hdr = (f"{'arm':<44}{'ckpt':>6}{'n':>6}{'uMSE':>8}{'uLCC':>8}{'uSRCC':>8}"
           f"{'sMSE':>8}{'sLCC':>8}{'sSRCC':>8}")
    print(f"{metric}\n{hdr}\n{'-'*len(hdr)}")
    rows = []
    for a in arms:
        enc, loss, ix = a.split(":")
        tag = f"{enc}-{loss}-{ix}_{metric}"
        for kind in ("best", "final"):
            f = f"{outroot}/{tag}/" + pattern.format(metric=metric, kind=kind)
            if not os.path.exists(f):
                print(f"{enc+'-'+loss+'-'+ix:<44}{kind:>6}{'MISSING':>9}"); continue
            s = score(f, gt)
            if s is None:
                print(f"{enc+'-'+loss+'-'+ix:<44}{kind:>6}{'no rows':>9}"); continue
            rows.append((f"{enc}-{loss}-{ix}", kind, s))
            print(f"{enc+'-'+loss+'-'+ix:<44}{kind:>6}{s[0]:>6}"
                  + "".join(f"{v:>8.3f}" for v in s[1:]))
    if rows:
        print(f"\nranked by UTT-SRCC ({name}):")
        for nm, kind, s in sorted(rows, key=lambda r: -r[2][3])[:10]:
            print(f"  {s[3]:.3f}  {nm} [{kind}]")
PY

echo ""
echo "Reference on the same test set (600 pairs), from the speaker/accent-ID grid:"
echo "  spk_sim  best single deep cell        uSRCC 0.588   deep top-8 ensemble 0.575"
echo "  acc_sim  best single deep cell        uSRCC 0.478   deep top-8 ensemble 0.478"
echo "  spk_sim  weak top-16 (frozen SSL)     uSRCC 0.606   + deep top-8  0.609"
echo "  acc_sim  weak top-16 (frozen SSL)     uSRCC 0.523   + deep top-8  0.528"
echo ""
echo "The weak rows are ridge regressions on THESE encoders' frozen features. A fine-tuned"
echo "cell below that lands under them is evidence that gradient fine-tuning is what hurts,"
echo "not the representation."
echo ""
echo "TensorBoard:  tensorboard --logdir $(pwd)/$OUTROOT"

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""; echo "FAILURES: ${FAILED[*]}"
fi

ELAPSED=$((SECONDS - START_TIME))
echo ""
echo "Job $SLURM_JOB_ID finished at $(date) after $((ELAPSED / 3600))h $((ELAPSED % 3600 / 60))m"
