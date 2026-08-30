#!/usr/bin/env bash
#SBATCH --job-name=voicemos-track3-strong-traindev-small-accent
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=david.guzman@mila.quebec

# The strong-learner grid refit on TRAIN + DEV, ACCENT similarity, the three cheaper encoders.
#
#   sbatch track3/jobs/strong/voicemos-track3-strong-traindev-small-accent.sh
#
# Same factorial as track3/jobs/ensemble/, axis for axis:
#
#   encoder      ecapa-voxceleb | commonaccent-ecapa | eres2netv2 | eres2netv2-w24s4ep4
#   objective    mse | coral
#   interaction  baseline | bilinear
#
# 4 x 2 x 2 = 16 cells per target. This file runs 12 of them for acc_sim;
# voicemos-track3-strong-traindev-w24-accent.sh runs the other 4.
#
# THE ONLY CHANGE FROM THE ensemble/ JOBS IS THE FITTING SET: sets/train_plus_dev.csv instead
# of sets/train.csv, i.e. 2,800 -> 3,400 unique pairs and 21 -> 23 systems, which the challenge
# permits. Head (moe), freeze schedule, backbone-lr-mult, steps, lr, effective batch and
# --best-metric srcc_utt are all held at the values the train-only grid used, so a cell here and
# the same cell in egs/ensemble_runs/ differ in exactly one thing.
#
# THERE IS NO HELD-OUT SET LEFT, and three things follow:
#
#   1. DEV IS IN THE FITTING SET. Selection still happens on dev UTT-SRCC -- --best-metric
#      srcc_utt, unchanged -- but that number is now IN-SAMPLE and estimates nothing. The dev
#      prediction files are named dev_<metric>_<kind>_IN-SAMPLE.csv for that reason. For the
#      scale of the illusion, see the weak-learner train+dev job: the same pool reads 0.875
#      in-sample against a true held-out 0.622.
#
#   2. BOTH CHECKPOINTS ARE KEPT, and the final-step one is the safer default downstream.
#      model_best is now the step that fit dev hardest, which is the failure mode that
#      make_submission.py exists to avoid; finetuned_model_<metric>_final.pt is untouched by
#      in-sample selection.
#
#   3. DO NOT RE-RANK ENSEMBLE MEMBERS ON THESE DEV NUMBERS. Member lists were chosen on
#      held-out dev by the train-only grid and are frozen in ../../weak/make_submission.py.
#      Re-ranking on in-sample scores picks whichever cell memorised hardest.
#
# TEST PREDICTIONS ARE WRITTEN IN THIS JOB, right after each arm trains, for both kept
# checkpoints. The train-only grid needed a separate voicemos-track3-deep-test-inference.sh
# pass because it stopped at dev; doing it inline costs ~2 min per checkpoint here and means a
# finished arm is immediately usable by make_submission.py. Test wavs live under the EVAL
# distribution, so those two calls pass a different --data-root than the dev ones.
#
# TEST SCORING AT THE END IS REPORT-ONLY. sets/vmc2026_track3_test_with_labels.csv is on disk,
# so the summary prints held-out test metrics when it is present. That table is the only honest
# number this job produces -- and it must never feed selection, ensembling or early stopping,
# or the last untouched set is gone too. Nothing in the loop above reads it.
#
# NOTHING IS OVERWRITTEN. Checkpoints go to egs/ensemble_runs_traindev/, a sibling of the
# train-only egs/ensemble_runs/, with identical <encoder>-<loss>-<interaction>_<metric> tags so
# the two grids can be diffed cell by cell.
#
# RESUMABLE. An arm whose final checkpoint already exists is skipped, and so is any prediction
# file already on disk, so a preemption or a timeout costs only the arm in flight -- just
# resubmit the same script.
#
# MEASURED COST. Training is a fixed 20,000 steps, so an arm costs what it cost on the
# train-only grid despite the 21% larger fitting set:
#
#   ecapa-voxceleb       4 arms x ~50 min   = ~3h20
#   commonaccent-ecapa   4 arms x ~47 min   = ~3h10
#   eres2netv2           4 arms x ~2h20     = ~9h20   (batch 4x4; it OOMs at 16)
#   inference            12 arms x 4 passes x ~2 min = ~1h40
#   total                                   ~17h30, hence 24 h with slack for a slow node.
#
# To run one cell, or to split this job across shorter allocations:
#   sbatch --export=ALL,ARMS="eres2netv2:coral:bilinear" --time=04:00:00 \
#       track3/jobs/strong/voicemos-track3-strong-traindev-small-accent.sh
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
METRIC=acc_sim

# Train and dev wavs both resolve under the TRAINING distribution; test wavs do not, and come
# from the eval one.
DR=../baseline/data/vmc2026_track3_train_phase_distro_v3_syn
EV=../baseline/data/vmc2026_track3_eval_phase_distro_v3_syn
TRAIN_CSV=${TRAIN_CSV:-$DR/sets/train_plus_dev.csv}
DEV_CSV=${DEV_CSV:-$DR/sets/dev.csv}
DEV_LABELS=${DEV_LABELS:-$EV/sets/dev_with_labels.csv}
TEST_CSV=${TEST_CSV:-$EV/sets/test.csv}
TEST_LABELS=${TEST_LABELS:-$EV/sets/vmc2026_track3_test_with_labels.csv}

# encoder:objective:interaction, cheapest encoder first.
ALL_ARMS="ecapa-voxceleb:mse:baseline ecapa-voxceleb:mse:bilinear \
ecapa-voxceleb:coral:baseline ecapa-voxceleb:coral:bilinear \
commonaccent-ecapa:mse:baseline commonaccent-ecapa:mse:bilinear \
commonaccent-ecapa:coral:baseline commonaccent-ecapa:coral:bilinear \
eres2netv2:mse:baseline eres2netv2:mse:bilinear \
eres2netv2:coral:baseline eres2netv2:coral:bilinear"
ARMS=${ARMS:-"$ALL_ARMS"}

LR=${LR:-1e-3}
BILINEAR_RANK=${BILINEAR_RANK:-64}
TRAIN_STEPS=${TRAIN_STEPS:-20000}
FREEZE_STEPS=${FREEZE_STEPS:-5000}
EVAL_STEPS=${EVAL_STEPS:-1000}
# Above TRAIN_STEPS on purpose: keep only model_best and the final-step checkpoint.
SAVE_STEPS=${SAVE_STEPS:-25000}
BEST_METRIC=${BEST_METRIC:-srcc_utt}

OUTROOT=${OUTROOT:-egs/ensemble_runs_traindev}
mkdir -p "$OUTROOT"

if [ ! -f "$DEV_LABELS" ]; then
    echo "ERROR: no labelled dev set at $DEV_LABELS"; exit 1
fi
if [ ! -f "$TEST_CSV" ]; then
    echo "ERROR: no test csv at $TEST_CSV"; exit 1
fi

# sets/train_plus_dev.csv concatenates train.csv and dev_with_labels.csv, with an empty
# listener_id on the dev rows. Build it if it is missing; it needs no new audio, and every dev
# wav already resolves under the training distribution's root. Same builder as
# track3/jobs/weak/voicemos-track3-weak-ensemble-traindev.sh, so both halves of the system are
# fitted on byte-identical data.
if [ ! -f "$TRAIN_CSV" ]; then
    echo "NOTE: $TRAIN_CSV missing, building it"
    python - "$TRAIN_CSV" "$DR" "$EV" <<'PY'
import csv, sys
out, TR, EV = sys.argv[1], sys.argv[2], sys.argv[3]
HDR = ["system_id", "utterance_id", "listener_id", "wav_a_path", "wav_b_path",
       "spk_sim", "acc_sim"]
n_tr = n_dev = 0
with open(out, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=HDR); w.writeheader()
    for r in csv.DictReader(open(f"{TR}/sets/train.csv")):
        w.writerow({k: r.get(k, "") for k in HDR}); n_tr += 1
    for r in csv.DictReader(open(f"{EV}/sets/dev_with_labels.csv")):
        row = {k: r.get(k, "") for k in HDR}; row["listener_id"] = ""
        w.writerow(row); n_dev += 1
print(f"wrote {out}: {n_tr} train rows + {n_dev} dev rows")
PY
    [ -f "$TRAIN_CSV" ] || { echo "ERROR: could not build $TRAIN_CSV"; exit 1; }
fi

# Guard against pointing this job at a train-only CSV by accident: the whole design below
# assumes dev is inside the fitting set, and the file naming would then lie.
python - "$TRAIN_CSV" "$DEV_LABELS" <<'PY' || exit 1
import csv, sys
tr = {(r["wav_a_path"], r["wav_b_path"]) for r in csv.DictReader(open(sys.argv[1]))}
dv = {(r["wav_a_path"], r["wav_b_path"]) for r in csv.DictReader(open(sys.argv[2]))}
n = len(tr & dv)
print(f"fitting set: {len(tr)} unique pairs, {n}/{len(dv)} dev pairs inside it")
if n != len(dv):
    raise SystemExit(
        "ERROR: dev is not fully inside the fitting set. This job is the TRAIN+DEV variant; "
        "for a held-out run use track3/jobs/ensemble/ instead."
    )
PY

echo "=================================================================="
echo "strong grid, fitted on TRAIN + DEV   metric=$METRIC"
echo "arms: $ARMS"
echo "train : $TRAIN_CSV"
echo "held fixed: moe head, freeze $FREEZE_STEPS/$TRAIN_STEPS, backbone-lr-mult 0.1, lr=$LR"
echo "select: best $BEST_METRIC on dev -- IN-SAMPLE, dev is in the fitting set"
echo "out   : $OUTROOT/<encoder>-<loss>-<interaction>_$METRIC/"
echo "        model_best_$METRIC.pt + finetuned_model_${METRIC}_final.pt"
echo "        dev_${METRIC}_<kind>_IN-SAMPLE.csv + test_${METRIC}_<kind>.csv"
echo "=================================================================="

FAILED=()

##################################################################
# Grid
#
# One loop over an explicit arm table rather than 12 copies: the arms differ only in
# encoder / objective / interaction, so the command below IS what runs, with those three
# values substituted. Batch size is per-encoder because ERes2NetV2 OOMs at batch 16.
##################################################################
for ARM in $ARMS; do
    IFS=':' read -r ENC LOSS IX <<< "$ARM"
    case "$ENC" in
        eres2netv2|eres2netv2-w24s4ep4) BATCH=4; ACCUM=4 ;;   # effective batch 16
        *)                              BATCH=16; ACCUM=1 ;;
    esac

    TAG="${ENC}-${LOSS}-${IX}_${METRIC}"
    OUT="$OUTROOT/$TAG"
    echo ""
    echo "##################################################################"
    echo "# $TAG   ($(date))   batch ${BATCH}x${ACCUM}"
    echo "##################################################################"

    # The final-step checkpoint is written last, so its presence means the arm finished.
    if [ -f "$OUT/finetuned_model_${METRIC}_final.pt" ]; then
        echo "already trained, skipping training (delete $OUT to force a rerun)"
    else
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
    fi

    # Predict with BOTH kept checkpoints, on dev and on test. The dev pass is in-sample and is
    # kept only as a sanity trace, hence the _IN-SAMPLE suffix; the test pass is what
    # ../../weak/make_submission.py consumes.
    for KIND in best final; do
        case "$KIND" in
            best)  CKPT="$OUT/model_best_${METRIC}.pt" ;;
            final) CKPT="$OUT/finetuned_model_${METRIC}_final.pt" ;;
        esac
        if [ ! -f "$CKPT" ]; then
            echo "NOTE: $CKPT missing, skipping."; FAILED+=("$TAG:$KIND:missing"); continue
        fi

        PRED="$OUT/dev_${METRIC}_${KIND}_IN-SAMPLE.csv"
        if [ -f "$PRED" ]; then
            echo "dev [$KIND] already predicted"
        else
            python inference.py \
                --data-root "$DR" \
                --csv-path "$DEV_CSV" \
                --checkpoint "$CKPT" \
                --out "$PRED"
            if [ $? -ne 0 ]; then
                echo "DEV INFERENCE FAILED for $TAG/$KIND"; FAILED+=("$TAG:$KIND:dev"); rm -f "$PRED"
            fi
        fi

        TPRED="$OUT/test_${METRIC}_${KIND}.csv"
        if [ -f "$TPRED" ]; then
            echo "test [$KIND] already predicted"
        else
            python inference.py \
                --data-root "$EV" \
                --csv-path "$TEST_CSV" \
                --checkpoint "$CKPT" \
                --out "$TPRED"
            if [ $? -ne 0 ]; then
                echo "TEST INFERENCE FAILED for $TAG/$KIND"; FAILED+=("$TAG:$KIND:test"); rm -f "$TPRED"
            fi
        fi
    done

    E=$((SECONDS - START_TIME))
    echo "[$TAG done, $((E / 3600))h $((E % 3600 / 60))m into the job]"
done

##################################################################
# Summary
#
# Two tables. The dev one is in-sample and is printed only so a broken arm is visible; the test
# one is the held-out reference, and is report-only -- nothing above reads the test labels.
##################################################################
echo ""
echo "=================================================================="
echo "All six metrics per arm and checkpoint (the final ranking combines"
echo "several of them, so do not read a single column):"
echo ""
python - "$METRIC" "$OUTROOT" "$ARMS" \
    "dev|$DEV_LABELS|dev_{metric}_{kind}_IN-SAMPLE.csv|IN-SAMPLE -- dev is in the fitting set, this estimates nothing" \
    "test|$TEST_LABELS|test_{metric}_{kind}.csv|held out -- report only, never select on this" <<'PY'
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
    gt = {}
    for r in csv.DictReader(open(labels)):
        gt[(r["wav_a_path"], r["wav_b_path"])] = r
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
echo "Reference from the train-only grid, on held-out dev:"
echo "  acc_sim  best single, 16-cell grid   uSRCC 0.552 (w24s4ep4 + coral + bilinear)"
echo "  acc_sim  top-8 grid ensemble         uSRCC 0.563"
echo "  acc_sim  heterogeneous top-8         uSRCC 0.579"
echo ""
echo "Those dev figures are NOT comparable to the dev table above, which is in-sample. The"
echo "test table is; a train+dev cell should land at or slightly above its train-only twin."
echo ""
echo "TensorBoard:  tensorboard --logdir $(pwd)/$OUTROOT"

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""; echo "FAILURES: ${FAILED[*]}"
fi

ELAPSED=$((SECONDS - START_TIME))
echo ""
echo "Job $SLURM_JOB_ID finished at $(date) after $((ELAPSED / 3600))h $((ELAPSED % 3600 / 60))m"
