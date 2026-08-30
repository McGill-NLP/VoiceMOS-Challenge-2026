#!/usr/bin/env bash
#SBATCH --job-name=voicemos-track3-ssl-layer-probe-wavlm-base
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=09:00:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=david.guzman@mila.quebec

# LAYER PROBE for WavLM Base+: which transformer layer should the full SSL grid read?
#
#   sbatch track3/jobs/ssl/voicemos-track3-ssl-layer-probe-wavlm-base.sh
#
# THE QUESTION. On FROZEN features layer 4 wins for every bundle on spk_sim and for two of the
# three on acc_sim, with the last layer far behind. For this bundle, ridge scores
# l4 0.535, l8 0.421, l12 0.343 on spk_sim; l4 0.531, l8 0.469, l12 0.327 on acc_sim.
#
# Two reasons not to just inherit that. It was measured where nothing can adapt -- under
# fine-tuning the upper layers get to reorganise, which is why SSL-MOS and UTMOS-style systems
# usually read the last layer instead. And the rule is already not universal: XLS-R prefers
# layer 8 to layer 4 on acc_sim (0.523 vs 0.505). Committing the 24-run grid in
# voicemos-track3-ssl-traindev-*.sh to layer 4 would bet the whole ablation on a result
# measured in a different training regime that does not hold everywhere even in that regime.
#
# This job buys that bet down. One cell per layer -- the cheapest and most representative of
# the four, mse + baseline -- so the layer axis is the only thing varying:
#
#   wavlm-base-plus-l4      37.7M params    40.1 GFLOP/utt  batch 16 x 1  ~1.4 h  (measured)
#   wavlm-base-plus-l8      66.0M params    54.3 GFLOP/utt  batch 8 x 2  ~1.9 h  (extrapolated)
#   wavlm-base-plus-l12     94.4M params    68.4 GFLOP/utt  batch 8 x 2  ~2.4 h  (extrapolated)
#
# Everything else is held at the full grid's settings: MoE head, freeze 5000/20000,
# --backbone-lr-mult 0.1, 20,000 steps, lr 1e-3, effective batch 16, --best-metric srcc_utt,
# fitted on sets/train_plus_dev.csv, scored on the labelled test split.
#
# WHY ONE TARGET. spk_sim only, because probing both doubles the cost and the frozen-feature
# layer ranking agrees across targets for five of the six (bundle, target) pairs. The exception
# is XLS-R on acc_sim, so for that bundle in particular a METRIC=acc_sim rerun is worth the
# extra ~13 h before committing its half of the grid. Same for any bundle whose spk_sim spread
# comes back inside the noise.
#
# THE l4 CELL IS THE GRID'S OWN CELL. Output goes to egs/ssl_runs_traindev/ with the same
# <encoder>-<loss>-<interaction>_<metric> tags the full grid uses, and every job in this
# directory skips an arm whose final checkpoint exists. So the probe's layer-4 run IS the
# grid's mse:baseline run -- when the grid follows at layer 4, it starts three arms in, and
# nothing is computed twice. Different layers get different tags, so nothing collides either.
#
# COST. The l4 row is MEASURED by voicemos-track3-ssl-sizing.sh on an L40S. The l8 and
#        last-layer rows are EXTRAPOLATED from it in proportion to GFLOP/utt (counted with
#        FlopCounterMode at the corpus mean duration of 4.78 s), so treat them as +/-30%.
#        Their batch sizes are set from the l4 measurement scaled by depth -- l4 peaked at
#        11.7-14.8 GiB of 46 at batch 16, and activation memory is roughly linear in layers,
#        so the deeper arms step down to keep the effective batch at 16 without an OOM.
#
#   wavlm-base-plus-l4   ~1.4 h  (measured, batch 16 x 1)
#   wavlm-base-plus-l8   ~1.9 h  (extrapolated, batch 8 x 2)
#   wavlm-base-plus-l12  ~2.4 h  (extrapolated, batch 8 x 2)
#   inference           3 arms x 4 passes x ~2 min = ~25 min
#   total               ~6.2 h, hence 9 h with slack.
#
# HOW TO READ THE RESULT. The summary ranks the three layers by held-out test UTT-SRCC. The
# reference lines printed underneath are what the frozen features scored on dev and what the
# speaker/accent-ID grid scored on test, so a layer that lands below its own frozen-feature
# ridge is evidence that fine-tuning is hurting, not that the layer is wrong.
#
# DEV IS IN THE FITTING SET: --best-metric srcc_utt selects on an IN-SAMPLE score, dev
# predictions carry the _IN-SAMPLE suffix, both checkpoints are kept and the final-step one is
# the safer default. Test labels are read by the summary only -- never by training, never by
# selection.
#
# RESUMABLE and ARMS-overridable, like every job in this directory:
#   sbatch --export=ALL,ARMS="wavlm-base-plus-l12:mse:baseline" --time=05:00:00 \
#       track3/jobs/ssl/voicemos-track3-ssl-layer-probe-wavlm-base.sh
#
# Deliberately NOT using `set -e`: if one layer fails the others should still run.

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

echo "NVIDIA SMI:"; nvidia-smi
NUM_WORKERS=${SLURM_CPUS_PER_TASK:-8}

##################################################################
# Configuration
##################################################################
METRIC=${METRIC:-spk_sim}

DR=../baseline/data/vmc2026_track3_train_phase_distro_v3_syn
EV=../baseline/data/vmc2026_track3_eval_phase_distro_v3_syn
TRAIN_CSV=${TRAIN_CSV:-$DR/sets/train_plus_dev.csv}
DEV_CSV=${DEV_CSV:-$DR/sets/dev.csv}
DEV_LABELS=${DEV_LABELS:-$EV/sets/dev_with_labels.csv}
TEST_CSV=${TEST_CSV:-$EV/sets/test.csv}
TEST_LABELS=${TEST_LABELS:-$EV/sets/vmc2026_track3_test_with_labels.csv}

# One arm per layer, cheapest layer first, all at mse + baseline.
ALL_ARMS="wavlm-base-plus-l4:mse:baseline wavlm-base-plus-l8:mse:baseline wavlm-base-plus-l12:mse:baseline"
ARMS=${ARMS:-"$ALL_ARMS"}

# Batch is per-arm, not per-job: these three differ by up to 6x in activation memory. The
# effective batch stays 16 in every case, so the arms remain comparable to each other and to
# every other grid in this project. Captured here, before the loop, because the per-arm
# defaults are assigned inside it -- reading ${BATCH:-...} there would pin every later arm to
# whatever the first arm chose.
BATCH_OVERRIDE=${BATCH:-}
ACCUM_OVERRIDE=${ACCUM:-}
LR=${LR:-1e-3}
TRAIN_STEPS=${TRAIN_STEPS:-20000}
FREEZE_STEPS=${FREEZE_STEPS:-5000}
EVAL_STEPS=${EVAL_STEPS:-1000}
SAVE_STEPS=${SAVE_STEPS:-25000}
BEST_METRIC=${BEST_METRIC:-srcc_utt}

OUTROOT=${OUTROOT:-egs/ssl_runs_traindev}
mkdir -p "$OUTROOT"

for F in "$DEV_LABELS" "$TEST_CSV" "$TRAIN_CSV"; do
    [ -f "$F" ] || { echo "ERROR: missing $F"; exit 1; }
done

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
echo "LAYER PROBE, WavLM Base+, fitted on TRAIN + DEV   metric=$METRIC"
echo "arms: $ARMS"
echo "batch: per-arm by depth (16x1 / 8x2 / 4x4), effective 16; moe head, mse + baseline, lr=$LR"
echo "out : $OUTROOT/  (shared with the full grid; matching arms are skipped)"
echo "=================================================================="

FAILED=()

##################################################################
# Probe
##################################################################
for ARM in $ARMS; do
    IFS=':' read -r E LOSS IX <<< "$ARM"

    # Depth drives activation memory; keep effective batch = BATCH x ACCUM = 16.
    LAYER=${E##*-l}
    if [ "$LAYER" -le 4 ]; then   ARM_BATCH=16; ARM_ACCUM=1
    elif [ "$LAYER" -le 12 ]; then ARM_BATCH=8;  ARM_ACCUM=2
    else                           ARM_BATCH=4;  ARM_ACCUM=4
    fi
    BATCH=${BATCH_OVERRIDE:-$ARM_BATCH}
    ACCUM=${ACCUM_OVERRIDE:-$ARM_ACCUM}

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
# Summary -- the layer ranking this job exists to produce
##################################################################
echo ""
echo "=================================================================="
echo "LAYER PROBE RESULT, WavLM Base+"
echo ""
python - "$METRIC" "$OUTROOT" "$ARMS" \
    "dev|$DEV_LABELS|dev_{metric}_{kind}_IN-SAMPLE.csv|IN-SAMPLE -- dev is in the fitting set, this estimates nothing" \
    "test|$TEST_LABELS|test_{metric}_{kind}.csv|held out -- the number that decides the layer" <<'PY'
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
    hdr = (f"{'layer':<28}{'ckpt':>6}{'n':>6}{'uMSE':>8}{'uLCC':>8}{'uSRCC':>8}"
           f"{'sMSE':>8}{'sLCC':>8}{'sSRCC':>8}")
    print(f"{metric}\n{hdr}\n{'-'*len(hdr)}")
    rows = []
    for a in arms:
        enc, loss, ix = a.split(":")
        tag = f"{enc}-{loss}-{ix}_{metric}"
        for kind in ("best", "final"):
            f = f"{outroot}/{tag}/" + pattern.format(metric=metric, kind=kind)
            if not os.path.exists(f):
                print(f"{enc:<28}{kind:>6}{'MISSING':>9}"); continue
            s = score(f, gt)
            if s is None:
                print(f"{enc:<28}{kind:>6}{'no rows':>9}"); continue
            rows.append((enc, kind, s))
            print(f"{enc:<28}{kind:>6}{s[0]:>6}" + "".join(f"{v:>8.3f}" for v in s[1:]))
    if rows and name == "test":
        print(f"\nLAYER RANKING by held-out test UTT-SRCC:")
        for nm, kind, s in sorted(rows, key=lambda r: -r[2][3]):
            print(f"  {s[3]:.3f}  {nm} [{kind}]")
        winner = sorted(rows, key=lambda r: -r[2][3])[0]
        spread = winner[2][3] - sorted(rows, key=lambda r: r[2][3])[0][2][3]
        print(f"\n  winner: {winner[0]}   spread across layers: {spread:.3f}")
        print("  A spread under ~0.02 is inside the noise this test set can resolve;")
        print("  prefer the cheapest layer in that case rather than the nominal winner.")
PY

echo ""
echo "Reference, same encoders, FROZEN features + ridge, held-out dev UTT-SRCC:"
echo "  spk_sim  l4 0.535  l8 0.421  l12 0.343     acc_sim  l4 0.531  l8 0.469  l12 0.327"
echo ""
echo "Reference on the same test set (600 pairs):"
echo "  spk_sim  best single deep cell (w24s4ep4)   uSRCC 0.588"
echo "  spk_sim  deep top-8 ensemble                uSRCC 0.575"
echo "  spk_sim  weak top-16 (frozen SSL + ridge)   uSRCC 0.606"
echo ""
echo "NEXT STEP. Set the winning layer in the full grid, then submit it:"
echo "  sbatch --export=ALL,ARMS=\"<enc>-l<N>:mse:baseline <enc>-l<N>:mse:bilinear \\"
echo "     <enc>-l<N>:coral:baseline <enc>-l<N>:coral:bilinear\" \\"
echo "     track3/jobs/ssl/voicemos-track3-ssl-traindev-wavlm-base-speaker.sh"
echo "The mse:baseline arm is already on disk from this probe and will be skipped."
echo ""
echo "TensorBoard:  tensorboard --logdir $(pwd)/$OUTROOT"

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""; echo "FAILURES: ${FAILED[*]}"
fi

ELAPSED=$((SECONDS - START_TIME))
echo ""
echo "Job $SLURM_JOB_ID finished at $(date) after $((ELAPSED / 3600))h $((ELAPSED % 3600 / 60))m"
