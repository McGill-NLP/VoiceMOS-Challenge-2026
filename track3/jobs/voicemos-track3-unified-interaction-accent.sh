#!/usr/bin/env bash
#SBATCH --job-name=voicemos-track3-unified-interaction-accent
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=david.guzman@mila.quebec

# Ablation of --interaction for ACCENT similarity (acc_sim), 20,000 steps per arm.
#
#   sbatch track3/jobs/voicemos-track3-unified-interaction-accent.sh
#
# Nine arms, one per interaction mode, ~9 h in total. Pass a subset to run fewer:
#   sbatch --export=ALL,MODES="no-b bilinear" --time=03:00:00 ...
#
# EVERYTHING except --interaction is held at the `base` arm of the ECAPA ladder --
# ecapa-voxceleb, mlp head, MSE, one AdamW group at 1e-3, no freeze schedule, batch 16 --
# so the interaction vector is the only variable. `baseline` is the control: it is the
# vector the official baseline uses, so its row is what every other row is read against.
#
#   baseline         [a, b, |a-b|, a*b]                  1024   control
#   scalars          baseline + [cos, ||a-b||]           1026
#   normed           LayerNorm per block                 1024
#   normed-scalars   both of the above                   1026
#   signed           [a, b, a-b, a*b]                    1024
#   no-b             [a, |a-b|, a*b]                      768
#   symmetric        [a+b, |a-b|, a*b]                    768   f(a,b) == f(b,a)
#   bilinear         baseline + (Ua)*(Vb), rank 64       1088
#   no-b-bilinear    no-b + (Ua)*(Vb), rank 64            832   the two that helped
#
# ECAPA rather than ERes2NetV2 on purpose: at ~0.416 s/step the latter would make a full
# nine-arm ablation 21 h per target. ECAPA runs ~0.16 s/step, so 20,000 steps is ~60 min per
# arm. If a mode wins here, re-run that mode alone on the stronger encoder.
#
# NO freeze schedule, deliberately: `freeze` was the clearest loser in the ECAPA ladder on
# both targets (SYS-SRCC 0.889 against 0.970 for `base`), and a frozen phase would delay
# the head learning to use whatever the interaction provides.
#
# BEST_METRIC is srcc_utt, as in every job here now. It matters especially for this one:
# on the CORAL runs srcc_sys selected checkpoints as early as step 1000 -- a barely-trained
# head -- which would make an ablation of the head's INPUT close to meaningless. The
# summary reports both the selected checkpoint and the best-UTT-SRCC step from the dev log,
# so the choice does not hide the answer either way.
#
# To split across jobs, or to re-run one mode on another encoder:
#
#   sbatch --export=ALL,MODES="normed-scalars bilinear" ...
#   sbatch --export=ALL,MODES=scalars,ENCODER=eres2netv2,BATCH=4,ACCUM=4 --time=04:00:00 ...
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

# The TRAINING distribution is the right data root for train and dev alike: the eval
# distro is missing all 600 sys019 reference wavs, so inference against it drops every row.
DR=../baseline/data/vmc2026_track3_train_phase_distro_v3_syn
TRAIN_CSV=$DR/sets/train.csv
DEV_CSV=${DEV_CSV:-$DR/sets/dev.csv}
DEV_LABELS=${DEV_LABELS:-../baseline/data/vmc2026_track3_eval_phase_distro_v3_syn/sets/dev_with_labels.csv}

# All nine modes. `all` is accepted as a synonym, and any subset can be passed:
#   --export=ALL,MODES="no-b bilinear"
ALL_MODES="baseline scalars normed normed-scalars signed no-b symmetric bilinear no-b-bilinear"
MODES=${MODES:-"$ALL_MODES"}
[ "$MODES" = "all" ] && MODES="$ALL_MODES"

# Held fixed across every arm. This is the ladder's `base` configuration.
ENCODER=${ENCODER:-ecapa-voxceleb}
HEAD=${HEAD:-mlp}
OBJECTIVE=${OBJECTIVE:-mse}
BATCH=${BATCH:-16}
ACCUM=${ACCUM:-1}
LR=${LR:-1e-3}
BILINEAR_RANK=${BILINEAR_RANK:-64}

TRAIN_STEPS=${TRAIN_STEPS:-20000}
EVAL_STEPS=${EVAL_STEPS:-1000}
# Denser than the CORAL jobs' 5000: those runs peaked at steps 16000-19000 and the
# checkpoint was not kept. 10 checkpoints x ~89 MB per arm is nothing against the free
# space here.
SAVE_STEPS=${SAVE_STEPS:-2000}
BEST_METRIC=${BEST_METRIC:-srcc_utt}

if [ ! -f "$DEV_LABELS" ]; then
    echo "ERROR: no labelled dev set at $DEV_LABELS"; exit 1
fi

echo "=================================================================="
echo "--interaction ablation   metric=$METRIC"
echo "modes: $MODES"
echo "held fixed: encoder=$ENCODER head=$HEAD objective=$OBJECTIVE batch=${BATCH}x${ACCUM} lr=$LR"
echo "steps=$TRAIN_STEPS  eval every $EVAL_STEPS  save every $SAVE_STEPS  select on $BEST_METRIC"
echo "no freeze schedule; single AdamW group"
echo "=================================================================="

mkdir -p egs
FAILED=()

##################################################################
# Sweep
#
# One loop rather than eight copies: the arms differ in exactly one flag value, so the
# fine-tuning command below IS the command that runs, with $MODE substituted.
##################################################################
for MODE in $MODES; do
    TAG="ix_${MODE}_${METRIC}"
    OUT="egs/$TAG"
    echo ""
    echo "##################################################################"
    echo "# $TAG   ($(date))"
    echo "##################################################################"

    python finetune.py \
        --data-root "$DR" \
        --train-csv "$TRAIN_CSV" \
        --target-metric "$METRIC" \
        --outdir "$OUT" \
        --encoder "$ENCODER" \
        --head "$HEAD" \
        --objective "$OBJECTIVE" \
        --interaction "$MODE" \
        --bilinear-rank "$BILINEAR_RANK" \
        --batch-size "$BATCH" \
        --accumulate-steps "$ACCUM" \
        --lr "$LR" \
        --train-steps "$TRAIN_STEPS" \
        --save-steps "$SAVE_STEPS" \
        --eval-steps "$EVAL_STEPS" \
        --eval-batch-size 16 \
        --best-metric "$BEST_METRIC" \
        --dev-csv "$DEV_LABELS" \
        --dev-data-root "$DR" \
        --num-workers "$NUM_WORKERS"

    if [ $? -ne 0 ]; then echo "TRAINING FAILED for $TAG"; FAILED+=("$TAG:train"); continue; fi

    CKPT="$OUT/model_best_${METRIC}.pt"
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
    if [ $? -ne 0 ]; then echo "INFERENCE FAILED for $TAG"; FAILED+=("$TAG:dev"); continue; fi

    echo "--- scoring against the official dev labels ---"
    python calculate_metrics.py \
        --prediction-csv "$OUT/dev_${METRIC}.csv" \
        --ground-truth-csv "$DEV_LABELS"

    E=$((SECONDS - START_TIME))
    echo "[$TAG done, $((E / 3600))h $((E % 3600 / 60))m into the job]"
done

##################################################################
# Summary
##################################################################
echo ""
echo "=================================================================="
echo "--interaction ablation results (the final ranking combines several"
echo "metrics, so do not read a single column):"
echo ""
python - "$DEV_LABELS" "$METRIC" "$MODES" <<'PY'
import csv, sys, os
import numpy as np, scipy.stats
from collections import defaultdict

labels, metric, modes = sys.argv[1], sys.argv[2], sys.argv[3].split()
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

hdr = f"{'interaction':<18}{'n':>6}{'uMSE':>8}{'uLCC':>8}{'uSRCC':>8}{'sMSE':>8}{'sLCC':>8}{'sSRCC':>8}"

print(f"A) selected checkpoint, scored by inference.py\n\n{metric}\n{hdr}\n{'-'*len(hdr)}")
for m in modes:
    f = f"egs/ix_{m}_{metric}/dev_{metric}.csv"
    if not os.path.exists(f):
        print(f"{m:<18}{'MISSING':>6}"); continue
    s = score(f)
    if s is None:
        print(f"{m:<18}{'too few rows':>6}"); continue
    print(f"{m:<18}{s[0]:>6}{s[1]:>8.3f}{s[2]:>8.3f}{s[3]:>8.3f}{s[4]:>8.3f}{s[5]:>8.3f}{s[6]:>8.3f}")

# The dev log carries every evaluation, so the best-UTT-SRCC step can be reported even
# when its checkpoint was not among those saved. These are the in-training BATCHED
# evaluations, which track inference.py closely at utterance level but can differ by
# ~0.01 on system SRCC -- a rank statistic over only 23 systems.
hdr2 = f"{'interaction':<18}{'step':>7}{'uMSE':>8}{'uLCC':>8}{'uSRCC':>8}{'sMSE':>8}{'sLCC':>8}{'sSRCC':>8}"
print(f"\n\nB) best-UTT-SRCC step from the dev log (batched eval)\n\n{metric}\n{hdr2}\n{'-'*len(hdr2)}")
for m in modes:
    log = f"egs/ix_{m}_{metric}/dev_log_{metric}.csv"
    if not os.path.exists(log):
        print(f"{m:<18}{'MISSING':>7}"); continue
    rows = [r for r in csv.DictReader(open(log)) if int(r["step"]) > 0]
    rows = [r for r in rows if r["srcc_utt"] not in ("", "nan")]
    if not rows:
        print(f"{m:<18}{'no evals':>7}"); continue
    b = max(rows, key=lambda r: float(r["srcc_utt"]))
    print(f"{m:<18}{int(b['step']):>7}{float(b['mse_utt']):>8.3f}{float(b['lcc_utt']):>8.3f}"
          f"{float(b['srcc_utt']):>8.3f}{float(b['mse_sys']):>8.3f}{float(b['lcc_sys']):>8.3f}"
          f"{float(b['srcc_sys']):>8.3f}")
PY

echo ""
echo "For reference on the same dev set:"
echo "  acc_sim  Baseline 2 published      uMSE 0.418  uLCC 0.465  uSRCC 0.440  sMSE 0.060  sLCC 0.902  sSRCC 0.861"
echo "  acc_sim  ladder base/ecapa @8k     uMSE 0.372  uLCC 0.515  uSRCC 0.468  sMSE 0.039  sLCC 0.933  sSRCC 0.970"
echo "  acc_sim  best uSRCC anywhere       CORAL commonaccent-moe 0.502 @20000 (batched eval)"
echo ""
echo "TensorBoard (every arm appears as its own run):"
echo "  tensorboard --logdir $(pwd)/egs"

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""; echo "FAILURES: ${FAILED[*]}"
fi

ELAPSED=$((SECONDS - START_TIME))
echo ""
echo "Job $SLURM_JOB_ID finished at $(date) after $((ELAPSED / 3600))h $((ELAPSED % 3600 / 60))m"
