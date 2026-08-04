#!/usr/bin/env bash
#SBATCH --job-name=voicemos-track3-baseline-finetune
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=david.guzman@mila.quebec

# The official Baseline 2 recipe with the encoder ACTUALLY fine-tuned, swept over encoder
# learning rates, for BOTH targets.
#
# ../baseline/model.py passed `freeze_ssl=False  # Fine-tuning everything` to
# EncoderClassifier.from_hparams(), but never overrode SpeechBrain's
# `Pretrained.__init__(freeze_params=True)` default, which sets requires_grad=False on
# every backbone parameter. The published Baseline 2 numbers are therefore a FROZEN
# 22.15M-parameter ECAPA plus a trained 0.12M projection+MLP head -- not a fine-tuned
# encoder. model.py now passes freeze_params=False and decides freezing itself, so
# `--freeze-ssl` means what it says in both directions:
#
#     default        22.27M trainable, 22.15M of it inside the encoder
#     --freeze-ssl    0.12M trainable, encoder held in eval    <- published baseline
#
#   sbatch track3/jobs/voicemos-track3-baseline-finetune.sh
#
# ------------------------------------------------------------------------------------
# The sweep
#
# Everything except the encoder learning rate is held at the baseline's configuration:
# batch size 16, AdamW, head lr 1e-3, MSE on per-pair mean scores, repetitive padding,
# grad clip 1.0. The encoder lr is the one axis, anchored at both ends:
#
#   frozen      encoder lr 0. The published baseline, re-run through this code path so
#               every number in the table comes from one implementation. Also the control
#               that proves the model.py change did not alter the frozen behaviour.
#   lr1e-3      the baseline's literal setting: ONE AdamW group at 1e-3 over all 22.27M
#               parameters. This is "the same parameters" taken at face value. It was only
#               ever validated on a 0.12M head, and 1e-3 on a pretrained 22M-parameter
#               backbone for ~114 epochs over 2,800 pairs will most likely wash out the
#               VoxCeleb representation. Included because it is the honest like-for-like
#               answer, not because it is expected to win.
#   enclr1e-4   encoder lr 1e-4, head stays at 1e-3
#   enclr1e-5   encoder lr 1e-5, head stays at 1e-3   <- ../encoders/ found this range best
#   enclr1e-6   encoder lr 1e-6, head stays at 1e-3
#
# Any `enclr<VALUE>` tag works without editing the case statement, so adding a point to
# the sweep is just:
#
#   sbatch --export=ALL,CONFIGS="enclr3e-5 enclr5e-5" track3/jobs/voicemos-track3-baseline-finetune.sh
#
# Step counts differ by design. The frozen and lr1e-3 runs get the baseline's 20000 steps
# because that is the recipe being reproduced; the fine-tuning runs get FT_STEPS=4000
# (~23 epochs), which is already generous for a 22M-parameter backbone on 2,800 pairs.
#
# Because ../baseline/finetune.py has no in-training dev evaluation, EVERY saved checkpoint
# is scored rather than only the final one, and the summary at the end prints the whole
# grid. For a run that diverges, the step at which it happened is then visible instead of
# the config reporting one useless number.
#
# Deliberately NOT using `set -e`: if one configuration fails, the others should still run
# rather than wasting the whole allocation.

START_TIME=$SECONDS
echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

##################################################################
# Activate the environment by loading Python and required packages
##################################################################
module load miniconda/3
module load gcc/9.3.0
module load cuda/12.3.2

export HF_HOME=$SCRATCH/huggingface
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

conda activate VoiceMOS

# Fail fast rather than burning the allocation on the wrong interpreter.
if [ "$CONDA_DEFAULT_ENV" != "VoiceMOS" ]; then
    echo "ERROR: conda env is '${CONDA_DEFAULT_ENV:-none}', expected VoiceMOS"; exit 1
fi
python -c "import torch, speechbrain" || { echo "ERROR: torch/speechbrain not importable"; exit 1; }
echo "python: $(which python)"

# torchaudio >= 2.9 dispatches load() to torchcodec, which dies with
# "libnppicc.so.12: cannot open shared object file" unless the NVIDIA libs that ship
# with torch are on the loader path.
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH=$SITE_PACKAGES/nvidia/npp/lib:$LD_LIBRARY_PATH

REPO=/home/mila/g/guzmand/scratch/Repositories/VoiceMOS-Challenge-2026
cd "$REPO/track3/baseline" || exit 1

python -c "
import torchaudio, glob
f = sorted(glob.glob('data/vmc2026_track3_train_phase_distro_v3_syn/wav/*.wav'))[0]
torchaudio.load(f)
print('torchaudio.load OK')" || { echo "ERROR: torchaudio cannot load wavs"; exit 1; }

echo "NVIDIA SMI:"
nvidia-smi
echo "HF_HOME: $HF_HOME"

##################################################################
# Configuration
##################################################################
# The TRAINING distribution is the right data root for every split used here: the eval
# distro is missing all 600 sys019 reference wavs (they ship separately with VCTK), while
# the train distro has every train and dev wav. Inference against the eval distro would
# silently skip all 600 dev rows.
DR=data/vmc2026_track3_train_phase_distro_v3_syn
DEV_CSV=${DEV_CSV:-$DR/sets/dev.csv}

# Labelled dev set, from the evaluation-phase distribution -- metadata only, so the
# wav paths still resolve against $DR above. Already one row per pair.
DEV_LABELS=${DEV_LABELS:-data/vmc2026_track3_eval_phase_distro_v3_syn/sets/dev_with_labels.csv}

BATCH=${BATCH:-16}
HEAD_LR=${HEAD_LR:-1e-3}

# Baseline step count, used by the two configurations that reproduce the published recipe.
# Overridable so the script can be smoke-tested cheaply:
#   BASELINE_STEPS=2 BASELINE_SAVE=2 FT_STEPS=2 FT_SAVE=2 sbatch ...
BASELINE_STEPS=${BASELINE_STEPS:-20000}
BASELINE_SAVE=${BASELINE_SAVE:-5000}

# Real fine-tuning needs far fewer steps than a frozen head does.
FT_STEPS=${FT_STEPS:-4000}
FT_SAVE=${FT_SAVE:-1000}

METRICS=(${METRICS:-spk_sim acc_sim})
CONFIGS=(${CONFIGS:-frozen lr1e-3 enclr1e-4 enclr1e-5 enclr1e-6})

if [ ! -f "$DEV_LABELS" ]; then
    echo "ERROR: no labelled dev set at $DEV_LABELS"
    echo "Training would run blind. Fix the path or pass --export=ALL,DEV_LABELS=..."
    exit 1
fi

echo "=================================================================="
echo "Baseline 2 recipe, encoder learning-rate sweep"
echo "metrics=${METRICS[*]}"
echo "configs=${CONFIGS[*]}"
echo "batch=$BATCH  head lr=$HEAD_LR"
echo "steps: baseline/frozen=$BASELINE_STEPS  fine-tuning=$FT_STEPS"
echo "data root=$DR"
echo "dev labels=$DEV_LABELS"
echo "=================================================================="

mkdir -p egs

##################################################################
# Sweep: train, then score every checkpoint on the labelled dev set
##################################################################
FAILED=()
for METRIC in "${METRICS[@]}"; do
for CONFIG in "${CONFIGS[@]}"; do
    TAG="${METRIC}_${CONFIG}"
    OUT="egs/$TAG"
    echo ""
    echo "##################################################################"
    echo "# $TAG  ($(date))"
    echo "##################################################################"

    case "$CONFIG" in
        frozen)
            # What the published baseline actually trains: the 0.12M head only.
            TRAIN_ARGS=(--freeze-ssl --lr "$HEAD_LR")
            STEPS=$BASELINE_STEPS; SAVE=$BASELINE_SAVE
            ;;
        lr1e-3)
            # The baseline's literal setting: a single AdamW group over every parameter.
            TRAIN_ARGS=(--lr 1e-3)
            STEPS=$BASELINE_STEPS; SAVE=$BASELINE_SAVE
            ;;
        enclr*)
            # Encoder gets its own lr, parsed straight off the tag; head stays at HEAD_LR.
            ENCODER_LR="${CONFIG#enclr}"
            TRAIN_ARGS=(--encoder-lr "$ENCODER_LR" --lr "$HEAD_LR")
            STEPS=$FT_STEPS; SAVE=$FT_SAVE
            ;;
        *)
            echo "Unknown config '$CONFIG'"; FAILED+=("$TAG:config"); continue ;;
    esac

    echo "--- training: ${TRAIN_ARGS[*]} --train-steps $STEPS ---"
    python finetune.py \
        --data-root "$DR" \
        --target-metric "$METRIC" \
        --outdir "$OUT" \
        --batch-size "$BATCH" \
        "${TRAIN_ARGS[@]}" \
        --train-steps "$STEPS" \
        --save-steps "$SAVE"
    if [ $? -ne 0 ]; then echo "TRAINING FAILED for $TAG"; FAILED+=("$TAG:train"); continue; fi

    # finetune.py writes the last step twice, as model_<metric>_step<STEPS>.pt and again as
    # finetuned_model_<metric>_final.pt. Score the final one only when no step checkpoint
    # covers it, otherwise the same weights get 600 inference forwards for nothing.
    CKPTS=("$OUT"/model_${METRIC}_step*.pt)
    if [ ! -f "$OUT/model_${METRIC}_step${STEPS}.pt" ]; then
        CKPTS+=("$OUT/finetuned_model_${METRIC}_final.pt")
    fi

    for CKPT in "${CKPTS[@]}"; do
        [ -f "$CKPT" ] || continue
        STEP=$(basename "$CKPT" .pt | sed "s/.*_//")
        PRED="$OUT/dev_${METRIC}_${STEP}.csv"

        echo ""
        echo "--- inference on the official dev set: $(basename "$CKPT") ---"
        # Clear any output from a previous run first, so a failure here cannot leave a
        # stale CSV that the summary would then report as OK.
        rm -f "$PRED"
        python inference.py \
            --data-root "$DR" \
            --csv-path "$DEV_CSV" \
            --checkpoint "$CKPT" \
            --target-metric "$METRIC" \
            --out "$PRED"
        if [ $? -ne 0 ]; then
            echo "INFERENCE FAILED for $TAG at $STEP"; FAILED+=("$TAG:$STEP"); continue
        fi

        echo "--- scoring against the official dev labels ---"
        python calculate_metrics.py \
            --prediction-csv "$PRED" \
            --ground-truth-csv "$DEV_LABELS"
    done
done
done

##################################################################
# Summary
#
# One table over every prediction CSV the sweep produced, so the learning rates can be
# compared without grepping 30-odd metric blocks out of the log. Re-scores from the CSVs
# on CPU, so it costs nothing.
##################################################################
echo ""
echo "=================================================================="
echo "SWEEP SUMMARY"
echo "=================================================================="
python - "$DEV_LABELS" "${METRICS[*]}" "${CONFIGS[*]}" <<'PY'
import csv, glob, os, sys
from collections import defaultdict

import numpy as np

from calculate_metrics import compute_metrics

dev_labels, metrics, configs = sys.argv[1], sys.argv[2].split(), sys.argv[3].split()

gt = {}
with open(dev_labels, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        gt[(row["wav_a_path"], row["wav_b_path"])] = row

# Published Baseline 2, for the frozen-encoder reference row.
PUBLISHED = {
    "spk_sim": (0.438, 0.511, 0.451, 0.069, 0.916, 0.860),
    "acc_sim": (0.418, 0.465, 0.440, 0.060, 0.902, 0.861),
}

def score(path, metric):
    ut, up = [], []
    st, sp = defaultdict(list), defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["wav_a_path"], row["wav_b_path"])
            if key not in gt:
                continue
            t = gt[key].get(metric)
            if not t or not t.strip():
                continue
            t, p = float(t), float(row[f"pred_{metric}"])
            sys_id = gt[key]["system_id"]
            ut.append(t); up.append(p)
            st[sys_id].append(t); sp[sys_id].append(p)
    if not ut:
        return None
    u = compute_metrics(np.array(ut), np.array(up))
    s = compute_metrics(
        np.array([np.mean(st[k]) for k in st]), np.array([np.mean(sp[k]) for k in sp])
    )
    return (u[0], u[1], u[2], s[0], s[1], s[2])

header = f"{'config':<14}{'step':>7}  {'UTT-MSE':>8}{'UTT-LCC':>8}{'UTT-SRCC':>9}  {'SYS-MSE':>8}{'SYS-LCC':>8}{'SYS-SRCC':>9}"
for metric in metrics:
    print(f"\n### {metric}")
    print(header)
    print("-" * len(header))
    print(f"{'published':<14}{'20000':>7}  " + "".join(
        f"{v:>8.3f}" if i not in (2, 5) else f"{v:>9.3f}"
        for i, v in enumerate(PUBLISHED[metric])
    ))
    for config in configs:
        rows = []
        for path in glob.glob(f"egs/{metric}_{config}/dev_{metric}_*.csv"):
            tag = os.path.basename(path)[: -len(".csv")].split("_")[-1]
            step = int(tag[4:]) if tag.startswith("step") else 10**9
            rows.append((step, tag, path))
        if not rows:
            print(f"{config:<14}{'-':>7}  (no predictions)")
            continue
        for step, tag, path in sorted(rows):
            r = score(path, metric)
            label = tag[4:] if tag.startswith("step") else tag
            if r is None:
                print(f"{config:<14}{label:>7}  (no matching pairs)")
                continue
            print(f"{config:<14}{label:>7}  " + "".join(
                f"{v:>8.3f}" if i not in (2, 5) else f"{v:>9.3f}"
                for i, v in enumerate(r)
            ))
PY

echo ""
echo "Prediction CSVs written by this run:"
for METRIC in "${METRICS[@]}"; do
for CONFIG in "${CONFIGS[@]}"; do
    for F in "egs/${METRIC}_${CONFIG}"/dev_${METRIC}_*.csv; do
        [ -f "$F" ] && echo "  OK      $(pwd)/$F  ($(($(wc -l < "$F") - 1)) rows)"
    done
done
done

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    echo "FAILURES: ${FAILED[*]}"
fi

ELAPSED=$((SECONDS - START_TIME))
echo ""
echo "Job $SLURM_JOB_ID finished at $(date) after $((ELAPSED / 3600))h $((ELAPSED % 3600 / 60))m"
