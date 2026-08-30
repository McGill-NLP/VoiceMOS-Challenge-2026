#!/usr/bin/env bash
#SBATCH --job-name=voicemos-track3-deep-test-inference
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=david.guzman@mila.quebec

# Test-set inference for every checkpoint in egs/ensemble_runs/.
#
#   sbatch track3/jobs/weak/voicemos-track3-deep-test-inference.sh
#
# PACKAGING ONLY -- THIS JOB PRODUCES NO METRICS. test.csv is unlabelled (its columns are
# system_id, utterance_id, wav_a_path, wav_b_path), so nothing here can be scored. All model
# selection and every reported number come from dev_with_labels.csv instead; see
# ../../unified/calculate_metrics.py and ../../weak/analyze.py. The score for these
# predictions comes back from CodaBench after upload.
#
# WHAT IT IS FOR. The grid scripts scored the 32 factorial runs on dev and stopped, so their
# checkpoints have no test predictions. This job writes them, which is what lets stack.py
# --write-test, or any ensemble over the factorial pool, emit a submission CSV without a
# further inference pass.
#
# WHEN TO RUN IT. Once, any time before submitting -- it is NOT a prerequisite for any
# analysis. Selection happens entirely on dev. Running it early simply decouples "when the
# analysis finishes" from "when a submission can be uploaded", at a cost of ~20 minutes.
#
# The test wavs live under the EVAL distribution, not the training one, so --data-root differs
# from what the grid scripts passed for dev.
#
# By default only model_best_<metric>.pt is scored: 32 runs at ~1-3 min each is about an hour.
# Set KINDS="best final" to add the last-step checkpoints (64 runs) if there is time.
#
# Resumable: existing outputs are skipped, so a preemption costs only the run in flight.
#
# Deliberately NOT using `set -e`: one bad checkpoint must not kill the sweep.

START_TIME=$SECONDS
echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

module load miniconda/3
module load gcc/9.3.0
module load cuda/12.3.2

export HF_HOME=$SCRATCH/huggingface
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false

conda activate VoiceMOS
if [ "$CONDA_DEFAULT_ENV" != "VoiceMOS" ]; then
    echo "ERROR: conda env is '${CONDA_DEFAULT_ENV:-none}', expected VoiceMOS"; exit 1
fi

SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH=$SITE_PACKAGES/nvidia/npp/lib:$LD_LIBRARY_PATH

REPO=${REPO:-/home/mila/g/guzmand/scratch/Repositories/VoiceMOS-Challenge-2026}
cd "$REPO/track3/unified" || exit 1

# Test wavs are in the eval distribution.
EV=../baseline/data/vmc2026_track3_eval_phase_distro_v3_syn
TEST_CSV=${TEST_CSV:-$EV/sets/test.csv}
OUTROOT=${OUTROOT:-egs/ensemble_runs}
KINDS=${KINDS:-"best"}

if [ ! -f "$TEST_CSV" ]; then echo "ERROR: no test csv at $TEST_CSV"; exit 1; fi

echo "=================================================================="
echo "deep test inference"
echo "  runs under : $OUTROOT"
echo "  test csv   : $TEST_CSV   ($(($(wc -l < "$TEST_CSV") - 1)) pairs)"
echo "  data root  : $EV"
echo "  checkpoints: $KINDS"
echo "=================================================================="

DONE=0; SKIPPED=0; FAILED=()

for D in "$OUTROOT"/*/; do
    TAG=$(basename "$D")
    # Directory names end in _spk_sim or _acc_sim; the metric drives both the checkpoint
    # filename and the prediction column.
    case "$TAG" in
        *_spk_sim) METRIC=spk_sim ;;
        *_acc_sim) METRIC=acc_sim ;;
        *) echo "SKIP $TAG (cannot infer target metric from name)"; continue ;;
    esac

    for KIND in $KINDS; do
        case "$KIND" in
            best)  CKPT="$D/model_best_${METRIC}.pt" ;;
            final) CKPT="$D/finetuned_model_${METRIC}_final.pt" ;;
            *) echo "unknown KIND '$KIND'"; continue ;;
        esac
        OUT="$D/test_${METRIC}_${KIND}.csv"

        if [ ! -f "$CKPT" ]; then
            echo "NOTE: $CKPT missing"; FAILED+=("$TAG:$KIND:nockpt"); continue
        fi
        if [ -f "$OUT" ]; then
            SKIPPED=$((SKIPPED + 1)); continue
        fi

        echo ""
        echo "--- $TAG [$KIND] -> $(basename "$OUT") ($(date +%H:%M:%S)) ---"
        python inference.py \
            --data-root "$EV" \
            --csv-path "$TEST_CSV" \
            --checkpoint "$CKPT" \
            --out "$OUT"
        if [ $? -ne 0 ]; then
            echo "INFERENCE FAILED for $TAG/$KIND"; FAILED+=("$TAG:$KIND"); rm -f "$OUT"; continue
        fi
        DONE=$((DONE + 1))
    done
done

echo ""
echo "=================================================================="
echo "wrote $DONE new prediction files, skipped $SKIPPED already present"
echo "test predictions on disk: $(ls "$OUTROOT"/*/test_*.csv 2>/dev/null | wc -l)"
if [ ${#FAILED[@]} -gt 0 ]; then echo "FAILURES: ${FAILED[*]}"; fi

ELAPSED=$((SECONDS - START_TIME))
echo "Job $SLURM_JOB_ID finished at $(date) after $((ELAPSED/3600))h $((ELAPSED%3600/60))m"
