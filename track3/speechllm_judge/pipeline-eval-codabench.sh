# Inference on the OFFICIAL (unlabeled) dev/eval set -> single CSV for CodaBench.
# Unlike pipeline-speech-llm.sh, there is no scoring step (no ground truth), and
# both pred_spk_sim and pred_acc_sim are merged into one submission file.
#
# Usage:  bash pipeline-eval-codabench.sh
# set -euo pipefail
module load miniconda/3 && conda activate speecheval

export DATA_ROOT=../data/vmc2026_track3_train_phase_distro_v3_syn
EVAL_CSV=$DATA_ROOT/sets/dev.csv
HERE=$(pwd)
SLM=$HERE/SpeechLLM-as-Judges             # upstream repo now lives inside this folder
CKPT=$SLM/checkpoint
OUTDIR=$HERE/predictions                  # all generated jsonl/csv go here
mkdir -p "$OUTDIR"
OUT=$OUTDIR/dev-eval.pred_submission.csv  # <- upload this to CodaBench

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "START eval-inference | EVAL_CSV=$EVAL_CSV | $(tail -n +2 $EVAL_CSV | wc -l) pairs"

for M in spk_sim acc_sim; do
  log "==================== METRIC: $M ===================="

  # 1. CSV -> CompareEval JSONL (dedups on the wav pair)
  log "[$M] step 1/3: building CompareEval JSONL ..."
  python csv_to_swift.py --data-root $DATA_ROOT --csv-path $EVAL_CSV \
      --target-metric $M --out "$OUTDIR/dev-eval.$M.jsonl"
  log "[$M] step 1/3: wrote predictions/dev-eval.$M.jsonl ($(wc -l < "$OUTDIR/dev-eval.$M.jsonl") pairs)"

  # 2. Run the judge (swift). Remove any stale results so it can't append/dup.
  log "[$M] step 2/3: running swift inference (slow part) ..."
  rm -f "$OUTDIR/dev-eval.$M.results.jsonl"
  cd "$SLM/script"
  CUDA_VISIBLE_DEVICES=0 bash inference.sh \
      "$CKPT" \
      "$OUTDIR/dev-eval.$M.jsonl" \
      "$OUTDIR/dev-eval.$M.results.jsonl"
  cd "$HERE"

  if [ ! -s "$OUTDIR/dev-eval.$M.results.jsonl" ]; then
    log "[$M] ERROR: predictions/dev-eval.$M.results.jsonl missing/empty; inference failed. Aborting."
    exit 1
  fi
  log "[$M] step 2/3: inference done ($(wc -l < "$OUTDIR/dev-eval.$M.results.jsonl") responses)"
done

# 3. Merge both predictions into one CSV: annotate dev.csv with pred_spk_sim,
#    then annotate THAT file with pred_acc_sim (columns are preserved).
log "step 3/3: merging predictions -> $OUT ..."
python swift_to_submission.py --results "$OUTDIR/dev-eval.spk_sim.results.jsonl" \
    --orig-csv $EVAL_CSV --data-root $DATA_ROOT \
    --target-metric spk_sim --out "$OUT"
python swift_to_submission.py --results "$OUTDIR/dev-eval.acc_sim.results.jsonl" \
    --orig-csv "$OUT" --data-root $DATA_ROOT \
    --target-metric acc_sim --out "$OUT"

log "DONE -> $OUT"
log "header: $(head -1 "$OUT")"
log "Upload $OUT to CodaBench."
