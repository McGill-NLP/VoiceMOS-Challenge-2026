# set -euo pipefail
module load miniconda/3 && conda activate speecheval
export DATA_ROOT=../data/vmc2026_track3_train_phase_distro_v3_syn
HERE=$(pwd)
SLM=$HERE/SpeechLLM-as-Judges   # upstream repo now lives inside this folder
CKPT=$SLM/checkpoint
OUTDIR=$HERE/predictions        # all generated jsonl/csv go here
mkdir -p "$OUTDIR"

# timestamped progress logger
log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "START pipeline | DATA_ROOT=$DATA_ROOT | CKPT=$CKPT | OUTDIR=$OUTDIR"

# For each metric in {spk_sim, acc_sim}:
for M in spk_sim acc_sim; do
  log "==================== METRIC: $M ===================="

  # 1. CSV -> CompareEval JSONL (dedups on the wav pair)
  log "[$M] step 1/3: building CompareEval JSONL ..."
  python csv_to_swift.py --data-root $DATA_ROOT --csv-path ../data/dev-ID.csv \
      --target-metric $M --out "$OUTDIR/dev-ID.$M.jsonl"
  log "[$M] step 1/3: wrote predictions/dev-ID.$M.jsonl ($(wc -l < "$OUTDIR/dev-ID.$M.jsonl") pairs)"

  # 2. Run the judge (swift, via local infer.sh with repetition_penalty).
  #    Tiny smoke test first: `head -5 predictions/dev-ID.$M.jsonl > tmp.jsonl`
  log "[$M] step 2/3: running swift inference (this is the slow part) ..."
  rm -f "$OUTDIR/dev-ID.$M.results.jsonl"
  CUDA_VISIBLE_DEVICES=0 bash infer.sh \
      "$CKPT" \
      "$OUTDIR/dev-ID.$M.jsonl" \
      "$OUTDIR/dev-ID.$M.results.jsonl"

  # swift crashes without a nonzero exit sometimes; make failures loud.
  if [ ! -s "$OUTDIR/dev-ID.$M.results.jsonl" ]; then
    log "[$M] ERROR: predictions/dev-ID.$M.results.jsonl missing/empty; inference failed. Aborting."
    exit 1
  fi
  log "[$M] step 2/3: inference done ($(wc -l < "$OUTDIR/dev-ID.$M.results.jsonl") responses)"

  # 3. Results JSONL -> submission CSV with pred_$M column
  #    (--data-root lets it rejoin on the absolute audios paths swift keeps)
  log "[$M] step 3/3: parsing responses -> predictions/dev-ID.pred_$M.csv ..."
  python swift_to_submission.py --results "$OUTDIR/dev-ID.$M.results.jsonl" \
      --orig-csv ../data/dev-ID.csv --data-root $DATA_ROOT \
      --target-metric $M --out "$OUTDIR/dev-ID.pred_$M.csv"
  log "[$M] step 3/3: wrote predictions/dev-ID.pred_$M.csv"
done

# 4. Score (dev-ID/dev-OOD carry labels; official dev.csv does not)
log "==================== step 4/4: scoring ===================="
python ../calculate_metrics.py --prediction-csv "$OUTDIR/dev-ID.pred_spk_sim.csv" \
    --ground-truth-csv ../data/dev-ID.csv

log "DONE"
