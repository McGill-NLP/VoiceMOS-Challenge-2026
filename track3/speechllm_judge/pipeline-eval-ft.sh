# Evaluate the FINE-TUNED adapters end to end on a Track 3 CSV.
# Auto-locates the best checkpoint per metric, uses the SAME prompt as training
# (csv_to_sft.py --mode infer), runs both metrics, merges into one CSV, and
# scores it if the CSV carries labels (dev-ID / dev-OOD) — otherwise it's a
# CodaBench submission (official dev.csv).
#
# Usage:
#   bash pipeline-eval-ft.sh                      # default: ../data/dev-ID.csv (scored)
#   bash pipeline-eval-ft.sh ../data/dev-OOD.csv
#   bash pipeline-eval-ft.sh ../data/vmc2026_track3_train_phase_distro_v3_syn/sets/dev.csv
# Override adapters with ADAPTER_spk_sim=/path ADAPTER_acc_sim=/path.
# set -euo pipefail
module load miniconda/3 && conda activate speecheval

export DATA_ROOT=../data/vmc2026_track3_train_phase_distro_v3_syn
CSV=${1:-../data/dev-ID.csv}
HERE=$(pwd)
BASE=$HERE/SpeechLLM-as-Judges/checkpoint
MODELDIR=$HERE/models
OUTDIR=$HERE/predictions
mkdir -p "$OUTDIR"

TAG=$(basename "$CSV" .csv)                 # dev-ID / dev-OOD / dev
OUT="$OUTDIR/$TAG.ft.pred.csv"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# Best checkpoint for a metric: trainer_state.json's best_model_checkpoint from
# the newest version dir, else the highest-step checkpoint. Env override wins.
best_ckpt() {
  local metric=$1
  local override
  override=$(eval echo "\${ADAPTER_$metric:-}")
  if [ -n "$override" ]; then echo "$override"; return; fi
  python - "$MODELDIR/sft_$metric" <<'PY'
import sys, glob, os, json
mdir = sys.argv[1]
vers = sorted(glob.glob(os.path.join(mdir, "v*")))
assert vers, f"no version dir under {mdir}"
base = vers[-1]
cks = sorted(glob.glob(os.path.join(base, "checkpoint-*")),
             key=lambda p: int(p.rsplit('-', 1)[-1]))
assert cks, f"no checkpoints under {base}"
best = None
ts = os.path.join(cks[-1], "trainer_state.json")
if os.path.exists(ts):
    best = json.load(open(ts)).get("best_model_checkpoint")
print(best if best and os.path.isdir(best) else cks[-1])
PY
}

log "START ft-eval | CSV=$CSV | $(tail -n +2 "$CSV" | wc -l) rows"

for M in spk_sim acc_sim; do
  ADAPTER=$(best_ckpt "$M")
  log "==================== METRIC: $M ===================="
  log "[$M] adapter: $ADAPTER"

  # 1. Build inference input with the SAME prompt used in training
  python csv_to_sft.py --mode infer --data-root "$DATA_ROOT" --target-metric "$M" \
      --csv "$CSV" --out "$OUTDIR/$TAG.$M.ft.jsonl"

  # 2. Fine-tuned inference (base + adapter)
  rm -f "$OUTDIR/$TAG.$M.ft.results.jsonl"
  bash infer_ft.sh "$BASE" "$ADAPTER" \
      "$OUTDIR/$TAG.$M.ft.jsonl" "$OUTDIR/$TAG.$M.ft.results.jsonl"

  if [ ! -s "$OUTDIR/$TAG.$M.ft.results.jsonl" ]; then
    log "[$M] ERROR: results missing/empty; inference failed. Aborting."; exit 1
  fi
  log "[$M] inference done ($(wc -l < "$OUTDIR/$TAG.$M.ft.results.jsonl") responses)"
done

# 3. Merge both metrics into one CSV (annotate CSV, then annotate that file)
log "merging predictions -> $OUT ..."
python swift_to_submission.py --results "$OUTDIR/$TAG.spk_sim.ft.results.jsonl" \
    --orig-csv "$CSV" --data-root "$DATA_ROOT" --target-metric spk_sim --out "$OUT"
python swift_to_submission.py --results "$OUTDIR/$TAG.acc_sim.ft.results.jsonl" \
    --orig-csv "$OUT" --data-root "$DATA_ROOT" --target-metric acc_sim --out "$OUT"

# 4. Score if labelled (dev-ID/dev-OOD carry spk_sim/acc_sim), else it's a submission
if head -1 "$CSV" | grep -q "spk_sim"; then
  log "==================== scoring ===================="
  python ../calculate_metrics.py --prediction-csv "$OUT" --ground-truth-csv "$CSV"
else
  log "Unlabelled CSV -> upload $OUT to CodaBench."
fi
log "DONE -> $OUT"
