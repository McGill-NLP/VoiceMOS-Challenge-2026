# set -euo pipefail
module load miniconda/3 && conda activate speecheval
export DATA_ROOT=../data/vmc2026_track3_train_phase_distro_v3_syn
CKPT=../../SpeechLLM-as-Judges/checkpoint
HERE=$(pwd)

# For each metric in {spk_sim, acc_sim}:
for M in spk_sim acc_sim; do
  # 1. CSV -> CompareEval JSONL (dedups on the wav pair)
  python csv_to_swift.py --data-root $DATA_ROOT --csv-path ../data/dev-ID.csv \
      --target-metric $M --out dev-ID.$M.jsonl

  # 2. Run the judge (swift). Tiny smoke test first: `head -5 dev-ID.$M.jsonl > tmp.jsonl`
  cd ../../SpeechLLM-as-Judges/script
  CUDA_VISIBLE_DEVICES=0 bash inference.sh \
      "$CKPT" \
      "$HERE/dev-ID.$M.jsonl" \
      "$HERE/dev-ID.$M.results.jsonl"
  cd "$HERE"

  # swift crashes without a nonzero exit sometimes; make failures loud.
  if [ ! -s "dev-ID.$M.results.jsonl" ]; then
    echo "ERROR: dev-ID.$M.results.jsonl missing/empty; inference failed. Aborting." >&2
    exit 1
  fi

  # 3. Results JSONL -> submission CSV with pred_$M column
  #    (--data-root lets it rejoin on the absolute audios paths swift keeps)
  python swift_to_submission.py --results dev-ID.$M.results.jsonl \
      --orig-csv ../data/dev-ID.csv --data-root $DATA_ROOT \
      --target-metric $M --out dev-ID.pred_$M.csv
done

# 4. Score (dev-ID/dev-OOD carry labels; official dev.csv does not)
python ../calculate_metrics.py --prediction-csv dev-ID.pred_spk_sim.csv \
    --ground-truth-csv ../data/dev-ID.csv