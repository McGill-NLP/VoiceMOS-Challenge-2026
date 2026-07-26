# SpeechLLM-as-Judges → Track 3 (zero-shot judge)

Glue for using [SpeechLLM-as-Judges](SpeechLLM-as-Judges) (the upstream repo,
vendored into this folder) as a zero-shot
judge for Track 3 speaker/accent similarity. The pretrained model does free-text
`CompareEval`; we prompt it for a 1–5 similarity rating (one prompt per target
metric) and parse the score back into a Track 3 submission CSV.

> ⚠️ The pretrained model was trained to compare **speech quality**, not
> speaker/accent similarity. Treat this as a zero-shot probe and inspect a few
> raw outputs before trusting the scores.

## Layout

Run everything from this folder (`track3/speechllm_judge/`):

```
speechllm_judge/
├── csv_to_swift.py            # Track 3 CSV -> CompareEval JSONL
├── swift_to_submission.py     # swift results JSONL -> prediction CSV
├── pipeline-speech-llm.sh     # dev-ID run + scoring (labelled split)
├── pipeline-eval-codabench.sh # official dev.csv run -> CodaBench submission
├── SpeechLLM-as-Judges/       # vendored upstream repo + checkpoint/
├── predictions/               # all generated .jsonl and .csv land here
└── logs/                      # tee'd run logs
```

## Pipeline

The two `pipeline-*.sh` scripts run this end to end and write everything to
`predictions/`. To run it by hand (illustrative — the labelled `dev-ID` split):

```bash
module load miniconda/3 && conda activate speecheval
export DATA_ROOT=../data/vmc2026_track3_train_phase_distro_v3_syn
HERE=$(pwd)
SLM=$HERE/SpeechLLM-as-Judges
CKPT=$SLM/checkpoint
OUTDIR=$HERE/predictions
mkdir -p "$OUTDIR"

# For each metric in {spk_sim, acc_sim}:
for M in spk_sim acc_sim; do
  # 1. CSV -> CompareEval JSONL (dedups on the wav pair)
  python csv_to_swift.py --data-root $DATA_ROOT --csv-path ../data/dev-ID.csv \
      --target-metric $M --out "$OUTDIR/dev-ID.$M.jsonl"

  # 2. Run the judge (swift). Smoke test first: `head -5 predictions/dev-ID.$M.jsonl > tmp.jsonl`
  cd "$SLM/script"
  CUDA_VISIBLE_DEVICES=0 bash inference.sh \
      "$CKPT" \
      "$OUTDIR/dev-ID.$M.jsonl" \
      "$OUTDIR/dev-ID.$M.results.jsonl"
  cd "$HERE"

  # 3. Results JSONL -> submission CSV with pred_$M column
  #    swift drops custom fields but keeps the absolute `audios` paths, so we
  #    pass --data-root to rejoin results onto the relative CSV wav paths.
  python swift_to_submission.py --results "$OUTDIR/dev-ID.$M.results.jsonl" \
      --orig-csv ../data/dev-ID.csv --data-root $DATA_ROOT \
      --target-metric $M --out "$OUTDIR/dev-ID.pred_$M.csv"
done

# 4. Score (dev-ID/dev-OOD carry labels; official dev.csv does not)
python ../calculate_metrics.py --prediction-csv "$OUTDIR/dev-ID.pred_spk_sim.csv" \
    --ground-truth-csv ../data/dev-ID.csv
```

`pipeline-eval-codabench.sh` runs the same steps on the unlabelled official
`dev.csv` and merges both metrics into `predictions/dev-eval.pred_submission.csv`
(both are keyed on `wav_a_path,wav_b_path`).

## Notes

- **Score parsing** (`swift_to_submission.py`): takes the last `Score: X`
  (1–5) in the response; falls back to keyword matching, then `--default-score`.
  The per-file stats line reports how many were explicit / keyword / default —
  watch it; a high `default` count means the prompt or parser needs tuning.
- **Prompts** live at the top of `csv_to_swift.py`; edit them there.
- For the unlabeled official `dev.csv`, skip step 4 and submit the CSV to
  CodaBench.

## Gotchas (already handled, noted for the record)

- **`swift: command not found`** — swift only exists inside the `speecheval`
  env; always `module load miniconda/3 && conda activate speecheval` first.
- **`AssertionError: py_dir: /apdcephfs/.../evaluate/plugin`** — the downloaded
  checkpoint's `args.json` hard-codes the authors' GRPO reward-plugin path. It's
  training-only; inference doesn't need it. We set `"external_plugins": []` in
  `SpeechLLM-as-Judges/checkpoint/args.json`. Re-apply if you re-download.
- **Rejoining outputs** — swift's result JSONL drops custom fields (our `key`)
  but keeps the absolute `audios` paths; `swift_to_submission.py` matches on
  those via `--data-root`. The model answers as `<answer>\nScore: N\n</answer>`,
  which the `Score: X` regex handles.
