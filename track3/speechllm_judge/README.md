# SpeechLLM-as-Judges → Track 3

Using [SpeechLLM-as-Judges](SpeechLLM-as-Judges) (SQ-LLM, a Qwen2.5-Omni judge;
upstream repo vendored into this folder) for Track 3 speaker/accent similarity.
Two experiments live here:

- **A. Zero-shot judge** — prompt the released checkpoint for a 1–5 similarity
  rating and parse it. *Result: poor* (see below).
- **B. LoRA fine-tuning** — SFT the checkpoint on Track 3 `train.csv`, one
  adapter per metric. *Result: much better than zero-shot, but the model
  collapses toward the majority class (5); needs class balancing.*

Run everything from this folder (`track3/speechllm_judge/`). All generated
`.jsonl`/`.csv` land in `predictions/`; run logs (via `tee`) in `logs/`.

## Layout

```
speechllm_judge/
├── csv_to_swift.py            # zero-shot: Track 3 CSV -> CompareEval JSONL
├── csv_to_sft.py              # fine-tune: Track 3 CSV -> SFT JSONL (--mode train|infer)
├── swift_to_submission.py     # results JSONL -> prediction CSV (shared)
├── infer.sh                   # zero-shot swift infer wrapper (+ repetition_penalty)
├── infer_ft.sh                # fine-tuned swift infer wrapper (base + --adapters)
├── pipeline-speech-llm.sh     # A: zero-shot dev-ID run + scoring
├── pipeline-eval-codabench.sh # A: zero-shot official dev.csv -> submission
├── pipeline-finetune.sh       # B: LoRA SFT core (parameterized by metric)
├── finetune-spk_sim.sh        # B: wrapper -> pipeline-finetune.sh spk_sim
├── finetune-acc_sim.sh        # B: wrapper -> pipeline-finetune.sh acc_sim
├── pipeline-eval-ft.sh        # B: evaluate/submit with the fine-tuned adapters
├── jobs/                      # Slurm scripts (one per metric) that run the wrappers
├── SpeechLLM-as-Judges/       # vendored upstream repo + checkpoint/
├── sft/                       # generated SFT train/val jsonl
├── models/                    # trained LoRA adapters: sft_spk_sim/, sft_acc_sim/
├── predictions/               # all generated .jsonl and .csv
└── logs/                      # tee'd run logs
```

---

## A. Zero-shot judge

```bash
module load miniconda/3 && conda activate speecheval
# labelled split (prints LCC/SRCC/MSE):
bash pipeline-speech-llm.sh
# official unlabelled dev.csv -> predictions/dev-eval.pred_submission.csv:
bash pipeline-eval-codabench.sh
```

Per metric these: `csv_to_swift.py` (CSV → CompareEval JSONL) → `infer.sh`
(swift infer) → `swift_to_submission.py` (→ CSV with `pred_<metric>`). Prompts
are at the top of `csv_to_swift.py`.

**Finding — zero-shot does not work for this task.** SQ-LLM was trained to judge
speech *quality*, not speaker/accent *similarity* (the paper lists "speaker
consistency" as future work). It scores "synthetic vs natural" instead of
"same accent?", so identical-accent pairs (GT 5) get 1–3. Prompt reshaping +
`repetition_penalty` fixed the format/looping but not the accuracy — the model
is answering the wrong question.

---

## B. LoRA fine-tuning (recommended)

Vanilla LoRA SFT of SQ-LLM on `train.csv`, **one separate adapter per metric**,
speech encoder frozen, initialised from the released checkpoint. Direct
`Score: X` supervision (empty `<think>` scaffold), no CoT, no GRPO.

**Prompts / format.** `csv_to_sft.py` holds the concise per-metric `SFT_PROMPTS`
and a fixed `system` turn (`You are a helpful assistant.`). Each training record
is system → user (metric prompt + two `<audio>`) → assistant
`<think>\n</think>\n\n<answer>\nScore: X\n</answer>`. The **same** file builds
inference input (`--mode infer`), so train/inference prompts always match — edit
`SFT_PROMPTS` only if you retrain.

### Train

Interactively (both metrics, sequentially):
```bash
bash pipeline-finetune.sh                 # or: bash pipeline-finetune.sh spk_sim
```

On Slurm, two parallel single-GPU jobs (one adapter each):
```bash
sbatch jobs/finetune-spk_sim.sh
sbatch jobs/finetune-acc_sim.sh
```
Adapters land in `models/sft_spk_sim/` and `models/sft_acc_sim/`. Tune via env,
e.g. `EPOCHS=3 bash pipeline-finetune.sh`.

### Evaluate / submit

```bash
bash pipeline-eval-ft.sh                      # dev-ID  -> LCC/SRCC/MSE
bash pipeline-eval-ft.sh ../data/dev-OOD.csv  # dev-OOD
bash pipeline-eval-ft.sh ../data/vmc2026_track3_train_phase_distro_v3_syn/sets/dev.csv
#   -> predictions/dev.ft.pred.csv  (CodaBench submission)
```

Uses the **latest** checkpoint per metric by default; `USE_BEST=1` uses the
lowest-val-loss one; `ADAPTER_spk_sim=/path ADAPTER_acc_sim=/path` pins exact
checkpoints. It auto-scores when the CSV carries labels, else prints the
submission path.

**Finding — fine-tuning fixes the zero-shot error but under-discriminates.**
Identical-accent pairs now correctly score 5 (zero-shot got 1–3). However the
model collapses toward the majority class: on the official dev set ~96% of
accent predictions were `5` (labels are ~47% fives, repeated across
listener-wise rows). Correlation on the labelled splits is the real test.
Levers if collapsed: class-balance the training data (downsample 5s / weight the
loss), train on mean-per-pair labels, tune LR/epochs, or switch to an ordinal
head (CORAL/CORN) like the official baseline.

---

## Notes

- **Score parsing** (`swift_to_submission.py`): prefers the `<answer>` block,
  takes the last `Score: X` (1–5), then an `X/5` fraction, then keywords, then
  `--default-score`. The stats line reports explicit/fraction/keyword/default —
  a high `default` count means the output format drifted.
- **Rejoining outputs**: swift's result JSONL drops custom fields but keeps the
  absolute `audios` paths; `swift_to_submission.py` rejoins on those via
  `--data-root` (keyed to `wav_a_path,wav_b_path`, matching `calculate_metrics.py`).
- **Two prompt sets, on purpose**: verbose reasoning prompts for zero-shot
  (`csv_to_swift.py`); concise no-reasoning prompts for fine-tuning
  (`csv_to_sft.py`, matching the empty-`<think>` target).

<!-- ## Gotchas (already handled, noted for the record)

- **`swift: command not found`** — swift only exists inside the `speecheval`
  env; always `module load miniconda/3 && conda activate speecheval` first.
- **`AssertionError: py_dir: /apdcephfs/.../evaluate/plugin`** — the downloaded
  checkpoint's `args.json` hard-codes the authors' GRPO reward-plugin path. It's
  training-only; inference doesn't need it. We set `"external_plugins": []` in
  `SpeechLLM-as-Judges/checkpoint/args.json`. Re-apply if you re-download.
- **`ValueError: remaining_argv: ['--train_type', 'lora']`** — swift 4.4.2 uses
  `--tuner_type`, not `--train_type` (the upstream `qwenomni_train_ratio.sh`
  predates this). Training scripts already use `--tuner_type`.
- **`MissingCUDAException: CUDA_HOME does not exist`** (training only) — `swift
  sft` pulls in deepspeed via accelerate, whose import needs `CUDA_HOME`. Fixed
  by `module load cudatoolkit/12.6` in the training scripts. Inference doesn't
  need it. -->
