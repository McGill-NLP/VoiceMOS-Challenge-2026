#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build swift SFT data (or matching inference input) from Track 3 CSVs.

Vanilla LoRA fine-tuning of SQ-LLM (Qwen2.5-Omni) for one similarity metric.
Each pair becomes a chat sample: a fixed system turn, a concise user prompt with
two <audio> placeholders, and — in train mode — an assistant target using the
model's native empty-<think> scaffold ending in "Score: X" (cf. the deepfake
example in SpeechLLM-as-Judges/example/output.json, which is exactly
"<think>\\n</think>\\n\\n<answer>\\nfake\\n</answer>").

The SAME prompt is used for training and for inference on the fine-tuned model,
so this one file is the single source of truth for both — pass --mode infer to
emit prompt-only records (no assistant turn) for a CSV with or without labels.

  # training data (labelled train.csv -> train + val jsonl, split by pair)
  python csv_to_sft.py --mode train --data-root DR --target-metric spk_sim \
      --csv ../data/train.csv --out-train sft/spk_sim.train.jsonl \
      --out-val sft/spk_sim.val.jsonl

  # inference input for the fine-tuned model (same prompt, no target)
  python csv_to_sft.py --mode infer --data-root DR --target-metric spk_sim \
      --csv ../data/dev-ID.csv --out predictions/dev-ID.spk_sim.ft.jsonl
"""

import argparse
import csv
import json
import os
import random

SYSTEM = "You are a helpful assistant."

# Concise, task-specific prompts. NO chain-of-thought instruction: the target
# uses an empty <think>, so the prompt must not ask the model to reason.
SFT_PROMPTS = {
    "spk_sim": (
        "Sample A: <audio> Sample B: <audio> "
        "Sample A is synthetic speech and Sample B is a natural reference "
        "recording from the target speaker. Rate how similar the SPEAKER "
        "IDENTITY (voice timbre and vocal characteristics) of Sample A is to "
        "Sample B, ignoring the spoken content, on a scale of 1 (clearly a "
        "different speaker) to 5 (the same speaker). Answer with a single line "
        "'Score: X'."
    ),
    "acc_sim": (
        "Sample A: <audio> Sample B: <audio> "
        "Sample A is synthetic speech and Sample B is a natural reference "
        "recording from the target speaker. Rate how similar the ACCENT "
        "(pronunciation, prosody, regional articulation) of Sample A is to "
        "Sample B, ignoring the spoken content, on a scale of 1 (clearly a "
        "different accent) to 5 (the same accent). Answer with a single line "
        "'Score: X'."
    ),
}

TARGET_TEMPLATE = "<think>\n</think>\n\n<answer>\nScore: {x}\n</answer>"


def make_record(metric, wav_a, wav_b, score=None):
    """Build one swift chat record; include an assistant turn iff score given."""
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": SFT_PROMPTS[metric]},
    ]
    if score is not None:
        messages.append({"role": "assistant",
                         "content": TARGET_TEMPLATE.format(x=score)})
    return {"messages": messages, "audios": [wav_a, wav_b]}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True, choices=["train", "infer"])
    p.add_argument("--data-root", required=True)
    p.add_argument("--target-metric", required=True, choices=["spk_sim", "acc_sim"])
    p.add_argument("--csv", required=True, help="Source Track 3 CSV.")
    # train mode
    p.add_argument("--out-train", help="[train] output train jsonl")
    p.add_argument("--out-val", help="[train] output val jsonl")
    p.add_argument("--val-frac", type=float, default=0.05,
                   help="[train] fraction of unique pairs held out for validation")
    p.add_argument("--seed", type=int, default=42)
    # infer mode
    p.add_argument("--out", help="[infer] output jsonl (prompt-only)")
    args = p.parse_args()

    data_root = os.path.abspath(args.data_root)
    metric = args.target_metric

    def abspair(row):
        return (os.path.join(data_root, row["wav_a_path"]),
                os.path.join(data_root, row["wav_b_path"]))

    with open(args.csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if args.mode == "infer":
        assert args.out, "--out required in infer mode"
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        seen = set()
        n = 0
        with open(args.out, "w", encoding="utf-8") as out:
            for r in rows:
                pair = (r["wav_a_path"], r["wav_b_path"])
                if pair in seen:
                    continue
                seen.add(pair)
                a, b = abspair(r)
                out.write(json.dumps(make_record(metric, a, b), ensure_ascii=False) + "\n")
                n += 1
        print(f"infer: wrote {n} unique-pair prompts to {args.out}")
        return

    # ---- train mode: pair-level split so no pair straddles train/val ----
    assert args.out_train and args.out_val, "--out-train and --out-val required in train mode"
    pairs = sorted({(r["wav_a_path"], r["wav_b_path"]) for r in rows})
    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    n_val = max(1, int(len(pairs) * args.val_frac))
    val_pairs = set(pairs[:n_val])
    print(f"{len(pairs)} unique pairs -> {len(val_pairs)} val / {len(pairs)-len(val_pairs)} train pairs")

    for path in (args.out_train, args.out_val):
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

    n_tr = n_va = 0
    with open(args.out_train, "w", encoding="utf-8") as ftr, \
         open(args.out_val, "w", encoding="utf-8") as fva:
        for r in rows:  # listener-wise rows: every rating is a training example
            score = r.get(metric, "")
            if score in (None, ""):
                continue
            a, b = abspair(r)
            line = json.dumps(make_record(metric, a, b, int(round(float(score)))),
                              ensure_ascii=False) + "\n"
            if (r["wav_a_path"], r["wav_b_path"]) in val_pairs:
                fva.write(line); n_va += 1
            else:
                ftr.write(line); n_tr += 1
    print(f"train: wrote {n_tr} rows -> {args.out_train}")
    print(f"val:   wrote {n_va} rows -> {args.out_val}")


if __name__ == "__main__":
    main()
