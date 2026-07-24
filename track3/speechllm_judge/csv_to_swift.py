#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert a Track 3 CSV into swift-style CompareEval JSONL for SpeechLLM-as-Judges.

Track 3 asks for a numeric speaker-similarity (spk_sim) and accent-similarity
(acc_sim) score between wav_a (synthetic) and wav_b (reference). The pretrained
SpeechLLM-as-Judges model does free-text CompareEval, so we phrase the question
as a similarity rating and force the model to end with "Score: X" for easy
parsing. We emit ONE JSONL per target metric (the model never sees listener_id,
so we deduplicate on the (wav_a_path, wav_b_path) pair).

Example:
    python csv_to_swift.py \
        --data-root ../data/vmc2026_track3_train_phase_distro_v3_syn \
        --csv-path ../data/dev-ID.csv \
        --target-metric spk_sim \
        --out dev-ID.spk_sim.jsonl
"""

import argparse
import csv
import json
import os

PROMPTS = {
    "spk_sim": (
        "Sample A: <audio> Sample B: <audio> "
        "Sample A is a synthetic speech sample and Sample B is a reference "
        "recording from the target speaker. Judge how similar the SPEAKER "
        "IDENTITY (voice timbre, vocal characteristics) of Sample A is to "
        "Sample B, ignoring differences in the spoken content. Reason briefly, "
        "then on the LAST line output an integer from 1 (clearly a different "
        "speaker) to 5 (indistinguishable, same speaker) in the exact format "
        "'Score: X'."
    ),
    "acc_sim": (
        "Sample A: <audio> Sample B: <audio> "
        "Sample A is a synthetic speech sample and Sample B is a reference "
        "recording from the target speaker. Judge how similar the ACCENT "
        "(pronunciation, prosody, regional articulation) of Sample A is to "
        "Sample B, ignoring differences in the spoken content. Reason briefly, "
        "then on the LAST line output an integer from 1 (clearly a different "
        "accent) to 5 (indistinguishable, same accent) in the exact format "
        "'Score: X'."
    ),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True,
                        help="Dataset root; relative wav paths in the CSV are joined to this.")
    parser.add_argument("--csv-path", required=True,
                        help="Track 3 CSV (train.csv / dev-ID.csv / dev-OOD.csv / dev.csv).")
    parser.add_argument("--target-metric", required=True, choices=["spk_sim", "acc_sim"])
    parser.add_argument("--out", required=True, help="Output JSONL path.")
    parser.add_argument("--absolute-audio", action="store_true", default=True,
                        help="Write absolute audio paths (default; swift needs resolvable paths).")
    args = parser.parse_args()

    data_root = os.path.abspath(args.data_root)
    prompt = PROMPTS[args.target_metric]

    seen = set()
    n_in = 0
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.csv_path, encoding="utf-8") as fin, \
         open(args.out, "w", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        for row in reader:
            n_in += 1
            wav_a_rel, wav_b_rel = row["wav_a_path"], row["wav_b_path"]
            pair = (wav_a_rel, wav_b_rel)
            if pair in seen:
                continue
            seen.add(pair)

            wav_a = os.path.join(data_root, wav_a_rel)
            wav_b = os.path.join(data_root, wav_b_rel)
            rec = {
                # keep the relative paths so swift_to_submission.py can rejoin
                # on the (wav_a_path, wav_b_path) key that calculate_metrics uses
                "key": f"{wav_a_rel}||{wav_b_rel}",
                "task": "CompareEval",
                "messages": [{"role": "user", "content": prompt}],
                "audios": [wav_a, wav_b],
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Read {n_in} rows -> wrote {len(seen)} unique pairs to {args.out}")


if __name__ == "__main__":
    main()
