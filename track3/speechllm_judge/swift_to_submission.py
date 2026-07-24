#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Parse SpeechLLM-as-Judges inference output into a Track 3 submission CSV.

Reads the swift `--result_path` JSONL (each line has the request plus a
generated `response`), extracts the integer the model was asked to emit as
"Score: X", and joins it back onto the original Track 3 CSV to produce a file
with a `pred_spk_sim` (or `pred_acc_sim`) column that calculate_metrics.py can
score. Pairs are matched on (wav_a_path, wav_b_path).

Example:
    python swift_to_submission.py \
        --results ../checkpoint/results_spk.jsonl \
        --orig-csv ../data/dev-ID.csv \
        --target-metric spk_sim \
        --out dev-ID.pred_spk.csv
"""

import argparse
import csv
import json
import os
import re

# 1-5 fallback keyword map, used only when no explicit "Score: X" is found.
KEYWORD_SCORES = [
    (re.compile(r"\b(identical|indistinguishable|exactly the same|same speaker|same accent)\b", re.I), 5),
    (re.compile(r"\b(very similar|highly similar|nearly identical|almost identical)\b", re.I), 4),
    (re.compile(r"\b(somewhat similar|moderately similar|fairly similar|similar)\b", re.I), 3),
    (re.compile(r"\b(slightly similar|somewhat different|noticeably different)\b", re.I), 2),
    (re.compile(r"\b(completely different|clearly different|very different|not similar|dissimilar)\b", re.I), 1),
]

SCORE_RE = re.compile(r"score\s*[:=]?\s*([1-5])(?:\s*/\s*5)?", re.I)


def extract_response(rec):
    """Pull the assistant's generated text out of a swift result record."""
    if isinstance(rec.get("response"), str):
        return rec["response"]
    # fallback: last assistant turn in messages
    for msg in reversed(rec.get("messages", [])):
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    return ""


def score_from_text(text, default):
    m = list(SCORE_RE.finditer(text))
    if m:
        return int(m[-1].group(1)), "explicit"
    for pat, val in KEYWORD_SCORES:
        if pat.search(text):
            return val, "keyword"
    return default, "default"


def pair_key_from_result(rec, data_root):
    """Recover (wav_a_path, wav_b_path) relative-path key from a result record.

    swift drops custom fields (our `key`) but preserves the resolved absolute
    `audios` paths, so we rebuild the relative CSV key from those.
    """
    audios = rec.get("audios")
    if isinstance(audios, list) and len(audios) == 2 and data_root:
        return (os.path.relpath(audios[0], data_root),
                os.path.relpath(audios[1], data_root))
    # fallback: legacy `key` field, if a record still carries it
    key = rec.get("key")
    if isinstance(key, str) and "||" in key:
        a, b = key.split("||", 1)
        return a, b
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, help="swift result JSONL.")
    parser.add_argument("--orig-csv", required=True, help="Original Track 3 CSV to annotate.")
    parser.add_argument("--data-root", required=True,
                        help="Same DATA_ROOT used in csv_to_swift.py; used to map the "
                             "absolute audios paths in the results back to relative CSV paths.")
    parser.add_argument("--target-metric", required=True, choices=["spk_sim", "acc_sim"])
    parser.add_argument("--out", required=True, help="Output submission CSV.")
    parser.add_argument("--default-score", type=float, default=3.0,
                        help="Score used when nothing parseable is found.")
    args = parser.parse_args()
    data_root = os.path.abspath(args.data_root)

    # 1. Build pair -> score map from the model outputs.
    scores = {}
    stats = {"explicit": 0, "keyword": 0, "default": 0, "unkeyed": 0}
    with open(args.results, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            pair = pair_key_from_result(rec, data_root)
            if pair is None:
                stats["unkeyed"] += 1
                continue
            score, how = score_from_text(extract_response(rec), args.default_score)
            scores[pair] = float(score)
            stats[how] += 1

    print(f"Parsed {len(scores)} pair scores from {args.results}: {stats}")

    # 2. Annotate the original CSV (one output row per input row).
    col = f"pred_{args.target_metric}"
    n_missing = 0
    with open(args.orig_csv, encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        fieldnames = list(reader.fieldnames)
        if col not in fieldnames:
            fieldnames.append(col)
        rows = list(reader)

    with open(args.out, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            pair = (row["wav_a_path"], row["wav_b_path"])
            if pair in scores:
                row[col] = scores[pair]
            else:
                row[col] = args.default_score
                n_missing += 1
            writer.writerow(row)

    if n_missing:
        print(f"WARNING: {n_missing} rows had no model score; filled with default {args.default_score}")
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
