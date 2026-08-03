#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Aggregate a listener-wise split into one row per (wav_a_path, wav_b_path) pair.

The local splits from ../corn-and-coral/build_splits.py (train.csv, dev-ID.csv,
dev-OOD.csv) carry one row per listener rating, so a pair appears ~5 times. Feeding
those straight to calculate_metrics.py silently keeps whichever listener happens to
come last, because it keys gt_dict on the pair. This writes the mean instead, which is
the quantity the challenge scores.

The output doubles as the inference input: it has the pair columns inference.py needs
and the label columns calculate_metrics.py needs, and it avoids re-predicting the same
pair once per listener.

    python make_eval_gt.py --in ../baseline/data/dev-OOD.csv --out egs/dev-OOD.mean.csv
"""

import argparse
import csv
from collections import OrderedDict


def main():
    parser = argparse.ArgumentParser(description="Average listener-wise ratings per pair.")
    parser.add_argument("--in", dest="inp", required=True, type=str, help="Listener-wise CSV.")
    parser.add_argument("--out", required=True, type=str, help="Per-pair CSV to write.")
    parser.add_argument("--metrics", nargs="+", default=["spk_sim", "acc_sim"],
                        help="Label columns to average (skipped if absent).")
    args = parser.parse_args()

    with open(args.inp, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{args.inp} is empty.")

    metrics = [m for m in args.metrics if m in rows[0]]
    carry = [c for c in ("system_id", "utterance_id", "ood_type") if c in rows[0]]

    pairs = OrderedDict()
    for row in rows:
        key = (row["wav_a_path"], row["wav_b_path"])
        if key not in pairs:
            pairs[key] = {c: row[c] for c in carry}
            pairs[key].update({m: [] for m in metrics})
        for m in metrics:
            if row[m].strip():
                pairs[key][m].append(float(row[m]))

    fieldnames = carry + ["wav_a_path", "wav_b_path"] + metrics
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (wav_a, wav_b), data in pairs.items():
            out = {c: data[c] for c in carry}
            out["wav_a_path"], out["wav_b_path"] = wav_a, wav_b
            for m in metrics:
                out[m] = sum(data[m]) / len(data[m]) if data[m] else ""
            writer.writerow(out)

    n_ratings = sum(len(d[metrics[0]]) for d in pairs.values()) if metrics else 0
    print(f"{args.inp}: {len(rows)} rows ({n_ratings} ratings) -> {len(pairs)} pairs -> {args.out}")


if __name__ == "__main__":
    main()
