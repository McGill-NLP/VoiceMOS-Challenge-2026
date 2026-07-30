#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Score a prediction CSV against a ground-truth CSV.

Same metric definitions as the baseline's `calculate_metrics.py`, with one fix:
the ground truth here is *listener-wise*, so multiple rows share a
(wav_a_path, wav_b_path) key. The baseline builds a dict keyed on that pair and
therefore keeps only whichever listener happens to come last; this aggregates by
mean over listeners instead, which is the quantity the models are trained on.
"""

import argparse
import csv
import logging
from collections import defaultdict

from metrics import evaluate, format_metrics


def load_truth(path, metric):
    """(wav_a, wav_b) -> (mean score, system_id, n_ratings)."""
    scores, systems = defaultdict(list), {}
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["wav_a_path"], row["wav_b_path"])
            systems[key] = row.get("system_id", "")
            val = row.get(metric, "")
            if val is not None and str(val).strip():
                scores[key].append(float(val))
    return {
        k: (sum(v) / len(v), systems[k], len(v))
        for k, v in scores.items() if v
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prediction-csv", required=True)
    p.add_argument("--ground-truth-csv", required=True)
    p.add_argument("--metrics", nargs="*", default=None,
                   help="Defaults to whichever pred_* columns are present.")
    p.add_argument("--held-out-from", default=None, metavar="TRAIN_CSV",
                   help="Additionally report metrics restricted to audio pairs "
                        "absent from this CSV. Useful for any split built at the "
                        "(system_id, listener_id) level: a pair rated by both a "
                        "training listener and a held-out listener lands in both "
                        "splits, so the overall number blends memorization with "
                        "generalization.")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    train_pairs = set()
    if args.held_out_from:
        with open(args.held_out_from, "r", encoding="utf-8") as f:
            train_pairs = {(r["wav_a_path"], r["wav_b_path"]) for r in csv.DictReader(f)}

    with open(args.prediction_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        preds = list(reader)

    metrics = args.metrics or [
        m for m in ("spk_sim", "acc_sim") if f"pred_{m}" in headers
    ]
    if not metrics:
        raise SystemExit("No pred_spk_sim / pred_acc_sim column in the prediction CSV.")

    for metric in metrics:
        truth = load_truth(args.ground_truth_csv, metric)
        rows = []
        for row in preds:
            key = (row["wav_a_path"], row["wav_b_path"])
            if key not in truth:
                continue
            value = row.get(f"pred_{metric}", "")
            if not str(value).strip():
                continue
            gold, system_id, _ = truth[key]
            rows.append((gold, float(value), system_id, key))

        if not rows:
            logging.warning(f"No overlapping pairs for {metric}; skipping.")
            continue

        n_missing = len(truth) - len(rows)
        if n_missing:
            logging.warning(f"{metric}: {n_missing} ground-truth pairs had no prediction.")

        def report(name, subset):
            if subset:
                print(format_metrics(name, evaluate(
                    [r[0] for r in subset], [r[1] for r in subset], [r[2] for r in subset]
                )))

        report(metric, rows)
        if train_pairs:
            unseen = [r for r in rows if r[3] not in train_pairs]
            seen = [r for r in rows if r[3] in train_pairs]
            print(f"  (of {len(rows)} pairs, {len(unseen)} are absent from "
                  f"{args.held_out_from} and {len(seen)} were seen in training)")
            report(f"{metric}/unseen", unseen)
            report(f"{metric}/seen", seen)


if __name__ == "__main__":
    main()
