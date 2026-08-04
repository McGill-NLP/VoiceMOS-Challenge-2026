#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fit an affine recalibration on a labelled split and apply it to a prediction CSV.

Why this exists. The contrastive term constrains only score *differences*, so it fixes no
absolute scale: with nothing anchoring the offset, predictions settle near the tanh centre
of the range-clipped head (3.0) while the Track 3 labels average ~4.0. Rank statistics do
not care -- SRCC and LCC are invariant under a positive affine map -- but MSE is wrecked,
which makes the contrastive arm look far worse than it is.

Fitting y = a * pred + b by least squares recovers the MSE and leaves SRCC/LCC untouched.
Measured on the contrastive arm with the fit taken from the training split:

    spk_sim   a=0.965 b=+0.979   utt MSE 0.885 -> 0.458   sys MSE 0.609 -> 0.067
    acc_sim   a=0.973 b=+1.132   utt MSE 1.364 -> 0.437   sys MSE 1.140 -> 0.051

Both slopes are ~1.0: this is almost purely an intercept, exactly as the theory predicts.

FIT ON TRAIN, NEVER ON THE SPLIT BEING SCORED. Fitting on dev and then reporting dev
metrics is contamination -- only two parameters, but it is still reading the answer.
--fit-csv must therefore be predictions over the training pairs.

    python inference.py --data-root $DR --csv-path train.mean.csv \\
        --checkpoint egs/spk_sim_contrastive/model_best_spk_sim.pt --out train_pred.csv
    python recalibrate.py --fit-csv train_pred.csv \\
        --apply-csv egs/spk_sim_contrastive/dev_spk_sim.csv \\
        --out egs/spk_sim_contrastive/dev_spk_sim.recal.csv
"""

import argparse
import csv
import logging


def infer_target_metric(fieldnames):
    """The prediction column is pred_<metric>; recover <metric> from it."""
    preds = [f for f in fieldnames if f.startswith("pred_")]
    if len(preds) != 1:
        raise SystemExit(
            f"Expected exactly one 'pred_*' column, found {preds or 'none'}."
        )
    return preds[0][len("pred_"):]


def read_pairs(path, target_metric, require_labels):
    """Returns (preds, labels) for rows that carry both. Labels may be absent."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{path} is empty.")

    pred_col = f"pred_{target_metric}"
    if pred_col not in rows[0]:
        raise SystemExit(f"{path} has no '{pred_col}' column.")

    preds, labels = [], []
    for row in rows:
        preds.append(float(row[pred_col]))
        raw = row.get(target_metric, "")
        labels.append(float(raw) if raw not in (None, "") else None)

    if require_labels and not any(l is not None for l in labels):
        raise SystemExit(
            f"{path} has no '{target_metric}' label column; the fit split must be labelled."
        )
    return rows, preds, labels


def fit_affine(preds, labels):
    """Least-squares y = a*x + b over rows where a label is present. Plain arithmetic so
    this stays dependency-free."""
    pairs = [(p, l) for p, l in zip(preds, labels) if l is not None]
    if len(pairs) < 2:
        raise SystemExit("Need at least 2 labelled rows to fit a line.")

    n = len(pairs)
    mx = sum(p for p, _ in pairs) / n
    my = sum(l for _, l in pairs) / n
    sxx = sum((p - mx) ** 2 for p, _ in pairs)
    sxy = sum((p - mx) * (l - my) for p, l in pairs)
    if sxx == 0:
        raise SystemExit("All predictions are identical; cannot fit a slope.")

    a = sxy / sxx
    b = my - a * mx
    return a, b, n


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fit-csv", required=True,
                   help="Predictions over a LABELLED split -- use the training pairs.")
    p.add_argument("--apply-csv", required=True, help="Predictions to recalibrate.")
    p.add_argument("--out", required=True, help="Where to write the recalibrated CSV.")
    p.add_argument("--target-metric", default=None,
                   help="Inferred from the pred_* column when omitted.")
    p.add_argument("--clip", action="store_true",
                   help="Clamp outputs to [1, 5]. Off by default: clamping creates ties at "
                        "the bounds and so can perturb SRCC, losing the guarantee that "
                        "recalibration leaves the ranking untouched.")
    p.add_argument("--verbose", type=int, default=1)
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARN,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    with open(args.fit_csv, newline="", encoding="utf-8") as f:
        fit_fields = csv.DictReader(f).fieldnames
    target_metric = args.target_metric or infer_target_metric(fit_fields)

    _, fit_preds, fit_labels = read_pairs(args.fit_csv, target_metric, require_labels=True)
    a, b, n = fit_affine(fit_preds, fit_labels)
    logging.info(
        f"Fitted on {n} labelled rows from {args.fit_csv}: "
        f"{target_metric} = {a:.4f} * pred + {b:.4f}"
    )
    if a <= 0:
        logging.warning(
            f"Slope is {a:.4f} <= 0, so the mapping REVERSES the ranking. That means the "
            "model is anti-correlated with the labels on the fit split; do not ship this."
        )

    rows, apply_preds, _ = read_pairs(args.apply_csv, target_metric, require_labels=False)
    pred_col = f"pred_{target_metric}"
    n_clipped = 0
    for row, value in zip(rows, apply_preds):
        recal = a * value + b
        if args.clip:
            clamped = min(5.0, max(1.0, recal))
            n_clipped += clamped != recal
            recal = clamped
        row[pred_col] = recal

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    logging.info(f"Wrote {len(rows)} recalibrated rows to {args.out}"
                 + (f" ({n_clipped} clamped to [1, 5])" if args.clip else ""))


if __name__ == "__main__":
    main()
