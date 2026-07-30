#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run a trained model over a CSV and write per-pair predictions.

Works on labelled split CSVs and on the unlabelled official dev.csv alike.
Output has one row per unique (wav_a_path, wav_b_path) pair with a
`pred_<target_metric>` column.
"""

import argparse
import csv
import logging
import os
import time

import torch

from data import build_loader
from metrics import evaluate as eval_metrics, format_metrics
from model import ECAPA, Model


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True)
    p.add_argument("--csv-path", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--checkpoint", required=True, help="Checkpoint from train_head.py.")
    p.add_argument("--target-metric", required=True, choices=["spk_sim", "acc_sim"])
    p.add_argument("--head", choices=["mlp", "linear"], default=None,
                   help="Defaults to the value stored in the checkpoint.")
    p.add_argument("--no-range-clipping", action="store_true")
    p.add_argument("--model-name", default=ECAPA)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-audio-sec", type=float, default=None)
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    saved = ckpt.get("args", {})
    head = args.head or saved.get("head", "mlp")
    range_clipping = not (args.no_range_clipping or saved.get("no_range_clipping", False))
    logging.info(f"Head={head} range_clipping={range_clipping} (step {ckpt.get('step')})")

    model = Model(
        model_name=args.model_name, use_projection=True,
        freeze_ecapa=not saved.get("unfreeze_ecapa", False),
        ecapa_eval_mode=saved.get("ecapa_eval_mode", False),
        head=head, range_clipping=range_clipping,
    )
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.to(device).eval()

    dataset, loader = build_loader(
        args.data_root, args.csv_path, args.target_metric,
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        max_audio_sec=args.max_audio_sec, train=False,
    )

    rows, start = [], time.time()
    with torch.no_grad():
        for batch in loader:
            preds = model(
                batch["wav_a"].to(device), batch["wav_b"].to(device),
                batch["wav_a_lengths"].to(device), batch["wav_b_lengths"].to(device),
                batch["b_index"].to(device),
            ).float().cpu().tolist()
            scores = batch["score"].tolist() if batch["score"] is not None else [None] * len(preds)
            for i, pred in enumerate(preds):
                rows.append({
                    "system_id": batch["system_id"][i],
                    "utterance_id": batch["utterance_id"][i],
                    "wav_a_path": batch["wav_a_path"][i],
                    "wav_b_path": batch["wav_b_path"][i],
                    f"pred_{args.target_metric}": pred,
                    args.target_metric: scores[i],
                })

    elapsed = time.time() - start
    logging.info(f"{len(rows)} pairs in {elapsed:.1f}s ({elapsed / max(len(rows), 1):.3f}s/pair)")

    labelled = [r for r in rows if r[args.target_metric] is not None]
    if labelled:
        m = eval_metrics(
            [r[args.target_metric] for r in labelled],
            [r[f"pred_{args.target_metric}"] for r in labelled],
            [r["system_id"] for r in labelled],
        )
        logging.info(format_metrics(os.path.basename(args.csv_path), m))

    # Keep the ground-truth column only when there is ground truth. On the
    # unlabelled official dev.csv this yields exactly the baseline's submission
    # header: system_id,utterance_id,wav_a_path,wav_b_path,pred_<metric>
    fieldnames = ["system_id", "utterance_id", "wav_a_path", "wav_b_path",
                  f"pred_{args.target_metric}"]
    if labelled:
        fieldnames.append(args.target_metric)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    logging.info(f"Wrote {args.out} ({len(rows)} rows, columns: {', '.join(fieldnames)})")


if __name__ == "__main__":
    main()
