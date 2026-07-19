#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import logging
import os
import time

import torch
import torchaudio
from tqdm import tqdm

from finetune import ModelV2


def main():
    parser = argparse.ArgumentParser(description="Inference for finetune_v2.py checkpoints.")
    parser.add_argument("--data-root", required=True, type=str)
    parser.add_argument("--csv-path", required=True, type=str)
    parser.add_argument("--out", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--listener-vocab", default=None, type=str,
                         help="Path to listener_vocab.json saved during training "
                              "(only needed to size the listener embedding table).")
    parser.add_argument("--target-metric", required=True, choices=["spk_sim", "acc_sim", "both"])
    parser.add_argument("--use-listener-bias", action="store_true",
                         help="Must match the flag used at training time.")
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose > 1 else logging.INFO if args.verbose > 0 else logging.WARN
    logging.basicConfig(level=log_level, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    target_metrics = ["spk_sim", "acc_sim"] if args.target_metric == "both" else [args.target_metric]

    num_listeners = 0
    if args.use_listener_bias:
        assert args.listener_vocab, "--listener-vocab is required when --use-listener-bias is set"
        with open(args.listener_vocab) as f:
            num_listeners = len(json.load(f))

    model = ModelV2(
        model_name="speechbrain/spkrec-ecapa-voxceleb",
        embedding_dim=256,
        use_projection=True,
        freeze_ssl=False,
        target_metrics=target_metrics,
        num_listeners=num_listeners,
        use_listener_bias=args.use_listener_bias,
    )
    logging.info(f"Loading weights from {args.checkpoint} ...")
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.to(device)
    model.eval()

    with open(args.csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    logging.info(f"Number of inference samples = {len(rows)}.")

    out_results = []
    start_time = time.time()
    for row in tqdm(rows, desc="[inference]"):
        wav_a_rel, wav_b_rel = row.get("wav_a_path"), row.get("wav_b_path")
        if not wav_a_rel or not wav_b_rel:
            logging.warning(f"Skipping row - missing audio paths: {row}")
            continue
        wav_a_path = os.path.join(args.data_root, wav_a_rel)
        wav_b_path = os.path.join(args.data_root, wav_b_rel)
        try:
            wav_a, sr_a = torchaudio.load(wav_a_path)
            wav_b, sr_b = torchaudio.load(wav_b_path)
            if sr_a != 16000:
                wav_a = torchaudio.functional.resample(wav_a, sr_a, 16000)
            if sr_b != 16000:
                wav_b = torchaudio.functional.resample(wav_b, sr_b, 16000)
            with torch.no_grad():
                # No listener_idx passed -> listener-independent mean head is used.
                out = model(wav_a.to(device), wav_b.to(device))
        except Exception as e:
            logging.error(f"Failed to process pair {wav_a_rel} and {wav_b_rel}: {e}")
            continue

        out_row = row.copy()
        for m in target_metrics:
            out_row[f"pred_{m}"] = out[m].item()
        out_results.append(out_row)

    total_time = time.time() - start_time
    logging.info(f"Total inference time = {total_time:.2f} secs.")
    if out_results:
        logging.info(f"Average speed = {total_time / len(out_results):.3f} sec / pair.")
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(out_results[0].keys()))
            writer.writeheader()
            writer.writerows(out_results)
        logging.info(f"Predictions saved to {args.out}")
    else:
        logging.warning("No results to save.")


if __name__ == "__main__":
    main()
