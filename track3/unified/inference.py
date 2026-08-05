#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inference for the unified Track 3 model.

The architecture is read back from the checkpoint's stored config -- encoder, head type,
objective, MoE width -- so a model can never be rebuilt wrong by forgetting a flag, and
the target metric it was trained on decides the output column name.

Zero-shot mode (no --checkpoint) falls back to raw cosine similarity between the two
backbone embeddings, matching ../baseline/inference.py. Note that this is objective- and
head-independent by construction, and identical for spk_sim and acc_sim.
"""

import argparse
import csv
import logging
import os
import time

import torch
import torchaudio
from tqdm import tqdm

from encoders import ENCODER_REGISTRY
from model import UnifiedModel, build_from_config


def main():
    parser = argparse.ArgumentParser(description="Inference for VoiceMOS 2026 Track 3 (unified model).")
    parser.add_argument("--data-root", required=True, type=str, help="Root directory of the dataset distribution.")
    parser.add_argument("--csv-path", required=True, type=str, help="CSV to run inference over (e.g. sets/dev.csv).")
    parser.add_argument("--out", required=True, type=str, help="Where to write the predictions CSV.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Trained checkpoint. Omit for zero-shot cosine.")
    parser.add_argument("--encoder", type=str, default="ecapa-voxceleb",
                        help=f"Backbone for zero-shot runs. One of: {', '.join(ENCODER_REGISTRY)}. Ignored when --checkpoint is given.")
    parser.add_argument("--target-metric", type=str, default="spk_sim", choices=["spk_sim", "acc_sim"],
                        help="Output column for zero-shot runs. Read from the checkpoint otherwise.")
    parser.add_argument("--cache-dir", type=str, default=None, help="Where to cache downloaded encoder weights.")
    parser.add_argument("--verbose", type=int, default=1, help="Logging level.")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose > 1 else logging.INFO if args.verbose > 0 else logging.WARN
    logging.basicConfig(level=log_level, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    is_zero_shot = args.checkpoint is None

    if is_zero_shot:
        logging.info(f"ZERO-SHOT mode with encoder '{args.encoder}' (raw cosine similarity).")
        target_metric = args.target_metric
        model = UnifiedModel(
            encoder_name=args.encoder,
            target_metric=target_metric,
            use_projection=False,   # an untrained Linear would only scramble the embeddings
            cache_dir=args.cache_dir,
        )
    else:
        obj = torch.load(args.checkpoint, map_location="cpu")
        if not (isinstance(obj, dict) and "config" in obj and "state_dict" in obj):
            raise SystemExit(
                f"{args.checkpoint} is not a unified checkpoint (expected keys 'config' and 'state_dict')."
            )
        config, state_dict = obj["config"], obj["state_dict"]
        target_metric = config["target_metric"]
        logging.info(
            f"Rebuilding from checkpoint config: encoder={config['encoder']} "
            f"head={config.get('head', 'mlp')} objective={config.get('objective', 'mse')} "
            f"target={target_metric}"
        )
        model = build_from_config(config, cache_dir=args.cache_dir)
        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    logging.info(f"Loading dataset from {args.csv_path}")
    with open(args.csv_path, "r", encoding="utf-8") as f:
        dataset = list(csv.DictReader(f))
    logging.info(f"Number of inference samples = {len(dataset)}.")

    out_results = []
    start_time = time.time()

    for row in tqdm(dataset, desc="[inference]"):
        wav_a_rel, wav_b_rel = row.get("wav_a_path"), row.get("wav_b_path")
        if not wav_a_rel or not wav_b_rel:
            logging.warning(f"Skipping row - missing audio paths: {row}")
            continue

        try:
            wav_a, sr_a = torchaudio.load(os.path.join(args.data_root, wav_a_rel))
            wav_b, sr_b = torchaudio.load(os.path.join(args.data_root, wav_b_rel))
            if sr_a != 16000:
                wav_a = torchaudio.functional.resample(wav_a, sr_a, 16000)
            if sr_b != 16000:
                wav_b = torchaudio.functional.resample(wav_b, sr_b, 16000)

            with torch.no_grad():
                outputs = model(wav_a.to(device), wav_b.to(device))
                key = "cos_sim" if is_zero_shot else target_metric
                pred_score = outputs[key].item()
        except Exception as e:
            logging.error(f"Failed to process pair {wav_a_rel} and {wav_b_rel}: {e}")
            continue

        out_row = row.copy()
        out_row[f"pred_{target_metric}"] = pred_score
        out_results.append(out_row)

    total_time = time.time() - start_time
    logging.info(f"Total inference time = {total_time:.2f} secs.")
    if out_results:
        logging.info(f"Average speed = {total_time / len(out_results):.3f} sec / pair.")
        with open(args.out, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(out_results[0].keys()))
            writer.writeheader()
            writer.writerows(out_results)
        logging.info(f"Predictions saved to {args.out}")
    else:
        logging.warning("No results to save.")


if __name__ == "__main__":
    main()
