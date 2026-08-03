#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inference for VoiceMOS 2026 Track 3, UTMOS-objective experiment.

Same behaviour as ../baseline/inference.py, except the encoder source and target metric are
read back from the checkpoint, so a model trained on acc_sim cannot be written out under a
pred_spk_sim column by forgetting a flag.
"""

import argparse
import csv
import logging
import os
import time

import torch
import torchaudio
from tqdm import tqdm

from model import Model


# ../baseline/model.py holds the backbone at `encoder.ssl_model`; here it lives at
# `encoder.backbone.encoder`. Everything else (projection, mlp_heads) is identical, so
# renaming the prefix makes the official-egs checkpoints loadable as-is.
_BASELINE_KEY_PREFIX = "encoder.ssl_model."
_ENCODERS_KEY_PREFIX = "encoder.backbone."


def load_checkpoint(path):
    """Returns (state_dict, config). Supports plain state_dict checkpoints too."""
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj and "config" in obj:
        return obj["state_dict"], obj["config"]

    # Baseline-style checkpoint: a bare state_dict with no provenance.
    if any(k.startswith(_BASELINE_KEY_PREFIX) for k in obj):
        logging.info("Detected a baseline-format checkpoint; remapping backbone keys.")
        obj = {
            (
                _ENCODERS_KEY_PREFIX + k[len(_BASELINE_KEY_PREFIX):]
                if k.startswith(_BASELINE_KEY_PREFIX)
                else k
            ): v
            for k, v in obj.items()
        }
    return obj, None


def main():
    parser = argparse.ArgumentParser(description="Inference for VoiceMOS 2026 Track 3 (Zero-shot or Fine-tuned).")
    parser.add_argument("--data-root", required=True, type=str, help="Root directory of the dataset distribution.")
    parser.add_argument("--csv-path", required=True, type=str, help="CSV file path to do inference (e.g. sets/dev.csv).")
    parser.add_argument("--out", type=str, required=True, help="Path to save the output predictions CSV.")
    parser.add_argument("--encoder-source", type=str, default=None, help="SpeechBrain encoder. Read from the checkpoint when fine-tuned; defaults to the baseline's encoder zero-shot.")
    parser.add_argument("--cache-dir", type=str, default=None, help="Where to cache downloaded encoder weights.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to fine-tuned .pt checkpoint. If not provided, runs zero-shot.")
    parser.add_argument("--target-metric", type=str, default=None, choices=["spk_sim", "acc_sim"], help="Metric to predict. Read from the checkpoint when available; defaults to spk_sim.")
    parser.add_argument("--verbose", type=int, default=1, help="logging level.")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose > 1 else logging.INFO if args.verbose > 0 else logging.WARN
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    is_zero_shot = args.checkpoint is None

    # 1. Load Model dynamically based on the mode
    if is_zero_shot:
        encoder_source = args.encoder_source or "speechbrain/spkrec-ecapa-voxceleb"
        target_metric = args.target_metric or "spk_sim"
        logging.info(f"Initializing Model in ZERO-SHOT mode using '{encoder_source}'...")
        model = Model(
            encoder_source=encoder_source,
            use_projection=False,
            freeze_encoder=True,
            mlp_heads=[],
            cache_dir=args.cache_dir,
        )
    else:
        state_dict, config = load_checkpoint(args.checkpoint)

        if config is not None:
            encoder_source = args.encoder_source or config["encoder_source"]
            target_metric = args.target_metric or config["target_metric"]
            if config.get("loss"):
                logging.info(f"Checkpoint was trained with --loss {config['loss']}.")
        else:
            head_names = sorted({
                k.split(".")[1] for k in state_dict if k.startswith("mlp_heads.")
            })
            encoder_source = args.encoder_source or "speechbrain/spkrec-ecapa-voxceleb"
            target_metric = args.target_metric or (head_names[0] if head_names else "spk_sim")
            logging.warning(
                f"Checkpoint has no config; assuming encoder_source='{encoder_source}' and "
                f"target_metric='{target_metric}' (heads found: {head_names})."
            )

        logging.info(f"Initializing Model in FINE-TUNED mode for {target_metric} with '{encoder_source}'...")
        model = Model(
            encoder_source=encoder_source,
            use_projection=True,
            freeze_encoder=False,
            mlp_heads=[target_metric],
            cache_dir=args.cache_dir,
        )
        logging.info(f"Loading weights from {args.checkpoint}...")
        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    # 2. Load Dataset Metadata
    logging.info(f"Loading Dataset from {args.csv_path}")
    dataset = []
    with open(args.csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset.append(row)
    logging.info(f"Number of inference samples = {len(dataset)}.")

    out_results = []
    start_time = time.time()

    # 3. Inference Loop
    for batch in tqdm(dataset, desc="[inference]"):
        wav_a_rel = batch.get("wav_a_path")
        wav_b_rel = batch.get("wav_b_path")

        if not wav_a_rel or not wav_b_rel:
            logging.warning(f"Skipping row - missing audio paths: {batch}")
            continue

        # Resolve absolute paths based on the provided data root
        wav_a_path = os.path.join(args.data_root, wav_a_rel)
        wav_b_path = os.path.join(args.data_root, wav_b_rel)

        try:
            # Load audio using torchaudio
            wav_a, sr_a = torchaudio.load(wav_a_path)
            wav_b, sr_b = torchaudio.load(wav_b_path)

            # Ensure 16kHz sample rate
            if sr_a != 16000: wav_a = torchaudio.functional.resample(wav_a, sr_a, 16000)
            if sr_b != 16000: wav_b = torchaudio.functional.resample(wav_b, sr_b, 16000)

            with torch.no_grad():
                # Pass batch=1 tensors to the model
                outputs = model(wav_a.to(device), wav_b.to(device))

                if is_zero_shot:
                    # In zero-shot, fallback to the raw cosine similarity computation
                    pred_score = outputs["cos_sim"].item()
                else:
                    # In fine-tuned mode, grab the specific metric projection head output
                    pred_score = outputs[target_metric].item()

        except Exception as e:
            logging.error(f"Failed to process pair {wav_a_rel} and {wav_b_rel}: {e}")
            continue

        # Build the output row: copy the original data and append our predictions
        out_row = batch.copy()
        out_row[f"pred_{target_metric}"] = pred_score

        out_results.append(out_row)

    total_time = time.time() - start_time
    logging.info(f"Total inference time = {total_time:.2f} secs.")
    if len(out_results) > 0:
        logging.info(f"Average speed = {total_time / len(out_results):.3f} sec / pair.")

    # 4. Save Results
    if len(out_results) > 0:
        fieldnames = list(out_results[0].keys())
        with open(args.out, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(out_results)
        logging.info(f"Predictions saved to {args.out}")
    else:
        logging.warning("No results to save.")

if __name__ == "__main__":
    main()
