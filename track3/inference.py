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

import finetune  # module reference, so we can override MOE_NUM_EXPERTS/MOE_TOP_K
from finetune import ModelV2


def main():
    parser = argparse.ArgumentParser(description="Inference for finetune.py checkpoints.")
    parser.add_argument("--data-root", required=True, type=str)
    parser.add_argument("--csv-path", required=True, type=str)
    parser.add_argument("--out", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--listener-vocab", default=None, type=str,
                         help="Path to listener_vocab.json saved during training "
                              "(only needed to size the listener embedding table). "
                              "If omitted, defaults to listener_vocab.json next to --checkpoint.")
    parser.add_argument("--target-metric", required=True, choices=["spk_sim", "acc_sim", "both"],
                         help="Which prediction column(s) to write to --out. Must be a subset of "
                              "what the checkpoint was actually trained on (see config.json).")
    parser.add_argument("--use-listener-bias", action="store_true",
                         help="Only used as a fallback for checkpoints with no config.json "
                              "(older checkpoints). Ignored otherwise.")
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose > 1 else logging.INFO if args.verbose > 0 else logging.WARN
    logging.basicConfig(level=log_level, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    requested_metrics = ["spk_sim", "acc_sim"] if args.target_metric == "both" else [args.target_metric]
    checkpoint_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    config_path = os.path.join(checkpoint_dir, "config.json")

    if os.path.exists(config_path):
        with open(config_path) as f:
            run_config = json.load(f)
        build_target_metrics = run_config["target_metrics"]
        build_use_listener_bias = run_config.get("use_listener_bias", False)
        # Make sure the MoE head sizes we build match exactly what this
        # checkpoint was trained with, even if the module-level constants
        # in finetune.py have since been changed for a newer run.
        finetune.MOE_NUM_EXPERTS = run_config.get("moe_num_experts", finetune.MOE_NUM_EXPERTS)
        finetune.MOE_TOP_K = run_config.get("moe_top_k", finetune.MOE_TOP_K)

        logging.info(
            f"Loaded {config_path}: target_metrics={build_target_metrics}, "
            f"use_listener_bias={build_use_listener_bias}, "
            f"moe_num_experts={finetune.MOE_NUM_EXPERTS}, moe_top_k={finetune.MOE_TOP_K}"
        )

        missing = [m for m in requested_metrics if m not in build_target_metrics]
        if missing:
            raise ValueError(
                f"--target-metric {args.target_metric} requires prediction head(s) {missing}, "
                f"but the checkpoint at {args.checkpoint} was trained with "
                f"target_metrics={build_target_metrics} (see {config_path}). "
                f"Re-run with --target-metric matching one of {build_target_metrics}, "
                f"or re-train a checkpoint that includes {missing}."
            )

        if args.listener_vocab is None and build_use_listener_bias:
            default_vocab = os.path.join(checkpoint_dir, "listener_vocab.json")
            if os.path.exists(default_vocab):
                args.listener_vocab = default_vocab
                logging.info(f"Auto-detected listener vocab at {default_vocab}")
    else:
        logging.warning(
            f"No config.json found at {config_path} (older checkpoint, trained before this "
            f"bugfix?). Falling back to --target-metric/--use-listener-bias as given -- if "
            f"these don't match what was actually used at training time, you'll hit a "
            f"state_dict mismatch. Re-training with the current finetune.py will fix this "
            f"for future checkpoints."
        )
        build_target_metrics = requested_metrics
        build_use_listener_bias = args.use_listener_bias

    num_listeners = 0
    if build_use_listener_bias:
        assert args.listener_vocab, (
            "This checkpoint was trained with listener-bias enabled, but no --listener-vocab "
            "was given and none could be auto-detected next to the checkpoint."
        )
        with open(args.listener_vocab) as f:
            num_listeners = len(json.load(f))

    model = ModelV2(
        model_name="speechbrain/spkrec-ecapa-voxceleb",
        embedding_dim=256,
        use_projection=True,
        freeze_ssl=False,
        target_metrics=build_target_metrics,
        num_listeners=num_listeners,
        use_listener_bias=build_use_listener_bias,
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
        for m in requested_metrics:
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
