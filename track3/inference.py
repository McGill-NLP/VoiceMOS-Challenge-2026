#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import csv
import json
import logging
import os
import re
import time

import torch
import torchaudio
from tqdm import tqdm

import finetune  # module reference, so we can override MOE_NUM_EXPERTS/MOE_TOP_K
from finetune import ModelV2


def infer_from_state_dict(state_dict):
    """
    Independently re-derives target_metrics, use_listener_bias, and
    moe_num_experts directly from the checkpoint's own weight keys/shapes.
    This is a cross-check against config.json, and the sole source of
    truth if config.json somehow disagrees on these particular fields
    (which ARE fully determined by parameter shapes, unlike model names or
    the wean alpha).
    """
    target_metrics = sorted({
        k.split(".")[1] for k in state_dict if k.startswith("mean_heads.")
    })
    if not target_metrics:
        raise ValueError(
            "Could not find any 'mean_heads.<metric>.*' keys in this checkpoint -- "
            "it doesn't look like a ModelV2 checkpoint produced by finetune.py. "
            f"First few keys found: {list(state_dict.keys())[:10]}"
        )
    use_listener_bias = any(k.startswith("listener_emb.") for k in state_dict)

    first_metric = target_metrics[0]
    expert_idx_pattern = re.compile(rf"^mean_heads\.{re.escape(first_metric)}\.experts\.(\d+)\.")
    expert_indices = set()
    for k in state_dict:
        m = expert_idx_pattern.match(k)
        if m:
            expert_indices.add(int(m.group(1)))
    if not expert_indices:
        raise ValueError(
            f"Could not find any 'mean_heads.{first_metric}.experts.<i>.*' keys -- "
            f"this checkpoint's mean head doesn't look like a MoEProjection."
        )
    num_experts = max(expert_indices) + 1

    return target_metrics, use_listener_bias, num_experts


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
                              "what the checkpoint was actually trained on.")
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose > 1 else logging.INFO if args.verbose > 0 else logging.WARN
    logging.basicConfig(level=log_level, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    requested_metrics = ["spk_sim", "acc_sim"] if args.target_metric == "both" else [args.target_metric]
    checkpoint_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    config_path = os.path.join(checkpoint_dir, "config.json")

    logging.info(f"Loading checkpoint from {args.checkpoint} ...")
    raw_state_dict = torch.load(args.checkpoint, map_location="cpu")
    ckpt_target_metrics, ckpt_use_listener_bias, ckpt_num_experts = infer_from_state_dict(raw_state_dict)
    logging.info(
        f"Inferred from checkpoint weights: target_metrics={ckpt_target_metrics}, "
        f"use_listener_bias={ckpt_use_listener_bias}, moe_num_experts={ckpt_num_experts}"
    )

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"No config.json found at {config_path}. This inference.py requires it -- "
            f"the dual-encoder architecture (separate model_name_spk/model_name_acc) and "
            f"the wean_max_alpha to apply at inference time are NOT recoverable from the "
            f"checkpoint's weight shapes alone. If this checkpoint predates the dual-encoder "
            f"architecture change, it is not compatible with this version of inference.py."
        )
    with open(config_path) as f:
        run_config = json.load(f)

    cfg_metrics = sorted(run_config.get("target_metrics", []))
    if cfg_metrics != ckpt_target_metrics:
        logging.warning(
            f"config.json says target_metrics={cfg_metrics}, but the checkpoint's own "
            f"weights indicate {ckpt_target_metrics}. Trusting the checkpoint's weights "
            f"for architecture, but using config.json for model names / wean alpha."
        )

    finetune.MOE_NUM_EXPERTS = ckpt_num_experts
    finetune.MOE_TOP_K = run_config.get("moe_top_k", finetune.MOE_TOP_K)
    model_name_spk = run_config.get("model_name_spk", finetune.DEFAULT_MODEL_NAME_SPK)
    model_name_acc = run_config.get("model_name_acc", finetune.DEFAULT_MODEL_NAME_ACC)
    adapter_hidden_dim = run_config.get("adapter_hidden_dim", 128)
    wean_max_alpha = run_config.get("wean_max_alpha", 1.0)
    logging.info(
        f"From config.json: model_name_spk={model_name_spk}, model_name_acc={model_name_acc}, "
        f"adapter_hidden_dim={adapter_hidden_dim}, wean_max_alpha={wean_max_alpha}"
    )

    missing = [m for m in requested_metrics if m not in ckpt_target_metrics]
    if missing:
        raise ValueError(
            f"--target-metric {args.target_metric} requires prediction head(s) {missing}, "
            f"but the checkpoint at {args.checkpoint} only contains heads for "
            f"{ckpt_target_metrics} (inferred directly from its weights). "
            f"Re-run with --target-metric matching one of {ckpt_target_metrics}, "
            f"or use a different checkpoint that includes {missing}."
        )

    if args.listener_vocab is None and ckpt_use_listener_bias:
        default_vocab = os.path.join(checkpoint_dir, "listener_vocab.json")
        if os.path.exists(default_vocab):
            args.listener_vocab = default_vocab
            logging.info(f"Auto-detected listener vocab at {default_vocab}")

    num_listeners = 0
    if ckpt_use_listener_bias:
        assert args.listener_vocab, (
            "This checkpoint was trained with listener-bias enabled (detected from its "
            "weights), but no --listener-vocab was given and none could be auto-detected "
            "next to the checkpoint."
        )
        with open(args.listener_vocab) as f:
            num_listeners = len(json.load(f))

    model = ModelV2(
        model_name_spk=model_name_spk,
        model_name_acc=model_name_acc,
        embedding_dim=256,
        use_projection=True,
        freeze_ssl=False,
        target_metrics=ckpt_target_metrics,
        num_listeners=num_listeners,
        use_listener_bias=ckpt_use_listener_bias,
        adapter_hidden_dim=adapter_hidden_dim,
    )
    model.load_state_dict(raw_state_dict)
    model.to(device)
    model.eval()
    logging.info("Checkpoint loaded successfully.")

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
                # Use the fully-weaned final alpha (the trained steady state),
                # NOT 0.0 (which would mean "as if training just started").
                out = model(wav_a.to(device), wav_b.to(device), adapter_alpha=wean_max_alpha)
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
