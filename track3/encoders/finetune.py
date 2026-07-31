#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fine-tuning for VoiceMOS 2026 Track 3 with a pluggable encoder.

Same recipe as ../baseline/finetune.py (per-pair score averaging, repetitive padding,
AdamW, MSE, fixed step count) so results are comparable. Added on top:

  --encoder             which backbone to fine-tune (see `python encoders.py --list`)
  --encoder-lr          separate, usually lower, learning rate for the pretrained backbone
  --freeze-encoder      train only the projection + MLP head
  --encoder-checkpoint  use local backbone weights instead of downloading

Checkpoints are saved as {"config": ..., "state_dict": ...} so that inference.py can
rebuild the right architecture without being told which encoder was used.
"""

import argparse
import csv
import logging
import os
from collections import defaultdict

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from calculate_metrics import compute_metrics
from encoders import ENCODER_REGISTRY
from model import Model


class SimilarityDataset(Dataset):
    def __init__(self, data_root, data_rows, target_metric, keep_system_id=False):
        self.data_root = data_root
        self.target_metric = target_metric
        # System-level metrics need the system_id to survive collation. Off during
        # training, where it is unused.
        self.keep_system_id = keep_system_id

        # Aggregate scores by unique audio pairs
        aggregated_data = {}
        for row in data_rows:
            pair_key = (row["wav_a_path"], row["wav_b_path"])
            if pair_key not in aggregated_data:
                aggregated_data[pair_key] = row.copy()
                aggregated_data[pair_key][target_metric] = []

            # Accumulate the scores for this specific pair
            aggregated_data[pair_key][target_metric].append(float(row[target_metric]))

        self.data_rows = []
        for pair_key, row_data in aggregated_data.items():
            # Calculate the mean score
            scores = row_data[target_metric]
            row_data[target_metric] = sum(scores) / len(scores)

            # Remove listener_id since this is an averaged sample
            if "listener_id" in row_data:
                del row_data["listener_id"]

            self.data_rows.append(row_data)

        logging.info(f"Aggregated {len(data_rows)} raw rows into {len(self.data_rows)} unique averaged pairs.")

    def __len__(self):
        return len(self.data_rows)

    def __getitem__(self, idx):
        row = self.data_rows[idx]
        wav_a_path = os.path.join(self.data_root, row["wav_a_path"])
        wav_b_path = os.path.join(self.data_root, row["wav_b_path"])

        # Load audio
        wav_a, sr_a = torchaudio.load(wav_a_path)
        wav_b, sr_b = torchaudio.load(wav_b_path)

        # Ensure 16kHz
        if sr_a != 16000: wav_a = torchaudio.functional.resample(wav_a, sr_a, 16000)
        if sr_b != 16000: wav_b = torchaudio.functional.resample(wav_b, sr_b, 16000)

        target = float(row[self.target_metric])

        # Return a dictionary. The collater will dynamically pack the target_metric.
        item = {
            "wav_a": wav_a.squeeze(0),
            "wav_b": wav_b.squeeze(0),
            self.target_metric: target
        }
        if self.keep_system_id and "system_id" in row:
            item["system_id"] = row["system_id"]
        return item


class SimilarityCollater:
    """Collates a batch of similarity dicts into padded tensors. Safely handles missing 'wav_b'."""

    def __init__(self, padding_mode="repetitive"):
        self.padding_mode = padding_mode

    def __call__(self, batch):
        batch_dict = {}

        # 1. Sort batch by wav_a length (longest first)
        sorted_batch = sorted(batch, key=lambda x: -x["wav_a"].shape[0])
        bs = len(sorted_batch)
        all_keys = list(sorted_batch[0].keys())

        # Helper function for padding (Zero / Repetitive)
        def pad_features(feats):
            feat_lengths = torch.tensor([feat.shape[0] for feat in feats], dtype=torch.long)
            if self.padding_mode == "zero_padding":
                feats_padded = pad_sequence(feats, batch_first=True, padding_value=0.0)
            elif self.padding_mode == "repetitive":
                max_len = feat_lengths.max().item() # Use max across the specific feature list
                feats_padded = []
                for feat in feats:
                    this_len = feat.shape[0]
                    if this_len == 0:
                        feats_padded.append(torch.zeros(max_len, dtype=feat.dtype))
                        continue
                    dup_times = max_len // this_len
                    remain = max_len - this_len * dup_times
                    to_dup = [feat for _ in range(dup_times)]
                    if remain > 0:
                        to_dup.append(feat[:remain])
                    duplicated_feat = torch.cat(to_dup, dim=0)
                    feats_padded.append(duplicated_feat)
                feats_padded = torch.stack(feats_padded, dim=0)
            else:
                raise NotImplementedError(f"Padding mode {self.padding_mode} not implemented.")
            return feats_padded, feat_lengths

        # 2. Process Audio
        wav_a_list = [sorted_batch[i]["wav_a"] for i in range(bs)]
        batch_dict["wav_a"], batch_dict["wav_a_lengths"] = pad_features(wav_a_list)
        wav_b_list = [item["wav_b"] for item in sorted_batch if item.get("wav_b") is not None]
        batch_dict["wav_b"], batch_dict["wav_b_lengths"] = pad_features(wav_b_list)

        # 3. Explicit ID packing
        if "system_id" in all_keys:
            batch_dict["system_ids"] = [sorted_batch[i]["system_id"] for i in range(bs)]
        if "sample_id" in all_keys:
            batch_dict["sample_ids"] = [sorted_batch[i]["sample_id"] for i in range(bs)]
        if "listener_id" in all_keys:
            batch_dict["listener_ids"] = [sorted_batch[i]["listener_id"] for i in range(bs)]

        # 4. Dynamically pack all metric targets (e.g., 'score', 'mos', 'spk', 'acc')
        ignore_keys = ["wav_path", "wav_a_path", "wav_b_path", "wav_a", "wav_b",
                       "system_id", "sample_id", "listener_id"]

        for key in all_keys:
            if key not in ignore_keys:
                if isinstance(sorted_batch[0][key], (int, float)):
                    batch_dict[key] = torch.tensor([item[key] for item in sorted_batch], dtype=torch.float32)
                elif isinstance(sorted_batch[0][key], str):
                    batch_dict[key] = [item[key] for item in sorted_batch]

        return batch_dict


def save_checkpoint(path, model, config):
    torch.save({"config": config, "state_dict": model.state_dict()}, path)


# Metric keys reported by evaluate(), in display order. "sys" variants average predictions
# and targets per system first, which is what the challenge's system-level scores do.
METRIC_KEYS = ["mse_utt", "lcc_utt", "srcc_utt", "mse_sys", "lcc_sys", "srcc_sys"]


@torch.no_grad()
def evaluate(model, loader, target_metric, device):
    """Run the model over a labelled dev set and return utterance- and system-level scores.

    Restores the model's previous train/eval mode on the way out so the training loop is
    unaffected. Uses the same compute_metrics as calculate_metrics.py.

    Note on batching: this runs batched, so the collater pads every clip in a batch up to
    the longest one. inference.py runs one pair at a time with no padding, so the two paths
    can disagree slightly. Utterance-level scores track closely (measured: srcc_utt 0.1691
    vs 0.1696 for the same checkpoint), but system-level SRCC is a rank statistic over only
    ~23 systems, so a hair of numerical difference can swap two adjacent systems and move it
    by ~0.01 (measured: srcc_sys 0.4308 batched vs 0.4407 unbatched). Treat the in-training
    curve as the monitoring signal and a standalone inference.py + calculate_metrics.py run
    as the number of record. --eval-batch-size 1 makes them agree exactly, at ~15x the cost.
    """
    was_training = model.training
    model.eval()

    preds, targets, system_ids = [], [], []
    for batch in loader:
        outputs = model(
            batch["wav_a"].to(device),
            batch["wav_b"].to(device),
            batch["wav_a_lengths"].to(device),
            batch["wav_b_lengths"].to(device),
        )
        preds.extend(outputs[target_metric].detach().cpu().tolist())
        targets.extend(batch[target_metric].tolist())
        system_ids.extend(batch.get("system_ids", [None] * len(batch[target_metric])))

    if was_training:
        model.train()

    utt_true = np.array(targets)
    utt_pred = np.array(preds)
    # Cast to plain floats: compute_metrics returns numpy scalars, and those end up in the
    # checkpoint config, where torch.load's weights_only=True default refuses to unpickle
    # numpy._core.multiarray.scalar.
    results = {k: float(v) for k, v in zip(METRIC_KEYS[:3], compute_metrics(utt_true, utt_pred))}

    if any(s is not None for s in system_ids):
        sys_true, sys_pred = defaultdict(list), defaultdict(list)
        for sid, t, p in zip(system_ids, targets, preds):
            sys_true[sid].append(t)
            sys_pred[sid].append(p)
        results.update({k: float(v) for k, v in zip(METRIC_KEYS[3:], compute_metrics(
            np.array([np.mean(sys_true[s]) for s in sys_true]),
            np.array([np.mean(sys_pred[s]) for s in sys_true]),
        ))})
        results["n_systems"] = len(sys_true)

    results["n_pairs"] = len(utt_true)
    return results


def format_metrics(results):
    parts = [f"{k}={results[k]:.4f}" for k in METRIC_KEYS if k in results]
    return "  ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Baseline for VoiceMOS 2026 Track 3.")
    parser.add_argument("--data-root", required=True, type=str, help="Root directory of the dataset distribution.")
    parser.add_argument("--target-metric", required=True, type=str, choices=["spk_sim", "acc_sim"], help="Metric to train the projection head on.")
    parser.add_argument("--outdir", required=True, type=str, help="Directory to save the trained checkpoints.")
    parser.add_argument("--train-csv", type=str, default=None, help="Training CSV. Defaults to <data-root>/sets/train.csv.")
    parser.add_argument("--encoder", type=str, default="ecapa-voxceleb", help=f"Encoder to fine-tune. One of: {', '.join(ENCODER_REGISTRY)}. Combine with '+'.")
    parser.add_argument("--encoder-checkpoint", type=str, default=None, help="Local backbone weights, instead of downloading.")
    parser.add_argument("--cache-dir", type=str, default=None, help="Where to cache downloaded encoder weights.")
    parser.add_argument("--freeze-encoder", action="store_true", help="Train only the projection head and MLP head.")
    parser.add_argument("--encoder-lr", type=float, default=None, help="Learning rate for the pretrained backbone. Defaults to --lr.")
    parser.add_argument("--dev-csv", type=str, default=None, help="Labelled dev CSV. If given, it is scored periodically during training.")
    parser.add_argument("--dev-data-root", type=str, default=None, help="Data root for --dev-csv. Defaults to --data-root.")
    parser.add_argument("--eval-steps", type=int, default=500, help="Evaluate --dev-csv every N optimizer steps.")
    parser.add_argument("--eval-batch-size", type=int, default=16, help="Batch size for dev evaluation. Use 1 to match inference.py exactly (see evaluate()).")
    parser.add_argument("--best-metric", type=str, default="srcc_sys", choices=METRIC_KEYS, help="Dev metric used to keep model_best.pt.")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--accumulate-steps", type=int, default=1, help="Number of gradient accumulation steps.")
    parser.add_argument("--train-steps", type=int, default=20000, help="Total number of training steps.")
    parser.add_argument("--save-steps", type=int, default=5000, help="Save a checkpoint every N steps.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed.")
    parser.add_argument("--verbose", type=int, default=1, help="logging level.")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose > 1 else logging.INFO if args.verbose > 0 else logging.WARN
    logging.basicConfig(level=log_level, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")

    torch.manual_seed(args.seed)

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # 1. Data Preparation
    train_csv_path = args.train_csv or os.path.join(args.data_root, "sets", "train.csv")
    logging.info(f"Loading {train_csv_path}...")

    train_data = []
    with open(train_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            train_data.append(row)

    logging.info(f"Loaded {len(train_data)} training samples.")

    train_dataset = SimilarityDataset(args.data_root, train_data, args.target_metric)
    collater = SimilarityCollater(padding_mode="repetitive")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collater,
        num_workers=args.num_workers,
        drop_last=True
    )

    # 1b. Dev set for in-training evaluation
    dev_loader = None
    if args.dev_csv:
        dev_root = args.dev_data_root or args.data_root
        logging.info(f"Loading dev set {args.dev_csv} (data root {dev_root})...")
        with open(args.dev_csv, 'r', encoding='utf-8') as f:
            dev_data = list(csv.DictReader(f))
        if args.target_metric not in (dev_data[0] if dev_data else {}):
            raise SystemExit(
                f"{args.dev_csv} has no '{args.target_metric}' column; it must be a labelled dev set."
            )
        # keep_system_id=True so system-level metrics can be computed. The aggregation is a
        # no-op when the CSV is already one row per pair, and averages it when it is not.
        dev_dataset = SimilarityDataset(dev_root, dev_data, args.target_metric, keep_system_id=True)
        dev_loader = DataLoader(
            dev_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=collater,
            num_workers=args.num_workers,
            drop_last=False,
        )
        logging.info(
            f"Dev evaluation every {args.eval_steps} steps on {len(dev_dataset)} pairs; "
            f"model_best.pt tracks {args.best_metric}."
        )

    # 2. Model Initialization
    logging.info(f"Initializing Model for {args.target_metric} prediction with encoder '{args.encoder}'...")
    config = {
        "encoder": args.encoder,
        "target_metric": args.target_metric,
        "use_projection": True,
        "freeze_encoder": args.freeze_encoder,
    }
    model = Model(
        encoder_name=args.encoder,
        use_projection=True,
        mlp_heads=[args.target_metric],
        freeze_encoder=args.freeze_encoder,
        cache_dir=args.cache_dir,
        encoder_checkpoint=args.encoder_checkpoint,
    )
    model.to(device)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(
        f"Encoder output dim {model.encoder.backbone.output_dim}, "
        f"{n_train / 1e6:.2f}M trainable parameters."
    )

    # 3. Optimizer & Criterion
    if args.freeze_encoder:
        param_groups = [{"params": list(model.head_parameters()), "lr": args.lr}]
    elif args.encoder_lr is not None:
        param_groups = [
            {"params": list(model.encoder_parameters()), "lr": args.encoder_lr},
            {"params": list(model.head_parameters()), "lr": args.lr},
        ]
        logging.info(f"Backbone lr={args.encoder_lr}, head lr={args.lr}")
    else:
        param_groups = [{"params": list(model.parameters()), "lr": args.lr}]

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    criterion = torch.nn.MSELoss()

    # 4. Training Loop
    model.train()
    global_step = 0
    forward_steps = 0
    train_loss = 0.0

    # Dev history, so the run can be plotted or grepped afterwards rather than only read
    # off the log. One row per evaluation.
    log_path = os.path.join(args.outdir, f"dev_log_{args.target_metric}.csv")
    log_file = None
    best_score = None
    if dev_loader is not None:
        log_file = open(log_path, "w", newline="", encoding="utf-8")
        log_writer = csv.writer(log_file)
        log_writer.writerow(["step", "train_mse"] + METRIC_KEYS)

    def run_eval(step, recent_loss, track_best=True):
        nonlocal best_score
        results = evaluate(model, dev_loader, args.target_metric, device)
        logging.info(f"[dev @ step {step}] {format_metrics(results)}")
        log_writer.writerow(
            [step, f"{recent_loss:.6f}"] + [f"{results.get(k, float('nan')):.6f}" for k in METRIC_KEYS]
        )
        log_file.flush()

        # MSE is better when lower; the correlations are better when higher.
        score = results.get(args.best_metric)
        if track_best and score is not None and not np.isnan(score):
            improved = (
                best_score is None
                or (score < best_score if args.best_metric.startswith("mse") else score > best_score)
            )
            if improved:
                best_score = score
                save_checkpoint(
                    os.path.join(args.outdir, f"model_best_{args.target_metric}.pt"),
                    model,
                    {**config, "best_metric": args.best_metric, "best_score": score, "best_step": step},
                )
                logging.info(f"[dev @ step {step}] new best {args.best_metric}={score:.4f}, saved model_best.")
        return results

    optimizer.zero_grad()
    pbar = tqdm(total=args.train_steps, desc="Training")

    if dev_loader is not None:
        # Step 0 is the untrained-head reference point every later number is read against.
        # Excluded from best-checkpoint tracking: a randomly initialised head can score well
        # by chance, and shipping it would be meaningless.
        run_eval(0, float("nan"), track_best=False)

    while global_step < args.train_steps:
        for batch in train_loader:
            if global_step >= args.train_steps:
                break

            wav_a = batch["wav_a"].to(device)
            len_a = batch["wav_a_lengths"].to(device)
            wav_b = batch["wav_b"].to(device)
            len_b = batch["wav_b_lengths"].to(device)
            targets = batch[args.target_metric].to(device)

            outputs = model(wav_a, wav_b, len_a, len_b)
            preds = outputs[args.target_metric]

            # Loss and Gradient Accumulation
            loss = criterion(preds, targets)
            scaled_loss = loss / args.accumulate_steps
            scaled_loss.backward()

            train_loss += loss.item()
            forward_steps += 1

            if forward_steps % args.accumulate_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

                global_step += 1
                pbar.update(1)

                # Update progress bar description every step
                recent_loss = train_loss / args.accumulate_steps
                pbar.set_postfix({"MSE": f"{recent_loss:.4f}"})
                train_loss = 0.0

                # Save checkpoint every N steps
                if global_step % args.save_steps == 0:
                    save_path = os.path.join(args.outdir, f"model_{args.target_metric}_step{global_step}.pt")
                    save_checkpoint(save_path, model, config)
                    logging.info(f"Checkpoint saved to {save_path}")

                # Score the dev set every N steps
                if dev_loader is not None and global_step % args.eval_steps == 0:
                    results = run_eval(global_step, recent_loss)
                    pbar.set_postfix({
                        "MSE": f"{recent_loss:.4f}",
                        args.best_metric: f"{results.get(args.best_metric, float('nan')):.4f}",
                    })

    pbar.close()

    # 5. Save Final Model
    save_path = os.path.join(args.outdir, f"finetuned_model_{args.target_metric}_final.pt")
    save_checkpoint(save_path, model, config)
    logging.info(f"Training complete. Final model saved to {save_path}")

    # 6. Final dev evaluation, unless the last step already ran one
    if dev_loader is not None:
        if global_step % args.eval_steps != 0:
            run_eval(global_step, float("nan"))
        logging.info(f"Dev history written to {log_path}")
        logging.info(
            f"Best {args.best_metric} = {best_score:.4f} "
            f"(model_best_{args.target_metric}.pt); final-step model is {os.path.basename(save_path)}."
        )
        log_file.close()

if __name__ == "__main__":
    main()
