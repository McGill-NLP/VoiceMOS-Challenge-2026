#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import logging
import os
from functools import partial

from coral_pytorch.dataset import levels_from_labelbatch
from coral_pytorch.losses import coral_loss, corn_loss

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import CLASSIFIER_TYPE, Model, decode_score

# spk_sim / acc_sim ratings live on a 1-5 scale; CORN/CORAL need 0-indexed
# integer ordinal classes, so this is also the number of rating buckets.
NUM_CLASSES = 5


def soft_survival_targets(targets, num_classes):
    """Converts continuous 1..num_classes ratings into soft marginal
    "survival" targets S(k) = P(y > k) for k = 0..num_classes-2, assuming a
    two-point mixture between floor(t) and ceil(t) matching the rating's
    fractional part. By construction, sum_k S(k) == t (0-indexed), so this
    is the continuous generalization of the hard extended-binary encoding
    (`levels_from_labelbatch`) used by CORAL, and reduces to it exactly when
    t is an integer.
    """
    t = torch.clamp(targets - 1, 0, num_classes - 1)
    thresholds = torch.arange(
        num_classes - 1, device=targets.device, dtype=targets.dtype
    )
    return torch.clamp(t.unsqueeze(1) - thresholds, 0, 1)


def soft_coral_loss(logits, targets, num_classes):
    """Soft-target analogue of coral_pytorch's coral_loss. CORAL's head
    already outputs marginal probabilities P(y>k) (rank-consistent by
    architecture, not by loss), so we can train directly against the soft
    marginal survival targets. Matches coral_loss's reduction (sum over the
    threshold dim, mean over the batch) so the loss scale - and hence
    learning rate - is comparable to the hard-label runs.
    """
    soft_targets = soft_survival_targets(targets, num_classes)
    elementwise_bce = F.binary_cross_entropy_with_logits(
        logits, soft_targets, reduction="none"
    )
    return elementwise_bce.sum(dim=1).mean()


def soft_corn_loss(logits, targets, num_classes):
    """Soft-target analogue of coral_pytorch's corn_loss. CORN's logits are
    *conditional* probabilities P(y>k | y>k-1) (see corn_label_from_logits,
    which cumprods them back into marginal probabilities), and corn_loss
    trains each threshold task only on examples that cleared the previous
    one. This mirrors that: it decomposes the soft marginal survival
    targets S(k) into conditional targets c(k) = S(k) / S(k-1) via the same
    chain rule, and weights each task's BCE term by S(k-1) - a soft version
    of "did this example clear the previous threshold" - instead of hard
    masking. Reduces to corn_loss exactly when targets are integers.
    """
    survival = soft_survival_targets(
        targets, num_classes
    )  # S(k), shape (batch, num_classes-1)
    prev_survival = F.pad(
        survival[:, :-1], (1, 0), value=1.0
    )  # S(k-1), with S(-1) := 1

    # Where prev_survival is 0 the conditional target is undefined (0/0), but
    # its weight is also 0, so it can't contribute to the loss regardless.
    cond_target = torch.where(
        prev_survival > 0,
        survival / prev_survival.clamp(min=1e-8),
        torch.zeros_like(survival),
    )

    elementwise_bce = F.binary_cross_entropy_with_logits(
        logits, cond_target, reduction="none"
    )
    return (prev_survival * elementwise_bce).sum() / prev_survival.sum().clamp(min=1e-8)


def compute_loss(args, criterion, preds, targets, device):
    """Shared loss computation for both the training step and validation,
    so the two can never drift out of sync with each other.
    """
    if args.classifier_type == CLASSIFIER_TYPE.CORAL:
        logits, _probas = preds
        if args.soft_labels:
            return soft_coral_loss(logits, targets, NUM_CLASSES)
        labels = torch.clamp(torch.round(targets - 1), 0, NUM_CLASSES - 1).to(
            torch.long
        )
        levels = levels_from_labelbatch(labels, num_classes=NUM_CLASSES).to(device)
        return criterion(logits, levels)
    elif args.classifier_type == CLASSIFIER_TYPE.CORN:
        if args.soft_labels:
            return soft_corn_loss(preds, targets, NUM_CLASSES)
        labels = torch.clamp(torch.round(targets - 1), 0, NUM_CLASSES - 1).to(
            torch.long
        )
        return criterion(preds, labels)
    else:
        return criterion(preds, targets)


@torch.no_grad()
def evaluate(model, val_loader, args, criterion, device):
    """Computes the mean per-example validation loss, and MSE in the
    original 1-5 rating scale (via the same decode_score used by
    inference.py), over val_loader.
    """
    model.eval()
    total_loss = 0.0
    total_squared_error = 0.0
    total_examples = 0

    for batch in val_loader:
        wav_a = batch["wav_a"].to(device)
        len_a = batch["wav_a_lengths"].to(device)
        wav_b = batch["wav_b"].to(device)
        len_b = batch["wav_b_lengths"].to(device)
        targets = batch[args.target_metric].to(device)

        outputs = model(wav_a, wav_b, len_a, len_b)
        preds = outputs[args.target_metric]
        loss = compute_loss(args, criterion, preds, targets, device)
        pred_scores = decode_score(args.classifier_type, preds)

        batch_size = targets.shape[0]
        total_loss += loss.item() * batch_size
        total_squared_error += ((pred_scores - targets) ** 2).sum().item()
        total_examples += batch_size

    model.train()
    return total_loss / total_examples, total_squared_error / total_examples


class SimilarityDataset(Dataset):
    def __init__(self, data_root, data_rows, target_metric):
        self.data_root = data_root
        self.target_metric = target_metric

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

        logging.info(
            f"Aggregated {len(data_rows)} raw rows into {len(self.data_rows)} unique averaged pairs."
        )

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
        if sr_a != 16000:
            wav_a = torchaudio.functional.resample(wav_a, sr_a, 16000)
        if sr_b != 16000:
            wav_b = torchaudio.functional.resample(wav_b, sr_b, 16000)

        target = float(row[self.target_metric])

        # Return a dictionary. The collater will dynamically pack the target_metric.
        return {
            "wav_a": wav_a.squeeze(0),
            "wav_b": wav_b.squeeze(0),
            self.target_metric: target,
        }


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
            feat_lengths = torch.tensor(
                [feat.shape[0] for feat in feats], dtype=torch.long
            )
            if self.padding_mode == "zero_padding":
                feats_padded = pad_sequence(feats, batch_first=True, padding_value=0.0)
            elif self.padding_mode == "repetitive":
                max_len = (
                    feat_lengths.max().item()
                )  # Use max across the specific feature list
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
                raise NotImplementedError(
                    f"Padding mode {self.padding_mode} not implemented."
                )
            return feats_padded, feat_lengths

        # 2. Process Audio
        wav_a_list = [sorted_batch[i]["wav_a"] for i in range(bs)]
        batch_dict["wav_a"], batch_dict["wav_a_lengths"] = pad_features(wav_a_list)
        wav_b_list = [
            item["wav_b"] for item in sorted_batch if item.get("wav_b") is not None
        ]
        batch_dict["wav_b"], batch_dict["wav_b_lengths"] = pad_features(wav_b_list)

        # 3. Explicit ID packing
        if "system_id" in all_keys:
            batch_dict["system_ids"] = [sorted_batch[i]["system_id"] for i in range(bs)]
        if "sample_id" in all_keys:
            batch_dict["sample_ids"] = [sorted_batch[i]["sample_id"] for i in range(bs)]
        if "listener_id" in all_keys:
            batch_dict["listener_ids"] = [
                sorted_batch[i]["listener_id"] for i in range(bs)
            ]

        # 4. Dynamically pack all metric targets (e.g., 'score', 'mos', 'spk', 'acc')
        ignore_keys = [
            "wav_path",
            "wav_a_path",
            "wav_b_path",
            "wav_a",
            "wav_b",
            "system_id",
            "sample_id",
            "listener_id",
        ]

        for key in all_keys:
            if key not in ignore_keys:
                if isinstance(sorted_batch[0][key], (int, float)):
                    batch_dict[key] = torch.tensor(
                        [item[key] for item in sorted_batch], dtype=torch.float32
                    )
                elif isinstance(sorted_batch[0][key], str):
                    batch_dict[key] = [item[key] for item in sorted_batch]

        return batch_dict


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Baseline for VoiceMOS 2026 Track 3."
    )
    parser.add_argument(
        "--data-root",
        required=True,
        type=str,
        help="Root directory of the dataset distribution.",
    )
    parser.add_argument(
        "--target-metric",
        required=True,
        type=str,
        choices=["spk_sim", "acc_sim"],
        help="Metric to train the projection head on.",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        type=str,
        help="Directory to save the trained checkpoints.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Training batch size."
    )
    parser.add_argument(
        "--classifier-type",
        type=CLASSIFIER_TYPE,
        default=CLASSIFIER_TYPE.REGULAR,
        help="Whether to train using a CORAL, CORN, or REGULAR classifier",
    )
    parser.add_argument(
        "--soft-labels",
        action="store_true",
        help="For CORAL/CORN, train against soft targets derived from the continuous rating instead of rounding to a hard class label. No effect for REGULAR.",
    )
    parser.add_argument(
        "--accumulate-steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps.",
    )
    parser.add_argument(
        "--train-steps", type=int, default=20000, help="Total number of training steps."
    )
    parser.add_argument(
        "--save-steps", type=int, default=5000, help="Save a checkpoint every N steps."
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=None,
        help="Compute validation loss every N steps (defaults to --save-steps). Ignored if no sets/val.csv is found under --data-root.",
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--verbose", type=int, default=1, help="logging level.")
    args = parser.parse_args()

    log_level = (
        logging.DEBUG
        if args.verbose > 1
        else logging.INFO
        if args.verbose > 0
        else logging.WARN
    )
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    tb_logdir = os.path.join(args.outdir, "tensorboard")
    writer = SummaryWriter(log_dir=tb_logdir)
    logging.info(f"Logging TensorBoard events to {tb_logdir}")

    # 1. Data Preparation
    train_csv_path = os.path.join(args.data_root, "sets", "train.csv")
    logging.info(f"Loading {train_csv_path}...")

    train_data = []
    with open(train_csv_path, "r", encoding="utf-8") as f:
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
        num_workers=4,
        drop_last=True,
    )

    val_csv_path = os.path.join(args.data_root, "sets", "val.csv")
    val_loader = None
    if os.path.exists(val_csv_path):
        logging.info(f"Loading {val_csv_path}...")

        val_data = []
        with open(val_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                val_data.append(row)

        logging.info(f"Loaded {len(val_data)} validation samples.")

        val_dataset = SimilarityDataset(args.data_root, val_data, args.target_metric)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collater,
            num_workers=4,
            drop_last=False,
        )
    else:
        logging.warning(
            f"No validation set found at {val_csv_path}; skipping validation loss."
        )

    eval_steps = args.eval_steps if args.eval_steps is not None else args.save_steps

    # 2. Model Initialization
    logging.info(f"Initializing Model for {args.target_metric} prediction...")
    model = Model(
        model_name="speechbrain/spkrec-ecapa-voxceleb",
        use_projection=True,
        mlp_heads=[args.target_metric],
        freeze_ssl=False,  # Fine-tuning everything
        classifier_type=args.classifier_type,
    )
    model.to(device)

    # 3. Optimizer & Criterion
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    match args.classifier_type:
        case CLASSIFIER_TYPE.CORAL:
            criterion = coral_loss
        case CLASSIFIER_TYPE.CORN:
            criterion = partial(corn_loss, num_classes=NUM_CLASSES)
        case CLASSIFIER_TYPE.REGULAR:
            criterion = torch.nn.MSELoss()

    # 4. Training Loop
    model.train()
    global_step = 0
    forward_steps = 0
    train_loss = 0.0

    optimizer.zero_grad()
    pbar = tqdm(total=args.train_steps, desc="Training")

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

            loss = compute_loss(args, criterion, preds, targets, device)

            # Loss and Gradient Accumulation
            scaled_loss = loss / args.accumulate_steps
            scaled_loss.backward()

            train_loss += loss.item()
            forward_steps += 1

            if forward_steps % args.accumulate_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

                global_step += 1
                pbar.update(1)

                # Update progress bar description every step
                avg_train_loss = train_loss / args.accumulate_steps
                pbar.set_postfix({"loss": f"{avg_train_loss:.4f}"})
                writer.add_scalar("train/loss", avg_train_loss, global_step)
                writer.add_scalar("train/grad_norm", grad_norm, global_step)
                writer.add_scalar(
                    "train/lr", optimizer.param_groups[0]["lr"], global_step
                )
                train_loss = 0.0

                # Save checkpoint every N steps
                if global_step % args.save_steps == 0:
                    save_path = os.path.join(
                        args.outdir, f"model_{args.target_metric}_step{global_step}.pt"
                    )
                    torch.save(model.state_dict(), save_path)
                    logging.info(f"Checkpoint saved to {save_path}")

                # Compute validation loss every N steps
                if val_loader is not None and global_step % eval_steps == 0:
                    val_loss, val_mse = evaluate(
                        model, val_loader, args, criterion, device
                    )
                    logging.info(
                        f"Step {global_step}: validation loss = {val_loss:.4f}, MSE = {val_mse:.4f}"
                    )
                    writer.add_scalar("val/loss", val_loss, global_step)
                    writer.add_scalar("val/mse", val_mse, global_step)

    pbar.close()

    # 5. Final Validation
    if val_loader is not None:
        val_loss, val_mse = evaluate(model, val_loader, args, criterion, device)
        logging.info(f"Final validation loss = {val_loss:.4f}, MSE = {val_mse:.4f}")
        writer.add_scalar("val/loss", val_loss, global_step)
        writer.add_scalar("val/mse", val_mse, global_step)

    writer.close()

    # 6. Save Final Model
    save_path = os.path.join(
        args.outdir, f"finetuned_model_{args.target_metric}_final.pt"
    )
    torch.save(model.state_dict(), save_path)
    logging.info(f"Training complete. Final model saved to {save_path}")


if __name__ == "__main__":
    main()
