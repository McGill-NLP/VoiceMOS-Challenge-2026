#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified fine-tuning for VoiceMOS 2026 Track 3.

Combines the ingredients that helped individually on the other branches into one script,
so they can be stacked and ablated from the command line:

  --encoder            pluggable backbone, composable with '+'      (dev.dg/contrastive)
  --head mlp|moe       mixture-of-experts prediction head           (dev.yj/empirical)
  --freeze-steps N     two-phase schedule: freeze, then unfreeze    (dev.yj/empirical)
  --backbone-lr-mult   lower learning rate for the backbone         (dev.yj/empirical)
  --objective          mse | corn | coral                           (dev.ap/CORN)
  --lambda-rnc         Rank-N-Contrast auxiliary on the interaction (dev.dg/contrastive)

The defaults reproduce the official Baseline 2 recipe with a genuinely trainable encoder:
`--encoder ecapa-voxceleb --head mlp --objective mse --lambda-rnc 0`, batch 16, AdamW at
1e-3 over one parameter group, 20,000 steps, MSE on per-pair mean scores.

Checkpoints are saved as {"config": ..., "state_dict": ...} so inference.py can rebuild
the architecture without being told which combination produced it.
"""

import argparse
import csv
import json
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
from interactions import INTERACTIONS
from model import UnifiedModel
from objectives import NUM_CLASSES, OBJECTIVES, rnc_loss, task_loss

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # tensorboard is optional -- a missing dep must not kill a long run
    SummaryWriter = None


class SimilarityDataset(Dataset):
    def __init__(self, data_root, data_rows, target_metric, keep_system_id=False):
        self.data_root = data_root
        self.target_metric = target_metric
        # System-level metrics need the system_id to survive collation. Off during
        # training, where it is unused.
        self.keep_system_id = keep_system_id

        aggregated_data = {}
        for row in data_rows:
            pair_key = (row["wav_a_path"], row["wav_b_path"])
            if pair_key not in aggregated_data:
                aggregated_data[pair_key] = row.copy()
                aggregated_data[pair_key][target_metric] = []
            aggregated_data[pair_key][target_metric].append(float(row[target_metric]))

        self.data_rows = []
        for _pair_key, row_data in aggregated_data.items():
            scores = row_data[target_metric]
            row_data[target_metric] = sum(scores) / len(scores)
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
        wav_a, sr_a = torchaudio.load(os.path.join(self.data_root, row["wav_a_path"]))
        wav_b, sr_b = torchaudio.load(os.path.join(self.data_root, row["wav_b_path"]))
        if sr_a != 16000:
            wav_a = torchaudio.functional.resample(wav_a, sr_a, 16000)
        if sr_b != 16000:
            wav_b = torchaudio.functional.resample(wav_b, sr_b, 16000)

        item = {
            "wav_a": wav_a.squeeze(0),
            "wav_b": wav_b.squeeze(0),
            self.target_metric: float(row[self.target_metric]),
        }
        if self.keep_system_id and "system_id" in row:
            item["system_id"] = row["system_id"]
        return item


class SimilarityCollater:
    """Pads a batch of similarity dicts. Repetitive padding, as in the baseline."""

    def __init__(self, padding_mode="repetitive"):
        self.padding_mode = padding_mode

    def __call__(self, batch):
        batch_dict = {}
        sorted_batch = sorted(batch, key=lambda x: -x["wav_a"].shape[0])
        bs = len(sorted_batch)
        all_keys = list(sorted_batch[0].keys())

        def pad_features(feats):
            feat_lengths = torch.tensor([f.shape[0] for f in feats], dtype=torch.long)
            if self.padding_mode == "zero_padding":
                feats_padded = pad_sequence(feats, batch_first=True, padding_value=0.0)
            elif self.padding_mode == "repetitive":
                max_len = feat_lengths.max().item()
                out = []
                for feat in feats:
                    this_len = feat.shape[0]
                    if this_len == 0:
                        out.append(torch.zeros(max_len, dtype=feat.dtype))
                        continue
                    dup_times = max_len // this_len
                    remain = max_len - this_len * dup_times
                    to_dup = [feat for _ in range(dup_times)]
                    if remain > 0:
                        to_dup.append(feat[:remain])
                    out.append(torch.cat(to_dup, dim=0))
                feats_padded = torch.stack(out, dim=0)
            else:
                raise NotImplementedError(f"Padding mode {self.padding_mode} not implemented.")
            return feats_padded, feat_lengths

        batch_dict["wav_a"], batch_dict["wav_a_lengths"] = pad_features(
            [sorted_batch[i]["wav_a"] for i in range(bs)]
        )
        batch_dict["wav_b"], batch_dict["wav_b_lengths"] = pad_features(
            [item["wav_b"] for item in sorted_batch if item.get("wav_b") is not None]
        )

        if "system_id" in all_keys:
            batch_dict["system_ids"] = [sorted_batch[i]["system_id"] for i in range(bs)]

        ignore_keys = ["wav_path", "wav_a_path", "wav_b_path", "wav_a", "wav_b",
                       "system_id", "sample_id", "listener_id"]
        for key in all_keys:
            if key in ignore_keys:
                continue
            if isinstance(sorted_batch[0][key], (int, float)):
                batch_dict[key] = torch.tensor([item[key] for item in sorted_batch], dtype=torch.float32)
            elif isinstance(sorted_batch[0][key], str):
                batch_dict[key] = [item[key] for item in sorted_batch]

        return batch_dict


def save_checkpoint(path, model, config):
    torch.save({"config": config, "state_dict": model.state_dict()}, path)


METRIC_KEYS = ["mse_utt", "lcc_utt", "srcc_utt", "mse_sys", "lcc_sys", "srcc_sys"]


@torch.no_grad()
def evaluate(model, loader, target_metric, device):
    """Score a labelled dev set, utterance- and system-level.

    Restores the model's previous train/eval mode on the way out. Runs batched, so the
    collater pads every clip up to the longest in its batch; inference.py runs one pair
    at a time with no padding, so the two can disagree slightly. Utterance-level scores
    track closely, but system-level SRCC is a rank statistic over ~23 systems and a hair
    of numerical difference can swap two adjacent systems. Treat the in-training curve as
    monitoring and a standalone inference.py run as the number of record;
    --eval-batch-size 1 makes them agree exactly, at much higher cost.
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

    utt_true, utt_pred = np.array(targets), np.array(preds)
    # Cast to plain floats: compute_metrics returns numpy scalars, and those end up in
    # the checkpoint config, where torch.load's weights_only=True default refuses to
    # unpickle numpy._core.multiarray.scalar.
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
    return "  ".join(f"{k}={results[k]:.4f}" for k in METRIC_KEYS if k in results)


def print_metrics_block(results, target_metric, step, tag=""):
    """Print the dev scores in the same layout as calculate_metrics.py.

    The one-line `format_metrics` version is easy to grep across a run; this block is what
    you actually read in a Slurm log, and matching the standalone tool's layout means the
    in-training number and the post-hoc number can be compared without re-reading either.
    """
    bar = "=" * 46
    print(f"\n{bar}")
    print(f"Results for {target_metric.upper()}  [dev @ step {step}]{tag}")
    print(bar)
    print(f"Evaluated Pairs   : {results.get('n_pairs', '?')}")
    print(f"Evaluated Systems : {results.get('n_systems', '?')}")
    print("-" * 46)
    print("[UTTERANCE LEVEL]")
    print(f"MSE  : {results.get('mse_utt', float('nan')):.4f}")
    print(f"LCC  : {results.get('lcc_utt', float('nan')):.4f}")
    print(f"SRCC : {results.get('srcc_utt', float('nan')):.4f}")
    print("-" * 46)
    print("[SYSTEM LEVEL]")
    print(f"MSE  : {results.get('mse_sys', float('nan')):.4f}")
    print(f"LCC  : {results.get('lcc_sys', float('nan')):.4f}")
    print(f"SRCC : {results.get('srcc_sys', float('nan')):.4f}")
    print(f"{bar}\n", flush=True)


def build_parser():
    p = argparse.ArgumentParser(
        description="Unified fine-tuning for VoiceMOS 2026 Track 3.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # -- data
    p.add_argument("--data-root", required=True, type=str, help="Root directory of the dataset distribution.")
    p.add_argument("--target-metric", required=True, type=str, choices=["spk_sim", "acc_sim"], help="Which score to predict.")
    p.add_argument("--outdir", required=True, type=str, help="Directory for checkpoints and logs.")
    p.add_argument("--train-csv", type=str, default=None, help="Training CSV. Defaults to <data-root>/sets/train.csv.")
    p.add_argument("--dev-csv", type=str, default=None, help="Labelled dev CSV, scored periodically during training.")
    p.add_argument("--dev-data-root", type=str, default=None, help="Data root for --dev-csv. Defaults to --data-root.")

    # -- encoder
    p.add_argument("--encoder", type=str, default="ecapa-voxceleb",
                   help=f"Backbone. One of: {', '.join(ENCODER_REGISTRY)}. Combine with '+'.")
    p.add_argument("--encoder-checkpoint", type=str, default=None, help="Local backbone weights instead of downloading.")
    p.add_argument("--cache-dir", type=str, default=None, help="Where to cache downloaded encoder weights.")
    p.add_argument("--embedding-dim", type=int, default=256, help="Projection width after the backbone.")

    # -- head
    p.add_argument("--head", type=str, default="mlp", choices=["mlp", "moe"], help="Prediction head trunk.")
    p.add_argument("--hidden-dim", type=int, default=64, help="Hidden width inside the head trunk.")
    p.add_argument("--num-experts", type=int, default=2, help="MoE experts. Ignored unless --head moe.")
    p.add_argument("--top-k", type=int, default=None, help="MoE top-k routing. Unset means dense (soft) gating.")
    p.add_argument("--lambda-moe-aux", type=float, default=0.01, help="Weight of the MoE load-balancing auxiliary loss.")
    p.add_argument("--ordinal-dim", type=int, default=128, help="Trunk output width feeding the CORN/CORAL layer.")
    p.add_argument("--no-range-clipping", action="store_true", help="Disable the Tanh*2+3 output clamp (mse only).")

    # -- interaction
    p.add_argument("--interaction", type=str, default="baseline", choices=INTERACTIONS,
                   help="How the two embeddings are combined before the head. See interactions.py.")
    p.add_argument("--bilinear-rank", type=int, default=64,
                   help="Rank of the learned bilinear term. Ignored unless --interaction bilinear.")

    # -- objective
    p.add_argument("--objective", type=str, default="mse", choices=OBJECTIVES, help="Primary training objective.")
    p.add_argument("--hard-labels", action="store_true",
                   help="CORN/CORAL: round the per-pair mean to an integer class instead of using soft targets.")
    p.add_argument("--lambda-rnc", type=float, default=0.0,
                   help="Weight of the Rank-N-Contrast auxiliary on the interaction vector. 0 disables it.")
    p.add_argument("--rnc-temperature", type=float, default=2.0, help="RNC temperature (paper default).")
    p.add_argument("--rnc-label-diff", type=str, default="l1", choices=["l1", "l2"], help="RNC label distance.")
    p.add_argument("--rnc-feature-sim", type=str, default="l2", choices=["l1", "l2", "cosine"], help="RNC feature similarity.")

    # -- freezing schedule
    p.add_argument("--freeze-steps", type=int, default=0,
                   help="Keep the backbone frozen for the first N optimizer steps, then unfreeze. 0 trains it from step 1.")
    p.add_argument("--freeze-encoder", action="store_true", help="Never train the backbone (frozen control).")
    p.add_argument("--encoder-lr", type=float, default=None, help="Explicit backbone learning rate. Overrides --backbone-lr-mult.")
    p.add_argument("--backbone-lr-mult", type=float, default=None,
                   help="Backbone learning rate as a multiple of --lr. Unset means one parameter group at --lr.")

    # -- optimisation
    p.add_argument("--batch-size", type=int, default=16, help="Training batch size.")
    p.add_argument("--accumulate-steps", type=int, default=1, help="Gradient accumulation steps.")
    p.add_argument("--train-steps", type=int, default=20000, help="Total optimizer steps.")
    p.add_argument("--save-steps", type=int, default=5000, help="Save a checkpoint every N steps.")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the head.")
    p.add_argument("--weight-decay", type=float, default=0.0, help="AdamW weight decay.")
    p.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm.")

    # -- evaluation / bookkeeping
    p.add_argument("--eval-steps", type=int, default=500, help="Score --dev-csv every N optimizer steps.")
    p.add_argument("--eval-batch-size", type=int, default=16, help="Batch size for dev evaluation.")
    p.add_argument("--best-metric", type=str, default="srcc_sys", choices=METRIC_KEYS,
                   help="Dev metric that decides which checkpoint is kept as model_best.")
    p.add_argument("--tensorboard-dir", type=str, default=None,
                   help="TensorBoard event directory. Defaults to <outdir>/tensorboard.")
    p.add_argument("--no-tensorboard", action="store_true", help="Disable TensorBoard logging.")
    p.add_argument("--log-every", type=int, default=50,
                   help="Write training scalars to TensorBoard every N optimizer steps.")
    p.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    p.add_argument("--seed", type=int, default=1337, help="Random seed.")
    p.add_argument("--verbose", type=int, default=1, help="Logging level.")
    return p


def main():
    args = build_parser().parse_args()

    log_level = logging.DEBUG if args.verbose > 1 else logging.INFO if args.verbose > 0 else logging.WARN
    logging.basicConfig(level=log_level, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # TensorBoard. Optional by design: a missing tensorboard install, or a full disk on the
    # event directory, must not take down a run that costs hours of GPU.
    writer = None
    if not args.no_tensorboard:
        tb_logdir = args.tensorboard_dir or os.path.join(args.outdir, "tensorboard")
        if SummaryWriter is None:
            logging.warning("tensorboard is not installed; continuing without event logging.")
        else:
            try:
                writer = SummaryWriter(log_dir=tb_logdir)
                logging.info(f"TensorBoard events -> {tb_logdir}")
                logging.info(f"  tensorboard --logdir {os.path.abspath(tb_logdir)}")
            except Exception as e:  # noqa: BLE001 - never fatal
                logging.warning(f"Could not open TensorBoard writer ({e}); continuing without it.")

    def tb_text(tag, body):
        """Record run provenance in TensorBoard's TEXT tab.

        Deliberately not add_hparams: its plugin metadata did not survive a round trip
        through EventAccumulator on this torch/tensorboard pairing, so the HPARAMS tab
        would have been silently empty. Text always works, and run_config.json next to the
        checkpoints remains the machine-readable copy.
        """
        if writer is not None:
            writer.add_text(tag, body, 0)

    if args.lambda_rnc > 0 and args.batch_size < 32:
        logging.warning(
            f"--lambda-rnc {args.lambda_rnc} with --batch-size {args.batch_size}: Rank-N-Contrast "
            "ranks every sample against every other sample IN THE BATCH, so a small batch gives it "
            "very little to rank. ../rank-n-contrast used batch 96. Consider raising --batch-size "
            "(with --accumulate-steps 1) or expect the term to be weak."
        )

    # 1. Data
    train_csv_path = args.train_csv or os.path.join(args.data_root, "sets", "train.csv")
    logging.info(f"Loading {train_csv_path}...")
    with open(train_csv_path, "r", encoding="utf-8") as f:
        train_data = list(csv.DictReader(f))
    logging.info(f"Loaded {len(train_data)} training samples.")

    collater = SimilarityCollater(padding_mode="repetitive")
    train_dataset = SimilarityDataset(args.data_root, train_data, args.target_metric)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collater,
        num_workers=args.num_workers, drop_last=True,
    )

    dev_loader = None
    if args.dev_csv:
        dev_root = args.dev_data_root or args.data_root
        logging.info(f"Loading dev set {args.dev_csv} (data root {dev_root})...")
        with open(args.dev_csv, "r", encoding="utf-8") as f:
            dev_data = list(csv.DictReader(f))
        if args.target_metric not in (dev_data[0] if dev_data else {}):
            raise SystemExit(f"{args.dev_csv} has no '{args.target_metric}' column; it must be labelled.")
        dev_dataset = SimilarityDataset(dev_root, dev_data, args.target_metric, keep_system_id=True)
        dev_loader = DataLoader(
            dev_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collater,
            num_workers=args.num_workers, drop_last=False,
        )
        logging.info(
            f"Dev evaluation every {args.eval_steps} steps on {len(dev_dataset)} pairs; "
            f"model_best tracks {args.best_metric}."
        )

    # 2. Model
    config = {
        "encoder": args.encoder,
        "target_metric": args.target_metric,
        "objective": args.objective,
        "head": args.head,
        "interaction": args.interaction,
        "bilinear_rank": args.bilinear_rank,
        "embedding_dim": args.embedding_dim,
        "hidden_dim": args.hidden_dim,
        "ordinal_dim": args.ordinal_dim,
        "num_classes": NUM_CLASSES,
        "num_experts": args.num_experts,
        "top_k": args.top_k,
        "range_clipping": not args.no_range_clipping,
        "soft_labels": not args.hard_labels,
        "lambda_rnc": args.lambda_rnc,
        "freeze_steps": args.freeze_steps,
        "freeze_encoder": args.freeze_encoder,
    }
    logging.info(
        f"Model: encoder={args.encoder}  head={args.head}"
        + (f"(experts={args.num_experts}, top_k={args.top_k})" if args.head == "moe" else "")
        + f"  interaction={args.interaction}"
        + (f"(rank={args.bilinear_rank})" if args.interaction == "bilinear" else "")
        + f"  objective={args.objective}"
        + ("" if args.objective == "mse" else f"(soft_labels={not args.hard_labels})")
        + (f"  +rnc(lambda={args.lambda_rnc})" if args.lambda_rnc > 0 else "")
    )
    model = UnifiedModel(
        encoder_name=args.encoder,
        target_metric=args.target_metric,
        objective=args.objective,
        head_type=args.head,
        interaction=args.interaction,
        bilinear_rank=args.bilinear_rank,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        ordinal_dim=args.ordinal_dim,
        num_experts=args.num_experts,
        top_k=args.top_k,
        range_clipping=not args.no_range_clipping,
        cache_dir=args.cache_dir,
        encoder_checkpoint=args.encoder_checkpoint,
    )
    model.to(device)

    # 3. Freezing schedule
    #
    # Two-phase: the backbone stays frozen while the randomly initialised head finds a
    # sensible region, then unfreezes. Without it, early head gradients flow into a
    # pretrained backbone that has no reason to accommodate them.
    freeze_until = args.train_steps + 1 if args.freeze_encoder else args.freeze_steps
    if freeze_until > 0:
        model.set_encoder_trainable(False)
        if args.freeze_encoder:
            logging.info("Backbone frozen for the whole run (--freeze-encoder).")
        else:
            logging.info(f"Backbone frozen for the first {freeze_until} steps, then unfrozen.")
    else:
        model.set_encoder_trainable(True)

    n_total = sum(p.numel() for p in model.parameters())
    n_head = sum(p.numel() for p in model.head_parameters())
    n_backbone = sum(p.numel() for p in model.encoder_parameters())
    logging.info(
        f"Parameters: {n_total / 1e6:.2f}M total = {n_backbone / 1e6:.2f}M backbone "
        f"+ {n_head / 1e6:.2f}M head/projection. Encoder output dim {model.encoder.backbone.output_dim}, "
        f"interaction dim {model.interaction_dim}."
    )

    # 4. Optimizer
    #
    # The backbone is always in a parameter group even while frozen: its params simply
    # receive no gradient until the unfreeze step, at which point they start updating
    # without the optimizer needing to be rebuilt.
    if args.encoder_lr is not None:
        backbone_lr = args.encoder_lr
    elif args.backbone_lr_mult is not None:
        backbone_lr = args.lr * args.backbone_lr_mult
    else:
        backbone_lr = None

    if args.freeze_encoder:
        param_groups = [{"params": list(model.head_parameters()), "lr": args.lr}]
        logging.info(f"Optimizing the head only, lr={args.lr}.")
    elif backbone_lr is not None:
        param_groups = [
            {"params": list(model.encoder_parameters()), "lr": backbone_lr},
            {"params": list(model.head_parameters()), "lr": args.lr},
        ]
        logging.info(f"Backbone lr={backbone_lr}, head lr={args.lr}.")
    else:
        # The baseline recipe: one group, one learning rate, everything in it.
        param_groups = [{"params": list(model.parameters()), "lr": args.lr}]
        logging.info(f"Single parameter group, lr={args.lr}.")

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)

    run_config = {**config, **vars(args)}
    with open(os.path.join(args.outdir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, default=str)

    tb_text("run_config", "```json\n" + json.dumps(run_config, indent=2, default=str) + "\n```")

    # 5. Training
    model.train()
    global_step, forward_steps = 0, 0
    run_task, run_rnc, run_aux = 0.0, 0.0, 0.0

    log_path = os.path.join(args.outdir, f"dev_log_{args.target_metric}.csv")
    log_file, log_writer, best_score = None, None, None
    if dev_loader is not None:
        log_file = open(log_path, "w", newline="", encoding="utf-8")
        log_writer = csv.writer(log_file)
        log_writer.writerow(["step", "train_task", "train_rnc", "train_moe_aux"] + METRIC_KEYS)

    def run_eval(step, losses, track_best=True):
        nonlocal best_score
        results = evaluate(model, dev_loader, args.target_metric, device)
        logging.info(f"[dev @ step {step}] {format_metrics(results)}")
        # Same six numbers calculate_metrics.py reports, in its layout, so the Slurm log
        # carries the full picture and not just the one-line summary.
        print_metrics_block(results, args.target_metric, step,
                            tag="  (untrained reference)" if not track_best else "")
        log_writer.writerow(
            [step] + [f"{v:.6f}" for v in losses]
            + [f"{results.get(k, float('nan')):.6f}" for k in METRIC_KEYS]
        )
        log_file.flush()

        if writer is not None:
            for k in METRIC_KEYS:
                if k in results and not np.isnan(results[k]):
                    # dev/utt/... and dev/sys/... so TensorBoard groups the two levels.
                    stat, level = k.split("_")
                    writer.add_scalar(f"dev_{level}/{stat}", results[k], step)
            writer.flush()

        score = results.get(args.best_metric)
        if track_best and score is not None and not np.isnan(score):
            improved = (
                best_score is None
                or (score < best_score if args.best_metric.startswith("mse") else score > best_score)
            )
            if improved:
                best_score = score
                save_checkpoint(
                    os.path.join(args.outdir, f"model_best_{args.target_metric}.pt"), model,
                    {**config, "best_metric": args.best_metric, "best_score": score, "best_step": step},
                )
                logging.info(f"[dev @ step {step}] new best {args.best_metric}={score:.4f}, saved model_best.")
        if writer is not None and best_score is not None:
            writer.add_scalar(f"dev_best/{args.best_metric}", best_score, step)
        return results

    optimizer.zero_grad()
    pbar = tqdm(total=args.train_steps, desc="Training")

    if dev_loader is not None:
        # Step 0 is the untrained-head reference every later number is read against.
        # Excluded from best tracking: a random head can score well by chance.
        run_eval(0, (float("nan"),) * 3, track_best=False)

    while global_step < args.train_steps:
        for batch in train_loader:
            if global_step >= args.train_steps:
                break

            wav_a, len_a = batch["wav_a"].to(device), batch["wav_a_lengths"].to(device)
            wav_b, len_b = batch["wav_b"].to(device), batch["wav_b_lengths"].to(device)
            targets = batch[args.target_metric].to(device)

            outputs = model(wav_a, wav_b, len_a, len_b)

            primary = task_loss(args.objective, outputs["raw"], targets,
                                soft_labels=not args.hard_labels, num_classes=NUM_CLASSES)
            loss = primary
            run_task += float(primary.item())

            if args.lambda_rnc > 0:
                rnc = rnc_loss(
                    outputs["interaction"], targets,
                    temperature=args.rnc_temperature,
                    label_diff=args.rnc_label_diff,
                    feature_sim=args.rnc_feature_sim,
                )
                loss = loss + args.lambda_rnc * rnc
                run_rnc += float(rnc.item())

            if args.head == "moe":
                aux = outputs["moe_aux_loss"]
                loss = loss + args.lambda_moe_aux * aux
                run_aux += float(aux.item())

            (loss / args.accumulate_steps).backward()
            forward_steps += 1

            if forward_steps % args.accumulate_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()

                global_step += 1
                pbar.update(1)

                losses = (run_task / args.accumulate_steps,
                          run_rnc / args.accumulate_steps,
                          run_aux / args.accumulate_steps)
                postfix = {"task": f"{losses[0]:.4f}"}
                if args.lambda_rnc > 0:
                    postfix["rnc"] = f"{losses[1]:.4f}"
                pbar.set_postfix(postfix)
                run_task = run_rnc = run_aux = 0.0

                if writer is not None and (global_step % args.log_every == 0 or global_step == 1):
                    writer.add_scalar("train/task_loss", losses[0], global_step)
                    if args.lambda_rnc > 0:
                        writer.add_scalar("train/rnc_loss", losses[1], global_step)
                        writer.add_scalar("train/total_loss",
                                          losses[0] + args.lambda_rnc * losses[1], global_step)
                    if args.head == "moe":
                        writer.add_scalar("train/moe_aux_loss", losses[2], global_step)
                    writer.add_scalar("train/grad_norm", float(grad_norm), global_step)
                    for gi, group in enumerate(optimizer.param_groups):
                        writer.add_scalar(f"train/lr_group{gi}", group["lr"], global_step)
                    # 0/1 trace of the two-phase schedule, so a change in the loss curve can
                    # be lined up with the unfreeze step rather than guessed at.
                    writer.add_scalar("train/backbone_trainable",
                                      float(model.encoder_trainable), global_step)

                # Unfreeze exactly once, at the scheduled step.
                if not args.freeze_encoder and freeze_until > 0 and global_step == freeze_until:
                    model.set_encoder_trainable(True)
                    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    logging.info(
                        f"Step {global_step}: backbone unfrozen, {n_train / 1e6:.2f}M parameters now trainable."
                    )

                if global_step % args.save_steps == 0:
                    save_path = os.path.join(args.outdir, f"model_{args.target_metric}_step{global_step}.pt")
                    save_checkpoint(save_path, model, config)
                    logging.info(f"Checkpoint saved to {save_path}")

                if dev_loader is not None and global_step % args.eval_steps == 0:
                    results = run_eval(global_step, losses)
                    postfix[args.best_metric] = f"{results.get(args.best_metric, float('nan')):.4f}"
                    pbar.set_postfix(postfix)

    pbar.close()

    save_path = os.path.join(args.outdir, f"finetuned_model_{args.target_metric}_final.pt")
    save_checkpoint(save_path, model, config)
    logging.info(f"Training complete. Final model saved to {save_path}")

    if dev_loader is not None:
        if global_step % args.eval_steps != 0:
            run_eval(global_step, (float("nan"),) * 3)
        logging.info(f"Dev history written to {log_path}")
        if best_score is not None:
            logging.info(
                f"Best {args.best_metric} = {best_score:.4f} (model_best_{args.target_metric}.pt); "
                f"final-step model is {os.path.basename(save_path)}."
            )
        log_file.close()

    if writer is not None:
        writer.close()
        logging.info(
            f"TensorBoard events written. View with:\n"
            f"  tensorboard --logdir {os.path.abspath(args.tensorboard_dir or os.path.join(args.outdir, 'tensorboard'))}"
        )


if __name__ == "__main__":
    main()
