#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 1: train the pair encoder with the Rank-N-Contrast loss alone.

No regression head, no regression loss. The only objective is to order the pair
representations by their similarity-score distances. Stage 2 (`train_head.py`)
then freezes this encoder and fits a predictor on top.

Example:
    DR=../baseline/data/vmc2026_track3_train_phase_distro_v3_syn
    python train_rnc.py \
        --data-root $DR --train-csv $DR/sets/train.csv \
        --target-metric spk_sim --outdir egs/rnc_spk_sim
"""

import argparse
import json
import logging
import os
import random
import time

import numpy as np
import torch

from data import build_loader
from loss import RnCLoss, feature_label_rank_corr, rnc_lower_bound
from model import ECAPA, PairEncoder


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True, help="Dataset root containing wav/.")
    p.add_argument("--train-csv", required=True, help="Labelled training split.")
    p.add_argument("--val-csv", default=None,
                   help="Optional LABELLED split for monitoring. The official "
                        "sets/dev.csv has no scores, so by default stage 1 trains "
                        "for a fixed number of steps and keeps encoder_last.pt.")
    p.add_argument("--target-metric", required=True, choices=["spk_sim", "acc_sim"])
    p.add_argument("--outdir", required=True)

    p.add_argument("--batch-size", type=int, default=64,
                   help="RNC benefits monotonically from more in-batch positives "
                        "(paper Table 6a); raise this as far as memory allows. "
                        "With ECAPA training, measured peaks on a 46 GiB L40S are "
                        "22.3 GiB at 64 and 31.6 GiB at 96, both with "
                        "--max-audio-sec 6; without a crop, peak follows the "
                        "longest clip in the batch and 64 already reaches 36.6 GiB.")
    p.add_argument("--train-steps", type=int, default=13000)
    p.add_argument("--save-steps", type=int, default=1000)
    p.add_argument("--eval-steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-5,
                   help="AdamW learning rate for the 22.15M-parameter ECAPA "
                        "backbone. The baseline's 1e-3 was calibrated for a 49k "
                        "projection on a frozen encoder and is far too high here.")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lr-schedule", choices=["cosine", "none"], default="cosine",
                   help="The paper uses cosine annealing; the baseline uses none.")
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--temperature", type=float, default=2.0, help="RNC temperature (paper default).")
    p.add_argument("--feature-sim", choices=["l2", "l1", "cosine"], default="l2",
                   help="Paper Table 6b: negative L2 beats cosine for regression.")
    p.add_argument("--label-diff", choices=["l1", "l2"], default="l1")

    p.add_argument("--model-name", default=ECAPA)
    p.add_argument("--freeze-ecapa", action="store_true",
                   help="Freeze the 22.15M ECAPA parameters, leaving RNC only the "
                        "49k projection to shape. This is an ABLATION, not the "
                        "default: with ECAPA frozen the contrastive objective has "
                        "almost nothing to act on and stage 1 barely moves (the "
                        "first sweep lost 0.07 nats over 8,750 steps). Stage 2 is "
                        "where the baseline's frozen-ECAPA behaviour is reproduced.")
    p.add_argument("--unfreeze-ecapa", action="store_true",
                   help=argparse.SUPPRESS)  # deprecated: now the default, kept so
                                            # existing commands keep working
    p.add_argument("--ecapa-eval-mode", action="store_true",
                   help="Pin ECAPA to eval() so its BatchNorm running statistics "
                        "stop drifting. The baseline lets them drift.")
    p.add_argument("--max-audio-sec", type=float, default=None,
                   help="Random-crop waveforms to this length during training.")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(batch, device):
    return (
        batch["wav_a"].to(device, non_blocking=True),
        batch["wav_b"].to(device, non_blocking=True),
        batch["wav_a_lengths"].to(device, non_blocking=True),
        batch["wav_b_lengths"].to(device, non_blocking=True),
        batch["b_index"].to(device, non_blocking=True),
    )


@torch.no_grad()
def evaluate(encoder, loader, criterion, device, label_diff):
    """Collects features over the whole split, then reports the RNC loss, its
    tight lower bound L*, the gap between them, and the feature/label rank
    correlation (the paper's Table 1 diagnostic)."""
    encoder.eval()
    feats, labels = [], []
    for batch in loader:
        wav_a, wav_b, len_a, len_b, b_index = to_device(batch, device)
        feats.append(encoder(wav_a, wav_b, len_a, len_b, b_index).float().cpu())
        labels.append(batch["score"])
    encoder.train()
    if not feats:
        return {}

    feats = torch.cat(feats).to(device)
    labels = torch.cat(labels).to(device)
    loss = criterion(feats, labels).item()
    lower = rnc_lower_bound(labels, label_diff=label_diff).item()
    return {
        "rnc": loss,
        "rnc_lower_bound": lower,
        "rnc_gap": loss - lower,
        "feat_label_srcc": feature_label_rank_corr(
            feats, labels, feature_sim=criterion.feature_sim, label_diff=label_diff
        ),
        "feat_norm": float(feats.norm(dim=-1).mean().item()),
    }


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[logging.FileHandler(os.path.join(args.outdir, "train_rnc.log")),
                  logging.StreamHandler()],
    )
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")
    logging.info(f"Args: {vars(args)}")

    _, train_loader = build_loader(
        args.data_root, args.train_csv, args.target_metric,
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        max_audio_sec=args.max_audio_sec, train=True, drop_last=True,
    )
    val_loader = None
    if args.val_csv:
        _, val_loader = build_loader(
            args.data_root, args.val_csv, args.target_metric,
            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
            max_audio_sec=args.max_audio_sec, train=False,
        )

    if args.unfreeze_ecapa:
        logging.warning("--unfreeze-ecapa is deprecated and now a no-op: stage 1 "
                        "fine-tunes ECAPA by default. Use --freeze-ecapa to opt out.")

    encoder = PairEncoder(
        model_name=args.model_name,
        use_projection=True,
        freeze_ecapa=args.freeze_ecapa,
        ecapa_eval_mode=args.ecapa_eval_mode,
    ).to(device)
    criterion = RnCLoss(
        temperature=args.temperature,
        label_diff=args.label_diff,
        feature_sim=args.feature_sim,
    )
    params = [p for p in encoder.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    n_total = sum(p.numel() for p in encoder.parameters())
    logging.info(
        f"Trainable parameters: {n_train:,} / {n_total:,} ({100 * n_train / n_total:.2f}%) "
        f"| ECAPA {'FROZEN (ablation -- RNC can only reshape the projection)' if args.freeze_ecapa else 'trainable'}"
    )
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_steps)
        if args.lr_schedule == "cosine" else None
    )

    encoder.train()
    step, running, seen, best = 0, 0.0, 0, -float("inf")
    history = []
    start = time.time()
    while step < args.train_steps:
        for batch in train_loader:
            if step >= args.train_steps:
                break
            wav_a, wav_b, len_a, len_b, b_index = to_device(batch, device)
            labels = batch["score"].to(device)

            features = encoder(wav_a, wav_b, len_a, len_b, b_index)
            loss = criterion(features, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            running += loss.item()
            seen += 1
            step += 1

            if step % 50 == 0:
                lower = rnc_lower_bound(labels, label_diff=args.label_diff).item()
                logging.info(
                    f"step {step:>6}/{args.train_steps} | loss {running / seen:.4f} "
                    f"| batch L* {lower:.4f} | gap {running / seen - lower:+.4f} "
                    f"| lr {optimizer.param_groups[0]['lr']:.2e} "
                    f"| {(time.time() - start) / step:.2f}s/step"
                )
                running, seen = 0.0, 0

            if val_loader is not None and step % args.eval_steps == 0:
                stats = evaluate(encoder, val_loader, criterion, device, args.label_diff)
                stats["step"] = step
                history.append(stats)
                logging.info(
                    f"  [val] step {step} | rnc {stats['rnc']:.4f} "
                    f"| L* {stats['rnc_lower_bound']:.4f} | gap {stats['rnc_gap']:+.4f} "
                    f"| feat/label srcc {stats['feat_label_srcc']:.4f} "
                    f"| |feat| {stats['feat_norm']:.3f}"
                )
                if stats["feat_label_srcc"] > best:
                    best = stats["feat_label_srcc"]
                    torch.save(
                        {"encoder": encoder.state_dict(), "step": step,
                         "args": vars(args), "val": stats},
                        os.path.join(args.outdir, "encoder_best.pt"),
                    )
                    logging.info(f"  [val] new best feat/label srcc {best:.4f} -> encoder_best.pt")

            if step % args.save_steps == 0:
                torch.save(
                    {"encoder": encoder.state_dict(), "step": step, "args": vars(args)},
                    os.path.join(args.outdir, f"encoder_step{step}.pt"),
                )

    torch.save(
        {"encoder": encoder.state_dict(), "step": step, "args": vars(args)},
        os.path.join(args.outdir, "encoder_last.pt"),
    )
    with open(os.path.join(args.outdir, "rnc_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    logging.info(f"Done in {(time.time() - start) / 60:.1f} min. Checkpoints in {args.outdir}")


if __name__ == "__main__":
    main()
