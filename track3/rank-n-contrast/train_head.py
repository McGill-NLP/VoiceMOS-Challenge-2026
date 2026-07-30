#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 2: fit a regression head on the pair representation.

Drives both arms of the comparison through identical data handling:

  RNC arm       --encoder-ckpt <stage-1 ckpt> --freeze-encoder
                Linear probing on frozen RNC features (the paper's best variant,
                Table 6c: probing 6.14 < fine-tuning 6.36 < joint 6.42).

  Baseline arm  (no --encoder-ckpt, no --freeze-encoder)
                End-to-end MSE fine-tuning of pretrained ECAPA with the MLP head
                and range clipping -- i.e. the official baseline 2.

When the encoder is frozen its features are deterministic, so they are extracted
once and cached; stage 2 then costs seconds rather than GPU-hours.

Example (RNC arm):
    DR=../baseline/data/vmc2026_track3_train_phase_distro_v3_syn
    python train_head.py \
        --data-root $DR --train-csv $DR/sets/train.csv \
        --target-metric spk_sim --outdir egs/rnc_spk_sim/head \
        --encoder-ckpt egs/rnc_spk_sim/encoder_last.pt --freeze-encoder \
        --head linear --loss l1
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
from metrics import evaluate as eval_metrics, format_metrics
from model import ECAPA, Model


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True)
    p.add_argument("--train-csv", required=True)
    p.add_argument("--val-csv", default=None,
                   help="Optional LABELLED split for checkpoint selection. The "
                        "official sets/dev.csv has no scores, so by default there "
                        "is no selection signal and the final step is kept -- the "
                        "same thing the baseline does with its fixed 20k steps.")
    p.add_argument("--eval-csv", nargs="*", default=[],
                   help="Additional labelled splits, reported but never selected on.")
    p.add_argument("--target-metric", required=True, choices=["spk_sim", "acc_sim"])
    p.add_argument("--outdir", required=True)

    p.add_argument("--encoder-ckpt", default=None, help="Stage-1 RNC checkpoint.")
    p.add_argument("--freeze-encoder", action="store_true", help="Linear probing.")
    p.add_argument("--head", choices=["mlp", "linear"], default="mlp",
                   help="'mlp' matches the baseline head; 'linear' matches the RNC paper.")
    p.add_argument("--loss", choices=["mse", "l1"], default="mse",
                   help="'mse' matches the baseline; 'l1' matches the RNC paper.")
    p.add_argument("--no-range-clipping", action="store_true")

    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--train-steps", type=int, default=20000)
    p.add_argument("--eval-steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--select-on", default="sys_srcc",
                   choices=["sys_srcc", "sys_lcc", "utt_srcc", "utt_lcc", "neg_utt_mse"])

    p.add_argument("--model-name", default=ECAPA)
    p.add_argument("--unfreeze-ecapa", action="store_true",
                   help="Genuinely fine-tune the 22.15M ECAPA parameters. Off by "
                        "default: the official baseline leaves ECAPA frozen "
                        "(SpeechBrain's freeze_params defaults to True) and trains "
                        "only the 115k projection + head parameters, so this off "
                        "is what reproduces baseline 2.")
    p.add_argument("--ecapa-eval-mode", action="store_true",
                   help="Pin ECAPA to eval() so its BatchNorm running statistics "
                        "stop drifting. The baseline lets them drift.")
    p.add_argument("--max-audio-sec", type=float, default=None)
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
def extract_features(encoder, loader, device):
    """One deterministic pass over a split -> (features, scores, system_ids)."""
    encoder.eval()
    feats, scores, systems = [], [], []
    for batch in loader:
        wav_a, wav_b, len_a, len_b, b_index = to_device(batch, device)
        feats.append(encoder(wav_a, wav_b, len_a, len_b, b_index).float().cpu())
        scores.append(batch["score"])
        systems.extend(batch["system_id"])
    return torch.cat(feats), torch.cat(scores), systems


@torch.no_grad()
def predict_loader(model, loader, device):
    model.eval()
    preds, trues, systems = [], [], []
    for batch in loader:
        wav_a, wav_b, len_a, len_b, b_index = to_device(batch, device)
        preds.append(model(wav_a, wav_b, len_a, len_b, b_index).float().cpu())
        trues.append(batch["score"])
        systems.extend(batch["system_id"])
    model.train()
    return torch.cat(preds).numpy(), torch.cat(trues).numpy(), systems


@torch.no_grad()
def predict_cached(head, feats, device, batch_size=1024):
    head.eval()
    out = []
    for i in range(0, feats.shape[0], batch_size):
        out.append(head(feats[i:i + batch_size].to(device)).float().cpu())
    head.train()
    return torch.cat(out).numpy()


def selection_score(metrics, key):
    if key == "neg_utt_mse":
        return -metrics["utt_mse"]
    value = metrics[key]
    return -float("inf") if np.isnan(value) else value


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[logging.FileHandler(os.path.join(args.outdir, "train_head.log")),
                  logging.StreamHandler()],
    )
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")
    logging.info(f"Args: {vars(args)}")

    model = Model(
        model_name=args.model_name,
        use_projection=True,
        freeze_ecapa=not args.unfreeze_ecapa,
        ecapa_eval_mode=args.ecapa_eval_mode,
        head=args.head,
        range_clipping=not args.no_range_clipping,
    ).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logging.info(
        f"Parameters: {n_train:,} trainable / {n_total:,} total "
        f"({100 * n_train / n_total:.2f}%) | ECAPA "
        f"{'trainable' if args.unfreeze_ecapa else 'FROZEN (baseline behaviour)'}"
    )

    if args.encoder_ckpt:
        ckpt = torch.load(args.encoder_ckpt, map_location="cpu")
        state = ckpt.get("encoder", ckpt)
        model.pair_encoder.load_state_dict(state)
        logging.info(f"Loaded stage-1 encoder from {args.encoder_ckpt} (step {ckpt.get('step')})")
    elif args.freeze_encoder:
        logging.warning("--freeze-encoder without --encoder-ckpt: probing the *pretrained* encoder.")

    # Caching is exact only when the encoder is frozen; crops are made
    # deterministic (centre crop) so a cached feature matches what inference sees.
    cache = args.freeze_encoder
    loader_kwargs = dict(
        num_workers=args.num_workers, max_audio_sec=args.max_audio_sec,
    )
    _, train_loader = build_loader(
        args.data_root, args.train_csv, args.target_metric,
        batch_size=args.batch_size, shuffle=not cache,
        train=not cache, drop_last=not cache, **loader_kwargs,
    )
    def split_name(path):
        return os.path.splitext(os.path.basename(path))[0]

    val_name = split_name(args.val_csv) if args.val_csv else None
    eval_loaders = {}
    for path in ([args.val_csv] if args.val_csv else []) + list(args.eval_csv):
        _, eval_loaders[split_name(path)] = build_loader(
            args.data_root, path, args.target_metric,
            batch_size=args.batch_size, shuffle=False, train=False, **loader_kwargs,
        )
    if val_name is None:
        logging.info("No --val-csv: no checkpoint selection, keeping the final step.")

    history, best_score, best_step = [], -float("inf"), -1
    start = time.time()
    criterion = torch.nn.MSELoss() if args.loss == "mse" else torch.nn.L1Loss()

    if cache:
        # ---- Linear probing on cached frozen features ----
        logging.info("Extracting frozen features...")
        tr_x, tr_y, _ = extract_features(model.pair_encoder, train_loader, device)
        cached = {}
        for name, loader in eval_loaders.items():
            fx, fy, fs = extract_features(model.pair_encoder, loader, device)
            cached[name] = (fx, fy.numpy(), fs)
        logging.info(f"Train features {tuple(tr_x.shape)}; " +
                     "; ".join(f"{k} {tuple(v[0].shape)}" for k, v in cached.items()))

        head = model.head
        optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_steps)
        tr_x_dev, tr_y_dev = tr_x.to(device), tr_y.to(device)
        n = tr_x_dev.shape[0]
        head.train()

        for step in range(1, args.train_steps + 1):
            idx = torch.randint(0, n, (min(args.batch_size, n),), device=device)
            loss = criterion(head(tr_x_dev[idx]), tr_y_dev[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            if step % args.eval_steps == 0 or step == args.train_steps:
                row = {"step": step, "train_loss": float(loss.item())}
                for name, (fx, fy, fs) in cached.items():
                    m = eval_metrics(fy, predict_cached(head, fx, device), fs)
                    row[name] = m
                    logging.info(f"step {step:>6} | " + format_metrics(name, m))
                history.append(row)
                if val_name and val_name in row:
                    score = selection_score(row[val_name], args.select_on)
                    if score > best_score:
                        best_score, best_step = score, step
                        torch.save({"model": model.state_dict(), "step": step,
                                    "args": vars(args), "metrics": row},
                                   os.path.join(args.outdir, "model_best.pt"))
    else:
        # ---- End-to-end fine-tuning (baseline arm) ----
        params = [p for p in model.parameters() if p.requires_grad]
        logging.info(f"Trainable parameters: {sum(p.numel() for p in params):,}")
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
        model.train()
        step = 0
        while step < args.train_steps:
            for batch in train_loader:
                if step >= args.train_steps:
                    break
                wav_a, wav_b, len_a, len_b, b_index = to_device(batch, device)
                preds = model(wav_a, wav_b, len_a, len_b, b_index)
                loss = criterion(preds, batch["score"].to(device))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
                optimizer.step()
                step += 1

                if step % 100 == 0:
                    logging.info(f"step {step:>6}/{args.train_steps} | loss {loss.item():.4f} "
                                 f"| {(time.time() - start) / step:.2f}s/step")
                if step % args.eval_steps == 0 or step == args.train_steps:
                    row = {"step": step, "train_loss": float(loss.item())}
                    for name, loader in eval_loaders.items():
                        p, t, s = predict_loader(model, loader, device)
                        m = eval_metrics(t, p, s)
                        row[name] = m
                        logging.info(f"step {step:>6} | " + format_metrics(name, m))
                    history.append(row)
                    if val_name and val_name in row:
                        score = selection_score(row[val_name], args.select_on)
                        if score > best_score:
                            best_score, best_step = score, step
                            torch.save({"model": model.state_dict(), "step": step,
                                        "args": vars(args), "metrics": row},
                                       os.path.join(args.outdir, "model_best.pt"))

    torch.save({"model": model.state_dict(), "step": args.train_steps, "args": vars(args)},
               os.path.join(args.outdir, "model_last.pt"))
    with open(os.path.join(args.outdir, "head_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    if val_name and best_step > 0:
        logging.info(f"Best {val_name} {args.select_on} = {best_score:.4f} at step {best_step}")
        best_row = next(r for r in history if r["step"] == best_step)
        for name in eval_loaders:
            if name in best_row:
                logging.info("BEST " + format_metrics(name, best_row[name]))
    logging.info(f"Done in {(time.time() - start) / 60:.1f} min. Checkpoints in {args.outdir}")


if __name__ == "__main__":
    main()
