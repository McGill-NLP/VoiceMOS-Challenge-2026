#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import logging
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model import SpeechEncoder, Projection  # reuse the baseline's encoder/head


# ----------------------------------------------------------------------------
# Method 1: listener-bias-aware model
# ----------------------------------------------------------------------------
class ModelV2(nn.Module):
    def __init__(
        self,
        model_name="speechbrain/spkrec-ecapa-voxceleb",
        embedding_dim=256,
        use_projection=True,
        freeze_ssl=False,
        target_metrics=("spk_sim",),
        mlp_dnn_dim=64,
        mlp_range_clipping=True,
        num_listeners=0,
        use_listener_bias=False,
        listener_emb_dim=16,
        listener_dropout=0.5,
    ):
        super().__init__()
        self.encoder = SpeechEncoder(model_name, embedding_dim, use_projection, freeze_ssl)
        interaction_dim = self.encoder.final_dim * 4
        self.target_metrics = list(target_metrics)
        self.use_listener_bias = use_listener_bias
        self.listener_dropout = listener_dropout

        # listener-independent "mean quality" head (this is what we submit)
        self.mean_heads = nn.ModuleDict(
            {
                m: Projection(interaction_dim, mlp_dnn_dim, activation=nn.ReLU,
                               range_clipping=mlp_range_clipping)
                for m in self.target_metrics
            }
        )

        if use_listener_bias:
            assert num_listeners > 0, "num_listeners must be > 0 when use_listener_bias=True"
            # index 0 is reserved for "unknown listener" -> used at inference time
            self.listener_emb = nn.Embedding(num_listeners + 1, listener_emb_dim, padding_idx=0)
            self.bias_heads = nn.ModuleDict(
                {
                    m: nn.Sequential(
                        nn.Linear(interaction_dim + listener_emb_dim, mlp_dnn_dim),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(mlp_dnn_dim, 1),
                    )
                    for m in self.target_metrics
                }
            )

    def set_backbone_trainable(self, trainable: bool):
        for p in self.encoder.ssl_model.parameters():
            p.requires_grad = trainable

    def forward(self, wav_a, wav_b, len_a=None, len_b=None, listener_idx=None):
        emb_a = self.encoder(wav_a, len_a)
        emb_b = self.encoder(wav_b, len_b)
        interaction = torch.cat(
            [emb_a, emb_b, torch.abs(emb_a - emb_b), emb_a * emb_b], dim=-1
        )

        outputs = {"cos_sim": F.cosine_similarity(emb_a, emb_b, dim=-1)}
        for m in self.target_metrics:
            mean_pred = self.mean_heads[m](interaction).squeeze(-1)
            outputs[f"{m}_mean"] = mean_pred

            if self.use_listener_bias and listener_idx is not None:
                l_idx = listener_idx
                if self.training and self.listener_dropout > 0:
                    # Randomly force "unknown listener" during training so the
                    # mean head is trained to carry the full signal in exactly
                    # the condition it will face at dev/test inference time.
                    drop_mask = torch.rand_like(l_idx.float()) < self.listener_dropout
                    l_idx = l_idx.clone()
                    l_idx[drop_mask] = 0
                l_emb = self.listener_emb(l_idx)
                bias_in = torch.cat([interaction, l_emb], dim=-1)
                bias_pred = self.bias_heads[m](bias_in).squeeze(-1)
                outputs[f"{m}_bias"] = bias_pred
                outputs[m] = mean_pred + bias_pred
            else:
                outputs[m] = mean_pred
        return outputs


# ----------------------------------------------------------------------------
# Dataset: keeps every listener-wise row + precomputes per-pair mean target
# ----------------------------------------------------------------------------
class ListenerAwareDataset(Dataset):
    def __init__(self, data_root, data_rows, target_metrics, listener2idx=None, augment=False):
        self.data_root = data_root
        self.target_metrics = target_metrics
        self.augment = augment

        # Pass 1: per-pair per-metric mean (used to supervise the mean head)
        pair_scores = {}
        for row in data_rows:
            key = (row["wav_a_path"], row["wav_b_path"])
            pair_scores.setdefault(key, {m: [] for m in target_metrics})
            for m in target_metrics:
                pair_scores[key][m].append(float(row[m]))
        pair_mean = {
            key: {m: sum(v) / len(v) for m, v in metrics.items()}
            for key, metrics in pair_scores.items()
        }

        if listener2idx is None:
            listeners = sorted({r["listener_id"] for r in data_rows if "listener_id" in r})
            listener2idx = {l: i + 1 for i, l in enumerate(listeners)}  # 0 = unknown
        self.listener2idx = listener2idx

        self.rows = []
        for row in data_rows:
            key = (row["wav_a_path"], row["wav_b_path"])
            item = {
                "wav_a_path": row["wav_a_path"],
                "wav_b_path": row["wav_b_path"],
                "listener_idx": self.listener2idx.get(row.get("listener_id"), 0),
            }
            for m in target_metrics:
                item[m] = float(row[m])
                item[f"{m}_mean"] = pair_mean[key][m]
            self.rows.append(item)

        logging.info(
            f"ListenerAwareDataset: {len(data_rows)} raw rows -> {len(self.rows)} "
            f"training samples, {len(self.listener2idx)} unique listeners."
        )

    def __len__(self):
        return len(self.rows)

    def _load(self, path):
        wav, sr = torchaudio.load(os.path.join(self.data_root, path))
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        wav = wav.squeeze(0)
        if self.augment:
            wav = _augment_waveform(wav)
        return wav

    def __getitem__(self, idx):
        row = self.rows[idx]
        item = {
            "wav_a": self._load(row["wav_a_path"]),
            "wav_b": self._load(row["wav_b_path"]),
            "listener_idx": row["listener_idx"],
        }
        for m in self.target_metrics:
            item[m] = row[m]
            item[f"{m}_mean"] = row[f"{m}_mean"]
        return item


def _augment_waveform(wav, p=0.5):
    """Method 5: cheap, dependency-free waveform augmentation."""
    if random.random() > p:
        return wav
    choice = random.choice(["noise", "gain", "speed"])
    if choice == "noise":
        snr_db = random.uniform(15, 30)
        sig_power = wav.pow(2).mean().clamp_min(1e-8)
        noise = torch.randn_like(wav)
        noise_power = noise.pow(2).mean().clamp_min(1e-8)
        factor = (sig_power / (noise_power * 10 ** (snr_db / 10))).sqrt()
        wav = wav + noise * factor
    elif choice == "gain":
        gain_db = random.uniform(-6, 6)
        wav = wav * (10 ** (gain_db / 20))
    elif choice == "speed":
        rate = random.uniform(0.95, 1.05)
        wav = torchaudio.functional.resample(
            wav.unsqueeze(0), int(16000 * rate), 16000
        ).squeeze(0)
    return wav.clamp(-1.0, 1.0)


class CollaterV2:
    def __init__(self, target_metrics, padding_mode="repetitive"):
        self.target_metrics = target_metrics
        self.padding_mode = padding_mode

    def _pad(self, feats):
        lengths = torch.tensor([f.shape[0] for f in feats], dtype=torch.long)
        if self.padding_mode == "zero_padding":
            return pad_sequence(feats, batch_first=True), lengths
        max_len = lengths.max().item()
        out = []
        for f in feats:
            n = f.shape[0]
            if n == 0:
                out.append(torch.zeros(max_len, dtype=f.dtype))
                continue
            reps = max_len // n
            rem = max_len - reps * n
            chunk = torch.cat([f] * reps + ([f[:rem]] if rem else []), dim=0)
            out.append(chunk)
        return torch.stack(out), lengths

    def __call__(self, batch):
        batch = sorted(batch, key=lambda x: -x["wav_a"].shape[0])
        out = {}
        out["wav_a"], out["wav_a_lengths"] = self._pad([b["wav_a"] for b in batch])
        out["wav_b"], out["wav_b_lengths"] = self._pad([b["wav_b"] for b in batch])
        out["listener_idx"] = torch.tensor([b["listener_idx"] for b in batch], dtype=torch.long)
        for m in self.target_metrics:
            out[m] = torch.tensor([b[m] for b in batch], dtype=torch.float32)
            out[f"{m}_mean"] = torch.tensor([b[f"{m}_mean"] for b in batch], dtype=torch.float32)
        return out


# ----------------------------------------------------------------------------
# Method 2: pairwise ranking / contrastive loss
# ----------------------------------------------------------------------------
def pairwise_ranking_loss(preds, targets, margin_scale=1.0):
    """
    For every pair (i, j) in the batch, penalize predictions whose signed
    difference is smaller than the true signed difference (scaled by
    margin_scale). This directly rewards correct relative ordering AND
    correct relative scale -- exactly what LCC/SRCC measure -- unlike plain
    MSE, which only cares about absolute per-sample error.
    """
    diff_pred = preds.unsqueeze(1) - preds.unsqueeze(0)
    diff_true = targets.unsqueeze(1) - targets.unsqueeze(0)
    margin = diff_true.abs() * margin_scale
    sign = torch.sign(diff_true)
    loss = F.relu(margin - sign * diff_pred)
    n = preds.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=preds.device)
    denom = mask.sum().clamp_min(1)
    return loss[mask].sum() / denom


def main():
    parser = argparse.ArgumentParser(description="Enhanced fine-tuning for VoiceMOS 2026 Track 3.")
    parser.add_argument("--data-root", required=True, type=str)
    parser.add_argument("--target-metric", required=True, choices=["spk_sim", "acc_sim", "both"])
    parser.add_argument("--outdir", required=True, type=str)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--accumulate-steps", type=int, default=1)
    parser.add_argument("--train-steps", type=int, default=20000)
    parser.add_argument("--save-steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr-mult", type=float, default=0.1,
                         help="Backbone LR = lr * this.")
    parser.add_argument("--freeze-steps", type=int, default=2000,
                         help="Steps to keep ECAPA frozen before unfreezing. 0 disables freezing.")
    parser.add_argument("--use-listener-bias", action="store_true",
                         help="Enable MBNet-style listener bias correction.")
    parser.add_argument("--listener-dropout", type=float, default=0.5)
    parser.add_argument("--lambda-rank", type=float, default=0.0,
                         help="Weight of the pairwise ranking/contrastive loss. 0 disables.")
    parser.add_argument("--lambda-mean", type=float, default=1.0,
                         help="Weight of the mean-head MSE loss when --use-listener-bias is set.")
    parser.add_argument("--augment", action="store_true", help="Enable waveform augmentation.")
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose > 1 else logging.INFO if args.verbose > 0 else logging.WARN
    logging.basicConfig(level=log_level, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    target_metrics = ["spk_sim", "acc_sim"] if args.target_metric == "both" else [args.target_metric]

    train_csv = os.path.join(args.data_root, "sets", "train.csv")
    logging.info(f"Loading {train_csv} ...")
    with open(train_csv, encoding="utf-8") as f:
        train_rows = list(csv.DictReader(f))
    logging.info(f"Loaded {len(train_rows)} raw training rows.")

    dataset = ListenerAwareDataset(args.data_root, train_rows, target_metrics, augment=args.augment)
    collater = CollaterV2(target_metrics)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collater, num_workers=4, drop_last=True,
    )

    num_listeners = len(dataset.listener2idx)
    with open(os.path.join(args.outdir, "listener_vocab.json"), "w") as f:
        json.dump(dataset.listener2idx, f)

    model = ModelV2(
        model_name="speechbrain/spkrec-ecapa-voxceleb",
        embedding_dim=256,
        use_projection=True,
        freeze_ssl=(args.freeze_steps > 0),
        target_metrics=target_metrics,
        num_listeners=num_listeners,
        use_listener_bias=args.use_listener_bias,
        listener_dropout=args.listener_dropout,
    ).to(device)

    # Method 4: discriminative learning rates (backbone vs. heads)
    backbone_params = list(model.encoder.ssl_model.parameters())
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.ssl_model.")]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr * args.backbone_lr_mult},
            {"params": head_params, "lr": args.lr},
        ]
    )
    mse = nn.MSELoss()

    model.train()
    if args.freeze_steps > 0:
        model.set_backbone_trainable(False)
        logging.info(f"Backbone frozen for the first {args.freeze_steps} steps.")

    global_step, forward_steps, running_loss = 0, 0, 0.0
    optimizer.zero_grad()
    pbar = tqdm(total=args.train_steps, desc="Training")

    while global_step < args.train_steps:
        for batch in loader:
            if global_step >= args.train_steps:
                break

            if args.freeze_steps > 0 and global_step == args.freeze_steps:
                model.set_backbone_trainable(True)
                logging.info(f"Step {global_step}: backbone unfrozen.")

            wav_a = batch["wav_a"].to(device)
            len_a = batch["wav_a_lengths"].to(device)
            wav_b = batch["wav_b"].to(device)
            len_b = batch["wav_b_lengths"].to(device)
            listener_idx = batch["listener_idx"].to(device)

            outputs = model(wav_a, wav_b, len_a, len_b, listener_idx=listener_idx)

            loss = 0.0
            for m in target_metrics:
                raw_target = batch[m].to(device)
                loss = loss + mse(outputs[m], raw_target)

                if args.use_listener_bias:
                    mean_target = batch[f"{m}_mean"].to(device)
                    loss = loss + args.lambda_mean * mse(outputs[f"{m}_mean"], mean_target)

                if args.lambda_rank > 0:
                    loss = loss + args.lambda_rank * pairwise_ranking_loss(outputs[m], raw_target)

            scaled_loss = loss / args.accumulate_steps
            scaled_loss.backward()
            running_loss += float(loss.item())
            forward_steps += 1

            if forward_steps % args.accumulate_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

                global_step += 1
                pbar.update(1)
                pbar.set_postfix({"loss": f"{running_loss / args.accumulate_steps:.4f}"})
                running_loss = 0.0

                if global_step % args.save_steps == 0:
                    path = os.path.join(args.outdir, f"model_v2_step{global_step}.pt")
                    torch.save(model.state_dict(), path)
                    logging.info(f"Saved checkpoint: {path}")
    pbar.close()

    final_path = os.path.join(args.outdir, "model_v2_final.pt")
    torch.save(model.state_dict(), final_path)
    logging.info(f"Training complete. Final model saved to {final_path}")


if __name__ == "__main__":
    main()
