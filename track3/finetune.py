#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import csv
import json
import logging
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from model import SpeechEncoder, ResidualAdapter  # reuse the baseline's encoder + new adapter
# Fixed MoE hyperparameters. NOT exposed as CLI flags -- see module docstring
# for why these must stay in sync with what inference.py implicitly expects.
MOE_NUM_EXPERTS = 2
MOE_TOP_K = None  # None = dense (soft) gating over all experts
DEFAULT_MODEL_NAME_SPK = "speechbrain/spkrec-ecapa-voxceleb"
DEFAULT_MODEL_NAME_ACC = "Jzuluaga/accent-id-commonaccent_ecapa"
# ----------------------------------------------------------------------------
# Mixture-of-Experts projection head
# ----------------------------------------------------------------------------
class MoEProjection(nn.Module):
    """
    Drop-in replacement for a single-MLP projection head: mixes several small
    expert MLPs via a learned gate instead of using one global MLP. Dense
    (softmax) gating by default -- more stable than sparse top-k routing on
    a small train set, where top-k risks expert collapse (gate always picks
    the same 1-2 experts, others never get gradient). A load-balancing
    auxiliary loss further discourages collapse.
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_experts: int = 4,
        top_k: int = None,
        activation=nn.ReLU,
        range_clipping: bool = True,
        expert_dropout: float = 0.3,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.range_clipping = range_clipping
        if range_clipping:
            self.out_act = nn.Tanh()
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    activation(),
                    nn.Dropout(expert_dropout),
                    nn.Linear(hidden_dim, 1),
                )
                for _ in range(num_experts)
            ]
        )
        self.gate = nn.Linear(in_dim, num_experts)
    def _load_balance_loss(self, gate_weights):
        # Encourages uniform average usage across experts over the batch.
        importance = gate_weights.mean(dim=0)  # [E]
        target = 1.0 / self.num_experts
        return ((importance - target) ** 2).sum() * self.num_experts
    def forward(self, x):
        gate_logits = self.gate(x)  # [B, E]
        expert_outs = torch.cat([e(x) for e in self.experts], dim=-1)  # [B, E]
        if self.top_k is None or self.top_k >= self.num_experts:
            gate_weights = F.softmax(gate_logits, dim=-1)
            combined = (gate_weights * expert_outs).sum(dim=-1)
            aux_loss = self._load_balance_loss(gate_weights)
        else:
            topk_vals, topk_idx = gate_logits.topk(self.top_k, dim=-1)
            topk_weights = F.softmax(topk_vals, dim=-1)
            gathered = torch.gather(expert_outs, 1, topk_idx)
            combined = (topk_weights * gathered).sum(dim=-1)
            full_gate_weights = F.softmax(gate_logits, dim=-1)
            aux_loss = self._load_balance_loss(full_gate_weights)
        if self.range_clipping:
            combined = self.out_act(combined) * 2.0 + 3
        return combined, aux_loss
# ----------------------------------------------------------------------------
# Model: dual dedicated backbones + annealed residual adapter + MoE mean head
# ----------------------------------------------------------------------------
class ModelV2(nn.Module):
    def __init__(
        self,
        model_name_spk=DEFAULT_MODEL_NAME_SPK,
        model_name_acc=DEFAULT_MODEL_NAME_ACC,
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
        adapter_hidden_dim=128,
        cache_dir=None,
    ):
        super().__init__()
        self.target_metrics = list(target_metrics)
        self.use_listener_bias = use_listener_bias
        self.listener_dropout = listener_dropout
        model_name_by_metric = {"spk_sim": model_name_spk, "acc_sim": model_name_acc}
        # One dedicated encoder + one residual adapter per active metric.
        self.encoders = nn.ModuleDict()
        self.adapters = nn.ModuleDict()
        for m in self.target_metrics:
            enc = SpeechEncoder(
                model_name_by_metric[m],
                embedding_dim,
                use_projection,
                freeze_ssl,
                cache_dir=cache_dir,
            )
            self.encoders[m] = enc
            self.adapters[m] = ResidualAdapter(enc.final_dim, hidden_dim=adapter_hidden_dim)
        self.mean_heads = nn.ModuleDict()
        if use_listener_bias:
            assert num_listeners > 0, "num_listeners must be > 0 when use_listener_bias=True"
            self.listener_emb = nn.Embedding(num_listeners + 1, listener_emb_dim, padding_idx=0)
            self.bias_heads = nn.ModuleDict()
        for m in self.target_metrics:
            interaction_dim = self.encoders[m].final_dim * 4
            self.mean_heads[m] = MoEProjection(
                interaction_dim, mlp_dnn_dim,
                num_experts=MOE_NUM_EXPERTS, top_k=MOE_TOP_K,
                activation=nn.ReLU, range_clipping=mlp_range_clipping,
            )
            if use_listener_bias:
                self.bias_heads[m] = nn.Sequential(
                    nn.Linear(interaction_dim + listener_emb_dim, mlp_dnn_dim),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(mlp_dnn_dim, 1),
                )
    def set_backbone_trainable(self, trainable: bool):
        """Applies to ALL active encoders' backbones (both spk_sim's and
        acc_sim's, if both are present) -- one shared freeze/unfreeze
        schedule across both, controlled by --freeze-steps."""
        for enc in self.encoders.values():
            for p in enc.ssl_model.parameters():
                p.requires_grad = trainable
    def forward(self, wav_a, wav_b, len_a=None, len_b=None, listener_idx=None, adapter_alpha=0.0):
        outputs = {}
        moe_aux_loss = 0.0
        for m in self.target_metrics:
            enc = self.encoders[m]
            emb_a_raw = enc(wav_a, len_a)
            emb_b_raw = enc(wav_b, len_b)
            emb_a = self.adapters[m](emb_a_raw, adapter_alpha)
            emb_b = self.adapters[m](emb_b_raw, adapter_alpha)
            outputs[f"{m}_cos_sim"] = F.cosine_similarity(emb_a, emb_b, dim=-1)
            interaction = torch.cat(
                [emb_a, emb_b, torch.abs(emb_a - emb_b), emb_a * emb_b], dim=-1
            )
            mean_pred, aux_loss = self.mean_heads[m](interaction)
            moe_aux_loss = moe_aux_loss + aux_loss
            outputs[f"{m}_mean"] = mean_pred
            if self.use_listener_bias and listener_idx is not None:
                l_idx = listener_idx
                if self.training and self.listener_dropout > 0:
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
        outputs["moe_aux_loss"] = moe_aux_loss
        return outputs
# ----------------------------------------------------------------------------
# Wean schedule: linear ramp of adapter_alpha from 0 -> max_alpha
# ----------------------------------------------------------------------------
def compute_wean_alpha(step: int, start: int, end: int, max_alpha: float) -> float:
    if step <= start:
        return 0.0
    if end <= start or step >= end:
        return max_alpha
    return max_alpha * (step - start) / (end - start)
# ----------------------------------------------------------------------------
# Dataset: keeps every listener-wise row + precomputes per-pair mean target
# ----------------------------------------------------------------------------
class ListenerAwareDataset(Dataset):
    def __init__(self, data_root, data_rows, target_metrics, listener2idx=None, augment=False):
        self.data_root = data_root
        self.target_metrics = target_metrics
        self.augment = augment
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
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
def _augment_waveform(wav, p=0.5):
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
def pairwise_ranking_loss(preds, targets, margin_scale=1.0):
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
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--train-steps", type=int, default=20000)
    parser.add_argument("--save-steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr-mult", type=float, default=0.1,
                         help="Backbone LR = lr * this.")
    parser.add_argument("--freeze-steps", type=int, default=5000,
                         help="Steps to keep both encoders' backbones frozen before unfreezing. "
                              "0 disables freezing. Default 5000 (found to work best).")
    parser.add_argument("--use-listener-bias", action="store_true",
                         help="Enable MBNet-style listener bias correction.")
    parser.add_argument("--listener-dropout", type=float, default=0.5)
    parser.add_argument("--lambda-rank", type=float, default=0.0,
                         help="Weight of the pairwise ranking/contrastive loss. 0 disables.")
    parser.add_argument("--lambda-mean", type=float, default=1.0,
                         help="Weight of the mean-head MSE loss when --use-listener-bias is set.")
    parser.add_argument("--lambda-moe-aux", type=float, default=0.08,
                         help="Weight of the MoE load-balancing auxiliary loss.")
    parser.add_argument("--augment", action="store_true", help="Enable waveform augmentation.")
    parser.add_argument("--model-name-spk", type=str, default=DEFAULT_MODEL_NAME_SPK,
                         help="Pretrained encoder for spk_sim.")
    parser.add_argument("--model-name-acc", type=str, default=DEFAULT_MODEL_NAME_ACC,
                         help="Pretrained encoder for acc_sim.")
    parser.add_argument("--adapter-hidden-dim", type=int, default=128,
                         help="Hidden dim of the residual adapter MLP.")
    parser.add_argument("--wean-start-step", type=int, default=None,
                         help="Step at which adapter_alpha starts ramping up from 0. "
                              "Defaults to --freeze-steps (start weaning right as backbones unfreeze).")
    parser.add_argument("--wean-end-step", type=int, default=None,
                         help="Step at which adapter_alpha reaches --wean-max-alpha. "
                              "Defaults to --train-steps (ramp across the rest of training).")
    parser.add_argument("--wean-max-alpha", type=float, default=1.0,
                         help="Final adapter mixing coefficient reached at --wean-end-step.")
    parser.add_argument("--cache-dir", type=str, default="/home/mila/j/jeony/scratch/voicemos_v2/track3/cache",
                         help="Directory to download/cache pretrained checkpoints into "
                              "(e.g. spkrec-ecapa-voxceleb, accent-id-commonaccent_ecapa). "
                              "Defaults to SpeechBrain's own default location "
                              "(./pretrained_models/<source>) if not set.")
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args()
    log_level = logging.DEBUG if args.verbose > 1 else logging.INFO if args.verbose > 0 else logging.WARN
    logging.basicConfig(level=log_level, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    os.makedirs(args.outdir, exist_ok=True)
    if args.cache_dir is not None:
        os.makedirs(args.cache_dir, exist_ok=True)
        logging.info(f"Pretrained checkpoints will be cached under: {args.cache_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    set_seed(args.seed)
    logging.info(f"Random seed set to {args.seed}")
    wean_start = args.wean_start_step if args.wean_start_step is not None else args.freeze_steps
    wean_end = args.wean_end_step if args.wean_end_step is not None else args.train_steps
    logging.info(
        f"Wean schedule: alpha 0.0 -> {args.wean_max_alpha} over steps [{wean_start}, {wean_end}]"
    )
    target_metrics = ["spk_sim", "acc_sim"] if args.target_metric == "both" else [args.target_metric]
    train_csv = os.path.join(args.data_root, "sets", "train.csv")
    logging.info(f"Loading {train_csv} ...")
    with open(train_csv, encoding="utf-8") as f:
        train_rows = list(csv.DictReader(f))
    logging.info(f"Loaded {len(train_rows)} raw training rows.")
    dataset = ListenerAwareDataset(args.data_root, train_rows, target_metrics, augment=args.augment)
    collater = CollaterV2(target_metrics)
    g = torch.Generator()
    g.manual_seed(args.seed)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collater, num_workers=4, drop_last=True,
        generator=g,
    )
    num_listeners = len(dataset.listener2idx)
    with open(os.path.join(args.outdir, "listener_vocab.json"), "w") as f:
        json.dump(dataset.listener2idx, f)
    # Record everything inference.py needs but can't recover from the
    # checkpoint's weight shapes alone (HF model names per encoder, the
    # wean schedule's final alpha to use at inference time, and where the
    # pretrained checkpoints were cached).
    run_config = {
        "target_metrics": target_metrics,
        "use_listener_bias": args.use_listener_bias,
        "moe_num_experts": MOE_NUM_EXPERTS,
        "moe_top_k": MOE_TOP_K,
        "model_name_spk": args.model_name_spk,
        "model_name_acc": args.model_name_acc,
        "adapter_hidden_dim": args.adapter_hidden_dim,
        "wean_max_alpha": args.wean_max_alpha,
        "cache_dir": args.cache_dir,
    }
    config_path = os.path.join(args.outdir, "config.json")
    with open(config_path, "w") as f:
        json.dump(run_config, f, indent=2)
    logging.info(f"Saved run config to {config_path}: {run_config}")
    model = ModelV2(
        model_name_spk=args.model_name_spk,
        model_name_acc=args.model_name_acc,
        embedding_dim=256,
        use_projection=True,
        freeze_ssl=(args.freeze_steps > 0),
        target_metrics=target_metrics,
        num_listeners=num_listeners,
        use_listener_bias=args.use_listener_bias,
        listener_dropout=args.listener_dropout,
        adapter_hidden_dim=args.adapter_hidden_dim,
        cache_dir=args.cache_dir,
    ).to(device)
    backbone_params = []
    for enc in model.encoders.values():
        backbone_params.extend(list(enc.ssl_model.parameters()))
    backbone_param_ids = {id(p) for p in backbone_params}
    head_params = [p for p in model.parameters() if id(p) not in backbone_param_ids]
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
        logging.info(f"Both encoders' backbones frozen for the first {args.freeze_steps} steps.")
    global_step, forward_steps, running_loss = 0, 0, 0.0
    optimizer.zero_grad()
    pbar = tqdm(total=args.train_steps, desc="Training")
    while global_step < args.train_steps:
        for batch in loader:
            if global_step >= args.train_steps:
                break
            if args.freeze_steps > 0 and global_step == args.freeze_steps:
                model.set_backbone_trainable(True)
                logging.info(f"Step {global_step}: both encoders' backbones unfrozen.")
            adapter_alpha = compute_wean_alpha(global_step, wean_start, wean_end, args.wean_max_alpha)
            wav_a = batch["wav_a"].to(device)
            len_a = batch["wav_a_lengths"].to(device)
            wav_b = batch["wav_b"].to(device)
            len_b = batch["wav_b_lengths"].to(device)
            listener_idx = batch["listener_idx"].to(device)
            outputs = model(wav_a, wav_b, len_a, len_b, listener_idx=listener_idx, adapter_alpha=adapter_alpha)
            loss = 0.0
            for m in target_metrics:
                raw_target = batch[m].to(device)
                mse_val = mse(outputs[m], raw_target)
                loss = loss + mse_val
                if args.use_listener_bias:
                    mean_target = batch[f"{m}_mean"].to(device)
                    loss = loss + args.lambda_mean * mse(outputs[f"{m}_mean"], mean_target)
                if args.lambda_rank > 0:
                    rank_val = pairwise_ranking_loss(outputs[m], raw_target)
                    loss = loss + args.lambda_rank * rank_val
                    if global_step % 500 == 0:
                        logging.info(
                            f"[{m}] mse={mse_val.item():.4f} rank={rank_val.item():.4f} "
                            f"adapter_alpha={adapter_alpha:.4f}"
                        )
                elif global_step % 500 == 0:
                    logging.info(f"[{m}] mse={mse_val.item():.4f} adapter_alpha={adapter_alpha:.4f}")
            loss = loss + args.lambda_moe_aux * outputs["moe_aux_loss"]
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
                pbar.set_postfix({"loss": f"{running_loss / args.accumulate_steps:.4f}", "alpha": f"{adapter_alpha:.3f}"})
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
