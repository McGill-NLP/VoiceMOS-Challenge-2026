#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prediction heads for the unified Track 3 model.

A head is a *trunk* followed by a *task layer*:

    interaction vector ──> trunk (MLP or MoE) ──> task layer ──> raw output
        4 * final_dim            trunk_out_dim        objective-specific

The split is what lets `--head` and `--objective` vary independently. The trunk is the
part borrowed from ../../dev.yj (single MLP vs mixture of experts); the task layer is the
part borrowed from ../corn-and-coral (scalar + range clipping, CORN logits, CORAL layer).

Shapes by objective, with K = num_classes = 5:

    mse    trunk_out_dim = 1        task layer = Tanh*2+3   ->  [B]
    corn   trunk_out_dim = 128      task layer = Linear     ->  [B, K-1]  conditional logits
    coral  trunk_out_dim = 128      task layer = CoralLayer ->  [B, K-1]  marginal logits

For `--objective mse --head mlp` this reduces exactly to the official baseline's
Linear(4d, 64) -> ReLU -> Dropout(0.3) -> Linear(64, 1) -> Tanh*2+3.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from coral_pytorch.layers import CoralLayer


class MLPTrunk(nn.Module):
    """The baseline's projection MLP, with a configurable output width."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.3,
                 activation=nn.ReLU):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        # (output, aux_loss) so the two trunks are interchangeable at the call site.
        return self.net(x), x.new_zeros(())


class MoETrunk(nn.Module):
    """Mixture-of-experts trunk, ported from dev.yj/empirical's `MoEProjection`.

    Several small expert MLPs mixed by a learned gate instead of one global MLP. Dense
    (softmax) gating by default: on a 2,800-pair training set top-k routing risks expert
    collapse, where the gate always picks the same one or two experts and the rest never
    receive gradient. A load-balancing auxiliary loss discourages that further.

    Generalised from the original in one way only: experts emit `out_dim` values rather
    than a scalar, so the same trunk feeds the CORN and CORAL task layers. At
    out_dim = 1 it is numerically the original.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_experts: int = 2,
                 top_k: int = None, dropout: float = 0.3, activation=nn.ReLU):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.out_dim = out_dim

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                activation(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim),
            )
            for _ in range(num_experts)
        ])
        self.gate = nn.Linear(in_dim, num_experts)

    def _load_balance_loss(self, gate_weights):
        """Encourages uniform average expert usage across the batch."""
        importance = gate_weights.mean(dim=0)                  # [E]
        target = 1.0 / self.num_experts
        return ((importance - target) ** 2).sum() * self.num_experts

    def forward(self, x):
        gate_logits = self.gate(x)                             # [B, E]
        expert_outs = torch.stack([e(x) for e in self.experts], dim=1)   # [B, E, out]

        full_gate_weights = F.softmax(gate_logits, dim=-1)
        aux_loss = self._load_balance_loss(full_gate_weights)

        if self.top_k is None or self.top_k >= self.num_experts:
            weights = full_gate_weights                        # dense / soft gating
        else:
            topk_vals, topk_idx = gate_logits.topk(self.top_k, dim=-1)
            weights = torch.zeros_like(full_gate_weights)
            weights.scatter_(1, topk_idx, F.softmax(topk_vals, dim=-1))

        combined = (weights.unsqueeze(-1) * expert_outs).sum(dim=1)      # [B, out]
        return combined, aux_loss


class Head(nn.Module):
    """Trunk + objective-specific task layer.

    `forward` returns (raw_output, aux_loss). The raw output is whatever the objective's
    loss consumes; `objectives.decode_score` turns it into a continuous 1..K rating.
    """

    def __init__(
        self,
        in_dim: int,
        objective: str,
        head_type: str = "mlp",
        hidden_dim: int = 64,
        ordinal_dim: int = 128,
        num_classes: int = 5,
        num_experts: int = 2,
        top_k: int = None,
        range_clipping: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.objective = objective
        self.range_clipping = range_clipping and objective == "mse"

        # Regression collapses straight to a scalar, exactly as the baseline does. The
        # ordinal objectives need a feature vector for their task layer to read.
        trunk_out_dim = 1 if objective == "mse" else ordinal_dim

        if head_type == "mlp":
            self.trunk = MLPTrunk(in_dim, hidden_dim, trunk_out_dim, dropout=dropout)
        elif head_type == "moe":
            self.trunk = MoETrunk(in_dim, hidden_dim, trunk_out_dim,
                                  num_experts=num_experts, top_k=top_k, dropout=dropout)
        else:
            raise ValueError(f"Unknown head type: {head_type}")

        if objective == "mse":
            self.task = nn.Identity()
        elif objective == "corn":
            self.task = nn.Linear(ordinal_dim, num_classes - 1)
        elif objective == "coral":
            # Rank consistency is a property of this layer's architecture (one shared
            # weight vector, K-1 ordered biases), so it must sit AFTER the mixture --
            # mixing per-expert CORAL logits would not preserve it.
            self.task = CoralLayer(size_in=ordinal_dim, num_classes=num_classes)
        else:
            raise ValueError(f"Unknown objective: {objective}")

    def forward(self, x):
        trunk_out, aux_loss = self.trunk(x)
        out = self.task(trunk_out)

        if self.range_clipping:
            # Scale Tanh [-1, 1] to MOS [1, 5]
            out = torch.tanh(out) * 2.0 + 3.0

        if self.objective == "mse":
            out = out.squeeze(-1)
        return out, aux_loss
