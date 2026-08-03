#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Loss functions from UTMOS (Saeki et al., Interspeech 2022), the VoiceMOS Challenge 2022
winner, ported to the Track 3 pair-similarity task.

Faithful ports of `strong/loss_function.py` in https://github.com/sarulab-speech/UTMOS22,
with the frame-level machinery removed: UTMOS scores every frame and averages at inference,
whereas the Track 3 baseline emits one scalar per pair, so scores here are already [B].

The paper's final objective (Eq. 1) is

    L = beta * L_reg + gamma * L_con

with the shipped configuration using beta=1.0, gamma=0.5, tau=0.25, margin=0.1
(`strong/configs/train/default.yaml`; note the code default for margin is 0.2, the config
overrides it to 0.1).

Why this should matter here: the contrastive term penalises getting the *sign* of a score
difference wrong, so it optimises ranking directly, which is what SRCC measures. The Track 3
baseline trains on plain MSE, which is indifferent to rank as long as the absolute error is
small.
"""

import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    """Pairwise score-difference margin loss over every ordered pair in the batch.

        L_con(i,j) = max(0, |d_ij - d_hat_ij| - margin)

    where d_ij = y_i - y_j is the true score difference and d_hat_ij the predicted one.

    Errors smaller than `margin` are ignored, in the spirit of an SVM margin. The mean is
    taken over the full B x B matrix (which double counts, since both (i,j) and (j,i) are
    present, and includes a zero diagonal) and then halved -- exactly as UTMOS does it.

    Larger batches give quadratically more pairs and a better signal. Note that gradient
    accumulation does NOT help: the loss is computed per micro-batch, so only the real batch
    size counts.
    """

    def __init__(self, margin: float = 0.1):
        super().__init__()
        self.margin = margin

    def forward(self, pred_score: torch.Tensor, gt_score: torch.Tensor) -> torch.Tensor:
        gt_diff = gt_score.unsqueeze(1) - gt_score.unsqueeze(0)
        pred_diff = pred_score.unsqueeze(1) - pred_score.unsqueeze(0)
        loss = torch.clamp(torch.abs(pred_diff - gt_diff) - self.margin, min=0.0)
        return loss.mean().div(2)


class ClippedMSELoss(nn.Module):
    """MSE with a dead zone: squared error is zeroed wherever |y - y_hat| <= tau.

        L_reg(y, y_hat) = 1(|y - y_hat| > tau) * (y - y_hat)^2

    From Tseng et al. (2021) via MBNet's clipped MSE. The point is that a rating of 4 means
    "about a 4", so driving the prediction to exactly 4.0 overfits the label. Following
    UTMOS, the mean is taken over all elements including the zeroed ones, so tau shrinks the
    loss magnitude as well as masking it.
    """

    def __init__(self, tau: float = 0.25):
        super().__init__()
        self.tau = tau
        self.criterion = nn.MSELoss(reduction="none")

    def forward(self, pred_score: torch.Tensor, gt_score: torch.Tensor) -> torch.Tensor:
        loss = self.criterion(pred_score, gt_score)
        threshold = torch.abs(pred_score - gt_score) > self.tau
        return torch.mean(threshold * loss)


class CombineLosses(nn.Module):
    """Weighted sum of losses, all sharing the (pred, gt) signature."""

    def __init__(self, loss_weights: list, loss_instances: list):
        super().__init__()
        self.loss_weights = loss_weights
        self.loss_instances = nn.ModuleList(loss_instances)

    def forward(self, pred_score: torch.Tensor, gt_score: torch.Tensor) -> torch.Tensor:
        loss = torch.zeros((), dtype=torch.float, device=pred_score.device)
        for weight, instance in zip(self.loss_weights, self.loss_instances):
            loss = loss + weight * instance(pred_score, gt_score)
        return loss

    def components(self, pred_score: torch.Tensor, gt_score: torch.Tensor) -> dict:
        """Unweighted value of each term, for logging."""
        return {
            type(inst).__name__: float(inst(pred_score, gt_score).detach())
            for inst in self.loss_instances
        }


# --------------------------------------------------------------------------------------
# Configurations for the ablation
# --------------------------------------------------------------------------------------

LOSS_CHOICES = ("mse", "clipped", "contrastive", "utmos")


def build_loss(name: str, tau: float = 0.25, margin: float = 0.1,
               beta: float = 1.0, gamma: float = 0.5) -> nn.Module:
    """Build one of the four ablation arms.

      mse          plain MSE -- what ../baseline/finetune.py trains on. The control.
      clipped      clipped MSE alone, isolating the effect of the dead zone.
      contrastive  contrastive alone. UTMOS Table 2a "w/o MSE loss" shows this is viable
                   on its own; it fixes no absolute scale, so expect good SRCC/LCC and a
                   poor MSE until the predictions are recalibrated.
      utmos        beta * clipped + gamma * contrastive -- the paper's Eq. 1.
    """
    if name == "mse":
        return nn.MSELoss()
    if name == "clipped":
        return ClippedMSELoss(tau=tau)
    if name == "contrastive":
        return ContrastiveLoss(margin=margin)
    if name == "utmos":
        return CombineLosses(
            loss_weights=[beta, gamma],
            loss_instances=[ClippedMSELoss(tau=tau), ContrastiveLoss(margin=margin)],
        )
    raise ValueError(f"Unknown loss '{name}'. Choose from {LOSS_CHOICES}.")
