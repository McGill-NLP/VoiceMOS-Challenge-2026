#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
The unified Track 3 model.

Structure is the official baseline's throughout -- two waveforms through one shared
encoder, L2-normalised 256-d projections, the 4-way interaction vector, a head on top --
with three axes made independent:

    --encoder     any backbone in encoders.ENCODER_REGISTRY, composable with '+'
    --head        mlp | moe
    --objective   mse | corn | coral   (+ --lambda-rnc as an auxiliary)

`--encoder ecapa-voxceleb --head mlp --objective mse` is the official Baseline 2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from encoders import build_encoder
from heads import Head
from interactions import Interaction
from objectives import NUM_CLASSES, decode_score


class SpeechEncoder(nn.Module):
    """Backbone + linear projection, returning an L2-normalised embedding."""

    def __init__(
        self,
        encoder_name: str = "ecapa-voxceleb",
        embedding_dim: int = 256,
        use_projection: bool = True,
        cache_dir: str = None,
        encoder_checkpoint: str = None,
    ):
        super().__init__()
        # Loaded on CPU; the caller moves the assembled model to GPU.
        self.backbone = build_encoder(encoder_name, cache_dir=cache_dir,
                                      checkpoint=encoder_checkpoint)
        self.encoder_name = encoder_name

        if use_projection:
            self.projection = nn.Linear(self.backbone.output_dim, embedding_dim)
            self.final_dim = embedding_dim
        else:
            self.projection = nn.Identity()
            self.final_dim = self.backbone.output_dim

    def forward(self, waveform, lengths=None):
        embeddings = self.backbone(waveform, lengths)
        return F.normalize(self.projection(embeddings), p=2, dim=-1)


class UnifiedModel(nn.Module):
    def __init__(
        self,
        encoder_name: str = "ecapa-voxceleb",
        target_metric: str = "spk_sim",
        objective: str = "mse",
        head_type: str = "mlp",
        interaction: str = "baseline",
        bilinear_rank: int = 64,
        embedding_dim: int = 256,
        hidden_dim: int = 64,
        ordinal_dim: int = 128,
        num_classes: int = NUM_CLASSES,
        num_experts: int = 2,
        top_k: int = None,
        range_clipping: bool = True,
        dropout: float = 0.3,
        use_projection: bool = True,
        cache_dir: str = None,
        encoder_checkpoint: str = None,
    ):
        super().__init__()
        self.target_metric = target_metric
        self.objective = objective
        self.num_classes = num_classes

        self.encoder = SpeechEncoder(
            encoder_name, embedding_dim, use_projection,
            cache_dir=cache_dir, encoder_checkpoint=encoder_checkpoint,
        )
        self.interaction = Interaction(
            self.encoder.final_dim, mode=interaction, bilinear_rank=bilinear_rank
        )
        self.interaction_dim = self.interaction.out_dim

        self.head = Head(
            in_dim=self.interaction_dim,
            objective=objective,
            head_type=head_type,
            hidden_dim=hidden_dim,
            ordinal_dim=ordinal_dim,
            num_classes=num_classes,
            num_experts=num_experts,
            top_k=top_k,
            range_clipping=range_clipping,
            dropout=dropout,
        )

        # Backbones arrive with grads enabled; the two-phase schedule drives this.
        self._encoder_trainable = True

    # -- parameter groups ---------------------------------------------------
    def encoder_parameters(self):
        return self.encoder.backbone.parameters()

    def head_parameters(self):
        """Everything that is not the pretrained backbone: projection + head."""
        backbone_ids = {id(p) for p in self.encoder.backbone.parameters()}
        return (p for p in self.parameters() if id(p) not in backbone_ids)

    # -- freezing -----------------------------------------------------------
    @property
    def encoder_trainable(self) -> bool:
        """Whether the backbone is currently unfrozen (traced in TensorBoard)."""
        return self._encoder_trainable

    def set_encoder_trainable(self, trainable: bool):
        """Freeze or unfreeze the pretrained backbone.

        Freezing sets requires_grad=False *and* pins the backbone to eval(). Both are
        needed: ECAPA carries 31 BatchNorm1d layers whose running statistics keep
        drifting on every forward pass in train mode even when the weights cannot move,
        so a backbone frozen by requires_grad alone still changes its own embeddings.
        This is the defect that made the official baseline's "frozen" arm irreproducible;
        see ../../BRANCHES.md section 3.
        """
        self._encoder_trainable = trainable
        for p in self.encoder.backbone.parameters():
            p.requires_grad = trainable
        self.encoder.backbone.train(trainable and self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        # Re-assert the freeze: nn.Module.train() would otherwise put a frozen backbone
        # back into training mode on every call.
        self.encoder.backbone.train(mode and self._encoder_trainable)
        return self

    # -- forward ------------------------------------------------------------
    def forward(self, wav_a, wav_b=None, len_a=None, len_b=None):
        outputs = {}
        emb_a = self.encoder(wav_a, len_a)
        outputs["emb_a"] = emb_a

        if wav_b is None:
            return outputs

        emb_b = self.encoder(wav_b, len_b)
        outputs["emb_b"] = emb_b
        outputs["cos_sim"] = F.cosine_similarity(emb_a, emb_b, dim=-1)

        interaction = self.interaction(emb_a, emb_b)
        # Kept on the output so the Rank-N-Contrast auxiliary can act on the same
        # feature the head consumes, which is what the RNC paper prescribes (no separate
        # projection head for the contrastive term).
        outputs["interaction"] = interaction

        raw, aux_loss = self.head(interaction)
        outputs["raw"] = raw
        outputs["moe_aux_loss"] = aux_loss
        # The continuous 1..5 prediction, whatever the objective.
        outputs[self.target_metric] = decode_score(self.objective, raw, self.num_classes)
        return outputs


def build_from_config(config, cache_dir=None):
    """Rebuild a model from a checkpoint's stored config (used by inference.py)."""
    return UnifiedModel(
        encoder_name=config["encoder"],
        target_metric=config["target_metric"],
        objective=config.get("objective", "mse"),
        head_type=config.get("head", "mlp"),
        # Defaults to "baseline" so checkpoints written before this flag existed still
        # rebuild correctly. That mode is parameter-free, so it adds no state_dict keys.
        interaction=config.get("interaction", "baseline"),
        bilinear_rank=config.get("bilinear_rank", 64),
        embedding_dim=config.get("embedding_dim", 256),
        hidden_dim=config.get("hidden_dim", 64),
        ordinal_dim=config.get("ordinal_dim", 128),
        num_classes=config.get("num_classes", NUM_CLASSES),
        num_experts=config.get("num_experts", 2),
        top_k=config.get("top_k", None),
        range_clipping=config.get("range_clipping", True),
        cache_dir=cache_dir,
    )
