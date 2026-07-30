#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Speech similarity model with a pluggable encoder.

Structurally identical to ../baseline/model.py — the same projection head, the same
4-way interaction vector, the same range clipping — so that swapping the encoder is
the only variable when comparing against the baseline numbers. The difference is that
the backbone comes from `encoders.build_encoder` instead of being hard-coded to
SpeechBrain's ECAPA.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from encoders import build_encoder


class Projection(nn.Module):
    """
    Simplified Projection module mapping the interaction vector to a final MOS scalar.
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        activation=nn.ReLU,
        range_clipping: bool = False,
    ):
        super(Projection, self).__init__()
        self.range_clipping = range_clipping

        if self.range_clipping:
            self.proj = nn.Tanh()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            activation(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        output = self.net(x)

        if self.range_clipping:
            # Scale Tanh [-1, 1] to MOS [1, 5]
            return self.proj(output) * 2.0 + 3
        else:
            return output


class SpeechEncoder(torch.nn.Module):
    """Core feature extractor: Outputs a normalized 1D embedding."""
    def __init__(
        self,
        encoder_name: str = "ecapa-voxceleb",
        embedding_dim: int = 256,
        use_projection: bool = True,
        freeze_encoder: bool = False,
        cache_dir: str = None,
        encoder_checkpoint: str = None,
    ):
        super().__init__()

        # Loaded on CPU; the caller moves the assembled model to GPU.
        self.backbone = build_encoder(
            encoder_name, cache_dir=cache_dir, checkpoint=encoder_checkpoint
        )
        self.encoder_name = encoder_name

        if freeze_encoder:
            for param in self.backbone.parameters():
                param.requires_grad = False

        output_dim = self.backbone.output_dim

        # In zero-shot, we do not want an uninitialized Linear layer scrambling our embeddings
        if use_projection:
            self.projection = nn.Linear(output_dim, embedding_dim)
            self.final_dim = embedding_dim
        else:
            self.projection = nn.Identity()
            self.final_dim = output_dim

    def forward(self, waveform, lengths=None):
        embeddings = self.backbone(waveform, lengths)
        projected_embeddings = self.projection(embeddings)
        normalized_embeddings = F.normalize(projected_embeddings, p=2, dim=-1)

        return normalized_embeddings


class Model(torch.nn.Module):
    """
    Speech Similarity Model.
    """
    def __init__(
        self,
        encoder_name: str = "ecapa-voxceleb",
        embedding_dim: int = 256,
        use_projection: bool = True,
        freeze_encoder: bool = False,
        mlp_heads: list = None,
        mlp_dnn_dim: int = 64,
        mlp_range_clipping: bool = True,
        cache_dir: str = None,
        encoder_checkpoint: str = None,
    ):
        super().__init__()
        self.encoder = SpeechEncoder(
            encoder_name,
            embedding_dim,
            use_projection,
            freeze_encoder,
            cache_dir=cache_dir,
            encoder_checkpoint=encoder_checkpoint,
        )

        self.mlp_heads = nn.ModuleDict()

        if mlp_heads is not None:
            # Interaction dimension uses emb_a, emb_b, abs(emb_a - emb_b), and emb_a * emb_b
            interaction_dim = self.encoder.final_dim * 4
            for head_name in mlp_heads:
                self.mlp_heads[head_name] = Projection(
                    in_dim=interaction_dim,
                    hidden_dim=mlp_dnn_dim,
                    activation=nn.ReLU,
                    range_clipping=mlp_range_clipping,
                )

    def encoder_parameters(self):
        return self.encoder.backbone.parameters()

    def head_parameters(self):
        """Everything that is not the pretrained backbone (projection + MLP heads)."""
        backbone_ids = {id(p) for p in self.encoder.backbone.parameters()}
        return (p for p in self.parameters() if id(p) not in backbone_ids)

    def forward(self, wav_a, wav_b=None, len_a=None, len_b=None):
        outputs = {}

        emb_a = self.encoder(wav_a, len_a)
        outputs["emb_a"] = emb_a

        if wav_b is not None:
            emb_b = self.encoder(wav_b, len_b)
            outputs["emb_b"] = emb_b

            outputs["cos_sim"] = F.cosine_similarity(emb_a, emb_b, dim=-1)

            if len(self.mlp_heads) > 0:
                interaction = torch.cat([
                    emb_a,
                    emb_b,
                    torch.abs(emb_a - emb_b),
                    emb_a * emb_b
                ], dim=-1)

                for head_name, mlp in self.mlp_heads.items():
                    head_output = mlp(interaction)
                    outputs[head_name] = head_output.squeeze(-1)

        return outputs
