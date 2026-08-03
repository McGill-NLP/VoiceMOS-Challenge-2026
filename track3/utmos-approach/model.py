#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Track 3 similarity model, architecturally identical to ../baseline/model.py.

This experiment changes the *objective*, not the network: same SpeechBrain ECAPA encoder,
same 256-d projection, same 4-way interaction vector, same range-clipped MLP head. That
keeps the comparison against the baseline honest -- any difference in the numbers is
attributable to the loss.

One deliberate deviation. ../baseline/model.py passes `freeze_ssl=False  # Fine-tuning
everything`, but SpeechBrain's `Pretrained.__init__` defaults to `freeze_params=True` and
sets requires_grad=False on the whole backbone, so the published baseline only ever trains
its 0.12M-parameter head. Here `freeze_params=False` is passed explicitly and freezing is
the caller's decision, via `freeze_encoder`. Use `--freeze-encoder` to reproduce what the
baseline actually does.
"""

import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

SAMPLE_RATE = 16000
DEFAULT_CACHE_DIR = os.environ.get(
    "VMC_ENCODER_CACHE", os.path.expanduser("~/.cache/vmc2026-track3-encoders")
)


class Projection(nn.Module):
    """Maps the interaction vector to a scalar score.

    Identical to ../baseline/model.py, which in turn matches UTMOS's `model.Projection`
    (Linear -> activation -> Dropout(0.3) -> Linear, then tanh*2+3 for range clipping).
    """

    def __init__(self, in_dim: int, hidden_dim: int, activation=nn.ReLU,
                 range_clipping: bool = False):
        super().__init__()
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
        return output


class SpeechEncoder(nn.Module):
    """SpeechBrain EncoderClassifier backbone -> L2-normalised embedding."""

    def __init__(self, encoder_source: str = "speechbrain/spkrec-ecapa-voxceleb",
                 embedding_dim: int = 256, use_projection: bool = True,
                 freeze_encoder: bool = False, cache_dir: str = None):
        super().__init__()
        from speechbrain.inference.classifiers import EncoderClassifier

        savedir = os.path.join(cache_dir or DEFAULT_CACHE_DIR,
                               encoder_source.replace("/", "--"))
        # See the module docstring: freeze_params=False is what makes --freeze-encoder mean
        # something. Without it SpeechBrain silently freezes the backbone.
        self.backbone = EncoderClassifier.from_hparams(
            source=encoder_source, savedir=savedir,
            run_opts={"device": "cpu"}, freeze_params=False,
        )
        self.encoder_source = encoder_source

        # freeze_params=False also skips SpeechBrain's own .eval() call; start in eval and
        # let the training loop's model.train() flip it.
        self.backbone.eval()

        with torch.no_grad():
            output_dim = self.backbone.encode_batch(torch.zeros(1, SAMPLE_RATE)).shape[-1]

        if freeze_encoder:
            for param in self.backbone.parameters():
                param.requires_grad = False

        if use_projection:
            self.projection = nn.Linear(output_dim, embedding_dim)
            self.final_dim = embedding_dim
        else:
            self.projection = nn.Identity()
            self.final_dim = output_dim

        self.output_dim = output_dim
        logging.info(f"Encoder '{encoder_source}' -> {output_dim}-d, final {self.final_dim}-d")

    def forward(self, waveform, lengths=None):
        wav_lens = None
        if lengths is not None:
            wav_lens = (lengths.float() / waveform.shape[1]).to(waveform.device)

        # SpeechBrain tracks its device on the object rather than from the tensors.
        self.backbone.device = waveform.device

        embeddings = self.backbone.encode_batch(waveform, wav_lens=wav_lens)
        if embeddings.dim() == 3:  # (B, 1, D) -> (B, D)
            embeddings = embeddings.squeeze(1)

        return F.normalize(self.projection(embeddings), p=2, dim=-1)


class Model(nn.Module):
    """Pair -> similarity score."""

    def __init__(self, encoder_source: str = "speechbrain/spkrec-ecapa-voxceleb",
                 embedding_dim: int = 256, use_projection: bool = True,
                 freeze_encoder: bool = False, mlp_heads: list = None,
                 mlp_dnn_dim: int = 64, mlp_range_clipping: bool = True,
                 cache_dir: str = None):
        super().__init__()
        self.encoder = SpeechEncoder(
            encoder_source, embedding_dim, use_projection, freeze_encoder, cache_dir
        )

        self.mlp_heads = nn.ModuleDict()
        if mlp_heads:
            # emb_a, emb_b, |emb_a - emb_b|, emb_a * emb_b
            interaction_dim = self.encoder.final_dim * 4
            for head_name in mlp_heads:
                self.mlp_heads[head_name] = Projection(
                    in_dim=interaction_dim, hidden_dim=mlp_dnn_dim,
                    activation=nn.ReLU, range_clipping=mlp_range_clipping,
                )

    def encoder_parameters(self):
        return self.encoder.backbone.parameters()

    def head_parameters(self):
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
                interaction = torch.cat(
                    [emb_a, emb_b, torch.abs(emb_a - emb_b), emb_a * emb_b], dim=-1
                )
                for head_name, mlp in self.mlp_heads.items():
                    outputs[head_name] = mlp(interaction).squeeze(-1)

        return outputs
