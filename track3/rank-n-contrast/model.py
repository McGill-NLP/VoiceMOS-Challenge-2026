#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model for the Rank-N-Contrast experiments.

Deliberately identical to the official baseline up to the head, so the RNC arm
and the baseline arm differ only in the training objective:

    ECAPA-TDNN -> Linear(192, 256) -> L2 norm -> interaction -> head

The pair representation `v = [e_a, e_b, |e_a - e_b|, e_a * e_b]` (1024-d) is the
feature the RNC loss is applied to, and the same feature the head consumes. Per
the paper there is no separate projection head for the contrastive loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from speechbrain.inference.classifiers import EncoderClassifier

ECAPA = "speechbrain/spkrec-ecapa-voxceleb"


class SpeechEncoder(nn.Module):
    """ECAPA-TDNN wrapper producing one L2-normalised embedding per waveform.

    NOTE on freezing. The baseline passes `freeze_ssl=False` with the comment
    "Fine-tuning everything", but SpeechBrain's `Pretrained.__init__` defaults to
    `freeze_params=True`, which sets `requires_grad=False` on all 22.15M ECAPA
    parameters. The official baseline therefore trains only 115k parameters
    (0.52%): the 192->256 projection plus the MLP head. `freeze_ecapa=True` here
    reproduces that; set it False to genuinely fine-tune the encoder.

    Separately, SpeechBrain's freeze also calls `.eval()`, but a later
    `model.train()` puts ECAPA back into training mode, so its BatchNorm running
    statistics keep drifting even though its weights are frozen. That is what the
    baseline does; `ecapa_eval_mode=True` pins ECAPA to eval and stops the drift.
    """

    def __init__(
        self,
        model_name=ECAPA,
        embedding_dim=256,
        use_projection=True,
        freeze_ecapa=True,
        ecapa_eval_mode=False,
    ):
        super().__init__()
        self.ssl_model = EncoderClassifier.from_hparams(
            source=model_name, run_opts={"device": "cpu"}, freeze_params=freeze_ecapa
        )
        self.freeze_ecapa = freeze_ecapa
        self.ecapa_eval_mode = ecapa_eval_mode
        for param in self.ssl_model.parameters():
            param.requires_grad = not freeze_ecapa

        with torch.no_grad():
            output_dim = self.ssl_model.encode_batch(torch.zeros(1, 16000)).shape[-1]

        if use_projection:
            self.projection = nn.Linear(output_dim, embedding_dim)
            self.final_dim = embedding_dim
        else:
            self.projection = nn.Identity()
            self.final_dim = output_dim

    def train(self, mode=True):
        super().train(mode)
        if self.ecapa_eval_mode:
            self.ssl_model.eval()
        return self

    def forward(self, waveform, lengths=None):
        wav_lens = None
        if lengths is not None:
            wav_lens = (lengths.float() / waveform.shape[1]).to(waveform.device)
        # SpeechBrain tracks its device internally; keep it in sync.
        self.ssl_model.device = waveform.device
        embeddings = self.ssl_model.encode_batch(waveform, wav_lens=wav_lens)
        if embeddings.dim() == 3:
            embeddings = embeddings.squeeze(1)
        return F.normalize(self.projection(embeddings), p=2, dim=-1)


class PairEncoder(nn.Module):
    """Maps a (wav_a, wav_b) pair to the interaction vector used by RNC."""

    def __init__(
        self,
        model_name=ECAPA,
        embedding_dim=256,
        use_projection=True,
        freeze_ecapa=True,
        ecapa_eval_mode=False,
    ):
        super().__init__()
        self.encoder = SpeechEncoder(
            model_name, embedding_dim, use_projection, freeze_ecapa, ecapa_eval_mode
        )
        self.out_dim = self.encoder.final_dim * 4

    def forward(self, wav_a, wav_b, len_a=None, len_b=None, b_index=None):
        emb_a = self.encoder(wav_a, len_a)
        emb_b = self.encoder(wav_b, len_b)
        if b_index is not None:
            # Reference waveforms were deduplicated by the collater; expand back.
            emb_b = emb_b.index_select(0, b_index)
        return torch.cat([emb_a, emb_b, torch.abs(emb_a - emb_b), emb_a * emb_b], dim=-1)


class LinearHead(nn.Module):
    """The paper's stage-2 predictor: a single linear layer on frozen features."""

    def __init__(self, in_dim, range_clipping=False):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)
        self.range_clipping = range_clipping

    def forward(self, x):
        out = self.fc(x)
        if self.range_clipping:
            out = torch.tanh(out) * 2.0 + 3.0
        return out.squeeze(-1)


class MLPHead(nn.Module):
    """The baseline's head: 2-layer MLP with dropout and tanh range clipping to
    [1, 5] (arXiv:2104.03017)."""

    def __init__(self, in_dim, hidden_dim=64, range_clipping=True):
        super().__init__()
        self.range_clipping = range_clipping
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        out = self.net(x)
        if self.range_clipping:
            out = torch.tanh(out) * 2.0 + 3.0
        return out.squeeze(-1)


def build_head(kind, in_dim, range_clipping=True):
    if kind == "linear":
        return LinearHead(in_dim, range_clipping=range_clipping)
    if kind == "mlp":
        return MLPHead(in_dim, range_clipping=range_clipping)
    raise ValueError(f"Unknown head: {kind}")


class Model(nn.Module):
    """Pair encoder plus a scalar regression head."""

    def __init__(
        self,
        model_name=ECAPA,
        embedding_dim=256,
        use_projection=True,
        freeze_ecapa=True,
        ecapa_eval_mode=False,
        head="mlp",
        range_clipping=True,
    ):
        super().__init__()
        self.pair_encoder = PairEncoder(
            model_name, embedding_dim, use_projection, freeze_ecapa, ecapa_eval_mode
        )
        self.head = build_head(head, self.pair_encoder.out_dim, range_clipping)

    def forward(self, wav_a, wav_b, len_a=None, len_b=None, b_index=None, return_features=False):
        features = self.pair_encoder(wav_a, wav_b, len_a, len_b, b_index)
        preds = self.head(features)
        return (preds, features) if return_features else preds
