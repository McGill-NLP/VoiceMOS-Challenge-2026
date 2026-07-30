#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from speechbrain.inference.classifiers import EncoderClassifier


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


class ResidualAdapter(nn.Module):
    """
    Small trainable residual transform applied on top of a frozen/pretrained
    encoder's embedding, mixed in via a scheduled coefficient `alpha` that
    the training loop ramps from 0 -> some max value over training.

    At alpha=0 this is an EXACT passthrough (returns emb unchanged, no
    adapter computation at all) -- the model starts out identical to using
    the raw pretrained encoder (ECAPA / CommonAccent-ECAPA) directly, i.e.
    "starts off with ECAPA" exactly as it is. As alpha ramps up, the
    trainable residual net's contribution grows, letting the model
    increasingly rely on a representation it has learned itself rather than
    being capped at whatever the frozen pretrained embedding alone can
    express -- "weaning off" the fixed backbone gradually rather than an
    abrupt, potentially destabilizing switch.

    Because it's a residual (emb + alpha*residual, renormalized), the
    pretrained embedding's information is never discarded outright, even at
    alpha=1 -- the adapter learns a correction ON TOP of it, not a full
    replacement. This mirrors why gradual backbone unfreezing (rather than
    unfreezing everything at once) was already found to work better in this
    codebase: abrupt architecture/capacity changes are riskier than
    scheduled ones.
    """

    def __init__(self, dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, emb, alpha: float):
        if alpha <= 0.0:
            return emb  # exact passthrough -- no adapter computation, no gradient to it either
        residual = self.net(emb)
        mixed = emb + alpha * residual
        return F.normalize(mixed, p=2, dim=-1)


class FrameAttentionPool(nn.Module):
    """
    Learnable-query attention pooling over frame-level features
    (shape [B, C, T] -> [B, out_dim]).

    NOTE: superseded by the dual dedicated-encoder approach in finetune.py
    (a separate CommonAccent-ECAPA backbone for acc_sim, rather than a
    hand-rolled frame-level pooling head on top of the speaker-ID encoder).
    Left in place, unused, in case you want to revisit this alternative
    later.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 256):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(in_dim, hidden_dim, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(hidden_dim, 1, kernel_size=1),
        )
        self.out_proj = nn.Linear(in_dim, out_dim)

    def forward(self, frame_feats):
        # frame_feats: [B, C, T]
        scores = self.attn(frame_feats)          # [B, 1, T]
        weights = F.softmax(scores, dim=-1)       # [B, 1, T]
        pooled = (frame_feats * weights).sum(-1)  # [B, C]
        out = self.out_proj(pooled)               # [B, out_dim]
        return F.normalize(out, p=2, dim=-1)


class SpeechEncoder(torch.nn.Module):
    """
    Core feature extractor: Outputs a normalized 1D embedding.

    New (backward-compatible) argument: `capture_frame_level`. When True,
    a forward hook is registered on ECAPA's internal multi-layer feature
    aggregation module (`mfa`), which runs just before ECAPA's own
    attentive-statistics pooling collapses the time dimension. This lets
    callers retrieve the pre-pooling frame-level activations via
    `get_frame_features()` right after calling `forward()`, without a
    second forward pass through the backbone. Default is False, so existing
    callers (e.g. the baseline `Model` class below) are unaffected.

    New (backward-compatible) argument: `cache_dir`. Controls where the
    pretrained checkpoint (e.g. speechbrain/spkrec-ecapa-voxceleb or
    Jzuluaga/accent-id-commonaccent_ecapa) gets downloaded to. Passed
    straight through as `savedir` to EncoderClassifier.from_hparams().
    If None (default), SpeechBrain falls back to its own default location
    (a `pretrained_models/<source>` folder relative to the current working
    directory), same as before this argument existed. Different model
    sources are placed in their own subfolder under cache_dir so that
    --model-name-spk and --model-name-acc checkpoints never collide.
    """

    def __init__(
        self,
        model_name: str = "speechbrain/spkrec-ecapa-voxceleb",
        embedding_dim: int = 256,
        use_projection: bool = True,
        freeze_ssl: bool = False,
        capture_frame_level: bool = False,
        cache_dir: str = None,
    ):
        super().__init__()
        # Load to CPU first; PyTorch's Trainer will automatically move it to GPU later
        savedir = None
        if cache_dir is not None:
            # Namespace by model_name so spk/acc checkpoints (or any other
            # models later) each get their own subfolder and never collide.
            savedir = os.path.join(cache_dir, model_name.replace("/", "__"))
            os.makedirs(savedir, exist_ok=True)
        try:
            self.ssl_model = EncoderClassifier.from_hparams(
                source=model_name,
                savedir=savedir,
                run_opts={"device": "cpu"},
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load '{model_name}' via speechbrain.inference.classifiers."
                f"EncoderClassifier.from_hparams(). If this is a community model whose "
                f"hparams.yaml defines a custom classifier class (some accent-ID recipes "
                f"do), you may need speechbrain.pretrained.interfaces.foreign_class instead "
                f"-- check the model card / hparams.yaml on its Hugging Face page for the "
                f"exact loading snippet it recommends, and adapt SpeechEncoder.__init__ "
                f"accordingly. Original error: {e}"
            ) from e
        if freeze_ssl:
            for param in self.ssl_model.parameters():
                param.requires_grad = False
        self.capture_frame_level = capture_frame_level
        self._frame_feats = None
        self.frame_channels = None
        if capture_frame_level:
            self._register_frame_hook()
        # Dynamically determine the embedding dimension using a dummy tensor
        with torch.no_grad():
            dummy_wav = torch.zeros(1, 16000)
            dummy_out = self.ssl_model.encode_batch(dummy_wav)
            output_dim = dummy_out.shape[-1]
            if capture_frame_level:
                # The hook above will have fired during the dummy forward too;
                # use it to determine the channel count of the frame-level features.
                frame_feats = self.get_frame_features()
                self.frame_channels = frame_feats.shape[1]
        # In zero-shot, we do not want an uninitialized Linear layer scrambling our embeddings
        if use_projection:
            self.projection = nn.Linear(output_dim, embedding_dim)
            self.final_dim = embedding_dim
        else:
            self.projection = nn.Identity()
            self.final_dim = output_dim

    def _register_frame_hook(self):
        """
        Hooks ECAPA's `mfa` submodule (multi-layer feature aggregation),
        which produces frame-level features [B, C, T] right before ECAPA's
        own attentive-statistics pooling. Raises a clear error if this
        submodule can't be found, rather than silently failing later.
        """
        embedding_model = self.ssl_model.mods.embedding_model
        target_module = getattr(embedding_model, "mfa", None)
        if target_module is None:
            raise RuntimeError(
                "Could not locate ECAPA's frame-level aggregation module "
                "(`embedding_model.mfa`) to hook into. Your installed "
                "speechbrain version's ECAPA_TDNN implementation may differ "
                "from the expected structure -- inspect "
                "`self.ssl_model.mods.embedding_model` to find the right "
                "submodule to hook (it should output [B, C, T] features "
                "right before the pooling layer) and update `_register_frame_hook`."
            )

        def _hook(module, inp, output):
            self._frame_feats = output

        target_module.register_forward_hook(_hook)

    def get_frame_features(self):
        """
        Returns the frame-level features [B, C, T] captured during the most
        recent call to forward(). Must call forward() first, and the
        encoder must have been constructed with capture_frame_level=True.
        """
        if self._frame_feats is None:
            raise RuntimeError(
                "No frame-level features captured yet -- call forward() first, "
                "and make sure capture_frame_level=True was passed to __init__."
            )
        feats = self._frame_feats
        if feats.dim() != 3:
            raise RuntimeError(
                f"Expected 3D frame-level features [B, C, T], got shape {tuple(feats.shape)}."
            )
        # Heuristic safety check: channel dim should be dim=1, not dim=2 (time).
        # ECAPA channel counts are typically >= 512; a small dim=1 alongside a
        # much larger dim=2 suggests [B, T, C] ordering instead -- fix it up.
        if feats.shape[1] < feats.shape[2] and feats.shape[1] < 64:
            feats = feats.transpose(1, 2)
        return feats

    def forward(self, waveform, lengths=None):
        device = waveform.device
        # SpeechBrain expects lengths as relative percentages (0.0 to 1.0)
        wav_lens = None
        if lengths is not None:
            wav_lens = (lengths.float() / waveform.shape[1]).to(device)
        # Fix SpeechBrain's internal device tracking:
        self.ssl_model.device = device
        # encode_batch outputs shape: [Batch, 1, EmbeddingDim]
        embeddings = self.ssl_model.encode_batch(waveform, wav_lens=wav_lens)
        # Squeeze out the extra dimension to [Batch, EmbeddingDim]
        if embeddings.dim() == 3:
            embeddings = embeddings.squeeze(1)
        projected_embeddings = self.projection(embeddings)
        normalized_embeddings = F.normalize(projected_embeddings, p=2, dim=-1)
        return normalized_embeddings


class Model(torch.nn.Module):
    """
    Speech Similarity Model. (Unchanged baseline class.)
    """

    def __init__(
        self,
        model_name: str = "speechbrain/spkrec-ecapa-voxceleb",
        embedding_dim: int = 256,
        use_projection: bool = True,
        freeze_ssl: bool = False,
        mlp_heads: list = None,
        mlp_dnn_dim: int = 64,
        mlp_range_clipping: bool = True,
        cache_dir: str = None,
    ):
        super().__init__()
        self.encoder = SpeechEncoder(
            model_name, embedding_dim, use_projection, freeze_ssl, cache_dir=cache_dir
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
