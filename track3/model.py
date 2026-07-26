#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from speechbrain.inference.classifiers import EncoderClassifier


class Projection(nn.Module):
    
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


class FrameAttentionPool(nn.Module):
   
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
    
    def __init__(
        self,
        model_name: str = "speechbrain/spkrec-ecapa-voxceleb",
        embedding_dim: int = 256,
        use_projection: bool = True,
        freeze_ssl: bool = False,
        capture_frame_level: bool = False,
    ):
        super().__init__()

        # Load to CPU first; PyTorch's Trainer will automatically move it to GPU later
        self.ssl_model = EncoderClassifier.from_hparams(source=model_name, run_opts={"device": "cpu"})

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
    ):
        super().__init__()
        self.encoder = SpeechEncoder(model_name, embedding_dim, use_projection, freeze_ssl)

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
