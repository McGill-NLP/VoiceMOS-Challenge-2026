#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pluggable speech encoders for VoiceMOS 2026 Track 3.

Every encoder exposes the same contract, so the rest of the pipeline never has to
know which backbone it is talking to:

    encoder = build_encoder("eres2netv2")
    emb = encoder(waveform, lengths)   # (B, T) float32 @ 16 kHz -> (B, encoder.output_dim)

`lengths` is the number of *valid* samples per row (the collater pads rows to a
common length), or None when every row is already full length.

Available encoders, listed by `python encoders.py --list`:

  ecapa-voxceleb        speechbrain/spkrec-ecapa-voxceleb        (the baseline's encoder)
  commonaccent-ecapa    Jzuluaga/accent-id-commonaccent_ecapa    (ECAPA fine-tuned for accent ID)
  eres2netv2            iic/speech_eres2netv2_sv_zh-cn_16k-common
  eres2netv2-w24s4ep4   iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common
  wavlm-base-plus-l{4,8,12}    torchaudio WAVLM_BASE_PLUS      (masked-prediction SSL)
  wavlm-large-l{4,8,24}        torchaudio WAVLM_LARGE
  xlsr-300m-l{4,8,24}          torchaudio WAV2VEC2_XLSR_300M

The four speaker/accent-ID encoders are discriminative nets trained to be INVARIANT to the
channel and phonetic variation accent lives in; the SSL bundles retain it. On frozen features
the SSL models were the strongest weak learners by a wide margin (see ../weak/README.md), which
is why they are fine-tunable here too. `-l<n>` selects the transformer layer to read, and the
stack is physically truncated there -- layer 4 of WavLM-Large is a 63.5M-parameter model, not a
315M one, so the layer choice is a cost decision as much as a quality one.

Encoders can be combined by joining names with '+', e.g. "eres2netv2+commonaccent-ecapa".
Each branch is L2-normalised before concatenation so that no branch dominates by scale.
"""

import argparse
import logging
import os
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.compliance.kaldi as Kaldi

from eres2netv2 import ERes2NetV2

SAMPLE_RATE = 16000

# 25 ms analysis window; Kaldi returns zero frames for anything shorter.
MIN_SAMPLES = int(0.025 * SAMPLE_RATE)

DEFAULT_CACHE_DIR = os.environ.get(
    "VMC_ENCODER_CACHE", os.path.expanduser("~/.cache/vmc2026-track3-encoders")
)


# --------------------------------------------------------------------------------------
# Checkpoint fetching
# --------------------------------------------------------------------------------------

def _modelscope_url(model_id: str, revision: str, file_path: str) -> str:
    return (
        f"https://www.modelscope.cn/api/v1/models/{model_id}/repo"
        f"?Revision={revision}&FilePath={file_path}"
    )


def download_checkpoint(urls, filename: str, cache_dir: str = None) -> Path:
    """Fetch `filename` from the first URL that works, caching it under `cache_dir`."""
    cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / filename

    if dest.exists() and dest.stat().st_size > 0:
        logging.info(f"Using cached checkpoint {dest}")
        return dest

    errors = []
    for url in urls:
        try:
            logging.info(f"Downloading {filename} from {url} ...")
            tmp = dest.with_suffix(dest.suffix + ".part")
            urllib.request.urlretrieve(url, tmp)
            tmp.rename(dest)
            logging.info(f"Saved to {dest}")
            return dest
        except Exception as e:  # noqa: BLE001 - try every mirror before giving up
            errors.append(f"  {url}\n    -> {type(e).__name__}: {e}")

    raise RuntimeError(
        f"Could not download '{filename}' from any mirror:\n" + "\n".join(errors)
        + f"\n\nDownload it manually and pass --encoder-checkpoint, or place it at {dest}."
    )


# --------------------------------------------------------------------------------------
# Encoder interface
# --------------------------------------------------------------------------------------

class BaseEncoder(nn.Module):
    """Waveform -> single utterance embedding.

    Subclasses must set `self.output_dim` and implement `forward`.
    """

    output_dim: int

    def forward(self, waveform: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        raise NotImplementedError


class SpeechBrainEncoder(BaseEncoder):
    """Any SpeechBrain `EncoderClassifier` checkpoint.

    Covers both the baseline's speaker encoder and the CommonAccent accent encoder:
    they share an identical hyperparams layout (compute_features / mean_var_norm /
    embedding_model / classifier), so `encode_batch` behaves the same for both.
    """

    def __init__(self, source: str, savedir: str = None):
        super().__init__()
        from speechbrain.inference.classifiers import EncoderClassifier

        savedir = savedir or os.path.join(DEFAULT_CACHE_DIR, source.replace("/", "--"))
        # Load on CPU; the training script moves the whole model to GPU afterwards.
        #
        # freeze_params=False matters. SpeechBrain's `Pretrained` defaults it to True,
        # which sets requires_grad=False on every backbone parameter. ../baseline/model.py
        # does not override it, so its `freeze_ssl=False  # Fine-tuning everything` is a
        # no-op and the baseline only ever trains the 0.12M-parameter head. Here freezing
        # is controlled explicitly by the caller instead (see Model(freeze_encoder=...)).
        self.encoder = EncoderClassifier.from_hparams(
            source=source,
            savedir=savedir,
            run_opts={"device": "cpu"},
            freeze_params=False,
        )
        self.source = source

        # freeze_params=False also skips SpeechBrain's own .eval() call, which would leave
        # the backbone in train mode. Start in eval; finetune.py's model.train() flips it.
        self.encoder.eval()

        with torch.no_grad():
            dummy = torch.zeros(1, SAMPLE_RATE)
            self.output_dim = self.encoder.encode_batch(dummy).shape[-1]

    def forward(self, waveform, lengths=None):
        # SpeechBrain wants lengths as a fraction of the padded length.
        wav_lens = None
        if lengths is not None:
            wav_lens = (lengths.float() / waveform.shape[1]).to(waveform.device)

        # SpeechBrain tracks its device on the object rather than from the tensors.
        self.encoder.device = waveform.device

        embeddings = self.encoder.encode_batch(waveform, wav_lens=wav_lens)
        if embeddings.dim() == 3:  # (B, 1, D) -> (B, D)
            embeddings = embeddings.squeeze(1)
        return embeddings

    @torch.no_grad()
    def classify(self, waveform, lengths=None):
        """Class posteriors from the pretrained head (16 accents for CommonAccent).

        Not used during fine-tuning; exposed because the divergence between the accent
        posteriors of the two utterances in a pair is a useful standalone feature.
        """
        emb = self.forward(waveform, lengths)
        logits = self.encoder.mods.classifier(emb.unsqueeze(1)).squeeze(1)
        return F.softmax(logits, dim=-1)


class ERes2NetV2Encoder(BaseEncoder):
    """ERes2NetV2 from 3D-Speaker (Chen et al., Interspeech 2024).

    The backbone consumes 80-dim Kaldi FBank with per-utterance mean normalisation,
    exactly as in `speakerlab/bin/infer_sv.py`. `torchaudio.compliance.kaldi.fbank`
    only accepts one utterance at a time, so features are computed per row on the
    *unpadded* waveform and then repetitively padded in the feature domain. This
    keeps CMN statistics free of padding and avoids the waveform-domain discontinuity
    that repetitive padding would otherwise introduce mid-frame.
    """

    def __init__(self, model_args: dict, checkpoint: str, n_mels: int = 80):
        super().__init__()
        self.n_mels = n_mels
        self.model = ERes2NetV2(feat_dim=n_mels, **model_args)

        state = torch.load(checkpoint, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.model.load_state_dict(state)
        logging.info(
            f"Loaded ERes2NetV2 weights from {checkpoint} "
            f"({sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M params)"
        )

        self.output_dim = model_args.get("embedding_size", 192)

    @torch.no_grad()
    def _fbank(self, waveform, lengths=None):
        feats = []
        for i in range(waveform.shape[0]):
            wav = waveform[i]
            if lengths is not None:
                wav = wav[: int(lengths[i])]
            if wav.shape[0] < MIN_SAMPLES:
                wav = F.pad(wav, (0, MIN_SAMPLES - wav.shape[0]))

            feat = Kaldi.fbank(
                wav.unsqueeze(0),
                num_mel_bins=self.n_mels,
                sample_frequency=SAMPLE_RATE,
                dither=0.0,
            )
            feats.append(feat - feat.mean(dim=0, keepdim=True))  # CMN

        max_frames = max(f.shape[0] for f in feats)
        return torch.stack([_repetitive_pad(f, max_frames) for f in feats])

    def forward(self, waveform, lengths=None):
        feats = self._fbank(waveform, lengths)  # (B, T', n_mels)
        return self.model(feats)


class SSLEncoder(BaseEncoder):
    """A `torchaudio.pipelines` SSL bundle read at one layer and mean-pooled over time.

    Matches what the weak learners consume (`../weak/extract_features.py`), so a fine-tuned
    run and a ridge on frozen features see the same representation and differ only in whether
    gradients reach the backbone.

    THE STACK IS TRUNCATED at the requested layer. `extract_features(num_layers=n)` would
    compute the same thing, but the unused layers would still be built, moved to the GPU,
    handed to AdamW and written into every checkpoint. Deleting them makes layer 4 of
    WavLM-Large a 63.5M-parameter backbone (verified identical outputs to the untruncated
    model at atol 1e-5), which is what puts SSL fine-tuning within the same budget as
    ERes2NetV2-w24s4ep4 at 53.5M.

    TWO QUIRKS of torchaudio's implementation are handled here rather than worked around
    upstream:

    1. `Encoder.extract_features` cannot mask a padded batch for WavLM: it builds an additive
       `attention_mask`, and `WavLMSelfAttention.forward` asserts that argument is None. WavLM
       instead takes a boolean `key_padding_mask`, which `Transformer.get_intermediate_outputs`
       never forwards. The layer loop is therefore written out here, passing whichever of the
       two each attention implementation actually honours. Without it a padded row's embedding
       moved by up to 6.1 absolute depending on what else was in its batch -- which would have
       made test predictions depend on batch composition.

    2. Bundles whose models were pretrained on normalised audio (`WAVLM_LARGE`,
       `WAV2VEC2_XLSR_300M`) are wrapped in `_Wav2Vec2Model`, which applies
       `layer_norm(waveform, waveform.shape)` over the WHOLE batch tensor -- padding included.
       The wrapper is unwrapped and the normalisation redone per row over valid samples only,
       which is what the batch-of-one path in the weak pipeline effectively did.

    `layer_drop` defaults to 0: dropping one layer of a 24-layer stack is a perturbation,
    dropping one of four is a lobotomy.
    """

    def __init__(self, bundle_name: str, layer: int, layer_drop: float = 0.0):
        super().__init__()
        import torchaudio.pipelines as pipelines

        if not hasattr(pipelines, bundle_name):
            raise ValueError(f"torchaudio.pipelines has no bundle '{bundle_name}'")
        bundle = getattr(pipelines, bundle_name)
        if bundle.sample_rate != SAMPLE_RATE:
            raise ValueError(
                f"{bundle_name} expects {bundle.sample_rate} Hz, pipeline audio is {SAMPLE_RATE}"
            )

        wrapper = bundle.get_model()
        self.normalize_waveform = bool(getattr(bundle, "_normalize_waveform", False))
        self.model = getattr(wrapper, "model", wrapper)

        transformer = self.model.encoder.transformer
        depth = len(transformer.layers)
        if not 1 <= layer <= depth:
            raise ValueError(f"{bundle_name} has {depth} layers; cannot read layer {layer}.")
        del transformer.layers[layer:]
        transformer.layer_drop = layer_drop

        self.bundle_name = bundle_name
        self.layer = layer
        self.wavlm_attention = any(
            type(l.attention).__name__ == "WavLMSelfAttention" for l in transformer.layers
        )
        with torch.no_grad():
            probe = self.forward(torch.zeros(1, SAMPLE_RATE))
        self.output_dim = probe.shape[-1]

        logging.info(
            f"{bundle_name}: layer {layer} of {depth}, dim {self.output_dim}, "
            f"{sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M params after truncation"
            + (", input layer-norm on" if self.normalize_waveform else "")
        )

    @staticmethod
    def _valid_mask(lengths, size, device):
        return torch.arange(size, device=device)[None, :] < lengths[:, None]

    def _normalize(self, waveform, lengths):
        """Per-row layer-norm over valid samples, zeroing the padding."""
        if lengths is None:
            return F.layer_norm(waveform, waveform.shape[-1:])
        mask = self._valid_mask(lengths, waveform.shape[1], waveform.device).to(waveform.dtype)
        n = mask.sum(dim=1, keepdim=True).clamp(min=1)
        mean = (waveform * mask).sum(dim=1, keepdim=True) / n
        var = (((waveform - mean) * mask) ** 2).sum(dim=1, keepdim=True) / n
        return ((waveform - mean) * torch.rsqrt(var + 1e-5)) * mask

    def _transformer(self, features, frame_lengths):
        """`Encoder.extract_features` with the padding mask each attention type accepts.

        Mirrors torchaudio's own `Encoder._preprocess` -> `Transformer.get_intermediate_outputs`
        path exactly when `frame_lengths` is None; see quirk 1 above for why it is spelled out.
        """
        encoder = self.model.encoder
        x = encoder.feature_projection(features)

        pad = None
        if frame_lengths is not None:
            pad = ~self._valid_mask(frame_lengths, x.shape[1], x.device)   # True where padded
            x = x.masked_fill(pad.unsqueeze(-1), 0.0)

        transformer = encoder.transformer
        x = transformer._preprocess(x)

        # WavLM honours key_padding_mask (bool, True = pad) and rejects attention_mask;
        # wav2vec 2.0 / XLS-R are the other way round and ignore key_padding_mask.
        # Additive, not boolean, in both branches: torch's mask canonicalisation converts a
        # bool mask to -inf, which would produce NaNs for any row that is entirely padding.
        attention_mask = key_padding_mask = None
        if pad is not None:
            bias = -10000.0 * pad.to(x.dtype)
            if self.wavlm_attention:
                key_padding_mask = bias
            else:
                b, t = pad.shape
                attention_mask = bias[:, None, None, :].expand(b, 1, t, t)

        position_bias = None
        for layer in transformer.layers:
            x, position_bias = layer(
                x, attention_mask, position_bias=position_bias,
                key_padding_mask=key_padding_mask,
            )
        return x

    def forward(self, waveform, lengths=None):
        if self.normalize_waveform:
            waveform = self._normalize(waveform, lengths)

        features, frame_lengths = self.model.feature_extractor(waveform, lengths)
        hidden = self._transformer(features, frame_lengths)

        if frame_lengths is None:
            return hidden.mean(dim=1)
        mask = self._valid_mask(frame_lengths, hidden.shape[1], hidden.device)
        mask = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


class ConcatEncoder(BaseEncoder):
    """Runs several encoders on the same audio and concatenates their embeddings.

    Each branch is L2-normalised first, otherwise a branch with a larger natural norm
    would dominate the downstream projection.
    """

    def __init__(self, encoders: dict):
        super().__init__()
        self.encoders = nn.ModuleDict(encoders)
        self.output_dim = sum(e.output_dim for e in encoders.values())

    def forward(self, waveform, lengths=None):
        parts = [
            F.normalize(enc(waveform, lengths), p=2, dim=-1)
            for enc in self.encoders.values()
        ]
        return torch.cat(parts, dim=-1)


def _repetitive_pad(feat: torch.Tensor, target_len: int) -> torch.Tensor:
    """Tile `feat` along dim 0 up to `target_len` frames (MBNet-style padding)."""
    this_len = feat.shape[0]
    if this_len == 0:
        return torch.zeros(target_len, feat.shape[1], dtype=feat.dtype, device=feat.device)
    if this_len >= target_len:
        return feat[:target_len]

    dup_times = target_len // this_len
    remain = target_len - this_len * dup_times
    to_dup = [feat] * dup_times
    if remain > 0:
        to_dup.append(feat[:remain])
    return torch.cat(to_dup, dim=0)


# --------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------

ENCODER_REGISTRY = {
    "ecapa-voxceleb": {
        "type": "speechbrain",
        "source": "speechbrain/spkrec-ecapa-voxceleb",
        "embedding_dim": 192,
        "description": "ECAPA-TDNN trained on VoxCeleb for speaker verification (baseline encoder).",
    },
    "commonaccent-ecapa": {
        "type": "speechbrain",
        "source": "Jzuluaga/accent-id-commonaccent_ecapa",
        "embedding_dim": 192,
        "description": "ECAPA-TDNN fine-tuned on CommonVoice for 16-way English accent ID.",
    },
    "eres2netv2": {
        "type": "eres2netv2",
        "model_args": {
            "embedding_size": 192,
            "baseWidth": 26,
            "scale": 2,
            "expansion": 2,
        },
        "checkpoint_file": "pretrained_eres2netv2.ckpt",
        "checkpoint_urls": [
            _modelscope_url(
                "iic/speech_eres2netv2_sv_zh-cn_16k-common",
                "v1.0.1",
                "pretrained_eres2netv2.ckpt",
            ),
            "https://huggingface.co/bandad/eres2netv2_pretrained/resolve/main/"
            "pretrained_eres2netv2.ckpt",
        ],
        "embedding_dim": 192,
        "description": "ERes2NetV2 (17.8M) trained on 200k speakers. EER 0.61% VoxCeleb1-O.",
    },
    "eres2netv2-w24s4ep4": {
        "type": "eres2netv2",
        "model_args": {
            "embedding_size": 192,
            "baseWidth": 24,
            "scale": 4,
            "expansion": 4,
        },
        "checkpoint_file": "pretrained_eres2netv2w24s4ep4.ckpt",
        "checkpoint_urls": [
            _modelscope_url(
                "iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common",
                "v1.0.1",
                "pretrained_eres2netv2w24s4ep4.ckpt",
            ),
        ],
        "embedding_dim": 192,
        "description": "Wider ERes2NetV2 (w24s4ep4), 53.5M params, 200k speakers. 3x eres2netv2; measured 1.065 s/step and 29.0 GiB at batch 4 x 4 accum.",
    },
}


# SSL bundles from torchaudio.pipelines. Layers 4 and 8 plus the last one, matching the layers
# the weak-learner sweep kept -- layer 4 won there for every bundle and both targets, and the
# last layer was close to the worst, so the cheapest entry is also the expected best.
SSL_BUNDLES = {
    "wavlm-base-plus": ("WAVLM_BASE_PLUS", 12, 768,
                        "WavLM Base+ (94.4M full), masked prediction + denoising on 94k h."),
    "wavlm-large": ("WAVLM_LARGE", 24, 1024,
                    "WavLM Large (315.5M full), masked prediction + denoising on 94k h."),
    "xlsr-300m": ("WAV2VEC2_XLSR_300M", 24, 1024,
                  "XLS-R 300M, wav2vec 2.0 pretrained on 436k h across 128 languages."),
}

for _prefix, (_bundle, _depth, _dim, _desc) in SSL_BUNDLES.items():
    for _layer in sorted({4, 8, _depth}):
        ENCODER_REGISTRY[f"{_prefix}-l{_layer}"] = {
            "type": "ssl",
            "bundle": _bundle,
            "layer": _layer,
            "embedding_dim": _dim,
            "description": f"{_desc} Truncated after layer {_layer} of {_depth}.",
        }


def build_encoder(spec: str, cache_dir: str = None, checkpoint: str = None) -> BaseEncoder:
    """Build an encoder from a registry name, or several joined by '+'.

    `checkpoint` overrides the downloaded weights and is only valid for a single
    (non-combined) encoder.
    """
    names = [n.strip() for n in spec.split("+") if n.strip()]
    if not names:
        raise ValueError("Empty encoder spec.")
    if len(names) > 1 and checkpoint is not None:
        raise ValueError("--encoder-checkpoint cannot be used with a combined '+' encoder.")

    built = {}
    for name in names:
        if name not in ENCODER_REGISTRY:
            raise ValueError(
                f"Unknown encoder '{name}'. Available: {', '.join(ENCODER_REGISTRY)}"
            )
        cfg = ENCODER_REGISTRY[name]

        if cfg["type"] == "speechbrain":
            savedir = os.path.join(
                cache_dir or DEFAULT_CACHE_DIR, cfg["source"].replace("/", "--")
            )
            built[name] = SpeechBrainEncoder(cfg["source"], savedir=savedir)

        elif cfg["type"] == "eres2netv2":
            ckpt = checkpoint or download_checkpoint(
                cfg["checkpoint_urls"], cfg["checkpoint_file"], cache_dir
            )
            built[name] = ERes2NetV2Encoder(cfg["model_args"], str(ckpt))

        elif cfg["type"] == "ssl":
            if checkpoint is not None:
                raise ValueError(
                    "--encoder-checkpoint is not supported for SSL bundles; torchaudio "
                    "downloads and caches their weights itself."
                )
            built[name] = SSLEncoder(cfg["bundle"], cfg["layer"])

        else:
            raise ValueError(f"Unknown encoder type '{cfg['type']}' for '{name}'.")

    if len(built) == 1:
        return next(iter(built.values()))
    return ConcatEncoder(built)


def main():
    parser = argparse.ArgumentParser(description="Inspect and smoke-test encoders.")
    parser.add_argument("--list", action="store_true", help="List available encoders.")
    parser.add_argument("--encoder", type=str, default=None, help="Encoder to smoke-test.")
    parser.add_argument("--cache-dir", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.list or args.encoder is None:
        width = max(len(n) for n in ENCODER_REGISTRY)
        print(f"\n{'NAME'.ljust(width)}  DIM  DESCRIPTION")
        print("-" * 100)
        for name, cfg in ENCODER_REGISTRY.items():
            print(f"{name.ljust(width)}  {cfg['embedding_dim']:<3}  {cfg['description']}")
        print("\nCombine encoders with '+', e.g. eres2netv2+commonaccent-ecapa\n")
        return

    encoder = build_encoder(args.encoder, cache_dir=args.cache_dir)
    encoder.eval()

    # Two rows of different true lengths, padded to a common width.
    waveform = torch.randn(2, SAMPLE_RATE * 3) * 0.05
    lengths = torch.tensor([SAMPLE_RATE * 3, SAMPLE_RATE * 2])
    with torch.no_grad():
        emb = encoder(waveform, lengths)

    print(f"\nencoder     : {args.encoder}")
    print(f"output_dim  : {encoder.output_dim}")
    print(f"embedding   : {tuple(emb.shape)}")
    print(f"parameters  : {sum(p.numel() for p in encoder.parameters()) / 1e6:.2f}M")
    print(f"cos(a, b)   : {F.cosine_similarity(emb[0:1], emb[1:2]).item():.4f}\n")


if __name__ == "__main__":
    main()
