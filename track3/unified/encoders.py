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
