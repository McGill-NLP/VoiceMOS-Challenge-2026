#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Frozen-encoder feature extraction for the UTMOS22-style weak-learner stack.

One no-grad pass over every unique wav referenced by train/dev/test, for each requested
encoder, cached to .npz keyed by wav BASENAME. Nothing here trains; the whole point of the
weak-learner track is that these features never move.

Two backends behind one flag:

    sv:<name>        anything in ../unified/encoders.py ENCODER_REGISTRY  (192-d, speaker/accent ID)
    ssl:<PIPELINE>   any torchaudio.pipelines bundle                      (768/1024-d, masked-prediction SSL)

SSL bundles expose every transformer layer from a single forward pass, so `--ssl-layers`
costs nothing extra and yields genuinely different representations: lower layers carry
speaker and channel, middle layers carry phonetic content. UTMOS22 used only the last layer.

Why SSL at all: the four registry encoders are all discriminative speaker-ID nets trained to
be INVARIANT to the channel and phonetic variation that accent lives in. They share that
inductive bias, which is the most likely source of the 0.92 residual correlation measured
across the deep model pool -- and why adding a fourth encoder of the same kind (ecapa-voxceleb)
did not decorrelate anything. Masked-prediction SSL models are the largest available departure.

    python extract_features.py --encoders sv:ecapa-voxceleb ssl:WAVLM_LARGE
    python extract_features.py --list

Basenames are unique across the two data distributions (4,160 of them) and the 137 that
appear in both roots are byte-identical, so one flat basename -> vector map is well defined.
"""

import argparse
import csv
import logging
import os
import sys
import time

import numpy as np
import torch
import torchaudio

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "unified"))

TARGET_SR = 16000

TRAIN_ROOT = "../baseline/data/vmc2026_track3_train_phase_distro_v3_syn"
EVAL_ROOT = "../baseline/data/vmc2026_track3_eval_phase_distro_v3_syn"

# (root, csv) pairs covering every wav the weak learners will ever need.
SOURCES = [
    (TRAIN_ROOT, f"{TRAIN_ROOT}/sets/train.csv"),
    (EVAL_ROOT, f"{EVAL_ROOT}/sets/dev_with_labels.csv"),
    (EVAL_ROOT, f"{EVAL_ROOT}/sets/test.csv"),
]

# The three SSL bundles worth starting with. WavLM base+/large are the strongest speaker-aware
# SSL models; XLSR-300M is multilingual, so accent variation is signal in its space rather
# than nuisance -- aimed squarely at acc_sim, the weaker of the two targets.
SUGGESTED_SSL = ["WAVLM_BASE_PLUS", "WAVLM_LARGE", "WAV2VEC2_XLSR_300M"]


def collect_wavs():
    """basename -> absolute path, over every wav in train/dev/test."""
    paths = {}
    for root, csv_path in SOURCES:
        with open(csv_path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                for col in ("wav_a_path", "wav_b_path"):
                    rel = row[col]
                    base = os.path.basename(rel)
                    if base in paths:
                        continue
                    full = os.path.join(root, rel)
                    if os.path.exists(full):
                        paths[base] = os.path.abspath(full)
    return paths


def load_wav(path):
    """Load and resample to 16 kHz mono.

    The corpus is 24 kHz. Every SSL bundle and every registry encoder expects 16 kHz, so
    skipping this silently feeds audio at 1.5x speed -- it does not error, it just produces
    quietly wrong embeddings. ../unified/finetune.py does the same resample.
    """
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    return wav  # (1, T)


class SVBackend:
    """A frozen encoder from ../unified/encoders.py."""

    def __init__(self, name, device):
        from encoders import build_encoder

        self.model = build_encoder(name).to(device).eval()
        self.device = device
        self.dims = {"": self.model.output_dim}

    @torch.inference_mode()
    def __call__(self, wav):
        emb = self.model(wav.to(self.device), None)
        return {"": emb.squeeze(0).float().cpu().numpy()}


class SSLBackend:
    """A torchaudio.pipelines SSL bundle, mean-pooled over time, one entry per layer."""

    def __init__(self, bundle_name, layers, device):
        import torchaudio.pipelines as pipelines

        if not hasattr(pipelines, bundle_name):
            raise ValueError(f"torchaudio.pipelines has no bundle '{bundle_name}'")
        bundle = getattr(pipelines, bundle_name)
        if bundle.sample_rate != TARGET_SR:
            raise ValueError(f"{bundle_name} expects {bundle.sample_rate} Hz, not {TARGET_SR}")
        self.model = bundle.get_model().to(device).eval()
        self.device = device
        self.layers = layers
        # One probe forward to learn the layer count and width.
        with torch.inference_mode():
            feats, _ = self.model.extract_features(torch.zeros(1, TARGET_SR, device=device))
        self.n_layers = len(feats)
        width = feats[-1].shape[-1]
        self.dims = {self._tag(l): width for l in layers}
        logging.info(f"{bundle_name}: {self.n_layers} layers, width {width}, taking {layers}")

    def _tag(self, layer):
        return f"l{self.n_layers if layer == -1 else layer}"

    @torch.inference_mode()
    def __call__(self, wav):
        feats, _ = self.model.extract_features(wav.to(self.device))
        out = {}
        for l in self.layers:
            h = feats[-1] if l == -1 else feats[l - 1]  # 1-indexed layers, -1 = last
            out[self._tag(l)] = h.squeeze(0).mean(dim=0).float().cpu().numpy()
        return out


def build_backend(spec, ssl_layers, device):
    kind, _, name = spec.partition(":")
    if kind == "sv":
        return name, SVBackend(name, device)
    if kind == "ssl":
        return name, SSLBackend(name, ssl_layers, device)
    raise ValueError(f"Encoder spec must start with 'sv:' or 'ssl:', got '{spec}'")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--encoders", nargs="+", default=["sv:ecapa-voxceleb"],
                   help="Specs like sv:eres2netv2 or ssl:WAVLM_LARGE.")
    p.add_argument("--ssl-layers", nargs="+", type=int, default=[4, 8, -1],
                   help="1-indexed transformer layers to keep for SSL bundles; -1 = last.")
    p.add_argument("--outdir", default="egs/features")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="Debug: only this many wavs.")
    p.add_argument("--list", action="store_true", help="Show available encoders and exit.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s (%(module)s:%(lineno)d) %(message)s")

    if args.list:
        from encoders import ENCODER_REGISTRY
        import torchaudio.pipelines as pipelines
        print("sv: encoders (../unified/encoders.py)")
        for k in ENCODER_REGISTRY:
            print(f"    sv:{k}")
        print("\nssl: bundles (torchaudio.pipelines)")
        for k in sorted(x for x in dir(pipelines)
                        if any(t in x for t in ("WAVLM", "HUBERT", "WAV2VEC2"))):
            star = "  <-- suggested" if k in SUGGESTED_SSL else ""
            print(f"    ssl:{k}{star}")
        return

    os.makedirs(args.outdir, exist_ok=True)
    wavs = collect_wavs()
    names = sorted(wavs)
    if args.limit:
        names = names[: args.limit]
    logging.info(f"{len(names)} unique wavs, device={args.device}")

    for spec in args.encoders:
        tag, backend = build_backend(spec, args.ssl_layers, args.device)

        targets = {sub: os.path.join(args.outdir, f"{tag}{'_' + sub if sub else ''}.npz")
                   for sub in backend.dims}
        if not args.overwrite and all(os.path.exists(f) for f in targets.values()):
            logging.info(f"{tag}: all outputs present, skipping (use --overwrite to redo)")
            continue

        store = {sub: {} for sub in backend.dims}
        t0 = time.time()
        for i, base in enumerate(names, 1):
            vecs = backend(load_wav(wavs[base]))
            for sub, v in vecs.items():
                store[sub][base] = v
            if i % 500 == 0 or i == len(names):
                rate = i / (time.time() - t0)
                logging.info(f"  {tag}: {i}/{len(names)}  {rate:.1f} wav/s  "
                             f"eta {(len(names) - i) / max(rate, 1e-6):.0f}s")

        for sub, path in targets.items():
            np.savez_compressed(path, **store[sub])
            dim = backend.dims[sub]
            logging.info(f"  wrote {path}  ({len(store[sub])} vectors, dim {dim}, "
                         f"{os.path.getsize(path) / 1e6:.1f} MB)")

        del backend
        if args.device == "cuda":
            torch.cuda.empty_cache()

    logging.info("done")


if __name__ == "__main__":
    main()
