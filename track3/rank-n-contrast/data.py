#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dataset and collater for the Rank-N-Contrast experiments.

Follows the official baseline's data handling (per-pair score averaging over
listeners, repetitive padding) so that the RNC arm and the baseline arm differ
only in the training objective.

One addition: because `wav_b` is always drawn from the single reference system
(sys019, 137 utterances), a batch of B pairs contains at most 137 distinct
b-side waveforms. The collater deduplicates them and returns an index map, so
the encoder runs on `U <= B` reference waveforms instead of `B`. At the large
batch sizes RNC wants this is a meaningful saving.
"""

import csv
import logging
import os

import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

TARGET_SR = 16000


def read_rows(csv_path):
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def aggregate_pairs(rows, target_metric):
    """Collapses listener-wise rows into one row per (wav_a, wav_b) pair.

    The score becomes the mean over listeners, matching the baseline. Rows
    without the target metric (e.g. the unlabelled official dev.csv) are kept
    with a score of None so the same code path can drive inference.
    """
    agg = {}
    for row in rows:
        key = (row["wav_a_path"], row["wav_b_path"])
        if key not in agg:
            agg[key] = {
                "system_id": row.get("system_id", ""),
                "utterance_id": row.get("utterance_id", ""),
                "wav_a_path": row["wav_a_path"],
                "wav_b_path": row["wav_b_path"],
                "scores": [],
            }
        val = row.get(target_metric, "")
        if val is not None and str(val).strip() != "":
            agg[key]["scores"].append(float(val))

    out = []
    for item in agg.values():
        scores = item.pop("scores")
        item["score"] = sum(scores) / len(scores) if scores else None
        item["n_ratings"] = len(scores)
        out.append(item)
    return out


class PairDataset(Dataset):
    """Yields (wav_a, wav_b, score) for each unique audio pair."""

    def __init__(self, data_root, rows, target_metric, max_audio_sec=None, train=True):
        self.data_root = data_root
        self.max_samples = int(max_audio_sec * TARGET_SR) if max_audio_sec else None
        self.train = train
        self.items = aggregate_pairs(rows, target_metric)
        n_scored = sum(1 for it in self.items if it["score"] is not None)
        logging.info(
            f"Aggregated {len(rows)} rows into {len(self.items)} unique pairs "
            f"({n_scored} with a {target_metric} score)."
        )

    def __len__(self):
        return len(self.items)

    def _load(self, rel_path):
        wav, sr = torchaudio.load(os.path.join(self.data_root, rel_path))
        if sr != TARGET_SR:
            wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
        wav = wav.squeeze(0)
        if self.max_samples and wav.shape[0] > self.max_samples:
            extra = wav.shape[0] - self.max_samples
            # Random crop while training, deterministic centre crop otherwise.
            start = torch.randint(0, extra + 1, (1,)).item() if self.train else extra // 2
            wav = wav[start:start + self.max_samples]
        return wav

    def __getitem__(self, idx):
        item = self.items[idx]
        return {
            "wav_a": self._load(item["wav_a_path"]),
            "wav_b": self._load(item["wav_b_path"]),
            "wav_b_path": item["wav_b_path"],
            "score": item["score"],
            "system_id": item["system_id"],
            "utterance_id": item["utterance_id"],
            "wav_a_path": item["wav_a_path"],
        }


def _pad_repetitive(feats):
    """Repetitive padding (MBNet, arXiv:2103.00110): tile each waveform up to the
    batch max instead of zero-padding, so batch-norm statistics stay clean."""
    lengths = torch.tensor([f.shape[0] for f in feats], dtype=torch.long)
    max_len = int(lengths.max().item())
    padded = []
    for feat in feats:
        this_len = feat.shape[0]
        if this_len == 0:
            padded.append(torch.zeros(max_len, dtype=feat.dtype))
            continue
        dup_times = max_len // this_len
        remain = max_len - this_len * dup_times
        chunks = [feat] * dup_times
        if remain > 0:
            chunks.append(feat[:remain])
        padded.append(torch.cat(chunks, dim=0))
    return torch.stack(padded, dim=0), lengths


def _pad_zero(feats):
    lengths = torch.tensor([f.shape[0] for f in feats], dtype=torch.long)
    return pad_sequence(feats, batch_first=True, padding_value=0.0), lengths


class PairCollater:
    def __init__(self, padding_mode="repetitive", dedup_reference=True):
        if padding_mode not in ("repetitive", "zero_padding"):
            raise ValueError(f"Unknown padding mode: {padding_mode}")
        self.pad = _pad_repetitive if padding_mode == "repetitive" else _pad_zero
        self.dedup_reference = dedup_reference

    def __call__(self, batch):
        out = {}
        out["wav_a"], out["wav_a_lengths"] = self.pad([b["wav_a"] for b in batch])

        if self.dedup_reference:
            # Collapse duplicate reference waveforms; b_index maps each pair back
            # to its row in the deduplicated reference batch.
            seen, uniq_wavs, b_index = {}, [], []
            for b in batch:
                path = b["wav_b_path"]
                if path not in seen:
                    seen[path] = len(uniq_wavs)
                    uniq_wavs.append(b["wav_b"])
                b_index.append(seen[path])
            out["wav_b"], out["wav_b_lengths"] = self.pad(uniq_wavs)
            out["b_index"] = torch.tensor(b_index, dtype=torch.long)
        else:
            out["wav_b"], out["wav_b_lengths"] = self.pad([b["wav_b"] for b in batch])
            out["b_index"] = torch.arange(len(batch), dtype=torch.long)

        scores = [b["score"] for b in batch]
        out["score"] = (
            torch.tensor(scores, dtype=torch.float32)
            if all(s is not None for s in scores)
            else None
        )
        for key in ("system_id", "utterance_id", "wav_a_path", "wav_b_path"):
            out[key] = [b[key] for b in batch]
        return out


def build_loader(
    data_root,
    csv_path,
    target_metric,
    batch_size,
    shuffle,
    num_workers=4,
    max_audio_sec=None,
    train=True,
    drop_last=False,
    padding_mode="repetitive",
):
    dataset = PairDataset(
        data_root, read_rows(csv_path), target_metric,
        max_audio_sec=max_audio_sec, train=train,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=PairCollater(padding_mode=padding_mode),
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=True,
    )
    return dataset, loader
