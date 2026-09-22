#!/usr/bin/env python3
"""Compare speaker–accent association in human MOS and model predictions."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DEFAULT_LABELS = (
    ROOT / "track3/baseline/data/vmc2026_track3_eval_phase_distro_v3_syn/sets/"
    "vmc2026_track3_test_with_labels.csv"
)
DEFAULT_PREDICTIONS = ROOT / "track3/weak/egs/submission_final/deep0-weak16_train_plus_dev"


def read_scores(path, columns):
    """Index by both waveform paths; never rely on CSV row order."""
    records = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = {"wav_a_path", "wav_b_path", *columns} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for line, row in enumerate(reader, start=2):
            key = (row["wav_a_path"], row["wav_b_path"])
            if not all(key) or key in records:
                raise ValueError(f"{path}, line {line}: empty or duplicate pair")
            for column in columns:
                value = float(row[column])
                if not np.isfinite(value) or not 1 <= value <= 5:
                    raise ValueError(f"{path}, line {line}: {column} must be in [1, 5]")
                row[column] = value
            records[key] = row
    if not records:
        raise ValueError(f"{path}: no scores found")
    return records


def align_scores(labels_path, speaker_path, accent_path):
    labels = read_scores(labels_path, ["spk_sim", "acc_sim"])
    speaker = read_scores(speaker_path, ["pred_spk_sim"])
    accent = read_scores(accent_path, ["pred_acc_sim"])
    for path, predictions in [(speaker_path, speaker), (accent_path, accent)]:
        missing = labels.keys() - predictions.keys()
        extra = predictions.keys() - labels.keys()
        if missing or extra:
            raise ValueError(f"{path}: {len(missing)} missing and {len(extra)} extra pairs")
    return [{
        "system_id": row.get("system_id", ""),
        "utterance_id": row.get("utterance_id", ""),
        "wav_a_path": key[0], "wav_b_path": key[1],
        "human_speaker": row["spk_sim"], "human_accent": row["acc_sim"],
        "model_speaker": speaker[key]["pred_spk_sim"],
        "model_accent": accent[key]["pred_acc_sim"],
    } for key, row in labels.items()]


def correlation(x, y):
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(spearmanr(x, y).statistic)


def summarize(values):
    speaker, accent = values.T
    return {
        "speaker_accent_srcc": correlation(speaker, accent),
        "mean_absolute_speaker_accent_gap": float(np.abs(accent - speaker).mean()),
        "mean_accent_minus_speaker": float((accent - speaker).mean()),
        "speaker_mean": float(speaker.mean()), "accent_mean": float(accent.mean()),
        "speaker_sd_across_pairs": float(speaker.std(ddof=1)),
        "accent_sd_across_pairs": float(accent.std(ddof=1)),
    }


def make_plot(human, model, output_dir, model_label):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 10,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharex=True, sharey=True,
                             layout="constrained")
    for ax, values, title in [
        (axes[0], human, "(a) Human mean ratings"),
        (axes[1], model, f"(b) {model_label}"),
    ]:
        # Plot each pair at its actual scores; transparency reveals overlap.
        ax.scatter(values[:, 0], values[:, 1], s=13, color="#0072B2",
                   alpha=0.4, edgecolors="none", zorder=3)
        ax.plot([1, 5], [1, 5], "--", color="#888888", linewidth=1, zorder=2)
        ax.set(title=title, xlabel="Speaker similarity", xlim=(0.95, 5.05),
               ylim=(0.95, 5.05), xticks=[1, 2, 3, 4, 5], yticks=[1, 2, 3, 4, 5],
               aspect="equal")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Accent similarity")
    for extension in ["pdf", "png", "svg"]:
        path = output_dir / f"human_vs_model.{extension}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--speaker-predictions", type=Path,
                        default=DEFAULT_PREDICTIONS / "test_spk_sim.csv")
    parser.add_argument("--accent-predictions", type=Path,
                        default=DEFAULT_PREDICTIONS / "test_acc_sim.csv")
    parser.add_argument("--model-label", default="Submitted weak-16 predictions")
    parser.add_argument("--output-dir", type=Path, default=HERE)
    args = parser.parse_args()
    rows = align_scores(args.labels, args.speaker_predictions, args.accent_predictions)
    human = np.array([[r["human_speaker"], r["human_accent"]] for r in rows])
    model = np.array([[r["model_speaker"], r["model_accent"]] for r in rows])
    summary = {
        "n_pairs": len(rows), "model_label": args.model_label,
        "human": summarize(human), "model": summarize(model),
        "prediction_srcc_against_mos": {
            "speaker": correlation(human[:, 0], model[:, 0]),
            "accent": correlation(human[:, 1], model[:, 1]),
        },
        "source_files": {k: str(getattr(args, k).resolve()) for k in
                         ["labels", "speaker_predictions", "accent_predictions"]},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "human_vs_model.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(summary, indent=2, allow_nan=False))
    make_plot(human, model, args.output_dir, args.model_label)


if __name__ == "__main__":
    main()
