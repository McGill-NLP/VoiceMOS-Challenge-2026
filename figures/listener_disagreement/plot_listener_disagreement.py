#!/usr/bin/env python3
"""Compare within-pair listener rating dispersion for speaker and accent."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_CSV = (
    HERE.parent.parent / "track3/baseline/data/"
    "vmc2026_track3_train_phase_distro_v3_syn/sets/train.csv"
)
SPEAKER_COLOR = "#0072B2"
ACCENT_COLOR = "#D55E00"


def pair_statistics(csv_path):
    """Sample SD across listeners; use the same listeners for both targets."""
    pairs = defaultdict(list)
    metadata = {}
    seen = set()
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"listener_id", "wav_a_path", "wav_b_path", "spk_sim", "acc_sim"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing CSV columns: {sorted(missing)}")
        for line, row in enumerate(reader, start=2):
            key = (row["wav_a_path"], row["wav_b_path"])
            listener = row["listener_id"].strip()
            rating_key = (key, listener)
            if not listener or rating_key in seen:
                raise ValueError(f"Line {line}: missing listener or duplicate listener/pair")
            scores = (float(row["spk_sim"]), float(row["acc_sim"]))
            if any(not np.isfinite(v) or not 1 <= v <= 5 for v in scores):
                raise ValueError(f"Line {line}: expected finite ratings in [1, 5]")
            seen.add(rating_key)
            pairs[key].append(scores)
            metadata[key] = (row.get("system_id", ""), row.get("utterance_id", ""))
    if not pairs:
        raise ValueError("No listener ratings found")

    rows = []
    for key, ratings in sorted(pairs.items()):
        if len(ratings) < 2:
            raise ValueError(f"Need at least two listeners to compute sample SD: {key}")
        values = np.asarray(ratings)
        means = values.mean(axis=0)
        sd = values.std(axis=0, ddof=1)
        rows.append({
            "system_id": metadata[key][0],
            "utterance_id": metadata[key][1],
            "wav_a_path": key[0],
            "wav_b_path": key[1],
            "n_ratings": len(ratings),
            "speaker_mean": float(means[0]),
            "accent_mean": float(means[1]),
            "speaker_sd": float(sd[0]),
            "accent_sd": float(sd[1]),
            "accent_minus_speaker_sd": float(sd[1] - sd[0]),
        })
    return rows


def summarize(rows):
    speaker = np.array([r["speaker_sd"] for r in rows])
    accent = np.array([r["accent_sd"] for r in rows])
    delta = accent - speaker
    equal = np.isclose(delta, 0, atol=1e-12, rtol=0)
    return {
        "n_pairs": len(rows),
        "n_ratings": sum(r["n_ratings"] for r in rows),
        "pairs_by_number_of_ratings": dict(sorted(Counter(r["n_ratings"] for r in rows).items())),
        "sd_definition": "sample standard deviation across listeners (ddof=1)",
        "aggregation": "each waveform pair has equal weight; all listeners retained",
        "speaker_mean_sd": float(speaker.mean()),
        "accent_mean_sd": float(accent.mean()),
        "speaker_median_sd": float(np.median(speaker)),
        "accent_median_sd": float(np.median(accent)),
        "mean_accent_minus_speaker_sd": float(delta.mean()),
        "median_accent_minus_speaker_sd": float(np.median(delta)),
        "accent_higher_percent": float(100 * np.mean((delta > 0) & ~equal)),
        "equal_percent": float(100 * np.mean(equal)),
        "accent_lower_percent": float(100 * np.mean((delta < 0) & ~equal)),
    }


def make_plot(rows, summary, output_dir):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })
    speaker = np.array([r["speaker_sd"] for r in rows])
    accent = np.array([r["accent_sd"] for r in rows])
    delta = accent - speaker
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), layout="constrained")
    sd_limit = max(2.5, np.ceil(max(speaker.max(), accent.max()) * 2) / 2)
    delta_limit = max(2.0, np.ceil(np.max(np.abs(delta)) * 2) / 2)

    # An empirical CDF avoids smoothing the discrete rating distributions.
    for values, color, name in [
        (speaker, SPEAKER_COLOR, "Speaker"), (accent, ACCENT_COLOR, "Accent")
    ]:
        unique, counts = np.unique(values, return_counts=True)
        x = np.r_[0, unique, sd_limit]
        y = np.r_[0, 100 * np.cumsum(counts) / len(values), 100]
        axes[0].step(x, y, where="post", color=color, linewidth=1.8,
                     label=f"{name} (mean SD = {values.mean():.3f})")
    axes[0].set(title="(a) Listener disagreement across pairs",
                xlabel="Within-pair rating standard deviation",
                ylabel="Pairs at or below this SD (%)",
                xlim=(0, sd_limit), ylim=(0, 103))
    axes[0].legend(loc="lower right", fontsize=8, frameon=False)

    # Center a bin on zero so equal-dispersion pairs do not straddle bins.
    bins = np.arange(-delta_limit - 0.05, delta_limit + 0.06, 0.1)
    axes[1].hist(delta, bins=bins, weights=np.full(len(delta), 100 / len(delta)),
                 color="#8F9CAA", edgecolor="white", linewidth=0.4)
    axes[1].axvline(0, color="#333333", linewidth=1)
    axes[1].axvline(delta.mean(), color=ACCENT_COLOR, linestyle="--", linewidth=1.5)
    axes[1].set(title="(b) Difference for the same pair",
                xlabel="Accent SD − speaker SD",
                ylabel="Pairs (%)", xlim=(-delta_limit - 0.05, delta_limit + 0.05))
    axes[1].text(
        0.03, 0.96,
        f"Mean ΔSD: {delta.mean():+.3f}\n"
        f"Accent higher: {summary['accent_higher_percent']:.2f}%\n"
        f"Equal: {summary['equal_percent']:.2f}%\n"
        f"Accent lower: {summary['accent_lower_percent']:.2f}%",
        transform=axes[1].transAxes, ha="left", va="top", fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9},
    )
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#E3E6E9", linewidth=0.6)
        ax.set_axisbelow(True)
    for extension in ["pdf", "png", "svg"]:
        path = output_dir / f"listener_disagreement.{extension}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV,
                        help="Individual listener ratings (default: official training CSV)")
    parser.add_argument("--output-dir", type=Path, default=HERE,
                        help="Output directory (default: this script's directory)")
    args = parser.parse_args()
    rows = pair_statistics(args.csv)
    summary = summarize(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "listener_disagreement.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary["source_csv"] = str(args.csv.resolve())
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(json.dumps(summary, indent=2))
    make_plot(rows, summary, args.output_dir)


if __name__ == "__main__":
    main()
