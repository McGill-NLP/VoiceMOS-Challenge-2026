#!/usr/bin/env python3
"""Plot the association between speaker and accent ratings for each listener."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


HERE = Path(__file__).resolve().parent
DEFAULT_CSV = (
    HERE.parent.parent / "track3/baseline/data/"
    "vmc2026_track3_train_phase_distro_v3_syn/sets/train.csv"
)


def listener_statistics(csv_path):
    """Use paired individual ratings, without averaging across listeners."""
    ratings = defaultdict(list)
    seen = set()
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"listener_id", "wav_a_path", "wav_b_path", "spk_sim", "acc_sim"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing CSV columns: {sorted(missing)}")
        for line, row in enumerate(reader, start=2):
            listener = row["listener_id"].strip()
            key = (listener, row["wav_a_path"], row["wav_b_path"])
            if not listener or key in seen:
                raise ValueError(f"Line {line}: missing listener or duplicate listener/pair")
            seen.add(key)
            scores = (float(row["spk_sim"]), float(row["acc_sim"]))
            if any(not np.isfinite(s) or s < 1 or s > 5 for s in scores):
                raise ValueError(f"Line {line}: expected finite ratings in [1, 5]")
            ratings[listener].append(scores)
    if not ratings:
        raise ValueError("No listener ratings found")

    summaries = []
    for listener, pairs in ratings.items():
        speaker, accent = np.asarray(pairs).T
        # Correlation is undefined if either dimension is constant.
        rho = (
            float(spearmanr(speaker, accent).statistic)
            if len(set(speaker)) > 1 and len(set(accent)) > 1
            else float("nan")
        )
        identical = int(np.count_nonzero(speaker == accent))
        summaries.append({
            "listener_id": listener,
            "n_ratings": len(pairs),
            "speaker_accent_srcc": rho,
            "n_identical": identical,
            "identical_percent": 100 * identical / len(pairs),
            "mean_absolute_difference": float(np.mean(np.abs(speaker - accent))),
        })
    # Both panels share one ordering; undefined correlations go last.
    summaries.sort(key=lambda r: (
        not np.isfinite(r["speaker_accent_srcc"]),
        -r["speaker_accent_srcc"] if np.isfinite(r["speaker_accent_srcc"]) else 0,
        r["listener_id"],
    ))
    return summaries


def make_plot(rows, output_dir):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(
        1, 2, figsize=(7.2, max(3.0, 0.21 * len(rows) + 1.1)),
        sharey=True, layout="constrained",
    )
    positions = np.arange(len(rows))
    correlations = np.array([r["speaker_accent_srcc"] for r in rows])
    percentages = np.array([r["identical_percent"] for r in rows])
    finite = correlations[np.isfinite(correlations)]
    lower = -1.02 if len(finite) and finite.min() < 0 else -0.02

    for ax, values, color, title, label, limits, ticks, pattern, offset in [
        (axes[0], correlations, "#0072B2", "(a) Association between ratings",
         "Speaker–accent Spearman correlation", (lower, 1.18),
         [-1, -0.5, 0, 0.5, 1] if lower < -1 else [0, 0.25, 0.5, 0.75, 1],
         ".3f", 0.025),
        (axes[1], percentages, "#D55E00", "(b) Identical ratings",
         "Identical speaker and accent scores (%)", (-2, 118),
         [0, 25, 50, 75, 100], ".1f", 2.5),
    ]:
        for y in positions[::2]:
            ax.axhspan(y - 0.5, y + 0.5, color="#F3F5F7", zorder=0)
        ax.scatter(values, positions, s=25, color=color, zorder=3)
        for y, value in zip(positions, values):
            if np.isfinite(value):
                ax.text(value + offset, y, format(value, pattern), va="center", fontsize=8)
            else:
                ax.text(0.02, y, "undefined", va="center", fontsize=8, color="#666666")
        ax.set(title=title, xlabel=label, xlim=limits, xticks=ticks)
        ax.grid(axis="x", color="#DDDDDD", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)

    axes[0].set_yticks(positions, [
        f"{r['listener_id']}  ({r['n_ratings']:,})" for r in rows
    ])
    axes[0].set_ylabel("Listener ID (number of rated pairs)")
    axes[0].set_ylim(len(rows) - 0.5, -0.5)
    for extension in ["pdf", "png", "svg"]:
        path = output_dir / f"listener_similarity.{extension}"
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
    rows = listener_statistics(args.csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "listener_similarity.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{sum(r['n_ratings'] for r in rows):,} ratings from {len(rows)} listeners")
    print(summary_path)
    make_plot(rows, args.output_dir)


if __name__ == "__main__":
    main()
