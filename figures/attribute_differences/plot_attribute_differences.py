#!/usr/bin/env python3
"""Plot recovery of the accent-minus-speaker MOS difference on held-out pairs."""

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import ScalarFormatter
import numpy as np

# Reuse validation and pair alignment; no generated files are required as inputs.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "human_vs_model"))
from plot_human_vs_model import DEFAULT_LABELS, DEFAULT_PREDICTIONS, align_scores, correlation


def difference_statistics(rows):
    # Restore mathematical ties such as 4.6 - 4.4 and 3.2 - 3.0 before ranking.
    human = np.round([r["human_accent"] - r["human_speaker"] for r in rows], 12)
    model = np.round([r["model_accent"] - r["model_speaker"] for r in rows], 12)
    nonzero = human != 0
    correct = np.sign(human[nonzero]) == np.sign(model[nonzero])
    summary = {
        "n_pairs": len(rows),
        "difference_definition": "accent similarity minus speaker similarity",
        "difference_rounding_decimals": 12,
        "gap_srcc": correlation(human, model),
        "gap_mae": float(np.abs(model - human).mean()),
        "zero_gap_baseline_mae": float(np.abs(human).mean()),
        "gap_rmse": float(np.sqrt(np.mean((model - human) ** 2))),
        "human_mean_absolute_gap": float(np.abs(human).mean()),
        "model_mean_absolute_gap": float(np.abs(model).mean()),
        "human_equal_score_pairs": int((~nonzero).sum()),
        "human_accent_higher_pairs": int((human > 0).sum()),
        "human_speaker_higher_pairs": int((human < 0).sum()),
        "direction_agreement_denominator": int(nonzero.sum()),
        "direction_agreement_n_correct": int(correct.sum()),
        "direction_agreement_percent": float(100 * correct.mean()) if nonzero.any() else None,
    }
    for row, observed, predicted in zip(rows, human, model):
        row["human_accent_minus_speaker"] = float(observed)
        row["model_accent_minus_speaker"] = float(predicted)
        row["absolute_gap_error"] = float(abs(predicted - observed))
    return human, model, summary


def make_plot(human, model, summary, output_dir, model_label):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 10,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })
    fig, ax = plt.subplots(figsize=(4.9, 4.9), layout="constrained")
    limit = max(0.5, np.ceil(max(np.max(np.abs(human)), np.max(np.abs(model))) * 2) / 2)
    density = ax.hexbin(human, model, gridsize=24,
                        extent=(-limit, limit, -limit, limit), mincnt=1,
                        cmap="viridis", linewidths=0)
    if int(density.get_array().sum()) != len(human):
        raise RuntimeError("Hexbin did not account for every pair")
    max_count = max(2, float(density.get_array().max()))
    density.set_norm(LogNorm(vmin=1, vmax=max_count))
    ax.plot([-limit, limit], [-limit, limit], "--", color="#888888", linewidth=1.2,
            label="Exact recovery (y = x)")
    ax.axhline(0, color="#999999", linewidth=0.7, zorder=0)
    ax.axvline(0, color="#999999", linewidth=0.7, zorder=0)
    ax.set(title=model_label,
           xlabel="Human MOS difference (accent − speaker)",
           ylabel="Predicted difference (accent − speaker)",
           xlim=(-limit - 0.05, limit + 0.05),
           ylim=(-limit - 0.05, limit + 0.05), aspect="equal")
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    rho = summary["gap_srcc"]
    rho_text = f"{rho:.3f}" if rho is not None else "undefined"
    agreement = summary["direction_agreement_percent"]
    agreement_text = f"{agreement:.1f}%" if agreement is not None else "undefined"
    ax.text(0, -0.23,
            f"{len(human):,} pairs; gap SRCC = {rho_text}; MAE = {summary['gap_mae']:.3f}\n"
            f"Direction agreement = {agreement_text} "
            f"({summary['direction_agreement_denominator']} non-tied pairs)",
            transform=ax.transAxes, va="top", fontsize=8)
    colorbar = fig.colorbar(density, ax=ax, shrink=0.8, pad=0.025)
    colorbar.set_label("Pairs per hexagon (log scale)", fontsize=9)
    colorbar.set_ticks([v for v in [1, 2, 5, 10, 20, 50, 100, 200, 500] if v <= max_count])
    colorbar.ax.yaxis.set_major_formatter(ScalarFormatter())
    colorbar.minorticks_off()
    for extension in ["pdf", "png", "svg"]:
        path = output_dir / f"attribute_differences.{extension}"
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
    parser.add_argument("--model-label", default="Submitted weak-16: recovery of attribute differences")
    parser.add_argument("--output-dir", type=Path, default=HERE)
    args = parser.parse_args()
    rows = align_scores(args.labels, args.speaker_predictions, args.accent_predictions)
    human, model, summary = difference_statistics(rows)
    summary["model_label"] = args.model_label
    summary["source_files"] = {k: str(getattr(args, k).resolve()) for k in
                               ["labels", "speaker_predictions", "accent_predictions"]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "attribute_differences.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(summary, indent=2, allow_nan=False))
    make_plot(human, model, summary, args.output_dir, args.model_label)


if __name__ == "__main__":
    main()
