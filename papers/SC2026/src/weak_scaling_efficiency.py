#!/usr/bin/env python3
"""Weak Scaling — Parallel efficiency line plot.

Standalone single-column IEEE subfigure. Simple line plot of parallel
efficiency vs. rank count, with an in-plot reference line at 100% (ideal
weak scaling) and a corner pill carrying the three headline numbers
(efficiency at peak, sustained per-tick, peak GPU count).

The chart type is a clean efficiency line plot — the per-phase
decomposition lives in `phase_breakdown.py` (the merged 2-panel figure),
so this plot focuses on the efficiency claim alone. Mirrors the
`strong_scaling_speedup.py` figure (line plot with corner pill) for
visual symmetry across the two scaling experiments.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _scaling_common import (
    FIG_W, FIG_H, PHASE_COLORS, COLOR_IDEAL, IDEAL_LW,
    FONT_LABEL, FONT_TICK, FONT_LEG, FONT_PILL,
    read_csv, derive_metrics, get_default_paths,
    setup_log2_xaxis, save_figure, add_cli_args,
)


def plot(gpus, avg, metrics, out_pdf, out_png, dpi):
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    g_arr = np.array(gpus)
    eff = np.array([metrics[g]["efficiency"] for g in gpus])
    # Mean steady-state per-tick rate across the sweep — the second
    # headline number after parallel efficiency.
    mean_tick_ms = float(np.mean([avg[g]["mean_tick_time"] for g in gpus]) * 1000)

    # Ideal-weak-scaling reference at 100% (matches the dashed-gray
    # convention used by strong_scaling_speedup.py for the ideal line).
    line_ideal, = ax.plot(g_arr, [100.0] * len(g_arr),
                          "--", color=COLOR_IDEAL, linewidth=IDEAL_LW,
                          label="Ideal", zorder=2)
    # Measured efficiency line. Reuses the project's "Steady State" teal
    # so it ties visually to the (b) panel's Steady State segment in the
    # Weak A subfigure pair.
    line_measured, = ax.plot(g_arr, eff, "o-",
                             color=PHASE_COLORS["Steady State"],
                             linewidth=1.8, markersize=5,
                             label="Measured efficiency", zorder=3)

    # Headline pill in the lower-left free quadrant (the data line sits at
    # ~96–101%, so the bottom half of the plot is empty across all x).
    peak = gpus[-1]
    peak_eff = metrics[peak]["efficiency"]
    ax.text(
        0.025, 0.04,
        f"{peak_eff:.1f}% efficiency\n"
        f"~{mean_tick_ms:.0f} ms/tick sustained\n"
        f"at {peak} GPUs",
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=FONT_PILL,
        color="#1A202C",
        zorder=6,
    )

    # log2 GPU-count x-axis matching strong_scaling_speedup.py for visual
    # symmetry across the (a) panels of the two subfigure pairs.
    setup_log2_xaxis(ax, gpus)
    ax.set_ylabel("Parallel Efficiency (%)", fontsize=FONT_LABEL)
    ax.set_ylim(90, 105)
    # Legend INSIDE the plot — upper-right free corner (above the data line
    # near the dashed Ideal reference).
    ax.legend(fontsize=FONT_LEG, loc="upper right",
              framealpha=0.9, handletextpad=0.4)
    ax.grid(True, which="both", alpha=0.3)
    ax.tick_params(axis="y", labelsize=FONT_TICK)

    plt.tight_layout()
    save_figure(fig, out_pdf, out_png, dpi)
    plt.close(fig)


def main():
    csv_dir, figs_dir, _ = get_default_paths(__file__)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_cli_args(parser, "weak_scaling_efficiency")
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir) if args.csv_dir else csv_dir
    figs_dir = Path(args.figs_dir) if args.figs_dir else figs_dir
    figs_dir.mkdir(parents=True, exist_ok=True)

    gpus, avg = read_csv(csv_dir / "weak_scaling.csv")
    metrics = derive_metrics(gpus, avg, baseline_gpu=gpus[0], scaling_kind="weak")

    out_pdf = figs_dir / "weak_scaling_efficiency.pdf"
    out_png = figs_dir / "weak_scaling_efficiency.png"
    plot(gpus, avg, metrics, out_pdf, out_png, args.dpi)
    print(f"Wrote {out_pdf} and {out_png}")


if __name__ == "__main__":
    main()
