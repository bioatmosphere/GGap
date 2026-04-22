#!/usr/bin/env python3
"""Strong Scaling — Combined speedup + cross-rank communication intensity.

Standalone single-column IEEE subfigure for the fixed 2,048-site strong-scaling
experiment. Combines what used to be two separate plots (speedup and efficiency
+ cross-rank) into one twin-axis figure:

  - Left Y-axis (log2): speedup vs. 8-GPU baseline. Two lines: measured (green)
    and ideal-linear (gray dashed). The vertical gap between them at any GPU
    count visually IS the parallel-efficiency loss.
  - Right Y-axis (linear, 0-100%): cross-rank edge fraction (orange). Climbs
    from 1.2% to 75% as the per-GPU partition shrinks.

The colored y-axis labels (green left, orange right) carry the
line-color → metric mapping directly, so the figure does not draw a
legend or in-plot text annotation — headline numbers (peak speedup,
peak efficiency, cross-rank range) live in the LaTeX caption instead.

This single plot replaces the previous pair `strong_scaling_speedup` (just
speedup) + `strong_scaling_efficiency` (efficiency + cross-rank) — they were
mathematically equivalent (efficiency = speedup / ideal × 100), so two figures
were redundant.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _scaling_common import (
    FIG_W, FIG_H, COLOR_STRONG, COLOR_IDEAL, COLOR_XRANK, EXPECTED_STRONG,
    FONT_LABEL, FONT_TICK, IDEAL_LW,
    read_csv, derive_metrics, cross_rank_pct, get_default_paths,
    setup_log2_xaxis, save_figure, add_cli_args,
)


def plot(gpus, avg, metrics, out_pdf, out_png, dpi):
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    g_arr = np.array(gpus)
    speedup = np.array([metrics[g]["speedup"] for g in gpus])
    ideal = g_arr / g_arr[0]
    xrank = np.array([
        cross_rank_pct(int(avg[g]["grid_height"]), int(avg[g]["sites_per_gpu"]))
        for g in gpus
    ])

    # Left axis: log2 speedup with ideal reference
    # Ideal line uses the SHARED COLOR_IDEAL + IDEAL_LW from _scaling_common
    # so it matches the dashed reference line in weak_scaling_efficiency.
    line_ideal,    = ax.plot(g_arr, ideal, "--", color=COLOR_IDEAL, linewidth=IDEAL_LW,
                             label="Ideal", zorder=2)
    line_measured, = ax.plot(g_arr, speedup, "o-", color=COLOR_STRONG, linewidth=1.8,
                             markersize=6, label="Measured speedup", zorder=3)

    setup_log2_xaxis(ax, EXPECTED_STRONG)
    ax.set_yscale("log", base=2)
    ax.set_ylabel("Speedup vs. 8-GPU baseline", fontsize=FONT_LABEL, color=COLOR_STRONG)
    ax.tick_params(axis="y", labelsize=FONT_TICK, labelcolor=COLOR_STRONG)
    ax.grid(True, which="both", alpha=0.3)

    # Right axis: cross-rank edge fraction (linear, 0-100%)
    ax2 = ax.twinx()
    line_xrank, = ax2.plot(g_arr, xrank, "s--", color=COLOR_XRANK, linewidth=1.5,
                           markersize=5, alpha=0.85, label="Cross-rank %", zorder=2)
    ax2.set_ylabel("Cross-rank edges (%)", fontsize=FONT_LABEL, color=COLOR_XRANK)
    ax2.tick_params(axis="y", labelsize=FONT_TICK, labelcolor=COLOR_XRANK)
    ax2.set_ylim([0, 100])

    # No legend, no in-plot annotation: the colored y-axis labels (green
    # left = "Speedup vs. 8-GPU baseline", orange right = "Cross-rank
    # edges (%)") carry the line-color → metric mapping, and headline
    # numbers (12.4× speedup, 19.3% efficiency at 512 GPUs, cross-rank
    # 1.2% → 75%) live in the LaTeX caption.

    plt.tight_layout()
    save_figure(fig, out_pdf, out_png, dpi)
    plt.close(fig)


def main():
    csv_dir, figs_dir, _ = get_default_paths(__file__)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_cli_args(parser, "strong_scaling_speedup")
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir) if args.csv_dir else csv_dir
    figs_dir = Path(args.figs_dir) if args.figs_dir else figs_dir
    figs_dir.mkdir(parents=True, exist_ok=True)

    gpus, avg = read_csv(csv_dir / "strong_scaling.csv")
    metrics = derive_metrics(gpus, avg, baseline_gpu=gpus[0], scaling_kind="strong")

    out_pdf = figs_dir / "strong_scaling_speedup.pdf"
    out_png = figs_dir / "strong_scaling_speedup.png"
    plot(gpus, avg, metrics, out_pdf, out_png, args.dpi)
    print(f"Wrote {out_pdf} and {out_png}")


if __name__ == "__main__":
    main()
