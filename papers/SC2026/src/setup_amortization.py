#!/usr/bin/env python3
"""Setup vs. Simulation — Amortization Curve.

Standalone subfigure: setup-fraction-of-total-wall-time vs. simulation length
(log x), showing the break-even point at which one-time setup becomes <5% of
the run. Uses the largest measured Weak Scaling B configuration.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _scaling_common import (
    FIG_W, FIG_H, COLOR_WEAK_B, COLOR_REF,
    FONT_LABEL, FONT_TICK, FONT_LEG, FONT_ANNOT,
    read_csv, get_default_paths, save_figure, add_cli_args,
)


def plot(g_rep, avg, out_pdf, out_png, dpi):
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    setup_total = sum(avg[g_rep][k] for k in [
        "model_creation_time", "load_globals_time", "partitioning_time",
        "site_init_time", "connectivity_time", "gpu_setup_time", "first_tick_time",
    ])
    mean_tick = avg[g_rep]["mean_tick_time"]

    n_ticks = np.logspace(1, 5, 200)
    # Steady-state ticks only: first_tick_time is already part of setup_total,
    # so the steady-state portion of an N-tick run is (N-1) * mean_tick.
    sim_time = mean_tick * np.maximum(n_ticks - 1, 0)
    setup_frac = setup_total / (setup_total + sim_time) * 100

    ax.semilogx(n_ticks, setup_frac, "-", color=COLOR_WEAK_B, linewidth=1.5)
    ax.axhline(5, color="red", linestyle="--", linewidth=1, label="5% threshold")
    ax.axvline(1000, color=COLOR_REF, linestyle=":", linewidth=1, label="1k-tick run")

    crossing_idx = np.where(setup_frac < 5.0)[0]
    if len(crossing_idx) > 0:
        crossing_ticks = n_ticks[crossing_idx[0]]
        # Round to 1 significant figure for a paper-friendly annotation
        # (avoids spurious precision like "~91,158 ticks").
        order = 10 ** int(np.floor(np.log10(crossing_ticks)))
        rounded = int(round(crossing_ticks / order) * order)
        label = f"~{rounded // 1000}k ticks"
        ax.axvline(crossing_ticks, color="red", linestyle=":", linewidth=1, alpha=0.7)
        ax.annotate(label,
                    xy=(crossing_ticks, 5),
                    xytext=(crossing_ticks * 0.25, 22),
                    fontsize=FONT_ANNOT,
                    arrowprops=dict(arrowstyle="->", color="red", lw=0.7))

    ax.set_xlabel("Simulation Length (ticks)", fontsize=FONT_LABEL)
    ax.set_ylabel("Setup Fraction of Total Wall Time (%)", fontsize=FONT_LABEL)
    ax.set_ylim([0, 100])
    # Legend at upper-right: in the right half the curve is well below
    # 70%, so the legend sits in clear space at the top of the plot.
    ax.legend(fontsize=FONT_LEG, loc="upper right", framealpha=0.9)
    # Major gridlines only — minor lines on a log-x make the figure noisy.
    ax.grid(True, which="major", alpha=0.3)
    ax.tick_params(axis="both", labelsize=FONT_TICK)

    plt.tight_layout()
    save_figure(fig, out_pdf, out_png, dpi)
    plt.close(fig)


def main():
    csv_dir, figs_dir, _ = get_default_paths(__file__)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_cli_args(parser, "setup_amortization")
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir) if args.csv_dir else csv_dir
    figs_dir = Path(args.figs_dir) if args.figs_dir else figs_dir
    figs_dir.mkdir(parents=True, exist_ok=True)

    gpus, avg = read_csv(csv_dir / "weak_scaling_b.csv")
    g_rep = gpus[-1]  # largest measured Weak B configuration

    out_pdf = figs_dir / "setup_amortization.pdf"
    out_png = figs_dir / "setup_amortization.png"
    plot(g_rep, avg, out_pdf, out_png, args.dpi)
    print(f"Wrote {out_pdf} and {out_png}")


if __name__ == "__main__":
    main()
