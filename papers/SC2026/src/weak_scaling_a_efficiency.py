#!/usr/bin/env python3
"""Weak Scaling A — Combined efficiency + first-tick/steady-state breakdown.

Standalone single-column IEEE subfigure. Stacked bar of 1000-tick simulation
time decomposed into the one-time first-tick warmup (orange) and the 999
steady-state ticks (green). A horizontal dashed line marks the 8-GPU baseline
(= ideal weak scaling). Parallel efficiency is annotated above each bar.

This single plot replaces the previous pair (line-plot efficiency + separate
stacked-bar breakdown) — the gap between each bar and the ideal line is the
efficiency loss, the orange segment growth at 1024+ GPUs is its cause.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _scaling_common import (
    FIG_W, FIG_H, PHASE_COLORS, COLOR_IDEAL, IDEAL_LW,
    FONT_LABEL, FONT_TICK, FONT_LEG, FONT_PILL, FONT_ANNOT,
    read_csv, derive_metrics, get_default_paths,
    save_figure, add_cli_args,
)

# Efficiency-label color thresholds: light gray when ~100%, dark red when below
COLOR_EFF_OK = "#6B7280"  # mid gray — minimal visual noise for "all is well"
COLOR_EFF_BAD = "#D32F2F"  # dark red — draws the eye to efficiency loss


def plot(gpus, avg, metrics, out_pdf, out_png, dpi):
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    g_labels = [str(g) for g in gpus]

    first_tick = np.array([avg[g]["first_tick_time"] for g in gpus])
    steady_state = np.array([avg[g]["steady_state_time"] for g in gpus])
    sim_time = first_tick + steady_state
    eff = np.array([metrics[g]["efficiency"] for g in gpus])

    # Mean steady-state per-tick rate (across all GPU counts) — this is the
    # sustained per-tick capability we want to highlight in the figure.
    mean_tick_ms = np.mean([avg[g]["mean_tick_time"] for g in gpus]) * 1000

    baseline_sim = avg[gpus[0]]["simulation_time"]

    # Stacked bars: first_tick (bottom) + steady_state (top)
    ax.bar(g_labels, first_tick, color=PHASE_COLORS["First Tick"],
           label="First Tick", edgecolor="white", linewidth=0.3, zorder=2)
    ax.bar(g_labels, steady_state, bottom=first_tick,
           color=PHASE_COLORS["Steady State"],
           label="Steady State (999 ticks)",
           edgecolor="white", linewidth=0.3, zorder=2)

    # Ideal-baseline reference line (shared style with strong scaling)
    ax.axhline(baseline_sim, color=COLOR_IDEAL, linestyle="--", linewidth=IDEAL_LW,
               label="Ideal", zorder=3)

    # In-plot annotation highlighting the sustained steady-state per-tick rate.
    # Placed at the vertical *middle* of the green Steady State region (data
    # coords), horizontally centered across all bars. Dark charcoal text in a
    # white pill for high contrast on the light-green segment.
    green_center_y = first_tick.mean() + steady_state.mean() / 2
    x_center = (len(gpus) - 1) / 2
    ax.text(
        x_center, green_center_y,
        f"sustained ≈ {mean_tick_ms:.0f} ms/tick",
        ha="center", va="center",
        fontsize=FONT_PILL,
        color="#1A202C",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                  edgecolor="#1A202C", linewidth=0.6, alpha=0.85),
        zorder=5,
    )

    # Efficiency annotations above each bar — rotated vertical to avoid horizontal
    # overlap (single-column figure has only ~0.39" per bar group). The label y
    # starts above whichever is higher (bar top or baseline) so labels never
    # touch the dashed Ideal reference line, with extra clearance for visual
    # breathing room.
    y_max = sim_time.max() * 1.35
    label_clearance = sim_time.max() * 0.06
    for i, (h, e) in enumerate(zip(sim_time, eff)):
        color = COLOR_EFF_BAD if e < 95.0 else COLOR_EFF_OK
        label_y = max(h, baseline_sim) + label_clearance
        ax.text(i, label_y, f"{e:.1f}%",
                ha="center", va="bottom", fontsize=FONT_ANNOT, color=color,
                rotation=90)

    ax.set_xlabel("Number of GPUs", fontsize=FONT_LABEL)
    ax.set_ylabel("Simulation Time (s)", fontsize=FONT_LABEL)
    ax.set_ylim([0, y_max])
    ax.legend(fontsize=FONT_LEG, loc="lower center", bbox_to_anchor=(0.5, 1.02),
              ncol=3, framealpha=0.9, columnspacing=0.8, handletextpad=0.4)
    ax.grid(axis="y", alpha=0.3, zorder=1)
    ax.tick_params(axis="x", labelsize=FONT_TICK, rotation=45)
    ax.tick_params(axis="y", labelsize=FONT_TICK)

    plt.tight_layout()
    save_figure(fig, out_pdf, out_png, dpi)
    plt.close(fig)


def main():
    csv_dir, figs_dir, _ = get_default_paths(__file__)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_cli_args(parser, "weak_scaling_a_efficiency")
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir) if args.csv_dir else csv_dir
    figs_dir = Path(args.figs_dir) if args.figs_dir else figs_dir
    figs_dir.mkdir(parents=True, exist_ok=True)

    gpus, avg = read_csv(csv_dir / "weak_scaling.csv")
    metrics = derive_metrics(gpus, avg, baseline_gpu=gpus[0], scaling_kind="weak")

    out_pdf = figs_dir / "weak_scaling_a_efficiency.pdf"
    out_png = figs_dir / "weak_scaling_a_efficiency.png"
    plot(gpus, avg, metrics, out_pdf, out_png, args.dpi)
    print(f"Wrote {out_pdf} and {out_png}")


if __name__ == "__main__":
    main()
