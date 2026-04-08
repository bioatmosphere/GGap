#!/usr/bin/env python3
"""Weak Scaling B — Combined efficiency + first-tick/steady-state breakdown.

Standalone single-column IEEE subfigure. Stacked bar of 1000-tick simulation
time decomposed into the one-time first-tick warmup (orange) and the 999
steady-state ticks (green). A horizontal dashed line marks the 8-GPU baseline
(= ideal weak scaling). Parallel efficiency is annotated above each bar.

Same combined design as `weak_scaling_a_efficiency.py`. The x-axis spans the
full 8–2048 GPU target range (matching Weak A); 256/512/1024/2048 are queued
on Frontier and currently appear as gaps that will fill in once those runs land.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _scaling_common import (
    FIG_W, FIG_H, PHASE_COLORS, EXPECTED_WEAK_B, COLOR_IDEAL, IDEAL_LW,
    FONT_LABEL, FONT_TICK, FONT_LEG, FONT_PILL, FONT_ANNOT,
    read_csv, derive_metrics, get_default_paths,
    save_figure, add_cli_args,
)

# Efficiency-label color thresholds
COLOR_EFF_OK = "#6B7280"  # mid gray
COLOR_EFF_BAD = "#D32F2F"  # dark red


def plot(gpus, avg, metrics, out_pdf, out_png, dpi):
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    # Use the FULL expected GPU range for the x-axis so missing runs show as gaps.
    g_labels = [str(g) for g in EXPECTED_WEAK_B]
    n_slots = len(EXPECTED_WEAK_B)

    first_tick = np.zeros(n_slots)
    steady_state = np.zeros(n_slots)
    sim_time = np.zeros(n_slots)
    eff = np.full(n_slots, np.nan)
    has_data = np.zeros(n_slots, dtype=bool)

    for i, g in enumerate(EXPECTED_WEAK_B):
        if g in avg:
            first_tick[i] = avg[g]["first_tick_time"]
            steady_state[i] = avg[g]["steady_state_time"]
            sim_time[i] = first_tick[i] + steady_state[i]
            eff[i] = metrics[g]["efficiency"]
            has_data[i] = True

    # Mean steady-state per-tick rate (across measured GPU counts) — sustained
    # per-tick capability we want to highlight in the figure.
    mean_tick_ms = np.mean([avg[g]["mean_tick_time"] for g in gpus]) * 1000

    baseline_sim = avg[gpus[0]]["simulation_time"]

    # Stacked bars: only draw segments where data exists; missing slots stay empty.
    ax.bar(g_labels, first_tick, color=PHASE_COLORS["First Tick"],
           label="First Tick", edgecolor="white", linewidth=0.3, zorder=2)
    ax.bar(g_labels, steady_state, bottom=first_tick,
           color=PHASE_COLORS["Steady State"],
           label="Steady State (999 ticks)",
           edgecolor="white", linewidth=0.3, zorder=2)

    # Ideal-baseline reference line (shared style with strong/weak A)
    ax.axhline(baseline_sim, color=COLOR_IDEAL, linestyle="--", linewidth=IDEAL_LW,
               label="Ideal", zorder=3)

    # In-plot annotation highlighting the sustained steady-state per-tick rate.
    # Same style as Weak A: dark charcoal text in a white pill, no arrow,
    # placed at the vertical *middle* of the green Steady State region of the
    # measured bars (skipping the empty queued slots).
    measured_indices = [i for i in range(n_slots) if has_data[i]]
    if measured_indices:
        measured_first_tick = first_tick[has_data]
        measured_steady = steady_state[has_data]
        green_center_y = measured_first_tick.mean() + measured_steady.mean() / 2
        x_center = (measured_indices[0] + measured_indices[-1]) / 2
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

    # Efficiency annotations above each measured bar — rotated vertical, with
    # extra clearance above the dashed Ideal baseline so labels don't touch it.
    sim_max_for_layout = max(sim_time.max(), baseline_sim)
    y_max = sim_max_for_layout * 1.35
    label_clearance = sim_max_for_layout * 0.06
    for i in range(n_slots):
        if has_data[i]:
            color = COLOR_EFF_BAD if eff[i] < 95.0 else COLOR_EFF_OK
            label_y = max(sim_time[i], baseline_sim) + label_clearance
            ax.text(i, label_y, f"{eff[i]:.1f}%",
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
    add_cli_args(parser, "weak_scaling_b_efficiency")
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir) if args.csv_dir else csv_dir
    figs_dir = Path(args.figs_dir) if args.figs_dir else figs_dir
    figs_dir.mkdir(parents=True, exist_ok=True)

    gpus, avg = read_csv(csv_dir / "weak_scaling_b.csv")
    metrics = derive_metrics(gpus, avg, baseline_gpu=gpus[0], scaling_kind="weak")

    missing = sorted(set(EXPECTED_WEAK_B) - set(gpus))
    if missing:
        print(f"WARNING: weak_scaling_b missing GPUs {missing}")

    out_pdf = figs_dir / "weak_scaling_b_efficiency.pdf"
    out_png = figs_dir / "weak_scaling_b_efficiency.png"
    plot(gpus, avg, metrics, out_pdf, out_png, args.dpi)
    print(f"Wrote {out_pdf} and {out_png}")


if __name__ == "__main__":
    main()
