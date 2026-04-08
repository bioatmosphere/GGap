#!/usr/bin/env python3
"""Strong Scaling — Phase Breakdown across the per-rank-workload sweep.

Standalone single-column IEEE subfigure. 7 stacked bars (one per rank count
from 8 → 512 GPUs), plotted on a **sites/GPU** x-axis (per-rank workload),
each bar split into four segments:

  1. Initialize  (Model Creation + Load Globals + Site Init + Connectivity)
                 — host-side Python construction of agents and torus edges.
                 Site Init dominates this bucket (>95% of the merged total).
  2. GPU Setup   — write-buffer allocation, breed-local array setup before
                   the first kernel launch.
  3. First Tick  — tick 1 inside `simulate()`: GPU buffer build, ghost
                   topology discovery, MPI communication-map handshake,
                   first kernel JIT, first ghost exchange.
  4. Steady State — sum of ticks 2..N (the actual repeating sim work).

**Why strong scaling?** In strong scaling, the per-rank workload shrinks
from 256 → 4 sites/GPU as the rank count grows from 8 → 512. Each phase
visibly scales differently with sites/GPU: Initialize and GPU Setup scale
near-linearly (pure per-rank Python / host-side bookkeeping), First Tick
scales sub-linearly (per-rank buffer construction plus smaller per-rank
components for kernel JIT and MPI handshake), and Steady State scales
least (the GPU kernel-granularity bound).

The companion `weak_scaling_a_phase_breakdown` figure provides the
orthogonal "constant in N" view (per-rank work fixed, varying rank count).
Together the two figures characterize setup cost as f(per-rank work) alone.

The bars shrink dramatically left-to-right (~636 s → ~37 s, 17× total),
and the composition shifts from setup-dominated (8 GPUs, ~85% setup) to
steady-state-dominated (512 GPUs, ~51% steady state). The crossover where
steady state catches up with setup is directly visible in the plot.

Intended LaTeX composition: this is the (b) panel of a double-column
subfigure pair with `strong_scaling_speedup` as (a).
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _scaling_common import (
    FIG_W, FIG_H, PHASE_COLORS,
    FONT_LABEL, FONT_TICK, FONT_LEG,
    read_csv, get_default_paths,
    save_figure, add_cli_args,
)

# Four merged segments and their colors. Initialize uses Site Init blue
# (Site Init dominates the merge); GPU Setup uses the project's "Load
# Globals" purple to keep it visually distinct from Initialize blue (the
# original cyan was too close to blue and the two were hard to tell apart
# in the stacked bar). Shared exactly with weak_scaling_a_phase_breakdown
# for visual cohesion across the two breakdown figures.
MERGED_ORDER = ["Initialize", "GPU Setup", "First Tick", "Steady State"]
MERGED_COLORS = {
    "Initialize":   PHASE_COLORS["Site Init"],     # blue   (Site Init dominates)
    "GPU Setup":    PHASE_COLORS["Load Globals"],  # purple
    "First Tick":   PHASE_COLORS["First Tick"],    # amber  (C_GAP)
    "Steady State": PHASE_COLORS["Steady State"],  # teal   (C_TREE)
}


def _segments_for_row(rec):
    """Return the 4 merged-segment values for a single CSV row dict."""
    init_merged = (rec["model_creation_time"] + rec["load_globals_time"]
                   + rec["site_init_time"] + rec["connectivity_time"])
    return {
        "Initialize":   init_merged,
        "GPU Setup":    rec["gpu_setup_time"],
        "First Tick":   rec["first_tick_time"],
        "Steady State": rec["steady_state_time"],
    }


def plot(gpus, avg, out_pdf, out_png, dpi):
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    # Build per-bar data, sorted by sites/GPU ascending (so the x-axis
    # reads "least per-rank work" → "most per-rank work" left to right).
    bars = []
    for g in gpus:
        rec = avg[g]
        bars.append({
            "sites_per_gpu": int(rec["sites_per_gpu"]),
            "segments": _segments_for_row(rec),
        })
    bars.sort(key=lambda b: b["sites_per_gpu"])

    n = len(bars)
    x_pos = np.arange(n)
    x_labels = [str(b["sites_per_gpu"]) for b in bars]

    bottoms = np.zeros(n)
    for cat in MERGED_ORDER:
        heights = np.array([b["segments"][cat] for b in bars])
        ax.bar(
            x_pos, heights, bottom=bottoms,
            color=MERGED_COLORS[cat],
            edgecolor="white", linewidth=0.3,
            label=cat,
        )
        bottoms += heights

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("sites / GPU (per-rank workload)", fontsize=FONT_LABEL)
    ax.set_ylabel("Wall-clock Time (s)", fontsize=FONT_LABEL)
    # Tight ceiling — no annotation to clear, so just give a small margin
    # above the tallest bar.
    y_max = max(sum(b["segments"].values()) for b in bars)
    ax.set_ylim(0, y_max * 1.03)

    ax.legend(fontsize=FONT_LEG, loc="lower center", bbox_to_anchor=(0.5, 1.02),
              ncol=4, framealpha=0.9, columnspacing=0.8, handletextpad=0.4)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", labelsize=FONT_TICK)
    ax.tick_params(axis="y", labelsize=FONT_TICK)

    plt.tight_layout()
    save_figure(fig, out_pdf, out_png, dpi)
    plt.close(fig)


def main():
    csv_dir, figs_dir, _ = get_default_paths(__file__)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_cli_args(parser, "strong_scaling_phase_breakdown")
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir) if args.csv_dir else csv_dir
    figs_dir = Path(args.figs_dir) if args.figs_dir else figs_dir
    figs_dir.mkdir(parents=True, exist_ok=True)

    gpus, avg = read_csv(csv_dir / "strong_scaling_b.csv")

    out_pdf = figs_dir / "strong_scaling_phase_breakdown.pdf"
    out_png = figs_dir / "strong_scaling_phase_breakdown.png"
    plot(gpus, avg, out_pdf, out_png, args.dpi)
    print(f"Wrote {out_pdf} and {out_png}")


if __name__ == "__main__":
    main()
