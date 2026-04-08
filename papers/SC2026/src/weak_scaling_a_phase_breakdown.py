#!/usr/bin/env python3
"""Weak Scaling A — Phase Breakdown (single representative bar).

Standalone single-column IEEE subfigure. Plots **one** stacked bar
representing the average wall-clock time decomposition for the Weak Scaling
A experiment, averaged across all 9 rank counts (8 → 2,048 GPUs). The bar
is split into four segments matching the strong-scaling phase breakdown:

  1. Initialize  (Model Creation + Load Globals + Site Init + Connectivity)
                 — host-side Python construction of agents and torus edges.
  2. GPU Setup   — write-buffer allocation, breed-local array setup before
                   the first kernel launch.
  3. First Tick  — tick 1 inside `simulate()`: GPU buffer build, ghost
                   topology discovery, MPI communication-map handshake,
                   first kernel JIT, first ghost exchange.
  4. Steady State — sum of ticks 2..N (the actual repeating sim work).

**Why one bar?** Weak scaling holds per-rank work fixed by design, so all 9
rank counts produce essentially the same per-rank wall-clock time. Plotting
9 identical bars adds no information; plotting one *averaged* bar with a
**variance annotation** ("avg of 8–2,048 GPUs, var <X%") lets the single
number make the "constant in N" claim directly. The companion
`strong_scaling_phase_breakdown` figure provides the orthogonal view (how
each phase scales with per-rank work). Together they characterize setup
cost as f(per-rank work) alone.

The bar is intended to be composed in LaTeX as the (b) panel of a
double-column subfigure pair with `weak_scaling_a_efficiency` as (a).
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _scaling_common import (
    FIG_W, FIG_H, PHASE_COLORS,
    FONT_LABEL, FONT_TICK, FONT_LEG, FONT_ANNOT,
    read_csv, get_default_paths,
    save_figure, add_cli_args,
)

# Four merged segments and their colors. Initialize uses Site Init blue
# (Site Init dominates the merge); GPU Setup uses the project's "Load
# Globals" purple to keep it visually distinct from Initialize blue (the
# original cyan was too close to blue and the two were hard to tell apart
# in the stacked bar).
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

    # Average each phase across all rank counts.
    segs_per_run = [_segments_for_row(avg[g]) for g in gpus]
    avg_segs = {
        cat: float(np.mean([s[cat] for s in segs_per_run]))
        for cat in MERGED_ORDER
    }
    # Variation across rank counts (relative spread of TOTAL bar height).
    totals = np.array([sum(s.values()) for s in segs_per_run])
    var_pct = (totals.max() - totals.min()) / totals.mean() * 100

    # One bar centered on x = 0, width 0.5 so it doesn't fill the whole panel
    # awkwardly (single-bar plots look ridiculous if the bar is full-width).
    bar_x = 0.0
    bar_w = 0.5
    bottom = 0.0
    for cat in MERGED_ORDER:
        h = avg_segs[cat]
        ax.bar(
            bar_x, h, width=bar_w, bottom=bottom,
            color=MERGED_COLORS[cat],
            edgecolor="white", linewidth=0.3,
            label=cat,
        )
        bottom += h

    total = sum(avg_segs.values())

    # Annotation above the bar — names the experiment, the rank count range,
    # and the variance across the sweep.
    ax.annotate(
        f"Weak A\navg of {gpus[0]}–{gpus[-1]} GPUs\n(var <{int(np.ceil(var_pct))}%)",
        xy=(bar_x, total),
        xytext=(0, 6), textcoords="offset points",
        ha="center", va="bottom",
        fontsize=FONT_ANNOT,
        color="#222",
    )

    # X-axis: single category label centered on the bar.
    ax.set_xticks([bar_x])
    ax.set_xticklabels(["Weak A (10 sites/GPU)"])
    # Tighten the x range so the single bar isn't lost in whitespace, but
    # leave some breathing room on each side.
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylabel("Wall-clock Time (s)", fontsize=FONT_LABEL)
    # Headroom for the annotation block (3 lines of FONT_ANNOT text).
    ax.set_ylim(0, total * 1.40)

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
    add_cli_args(parser, "weak_scaling_a_phase_breakdown")
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir) if args.csv_dir else csv_dir
    figs_dir = Path(args.figs_dir) if args.figs_dir else figs_dir
    figs_dir.mkdir(parents=True, exist_ok=True)

    gpus, avg = read_csv(csv_dir / "weak_scaling.csv")

    out_pdf = figs_dir / "weak_scaling_a_phase_breakdown.pdf"
    out_png = figs_dir / "weak_scaling_a_phase_breakdown.png"
    plot(gpus, avg, out_pdf, out_png, args.dpi)
    print(f"Wrote {out_pdf} and {out_png}")


if __name__ == "__main__":
    main()
