#!/usr/bin/env python3
"""Phase Breakdown — wall time scales with per-GPU workload (strong scaling).

Standalone single-column IEEE figure. 7 stacked bars across the strong-
scaling sweep at sites/GPU = 4, 8, 16, 32, 64, 128, 256 (one per rank
count from 8 → 512 GPUs). Bars grow left → right as the per-GPU
partition grows; the shrinkage IS the "wall time scales with per-GPU
workload" claim.

Each bar is split into the same 4 merged segments:

  1. Initialize  (Model Creation + Load Globals + Site Init + Connectivity)
                 — host-side Python construction of agents.
  2. GPU Setup   — write-buffer allocation, breed-local array setup before
                   the first kernel launch.
  3. First Tick  — tick 1 inside `simulate()`: GPU buffer build, ghost
                   topology discovery, MPI communication-map handshake,
                   first kernel JIT, first ghost exchange.
  4. Steady State — sum of ticks 2..N (the actual repeating sim work).

The complementary weak-scaling claim ("per-rank cost is independent of
rank count when communication is constant") is NOT shown here as a
panel. Instead it lives:
  - in `weak_scaling_efficiency.png` (the flat efficiency line at ~100%
    across 8 → 2,048 GPUs is the visual proof);
  - and in the §9.2 narrative as the explicit textual claim that all
    four merged phases vary by <5% across the 8 → 2,048 GPU sweep
    (with the per-rank decomposition table in §6 backing this up).

Per-tick MPI ghost-exchange is constant at ~5 μs/rank across the entire
8 → 2,048 GPU sweep — that constant-communication condition is the
prerequisite for the "time = f(per-rank workload), not f(rank count)"
claim and lives in the LaTeX caption, not the figure itself.
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

# Four merged segments and their colors. Initialize uses the existing
# Site Init blue (Site Init dominates the merge); GPU Setup uses the
# project's "Load Globals" purple to keep it visually distinct from
# Initialize blue.
MERGED_ORDER = ["Initialize", "GPU Setup", "First Tick", "Steady State"]
MERGED_COLORS = {
    "Initialize":   PHASE_COLORS["Site Init"],     # blue
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


def plot(strong_gpus, strong_avg, out_pdf, out_png, dpi):
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    # Build per-bar data, sorted ascending by sites/GPU.
    bars = []
    for g in strong_gpus:
        rec = strong_avg[g]
        bars.append({
            "sites_per_gpu": int(rec["sites_per_gpu"]),
            "segments": _segments_for_row(rec),
        })
    bars.sort(key=lambda b: b["sites_per_gpu"])

    n = len(bars)
    x_pos = np.arange(n)
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
    ax.set_xticklabels([str(b["sites_per_gpu"]) for b in bars])
    ax.set_xlabel("sites / GPU  (strong scaling)", fontsize=FONT_LABEL)
    ax.set_ylabel("Wall-clock Time (s)", fontsize=FONT_LABEL)
    # Tight ceiling — no annotation to clear, so just give a small margin
    # above the tallest bar.
    y_max = max(sum(b["segments"].values()) for b in bars)
    ax.set_ylim(0, y_max * 1.03)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", labelsize=FONT_TICK)
    ax.tick_params(axis="y", labelsize=FONT_TICK)

    # Legend INSIDE the plot — upper-left free corner. Bars grow
    # left-to-right (sites/GPU = 4 → 256), so the upper-left wedge is
    # the empty area.
    ax.legend(fontsize=FONT_LEG, loc="upper left", ncol=1,
              framealpha=0.9, handletextpad=0.4)

    plt.tight_layout()
    save_figure(fig, out_pdf, out_png, dpi)
    plt.close(fig)


def main():
    csv_dir, figs_dir, _ = get_default_paths(__file__)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_cli_args(parser, "phase_breakdown")
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir) if args.csv_dir else csv_dir
    figs_dir = Path(args.figs_dir) if args.figs_dir else figs_dir
    figs_dir.mkdir(parents=True, exist_ok=True)

    strong_gpus, strong_avg = read_csv(csv_dir / "strong_scaling_b.csv")

    out_pdf = figs_dir / "phase_breakdown.pdf"
    out_png = figs_dir / "phase_breakdown.png"
    plot(strong_gpus, strong_avg, out_pdf, out_png, args.dpi)
    print(f"Wrote {out_pdf} and {out_png}")


if __name__ == "__main__":
    main()
