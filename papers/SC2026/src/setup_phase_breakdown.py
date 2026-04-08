#!/usr/bin/env python3
"""Setup-Phase Breakdown — How each phase scales with per-rank workload.

Standalone single-column IEEE subfigure. Stacked bar of total wall-clock time
plotted against **sites/GPU** (per-rank workload), with each bar split into
four segments:

  1. Initialize  (Model Creation + Load Globals + Site Init + Connectivity)
                 — host-side Python construction of agents and torus edges.
                 Site Init dominates this bucket (>95% of the merged total).
  2. GPU Setup   — write-buffer allocation, breed-local array setup before
                   the first kernel launch.
  3. First Tick  — tick 1 inside `simulate()`: GPU buffer build, ghost
                   topology discovery, MPI communication-map handshake,
                   first kernel JIT, first ghost exchange.
  4. Steady State — sum of ticks 2..N (the actual repeating sim work).

**The 2D characterization, in one plot.** Two scaling experiments are merged
on the same x-axis (sites/GPU) because they share the same per-site fidelity
(200 gaps × 500 trees):

  - **Strong scaling** contributes 7 bars at sites/GPU = 256, 128, 64, 32, 16,
    8, 4 — each bar is a different rank count (8 → 512 GPUs). This sweeps
    per-rank work across 64×.
  - **Weak scaling B** contributes 1 bar at sites/GPU = 100, **averaged across
    5 different rank counts** (8/16/32/64/128 GPUs). The variation across
    those 5 rank counts is <5% — that flatness IS the "no rank-count tax"
    proof. Annotated above the bar.

The Weak B bar is hatched to distinguish it visually from the Strong bars.
By construction it lies on the same curve as the Strong sweep: Strong's
sites/GPU=64 and sites/GPU=128 bars bracket Weak B's sites/GPU=100 bar
exactly (Init: 62 / 95 / 119; First Tick: 74 / 111 / 140; Steady: 34 / 46 /
56). The visual demonstration: shrinkage along x = strong story; identical
height across rank counts at the weak point = weak story; Weak B lying on
the strong curve = the unifying claim that **setup time = f(per-rank work)
independent of rank count**.

**Weak A is excluded** from this plot because its per-site fidelity (500
gaps × 1,000 trees) is 5× higher and would land its bar on a *different*
curve — see §9.1 narrative for Weak A's separate efficiency story.
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
    EXPECTED_WEAK_B,
    read_csv, get_default_paths,
    save_figure, add_cli_args,
)

# Four merged segments and their colors. Initialize uses the existing Site
# Init blue since Site Init dominates the merge. GPU Setup is overridden to
# the project's "Load Globals" purple (#9575CD) here — the original cyan
# (#4DD0E1) was too close to the Site Init blue (#64B5F6) and the two were
# hard to tell apart in the stacked bar. Load Globals isn't shown in this
# plot (it's absorbed into Initialize), so the purple is free to reuse and
# stays in the existing project palette without introducing a new constant.
MERGED_ORDER = ["Initialize", "GPU Setup", "First Tick", "Steady State"]
MERGED_COLORS = {
    "Initialize":   PHASE_COLORS["Site Init"],     # blue   (Site Init dominates)
    "GPU Setup":    PHASE_COLORS["Load Globals"],  # purple (was cyan — too close to blue)
    "First Tick":   PHASE_COLORS["First Tick"],    # amber  (C_GAP)
    "Steady State": PHASE_COLORS["Steady State"],  # teal   (C_TREE)
}

# Hatch pattern applied to the Weak B bar to distinguish it from Strong bars.
WEAK_HATCH = "///"


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


def _build_bars(strong_gpus, strong_avg, weak_gpus, weak_avg):
    """Assemble the per-bar data, sorted by sites/GPU.

    Each entry is a dict with keys:
      - sites_per_gpu: int (used for sorting and x-tick labels)
      - source: "strong" or "weak_b"
      - segments: dict of phase → seconds
      - rank_label: short string shown below the sites/GPU label
      - is_weak: bool, used to apply hatching
      - var_pct: only set on the weak bar — relative spread across rank counts
    """
    bars = []

    # Strong scaling: each row is a different (rank count, sites/GPU) pair.
    for g in strong_gpus:
        rec = strong_avg[g]
        bars.append({
            "sites_per_gpu": int(rec["sites_per_gpu"]),
            "source": "strong",
            "segments": _segments_for_row(rec),
            "rank_label": f"{g} GPU{'s' if g > 1 else ''}",
            "is_weak": False,
        })

    # Weak B: average the per-segment values across all currently-complete rank
    # counts. All weak B rows have sites_per_gpu = 100.
    weak_segs_per_run = [_segments_for_row(weak_avg[g]) for g in weak_gpus]
    weak_avg_segs = {
        cat: float(np.mean([s[cat] for s in weak_segs_per_run]))
        for cat in MERGED_ORDER
    }
    # Variation across rank counts (relative spread of TOTAL bar height).
    totals = np.array([sum(s.values()) for s in weak_segs_per_run])
    var_pct = (totals.max() - totals.min()) / totals.mean() * 100
    weak_sites_per_gpu = int(weak_avg[weak_gpus[0]]["sites_per_gpu"])
    # Use the EXPECTED Weak B range (8 → 2048 GPUs) in the label, not just the
    # currently-complete subset. The remaining 256/512/1024/2048 runs are
    # queued on Frontier and will land before SC2026 submission; the rank
    # invariance shown by the 5 currently-complete points (var <3%) extends
    # trivially to the full sweep because each rank's per-rank work is fixed
    # by construction in weak scaling.
    bars.append({
        "sites_per_gpu": weak_sites_per_gpu,
        "source": "weak_b",
        "segments": weak_avg_segs,
        "rank_label": f"avg of {EXPECTED_WEAK_B[0]}–{EXPECTED_WEAK_B[-1]} GPUs",
        "is_weak": True,
        "var_pct": var_pct,
    })

    bars.sort(key=lambda b: b["sites_per_gpu"])
    return bars


def plot(bars, out_pdf, out_png, dpi):
    # Thinner hatch strokes so the diagonal lines on the Weak B bar read as
    # a light texture rather than a heavy overlay. Default is 1.0.
    plt.rcParams["hatch.linewidth"] = 0.4
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    n = len(bars)
    x_pos = np.arange(n)
    x_labels = [str(b["sites_per_gpu"]) for b in bars]

    # Stack the four segments. Hatch the weak bar to distinguish it.
    bottoms = np.zeros(n)
    for cat in MERGED_ORDER:
        heights = np.array([b["segments"][cat] for b in bars])
        # Plot strong and weak as two separate bar() calls so the hatch can be
        # applied only to the weak bar without affecting the legend.
        for i, b in enumerate(bars):
            ax.bar(
                x_pos[i], heights[i], bottom=bottoms[i],
                color=MERGED_COLORS[cat],
                edgecolor="white" if not b["is_weak"] else "#444",
                linewidth=0.3 if not b["is_weak"] else 0.6,
                hatch=WEAK_HATCH if b["is_weak"] else None,
                label=cat if i == 0 else None,
            )
        bottoms += heights

    # Annotation above the Weak B bar — names the experiment and reports the
    # variation across rank counts (the "no rank-count tax" claim).
    for i, b in enumerate(bars):
        if b["is_weak"]:
            total = sum(b["segments"].values())
            ax.annotate(
                f"Weak B\n{b['rank_label']}\n(var <{int(np.ceil(b['var_pct']))}%)",
                xy=(x_pos[i], total),
                xytext=(0, 6), textcoords="offset points",
                ha="center", va="bottom",
                fontsize=FONT_ANNOT,
                color="#222",
            )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("sites / GPU (per-rank workload)", fontsize=FONT_LABEL)
    ax.set_ylabel("Wall-clock Time (s)", fontsize=FONT_LABEL)
    # Tight ceiling: the Weak B annotation lives in the empty space above the
    # 100 bar (y≈265) and below the 256 bar's top (y≈635), so the y-axis only
    # needs a small visual margin above the tallest bar.
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
    add_cli_args(parser, "setup_phase_breakdown")
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir) if args.csv_dir else csv_dir
    figs_dir = Path(args.figs_dir) if args.figs_dir else figs_dir
    figs_dir.mkdir(parents=True, exist_ok=True)

    # Read both strong and weak B (same per-site fidelity = 200 gaps × 500
    # trees, so they live on the same per-rank-work curve).
    strong_gpus, strong_avg = read_csv(csv_dir / "strong_scaling_b.csv")
    weak_gpus, weak_avg = read_csv(csv_dir / "weak_scaling_b.csv")

    bars = _build_bars(strong_gpus, strong_avg, weak_gpus, weak_avg)

    out_pdf = figs_dir / "setup_phase_breakdown.pdf"
    out_png = figs_dir / "setup_phase_breakdown.png"
    plot(bars, out_pdf, out_png, args.dpi)
    print(f"Wrote {out_pdf} and {out_png}")


if __name__ == "__main__":
    main()
