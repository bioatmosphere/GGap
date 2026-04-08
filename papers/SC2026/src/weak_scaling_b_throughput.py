#!/usr/bin/env python3
"""Weak Scaling B — Throughput.

Standalone subfigure: aggregate throughput in billion agent-updates/second
vs. number of GPUs (log2 x-axis, log2 y-axis) for the compute-heavy
weak-scaling run (100 sites/GPU). Includes ideal-linear line.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _scaling_common import (
    FIG_W, FIG_H, COLOR_WEAK_B, COLOR_REF, EXPECTED_WEAK_B,
    read_csv, derive_metrics, get_default_paths,
    setup_log2_xaxis, save_figure, add_cli_args,
)


def plot(gpus, metrics, out_pdf, out_png, dpi):
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    g_arr = np.array(gpus)
    tput = np.array([metrics[g]["throughput"] / 1e9 for g in gpus])

    ideal_tput = (g_arr / gpus[0]) * tput[0]
    ax.plot(g_arr, ideal_tput, "--", color=COLOR_REF, linewidth=1, label="Ideal (linear)")
    ax.plot(g_arr, tput, "s-", color=COLOR_WEAK_B, linewidth=1.8, markersize=6, label="Measured")

    setup_log2_xaxis(ax, EXPECTED_WEAK_B)
    ax.set_ylabel("Throughput (B agent-updates/s)", fontsize=10)
    ax.set_yscale("log", base=2)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax.grid(True, which="both", alpha=0.3)
    ax.tick_params(axis="y", labelsize=8)

    plt.tight_layout()
    save_figure(fig, out_pdf, out_png, dpi)
    plt.close(fig)


def main():
    csv_dir, figs_dir, _ = get_default_paths(__file__)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_cli_args(parser, "weak_scaling_b_throughput")
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir) if args.csv_dir else csv_dir
    figs_dir = Path(args.figs_dir) if args.figs_dir else figs_dir
    figs_dir.mkdir(parents=True, exist_ok=True)

    gpus, avg = read_csv(csv_dir / "weak_scaling_b.csv")
    metrics = derive_metrics(gpus, avg, baseline_gpu=gpus[0], scaling_kind="weak")

    missing = sorted(set(EXPECTED_WEAK_B) - set(gpus))
    if missing:
        print(f"WARNING: weak_scaling_b missing GPUs {missing}")

    out_pdf = figs_dir / "weak_scaling_b_throughput.pdf"
    out_png = figs_dir / "weak_scaling_b_throughput.png"
    plot(gpus, metrics, out_pdf, out_png, args.dpi)
    print(f"Wrote {out_pdf} and {out_png}")


if __name__ == "__main__":
    main()
