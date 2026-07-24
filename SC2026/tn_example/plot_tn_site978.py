#!/usr/bin/env python3
"""
Reproduce the paper's per-site panels for site 978 (S. Appalachia) from OUR TN run.

Site 978 is one of the paper's 10 representative sites AND falls inside the TN box, so its
two panels can be compared directly to the paper figure (conus_10sites_with_bars):
  - LEFT : species composition over time (stacked area, biomass-C %, top-6 species + Other)
  - RIGHT: tree size (diameter) distribution at year 1000 (%)

Reads the time-series CSV produced by extract_timeseries.py. Column layout + colors are
copied verbatim from SC2026/figures/scripts/conus_10sites_with_bars.py so the look matches.

Usage:
    python plot_tn_site978.py            # site 978, defaults
"""
import argparse
import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- verbatim from conus_10sites_with_bars.py ----
TARGET_YEAR = 1000
TOP_N = 6
SIZE_CLASSES = ["0–8", "8–28", "28–48", "48–68", "68–88", ">88"]
SIZE_COL_INDICES = [5, 6, 7, 8, 9, 10]
BAR_COLORS = ["#90EE90", "#66BB6A", "#228B22", "#006400", "#8B4513", "#4E2A04"]
COMP_PALETTE = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]
OTHER_COLOR = "#cccccc"


def read_size_dist(path):
    counts = np.zeros(6)
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if int(row[1]) == TARGET_YEAR:
                for i, ci in enumerate(SIZE_COL_INDICES):
                    counts[i] += int(row[ci])
    total = counts.sum()
    return counts / total * 100.0 if total > 0 else counts


def read_species_biomass(path):
    data = defaultdict(lambda: defaultdict(float))
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            year = int(row[1])
            species = row[3]
            biomC = float(row[15])
            if biomC > 0:
                data[year][species] += biomC
    return data


def get_top_species(data, top_n):
    totals = defaultdict(float)
    for yd in data.values():
        for sp, bc in yd.items():
            totals[sp] += bc
    ranked = sorted(totals.items(), key=lambda x: -x[1])
    return [sp for sp, _ in ranked[:top_n]]


def build_stacked_data(data, top_species):
    years = sorted(data.keys())
    pcts = np.zeros((len(top_species) + 1, len(years)))
    for j, yr in enumerate(years):
        yd = data[yr]
        total = sum(yd.values())
        if total <= 0:
            continue
        for i, sp in enumerate(top_species):
            pcts[i, j] = yd.get(sp, 0) / total * 100.0
        other = total - sum(yd.get(sp, 0) for sp in top_species)
        pcts[-1, j] = other / total * 100.0
    return np.array(years), top_species + ["Other"], pcts


def render_panels(csv_path, site, out_base, suptitle=None, save_exts=("png", "pdf")):
    """Render the two site panels (size dist | composition). suptitle=None => no title
    (used by the side-by-side composer, whose row label already identifies the source)."""
    size_props = read_size_dist(csv_path)
    raw = read_species_biomass(csv_path)
    top_sp = get_top_species(raw, TOP_N)
    years, names, pcts = build_stacked_data(raw, top_sp)
    print(f"site {site}: {len(years)} years ({years[0]}..{years[-1]}), top species: {top_sp}")

    # Panel order size-dist | composition to match the paper's right-column site 7.
    fig, (axb, axc) = plt.subplots(1, 2, figsize=(8.0, 3.2))
    if suptitle:
        fig.suptitle(suptitle, fontsize=10, fontweight="bold")

    # RIGHT: species composition over time
    colors = COMP_PALETTE[:len(names) - 1] + [OTHER_COLOR]
    axc.stackplot(years, pcts, labels=names, colors=colors, alpha=0.85, linewidth=0)
    axc.set_xlim(years[0], years[-1])
    axc.set_ylim(0, 100)
    axc.set_xlabel("Year", fontsize=8)
    axc.set_ylabel("Biomass C (%)", fontsize=8)
    axc.set_title("Species composition over time", fontsize=8)
    axc.tick_params(labelsize=7)
    axc.spines["top"].set_visible(False)
    axc.spines["right"].set_visible(False)
    axc.legend(fontsize=6, loc="upper left", frameon=True, framealpha=0.5,
               edgecolor="none", handlelength=0.8, ncol=2)

    # LEFT: size distribution at year 1000
    x = np.arange(len(SIZE_CLASSES))
    axb.bar(x, size_props, color=BAR_COLORS, edgecolor="black", linewidth=0.3, width=0.7)
    axb.set_xticks(x)
    axb.set_xticklabels(SIZE_CLASSES, fontsize=7, rotation=45, ha="right")
    axb.set_ylabel("Trees (%)", fontsize=8)
    axb.set_xlabel("Diameter class (cm)", fontsize=8)
    axb.set_title(f"Size distribution at yr {TARGET_YEAR}", fontsize=8)
    axb.tick_params(labelsize=7)
    axb.spines["top"].set_visible(False)
    axb.spines["right"].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.95 if suptitle else 1.0])
    for ext in save_exts:
        fig.savefig(f"{out_base}.{ext}", dpi=200, bbox_inches="tight")
        print(f"saved {out_base}.{ext}")
    plt.close(fig)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description="Plot site-978 paper-style panels from the TN run.")
    p.add_argument("--site", type=int, default=978)
    p.add_argument("--csv", type=str, default=None,
                   help="species_data.csv time series (default: results/species_timeseries/site_0978/…)")
    p.add_argument("--out", type=str, default=os.path.join(script_dir, "figures", "tn_site978_panels"))
    p.add_argument("--suptitle", type=str,
                   default=None,
                   help="Figure title. Omit for the default; pass '' for no title.")
    args = p.parse_args()

    csv_path = args.csv or os.path.join(
        script_dir, "results", "species_timeseries", f"site_{args.site:04d}", "species_data.csv")
    if not os.path.exists(csv_path):
        raise SystemExit(f"ERROR: {csv_path} not found — run extract_timeseries.py first.")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    suptitle = (f"This work — 2-GPU TN reproduction · site {args.site} (S. Appalachia)"
                if args.suptitle is None else args.suptitle)
    render_panels(csv_path, args.site, args.out, suptitle=suptitle)


if __name__ == "__main__":
    main()
