#!/usr/bin/env python3
"""
Publication-quality CONUS size distribution and biomass figure
for SC2026 paper (IEEE two-column format, 7.16 inch width).

Two panels: (a) Size class proportions, (b) Biomass C by genus.

Usage:
    python plot_conus_size_dist_paper.py \
        --input_dir ../results/simulation/last_year_species \
        --output_dir ../results/simulation/conus_plots
"""

import argparse
import os
import sys
import csv
import glob
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# IEEE two-column width
FIG_WIDTH = 7.16  # inches
FIG_HEIGHT = 2.8  # inches

SIZE_BINS = ['0-8', '8-28', '-48', '-68', '-88', '>88']
BIN_LABELS = ['0\u20138', '8\u201328', '28\u201348', '48\u201368', '68\u201388', '>88']
BIN_COLORS = ['#4daf4a', '#377eb8', '#ff7f00', '#e41a1c', '#984ea3', '#a65628']

GENUS_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
]


def load_all_sites(input_dir):
    """Load species_data.csv from all site directories."""
    site_dirs = sorted(glob.glob(os.path.join(input_dir, "site_*")))
    rows = []
    for sd in site_dirs:
        filepath = os.path.join(sd, "species_data.csv")
        if not os.path.exists(filepath):
            continue
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    return rows


def make_figure(rows, output_dir):
    total_bins = np.zeros(len(SIZE_BINS))
    genus_biomass = defaultdict(float)

    for row in rows:
        for i, col in enumerate(SIZE_BINS):
            total_bins[i] += float(row[col])
        genus_biomass[row['genus']] += float(row['total_biomC'])

    # --- Setup ---
    plt.rcParams.update({
        'font.size': 8,
        'font.family': 'serif',
        'axes.labelsize': 8,
        'axes.titlesize': 9,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 6.5,
        'axes.linewidth': 0.6,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_WIDTH, FIG_HEIGHT))
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.18, top=0.90, wspace=0.30)

    # --- Panel (a): Size class proportions ---
    total_sum = total_bins.sum()
    pcts = 100.0 * total_bins / total_sum if total_sum > 0 else np.zeros(len(SIZE_BINS))

    bars = ax1.bar(BIN_LABELS, pcts, color=BIN_COLORS, edgecolor='black', linewidth=0.4,
                   width=0.7)
    ax1.set_xlabel('Diameter class (cm)')
    ax1.set_ylabel('Proportion of trees (%)')
    ax1.set_title('(a) Tree size class distribution', fontweight='bold', loc='left')
    ax1.grid(True, alpha=0.15, axis='y', linewidth=0.4)
    ax1.set_axisbelow(True)
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=6))

    # Expand y-limit to make room for labels
    ax1.set_ylim(0, max(pcts) * 1.15)

    for bar, pct in zip(bars, pcts):
        if pct > 0.3:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f'{pct:.1f}%', ha='center', va='bottom', fontsize=6.5)

    # --- Panel (b): Biomass C by genus (top 8 + Others) ---
    sorted_genera = sorted(genus_biomass.keys(), key=lambda g: -genus_biomass[g])
    top_n = 8
    top_genera = [g for g in sorted_genera if genus_biomass[g] > 0][:top_n]

    # Clean genus names (remove trailing apostrophe)
    def clean_name(g):
        return g.rstrip("'")

    top_vals = [genus_biomass[g] for g in top_genera]
    other_val = sum(genus_biomass[g] for g in genus_biomass if g not in top_genera and genus_biomass[g] > 0)

    labels = [clean_name(g) for g in top_genera]
    values = list(top_vals)
    colors = GENUS_COLORS[:len(top_genera)]

    if other_val > 0:
        labels.append('Others')
        values.append(other_val)
        colors.append('#cccccc')

    total_biomass = sum(values)
    values_pct = [100.0 * v / total_biomass for v in values]

    # Horizontal bar chart (easier to read genus names)
    y_pos = np.arange(len(labels))
    bars2 = ax2.barh(y_pos, values_pct, color=colors, edgecolor='black', linewidth=0.4,
                     height=0.65)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, fontsize=7)
    ax2.invert_yaxis()
    ax2.set_xlabel('Proportion of biomass C (%)')
    ax2.set_title('(b) Biomass carbon by genus', fontweight='bold', loc='left')
    ax2.grid(True, alpha=0.15, axis='x', linewidth=0.4)
    ax2.set_axisbelow(True)
    ax2.xaxis.set_major_locator(MaxNLocator(nbins=6))

    # Expand x-limit to make room for labels
    ax2.set_xlim(0, max(values_pct) * 1.15)

    for bar, pct in zip(bars2, values_pct):
        if pct > 1.0:
            ax2.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                     f'{pct:.1f}%', ha='left', va='center', fontsize=6.5)

    # Save in multiple formats
    for fmt in ['png', 'pdf']:
        dpi = 300 if fmt == 'png' else None
        outpath = os.path.join(output_dir, f'conus_size_biomass_paper.{fmt}')
        fig.savefig(outpath, dpi=dpi, format=fmt)
        print(f"  Saved {outpath}")

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Publication figure: CONUS size distribution and biomass by genus"
    )
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Directory with site_NNNN/species_data.csv files")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading all site data...")
    rows = load_all_sites(args.input_dir)
    print(f"  {len(rows):,} rows from {len(set(r['siteID'] for r in rows))} sites")

    print("Generating figure...")
    make_figure(rows, args.output_dir)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
