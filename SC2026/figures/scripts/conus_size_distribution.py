#!/usr/bin/env python3
"""
Publication-quality CONUS size distribution figure for SC2026 paper.

Bar chart of diameter-class proportions across all CONUS sites.

Usage:
    python conus_size_distribution.py
    python conus_size_distribution.py --input_dir /path/to/last_year_species
"""

import argparse
import os
import sys
import csv
import glob

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIG_W = 3.3
FIG_H = 1.8
FONT_LABEL = 6
FONT_TICK  = 6
FONT_ANNOT = 6

DPI = 600

SIZE_COLS = ['0-8', '8-28', '-48', '-68', '-88', '>88']
BIN_LABELS = ['0\u20138', '8\u201328', '28\u201348', '48\u201368', '68\u201388', '>88']


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


def plot_size_distribution(rows, output_dir):
    """Bar chart of diameter-class proportions."""
    total_bins = np.zeros(len(SIZE_COLS))
    for row in rows:
        for i, col in enumerate(SIZE_COLS):
            total_bins[i] += float(row[col])

    total_sum = total_bins.sum()
    pcts = 100.0 * total_bins / total_sum if total_sum > 0 else np.zeros(len(SIZE_COLS))

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    bars = ax.bar(BIN_LABELS, pcts, color='#64B5F6', edgecolor='#424242',
                  linewidth=0.5, width=0.65, zorder=3)

    for bar, pct in zip(bars, pcts):
        if pct > 0.3:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                    f'{pct:.1f}%', ha='center', va='bottom', fontsize=FONT_ANNOT,
                    color='#1A202C')

    ax.set_ylabel('Proportion of trees (%)', fontsize=FONT_LABEL)
    ax.tick_params(axis='both', labelsize=FONT_TICK)
    ax.set_ylim(0, max(pcts) * 1.18)
    ax.grid(True, which='major', alpha=0.3, axis='y', zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()

    outpath_png = os.path.join(output_dir, 'conus_size_distribution.png')
    outpath_pdf = os.path.join(output_dir, 'conus_size_distribution.pdf')
    fig.savefig(outpath_png, dpi=DPI, bbox_inches='tight')
    fig.savefig(outpath_pdf, format='pdf', bbox_inches='tight', transparent=True)
    print(f"  Saved {outpath_png}")
    print(f"  Saved {outpath_pdf}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Publication figure: CONUS size distribution"
    )
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--input_dir", type=str,
                        default=os.path.join(script_dir, "..", "..", "conus_simulation",
                                             "results", "last_year_species"),
                        help="Directory with site_NNNN/species_data.csv files")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(script_dir, "..", "figs"),
                        help="Output directory for figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading all site data...")
    rows = load_all_sites(args.input_dir)
    print(f"  {len(rows):,} rows from {len(set(r['siteID'] for r in rows))} sites")

    print("Generating size distribution figure...")
    plot_size_distribution(rows, args.output_dir)

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
