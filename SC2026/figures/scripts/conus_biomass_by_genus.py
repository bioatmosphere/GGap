#!/usr/bin/env python3
"""
Publication-quality CONUS biomass-by-genus figure for SC2026 paper.

Pie chart of biomass C by genus across all CONUS sites.

Usage:
    python conus_biomass_by_genus.py
    python conus_biomass_by_genus.py --input_dir /path/to/last_year_species
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

FIG_W = 3.3
FIG_H = 1.8
FONT_LEG   = 6

DPI = 600

GENUS_COLORS = [
    '#43A047',   # green
    '#1E88E5',   # blue
    '#E53935',   # red
    '#FB8C00',   # orange
    '#8E24AA',   # purple
    '#00ACC1',   # cyan
    '#F4511E',   # deep orange
    '#7CB342',   # light green
    '#9E9E9E',   # gray — "Others"
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


def plot_biomass_by_genus(rows, output_dir):
    """Pie chart of biomass C by genus."""
    genus_biomass = defaultdict(float)
    for row in rows:
        genus_biomass[row['genus']] += float(row['total_biomC'])

    sorted_genera = sorted(
        [g for g in genus_biomass if genus_biomass[g] > 0],
        key=lambda g: -genus_biomass[g]
    )
    top_n = 8
    top_genera = sorted_genera[:top_n]
    other_val = sum(genus_biomass[g] for g in sorted_genera[top_n:])

    def clean(g):
        return g.rstrip("'")

    labels = [clean(g) for g in top_genera]
    values = [genus_biomass[g] for g in top_genera]

    if other_val > 0:
        labels.append('Others')
        values.append(other_val)

    colors = GENUS_COLORS[:len(labels)]

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_position([0.0, 0.12, 0.55, 0.76])

    pcts = [100.0 * v / sum(values) for v in values]
    wedges, texts, autotexts = ax.pie(
        values, labels=None, colors=colors,
        autopct='', pctdistance=0.75, startangle=90, counterclock=False,
        wedgeprops=dict(edgecolor='#424242', linewidth=0.5),
        textprops=dict(fontsize=7, fontweight='bold', color='white'),
    )

    for i, (wedge, pct) in enumerate(zip(wedges, pcts)):
        ang = (wedge.theta2 + wedge.theta1) / 2.0
        rad = np.deg2rad(ang)
        if pct >= 5.0:
            x = 0.65 * np.cos(rad)
            y = 0.65 * np.sin(rad)
            ax.text(x, y, f'{pct:.1f}%', ha='center', va='center',
                    fontsize=7, fontweight='bold', color='white')
        elif pct >= 1.5:
            x_in = 0.95 * np.cos(rad)
            y_in = 0.95 * np.sin(rad)
            x_out = 1.25 * np.cos(rad)
            y_out = 1.25 * np.sin(rad)
            ax.annotate(f'{pct:.1f}%', xy=(x_in, y_in), xytext=(x_out, y_out),
                        fontsize=6, color='#1A202C',
                        arrowprops=dict(arrowstyle='-', color='#757575', lw=0.5),
                        ha='center', va='center')

    ax.legend(wedges, [f'$\\it{{{l}}}$' if l != 'Others' else l for l in labels],
              loc='center left', bbox_to_anchor=(1.05, 0.5),
              fontsize=FONT_LEG, frameon=False)

    outpath_png = os.path.join(output_dir, 'conus_biomass_by_genus.png')
    outpath_pdf = os.path.join(output_dir, 'conus_biomass_by_genus.pdf')
    fig.savefig(outpath_png, dpi=DPI, bbox_inches='tight')
    fig.savefig(outpath_pdf, format='pdf', bbox_inches='tight', transparent=True)
    print(f"  Saved {outpath_png}")
    print(f"  Saved {outpath_pdf}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Publication figure: CONUS biomass by genus"
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

    print("Generating biomass-by-genus figure...")
    plot_biomass_by_genus(rows, args.output_dir)

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
