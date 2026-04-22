#!/usr/bin/env python3
"""
Publication-quality CONUS size distribution and biomass-by-genus figures
for SC2026 paper.

Style follows strong_scaling_speedup.py: same figure dimensions (FIG_W × FIG_H),
font sizes (FONT_LABEL, FONT_TICK, FONT_LEG, FONT_PILL, FONT_ANNOT), grid
alpha, tight_layout, and annotation-pill convention from _scaling_common.

Two standalone subfigures (LaTeX handles panel layout via \\subfigure):
  (a) conus_size_distribution_paper — bar chart of diameter-class proportions
  (b) conus_biomass_by_genus_paper  — horizontal bar chart of biomass C by genus

Usage:
    python plot_conus_size_dist_paper.py
    python plot_conus_size_dist_paper.py --input_dir /path/to/last_year_species
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

# ── Shared constants from the scaling figures ──────────────────────
# Duplicated here (rather than importing _scaling_common) so this script
# can run from the conus/ directory without sys.path hacking.
FIG_W = 3.3       # inches — matches _scaling_common.FIG_W
FIG_H = 1.8       # inches — matches _scaling_common.FIG_H (was 2.6, then 2.0, now 1.8)
FONT_LABEL = 6    # axis labels — matches _scaling_common
FONT_TICK  = 6    # tick labels
FONT_LEG   = 6    # legend text
FONT_PILL  = 6    # in-plot annotation pill
FONT_ANNOT = 6    # bar-top value labels

DPI = 600

# ── Color palettes (Material Design, consistent with scaling figures) ──
# Size-class bars: green gradient from light to dark
SIZE_COLORS = ['#A5D6A7', '#66BB6A', '#43A047', '#2E7D32', '#1B5E20', '#0D3B0F']

# Genus bars: use the same Material Design hue set as the scaling figures
GENUS_COLORS = [
    '#43A047',   # green  (matches COLOR_STRONG)
    '#1E88E5',   # blue   (matches COLOR_WEAK_B)
    '#E53935',   # red    (matches COLOR_WEAK_A)
    '#FB8C00',   # orange (matches COLOR_XRANK)
    '#8E24AA',   # purple
    '#00ACC1',   # cyan
    '#F4511E',   # deep orange
    '#7CB342',   # light green
    '#9E9E9E',   # gray — "Others"
]

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
    """Bar chart of diameter-class proportions (style: strong_scaling_speedup)."""
    total_bins = np.zeros(len(SIZE_COLS))
    for row in rows:
        for i, col in enumerate(SIZE_COLS):
            total_bins[i] += float(row[col])

    total_sum = total_bins.sum()
    pcts = 100.0 * total_bins / total_sum if total_sum > 0 else np.zeros(len(SIZE_COLS))

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    bars = ax.bar(BIN_LABELS, pcts, color='#64B5F6', edgecolor='#424242',
                  linewidth=0.5, width=0.65, zorder=3)

    # Bar-top percentage labels (matches FONT_ANNOT from scaling figures)
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

    outpath_png = os.path.join(output_dir, 'conus_size_distribution_paper.png')
    outpath_pdf = os.path.join(output_dir, 'conus_size_distribution_paper.pdf')
    fig.savefig(outpath_png, dpi=DPI, bbox_inches='tight')
    fig.savefig(outpath_pdf, format='pdf', bbox_inches='tight', transparent=True)
    print(f"  Saved {outpath_png}")
    print(f"  Saved {outpath_pdf}")
    plt.close(fig)


def plot_biomass_by_genus(rows, output_dir):
    """Pie chart of biomass C by genus (style: strong_scaling_speedup)."""
    genus_biomass = defaultdict(float)
    for row in rows:
        genus_biomass[row['genus']] += float(row['total_biomC'])

    # Top 8 genera + Others
    sorted_genera = sorted(
        [g for g in genus_biomass if genus_biomass[g] > 0],
        key=lambda g: -genus_biomass[g]
    )
    top_n = 8
    top_genera = sorted_genera[:top_n]
    other_val = sum(genus_biomass[g] for g in sorted_genera[top_n:])
    total_biomass = sum(genus_biomass[g] for g in sorted_genera)

    def clean(g):
        return g.rstrip("'")

    labels = [clean(g) for g in top_genera]
    values = [genus_biomass[g] for g in top_genera]

    if other_val > 0:
        labels.append('Others')
        values.append(other_val)

    colors = GENUS_COLORS[:len(labels)]

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    # Keep pie visually matched to bar chart height when paired as subfigures
    ax.set_position([0.0, 0.12, 0.55, 0.76])

    pcts = [100.0 * v / sum(values) for v in values]
    wedges, texts, autotexts = ax.pie(
        values, labels=None, colors=colors,
        autopct='', pctdistance=0.75, startangle=90, counterclock=False,
        wedgeprops=dict(edgecolor='#424242', linewidth=0.5),
        textprops=dict(fontsize=7, fontweight='bold', color='white'),
    )

    # Place percentage labels: large wedges inside (white), small wedges outside with leader lines
    for i, (wedge, pct) in enumerate(zip(wedges, pcts)):
        ang = (wedge.theta2 + wedge.theta1) / 2.0
        rad = np.deg2rad(ang)
        if pct >= 5.0:
            # Inside label
            x = 0.65 * np.cos(rad)
            y = 0.65 * np.sin(rad)
            ax.text(x, y, f'{pct:.1f}%', ha='center', va='center',
                    fontsize=7, fontweight='bold', color='white')
        elif pct >= 1.5:
            # Outside label with leader line
            x_in = 0.95 * np.cos(rad)
            y_in = 0.95 * np.sin(rad)
            x_out = 1.25 * np.cos(rad)
            y_out = 1.25 * np.sin(rad)
            ax.annotate(f'{pct:.1f}%', xy=(x_in, y_in), xytext=(x_out, y_out),
                        fontsize=6, color='#1A202C',
                        arrowprops=dict(arrowstyle='-', color='#757575', lw=0.5),
                        ha='center', va='center')

    # Italic genus labels in legend
    ax.legend(wedges, [f'$\\it{{{l}}}$' if l != 'Others' else l for l in labels],
              loc='center left', bbox_to_anchor=(1.05, 0.5),
              fontsize=FONT_LEG, frameon=False)

    outpath_png = os.path.join(output_dir, 'conus_biomass_by_genus_paper.png')
    outpath_pdf = os.path.join(output_dir, 'conus_biomass_by_genus_paper.pdf')
    fig.savefig(outpath_png, dpi=DPI, bbox_inches='tight')
    fig.savefig(outpath_pdf, format='pdf', bbox_inches='tight', transparent=True)
    print(f"  Saved {outpath_png}")
    print(f"  Saved {outpath_pdf}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Publication figure: CONUS size distribution and biomass by genus"
    )
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--input_dir", type=str,
                        default=os.path.join(script_dir, "last_year_species"),
                        help="Directory with site_NNNN/species_data.csv files")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(script_dir),
                        help="Output directory for figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading all site data...")
    rows = load_all_sites(args.input_dir)
    print(f"  {len(rows):,} rows from {len(set(r['siteID'] for r in rows))} sites")

    print("Generating size distribution figure...")
    plot_size_distribution(rows, args.output_dir)

    print("Generating biomass-by-genus figure...")
    plot_biomass_by_genus(rows, args.output_dir)

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
