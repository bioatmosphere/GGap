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
FIG_H = 2.6       # inches — matches _scaling_common.FIG_H
FONT_LABEL = 8    # axis labels
FONT_TICK  = 7    # tick labels
FONT_LEG   = 7    # legend text
FONT_PILL  = 8    # in-plot annotation pill
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

    bars = ax.bar(BIN_LABELS, pcts, color=SIZE_COLORS, edgecolor='#424242',
                  linewidth=0.5, width=0.65, zorder=3)

    # Bar-top percentage labels (matches FONT_ANNOT from scaling figures)
    for bar, pct in zip(bars, pcts):
        if pct > 0.3:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                    f'{pct:.1f}%', ha='center', va='bottom', fontsize=FONT_ANNOT,
                    color='#1A202C')

    ax.set_xlabel('Diameter class (cm)', fontsize=FONT_LABEL)
    ax.set_ylabel('Proportion of trees (%)', fontsize=FONT_LABEL)
    ax.tick_params(axis='both', labelsize=FONT_TICK)
    ax.set_ylim(0, max(pcts) * 1.18)
    ax.grid(True, which='major', alpha=0.3, axis='y', zorder=0)
    ax.set_axisbelow(True)

    # In-plot annotation pill: headline metric (matches strong_scaling pill)
    below_28 = pcts[0] + pcts[1]
    ax.text(0.97, 0.96,
            f'{below_28:.0f}% of stems < 28 cm\n{int(total_sum):,} total trees',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=FONT_PILL, color='#1A202C', zorder=6)

    plt.tight_layout()

    for fmt in ['png', 'pdf']:
        outpath = os.path.join(output_dir, f'conus_size_distribution_paper.{fmt}')
        fig.savefig(outpath, dpi=DPI, format=fmt, bbox_inches='tight')
        print(f"  Saved {outpath}")
    plt.close(fig)


def plot_biomass_by_genus(rows, output_dir):
    """Horizontal bar chart of biomass C by genus (style: strong_scaling_speedup)."""
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
    pcts = [100.0 * v / total_biomass for v in values]

    if other_val > 0:
        labels.append('Others')
        values.append(other_val)
        pcts.append(100.0 * other_val / total_biomass)

    # Reverse for bottom-to-top display (largest at top)
    labels = labels[::-1]
    values = values[::-1]
    pcts = pcts[::-1]
    colors = GENUS_COLORS[:len(labels)][::-1]

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, pcts, color=colors, edgecolor='#424242',
                   linewidth=0.5, height=0.6, zorder=3)

    # Value labels at end of each bar
    for bar, pct in zip(bars, pcts):
        if pct > 1.5:
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f'{pct:.1f}%', ha='left', va='center', fontsize=FONT_ANNOT,
                    color='#1A202C')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=FONT_TICK, fontstyle='italic')
    ax.set_xlabel('Share of total biomass C (%)', fontsize=FONT_LABEL)
    ax.tick_params(axis='x', labelsize=FONT_TICK)
    ax.set_xlim(0, max(pcts) * 1.18)
    ax.grid(True, which='major', alpha=0.3, axis='x', zorder=0)
    ax.set_axisbelow(True)

    # In-plot annotation pill: total biomass
    total_Mg = total_biomass
    ax.text(0.97, 0.06,
            f'{total_Mg:,.0f} Mg C total\n1,424 sites',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=FONT_PILL, color='#1A202C', zorder=6)

    plt.tight_layout()

    for fmt in ['png', 'pdf']:
        outpath = os.path.join(output_dir, f'conus_biomass_by_genus_paper.{fmt}')
        fig.savefig(outpath, dpi=DPI, format=fmt, bbox_inches='tight')
        print(f"  Saved {outpath}")
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
