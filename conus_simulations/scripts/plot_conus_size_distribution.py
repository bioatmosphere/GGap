#!/usr/bin/env python3
"""
Plot CONUS-level tree size distribution by aggregating species_data.csv
from all sites (last-year extraction).

Usage:
    python plot_conus_size_distribution.py \
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
import matplotlib.pyplot as plt


SIZE_BINS = ['0-8', '8-28', '-48', '-68', '-88', '>88']
BIN_LABELS = ['0-8', '8-28', '28-48', '48-68', '68-88', '>88']
BIN_COLORS = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728', '#9467bd', '#8c564b']

SPECIES_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#1a9850', '#d73027', '#4575b4', '#fee08b', '#313695',
    '#a50026', '#006837', '#542788', '#f46d43', '#74add1'
]


def load_all_sites(input_dir):
    """Load species_data.csv from all site directories.

    Returns:
        List of dicts, one per CSV row, with float-converted size bin columns.
    """
    site_dirs = sorted(glob.glob(os.path.join(input_dir, "site_*")))
    print(f"Found {len(site_dirs)} site directories")

    rows = []
    for sd in site_dirs:
        filepath = os.path.join(sd, "species_data.csv")
        if not os.path.exists(filepath):
            continue
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

    print(f"Loaded {len(rows):,} species-site rows")
    return rows


def aggregate_by_genus(rows):
    """Aggregate size bins and biomass by genus across all sites."""
    genus_bins = defaultdict(lambda: np.zeros(len(SIZE_BINS)))
    genus_biomass = defaultdict(float)
    genus_ntrees = defaultdict(float)

    for row in rows:
        genus = row['genus']
        for i, col in enumerate(SIZE_BINS):
            genus_bins[genus][i] += float(row[col])
        genus_biomass[genus] += float(row['total_biomC'])
        genus_ntrees[genus] += sum(float(row[col]) for col in SIZE_BINS)

    return genus_bins, genus_biomass, genus_ntrees


def aggregate_total(rows):
    """Aggregate size bins across all species and sites."""
    total_bins = np.zeros(len(SIZE_BINS))
    for row in rows:
        for i, col in enumerate(SIZE_BINS):
            total_bins[i] += float(row[col])
    return total_bins


def plot_conus_size_distribution(rows, output_dir):
    """Create CONUS-level size distribution plots."""

    total_bins = aggregate_total(rows)
    genus_bins, genus_biomass, genus_ntrees = aggregate_by_genus(rows)

    # Sort genera by total tree count
    sorted_genera = sorted(genus_ntrees.keys(), key=lambda g: -genus_ntrees[g])
    # Top genera for detailed plots
    top_n = 15
    top_genera = [g for g in sorted_genera if genus_ntrees[g] > 0][:top_n]

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle('CONUS Forest Size Distribution (Final Year)', fontsize=16, fontweight='bold')
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # --- Plot 1: Total CONUS size distribution (bar) ---
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(BIN_LABELS, total_bins, color=BIN_COLORS, alpha=0.8, edgecolor='black')
    ax1.set_xlabel('Diameter Class (cm)')
    ax1.set_ylabel('Total Number of Trees')
    ax1.set_title('CONUS Size Distribution', fontweight='bold')
    ax1.grid(True, alpha=0.2, axis='y')
    for bar, count in zip(bars, total_bins):
        if count > 0:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f'{int(count):,}', ha='center', va='bottom', fontsize=7)

    # --- Plot 2: Proportional size distribution (%) ---
    ax2 = fig.add_subplot(gs[0, 1])
    total_sum = total_bins.sum()
    if total_sum > 0:
        pcts = 100.0 * total_bins / total_sum
    else:
        pcts = np.zeros(len(SIZE_BINS))
    bars2 = ax2.bar(BIN_LABELS, pcts, color=BIN_COLORS, alpha=0.8, edgecolor='black')
    ax2.set_xlabel('Diameter Class (cm)')
    ax2.set_ylabel('Percentage of Trees (%)')
    ax2.set_title('Size Class Proportions', fontweight='bold')
    ax2.grid(True, alpha=0.2, axis='y')
    for bar, pct in zip(bars2, pcts):
        if pct > 0.5:
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)

    # --- Plot 3: Top genera by tree count (horizontal bar) ---
    ax3 = fig.add_subplot(gs[0, 2])
    top_counts = [genus_ntrees[g] for g in top_genera]
    y_pos = np.arange(len(top_genera))
    colors3 = [SPECIES_COLORS[i % len(SPECIES_COLORS)] for i in range(len(top_genera))]
    ax3.barh(y_pos, top_counts, color=colors3, alpha=0.8, edgecolor='black')
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(top_genera, fontsize=8)
    ax3.invert_yaxis()
    ax3.set_xlabel('Total Number of Trees')
    ax3.set_title(f'Top {len(top_genera)} Genera by Tree Count', fontweight='bold')
    ax3.grid(True, alpha=0.2, axis='x')

    # --- Plot 4: Stacked bar - size distribution by top genera ---
    ax4 = fig.add_subplot(gs[1, :2])
    bottom = np.zeros(len(SIZE_BINS))
    for i, genus in enumerate(top_genera):
        vals = genus_bins[genus]
        color = SPECIES_COLORS[i % len(SPECIES_COLORS)]
        ax4.bar(BIN_LABELS, vals, bottom=bottom, label=genus, color=color, alpha=0.8)
        bottom += vals
    # Add "Others"
    others = np.zeros(len(SIZE_BINS))
    for genus in genus_bins:
        if genus not in top_genera:
            others += genus_bins[genus]
    if others.sum() > 0:
        ax4.bar(BIN_LABELS, others, bottom=bottom, label='Others', color='#cccccc', alpha=0.8)
    ax4.set_xlabel('Diameter Class (cm)')
    ax4.set_ylabel('Number of Trees')
    ax4.set_title('Size Distribution by Genus', fontweight='bold')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7, frameon=False)
    ax4.grid(True, alpha=0.2, axis='y')

    # --- Plot 5: Biomass by top genera (pie) ---
    ax5 = fig.add_subplot(gs[1, 2])
    top_biomass = [genus_biomass[g] for g in top_genera]
    other_biomass = sum(genus_biomass[g] for g in genus_biomass if g not in top_genera)
    pie_labels = list(top_genera)
    pie_values = list(top_biomass)
    pie_colors = [SPECIES_COLORS[i % len(SPECIES_COLORS)] for i in range(len(top_genera))]
    if other_biomass > 0:
        pie_labels.append('Others')
        pie_values.append(other_biomass)
        pie_colors.append('#cccccc')
    # Filter out zero entries
    nonzero = [(l, v, c) for l, v, c in zip(pie_labels, pie_values, pie_colors) if v > 0]
    if nonzero:
        pie_labels, pie_values, pie_colors = zip(*nonzero)
        ax5.pie(pie_values, labels=pie_labels, colors=pie_colors, autopct='%1.1f%%',
                pctdistance=0.85, textprops={'fontsize': 7})
    ax5.set_title('Biomass C by Genus', fontweight='bold')

    fig.savefig(os.path.join(output_dir, 'conus_size_distribution.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved conus_size_distribution.png")

    # --- Summary stats ---
    n_sites = len(set(row['siteID'] for row in rows))
    print(f"\n  Summary:")
    print(f"    Sites: {n_sites}")
    print(f"    Total trees: {int(total_sum):,}")
    print(f"    Total biomass C: {sum(genus_biomass.values()):,.0f} kg C/m²")
    for label, count in zip(BIN_LABELS, total_bins):
        print(f"    {label:>6s} cm: {int(count):>12,} ({100*count/total_sum:5.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Plot CONUS-level tree size distribution from all sites"
    )
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Directory with site_NNNN/species_data.csv files")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading all site data...")
    rows = load_all_sites(args.input_dir)

    if not rows:
        print("ERROR: No data loaded")
        return 1

    print("\nGenerating CONUS plots...")
    plot_conus_size_distribution(rows, args.output_dir)

    print(f"\nPlots saved to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
