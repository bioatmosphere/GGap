#!/usr/bin/env python3
"""
Simple GGap Model Output Visualization (No Pandas)

Creates plots for CONUS simulation results using only numpy and matplotlib.

Usage:
    python plot_conus_site.py <site_dir> <output_plots_dir>
"""

import sys
import os
import csv
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

SPECIES_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#1a9850', '#d73027', '#4575b4', '#fee08b', '#313695',
    '#a50026', '#006837', '#542788', '#f46d43', '#74add1'
]
LINE_STYLES = ['-', '--', '-.', ':']
MARKERS = ['o', 's', '^', 'v', 'D', 'p', '*', 'h']


def read_csv(filepath):
    """Read CSV file into dict of arrays."""
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        data = {}
        for row in reader:
            for key, val in row.items():
                if key not in data:
                    data[key] = []
                try:
                    data[key].append(float(val))
                except ValueError:
                    data[key].append(val)

    # Convert to numpy arrays where possible
    for key in data:
        try:
            data[key] = np.array(data[key], dtype=float)
        except:
            pass  # Keep as list if can't convert

    return data


def pivot_species_data(species_data, value_col):
    """Pivot species_data into {species_name: array of values} keyed by unique years.

    Aggregates by species (summing across rows with the same year+species).
    Returns (sorted_years, dict of species_name -> values_array).
    """
    years_list = species_data['year']
    species_list = species_data['species']
    values = species_data[value_col]

    # Aggregate by (year, species)
    agg = defaultdict(lambda: defaultdict(float))
    for i in range(len(years_list)):
        agg[years_list[i]][species_list[i]] += values[i]

    sorted_years = np.array(sorted(agg.keys()))
    all_species = sorted({sp for yr in agg.values() for sp in yr.keys()})

    result = {}
    for sp in all_species:
        result[sp] = np.array([agg[yr].get(sp, 0.0) for yr in sorted_years])

    return sorted_years, result


def plot_forest_dynamics(species_data, site_data, save_path):
    """Plot forest composition and biomass by species."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Forest Dynamics Over Time', fontsize=14, fontweight='bold')

    # Pivot per-species data
    years, biomass_by_sp = pivot_species_data(species_data, 'total_biomC')
    _, basal_by_sp = pivot_species_data(species_data, 'basal_area')
    # Compute n_trees by summing diameter class columns
    diam_cols = ['<0', '0-8', '8-28', '-48', '-68', '-88', '>88']
    available_diam_cols = [c for c in diam_cols if c in species_data]
    if available_diam_cols:
        species_data['n_trees'] = sum(species_data[c] for c in available_diam_cols)
    _, ntrees_by_sp = pivot_species_data(species_data, 'n_trees') if 'n_trees' in species_data else (years, {})

    # Filter to species with nonzero biomass
    active_species = [sp for sp, vals in biomass_by_sp.items() if vals.max() > 0]
    color_map = {sp: SPECIES_COLORS[i % len(SPECIES_COLORS)] for i, sp in enumerate(active_species)}

    # Plot 1: Stacked area - Biomass Carbon by Species
    ax1 = axes[0, 0]
    stack_arrays = [biomass_by_sp[sp] for sp in active_species]
    if stack_arrays:
        ax1.stackplot(years, *stack_arrays, labels=active_species,
                      colors=[color_map[sp] for sp in active_species], alpha=0.7)
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Biomass Carbon (kg/m²)')
    ax1.set_title('Biomass Carbon by Species')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7, frameon=False)
    ax1.grid(True, alpha=0.2)

    # Plot 2: Tree Population by Species
    ax2 = axes[0, 1]
    for idx, sp in enumerate(active_species):
        if sp not in ntrees_by_sp:
            continue
        ls = LINE_STYLES[idx % len(LINE_STYLES)]
        mk = MARKERS[idx % len(MARKERS)] if len(active_species) <= 10 else None
        ax2.plot(years, ntrees_by_sp[sp], linewidth=2, label=sp, color=color_map[sp],
                 linestyle=ls, marker=mk, markersize=4,
                 markevery=max(1, len(years) // 10), alpha=0.9)
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Number of Trees')
    ax2.set_title('Tree Population by Species')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7, frameon=False)
    ax2.grid(True, alpha=0.2)

    # Plot 3: Basal Area by Species
    ax3 = axes[1, 0]
    for idx, sp in enumerate(active_species):
        ls = LINE_STYLES[idx % len(LINE_STYLES)]
        mk = MARKERS[idx % len(MARKERS)] if len(active_species) <= 10 else None
        ax3.plot(years, basal_by_sp[sp], linewidth=2, label=sp, color=color_map[sp],
                 linestyle=ls, marker=mk, markersize=4,
                 markevery=max(1, len(years) // 10), alpha=0.9)
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Basal Area (cm²)')
    ax3.set_title('Basal Area by Species')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7, frameon=False)
    ax3.grid(True, alpha=0.2)

    # Plot 4: Shannon Diversity Index
    ax4 = axes[1, 1]
    shannon = []
    for yi in range(len(years)):
        total = sum(biomass_by_sp[sp][yi] for sp in active_species)
        if total > 0:
            h = 0.0
            for sp in active_species:
                p = biomass_by_sp[sp][yi] / total
                if p > 0:
                    h -= p * np.log(p)
            shannon.append(h)
        else:
            shannon.append(0.0)
    ax4.plot(years, shannon, linewidth=2, color='#2E7D32')
    ax4.fill_between(years, shannon, alpha=0.2, color='#4CAF50')
    ax4.set_xlabel('Year')
    ax4.set_ylabel('Shannon Index')
    ax4.set_title('Species Diversity')
    ax4.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {os.path.basename(save_path)}")


def plot_soil_biogeochemistry(soil_data, save_path):
    """Plot soil carbon and nitrogen dynamics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Soil Biogeochemistry Over Time', fontsize=14, fontweight='bold')

    years = soil_data['year']
    a0c0 = soil_data['a0c0']
    ac0 = soil_data['ac0']
    a0n0 = soil_data['a0n0']
    an0 = soil_data['an0']

    # Plot 1: Soil Carbon Pools
    ax1 = axes[0, 0]
    ax1.plot(years, a0c0, linewidth=2, label='Surface C (A0)', color='#8B4513')
    ax1.plot(years, ac0, linewidth=2, label='Mineral C (A)', color='#D2691E')
    total_c = a0c0 + ac0
    ax1.plot(years, total_c, linewidth=3, label='Total C', color='#000000', linestyle='--')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Carbon Pool (kg C/m²)')
    ax1.set_title('Soil Carbon Pools')
    ax1.legend(frameon=False)
    ax1.grid(True, alpha=0.2)

    # Plot 2: Soil Nitrogen Pools
    ax2 = axes[0, 1]
    ax2.plot(years, a0n0, linewidth=2, label='Surface N (A0)', color='#1E90FF')
    ax2.plot(years, an0, linewidth=2, label='Mineral N (A)', color='#4169E1')
    total_n = a0n0 + an0
    ax2.plot(years, total_n, linewidth=3, label='Total N', color='#00008B', linestyle='--')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Nitrogen Pool (kg N/m²)')
    ax2.set_title('Soil Nitrogen Pools')
    ax2.legend(frameon=False)
    ax2.grid(True, alpha=0.2)

    # Plot 3: Available Nitrogen
    ax3 = axes[1, 0]
    if 'avail_n' in soil_data:
        avail_n = soil_data['avail_n'] * 1000  # Convert to g/m²
        ax3.plot(years, avail_n, linewidth=2, color='#228B22')
        ax3.set_ylabel('Available N (g N/m²)')
    ax3.set_xlabel('Year')
    ax3.set_title('Available Nitrogen')
    ax3.grid(True, alpha=0.2)

    # Plot 4: Soil C:N Ratios
    ax4 = axes[1, 1]
    cn_ratio_A0 = a0c0 / np.maximum(a0n0, 1e-10)
    cn_ratio_A = ac0 / np.maximum(an0, 1e-10)
    cn_ratio_total = total_c / np.maximum(total_n, 1e-10)

    ax4.plot(years, cn_ratio_A0, linewidth=2, label='Surface C:N', color='#FF8C00')
    ax4.plot(years, cn_ratio_A, linewidth=2, label='Mineral C:N', color='#DC143C')
    ax4.plot(years, cn_ratio_total, linewidth=3, label='Total C:N', color='#8B0000', linestyle='--')
    ax4.set_xlabel('Year')
    ax4.set_ylabel('C:N Ratio')
    ax4.set_title('Soil C:N Ratios')
    ax4.legend(frameon=False)
    ax4.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {os.path.basename(save_path)}")


def plot_environmental_conditions(site_data, save_path):
    """Plot environmental and climate conditions."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Environmental Conditions Over Time', fontsize=14, fontweight='bold')

    years = site_data['year']

    # Plot 1: Temperature
    ax1 = axes[0, 0]
    ax1.plot(years, site_data['degd'], linewidth=2, color='#DC143C', label='Degree Days')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Degree Days', color='#DC143C')
    ax1.set_title('Temperature Conditions')
    ax1.tick_params(axis='y', labelcolor='#DC143C')
    ax1.grid(True, alpha=0.2)

    if 'grow' in site_data:
        ax1_twin = ax1.twinx()
        ax1_twin.plot(years, site_data['grow'], linewidth=2, color='#2CA02C', label='Growing Days')
        ax1_twin.set_ylabel('Growing Days', color='#2CA02C')
        ax1_twin.tick_params(axis='y', labelcolor='#2CA02C')

    # Plot 2: Water balance
    ax2 = axes[0, 1]
    ax2.plot(years, site_data['rain'], linewidth=2, label='Rainfall', color='#1E90FF')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Rainfall (m/year)')
    ax2.set_title('Water Balance')
    ax2.legend(frameon=False)
    ax2.grid(True, alpha=0.2)

    # Plot 3: Drought stress
    ax3 = axes[1, 0]
    if 'dryd_upper' in site_data:
        ax3.plot(years, site_data['dryd_upper'], linewidth=2, label='Drought Days', color='#DC143C')
        ax3.set_ylabel('Drought Days')
    ax3.set_xlabel('Year')
    ax3.set_title('Drought Stress')
    if 'dryd_upper' in site_data:
        ax3.legend(frameon=False)
    ax3.grid(True, alpha=0.2)

    # Plot 4: Other environmental stress
    ax4 = axes[1, 1]
    if 'flood_d' in site_data:
        ax4.plot(years, site_data['flood_d'], linewidth=2, label='Flood Days', color='#1E90FF')
        ax4.set_ylabel('Days')
        ax4.legend(frameon=False)
    ax4.set_xlabel('Year')
    ax4.set_title('Other Environmental Stress')
    ax4.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {os.path.basename(save_path)}")


def plot_summary_dashboard(species_data, soil_data, site_data, save_path):
    """Create a comprehensive dashboard view."""
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('GGap Model Output Dashboard', fontsize=14, fontweight='bold')

    # Create grid
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    sp_years = species_data['year']
    biomass_c = species_data['total_biomC']
    n_trees = species_data['n_trees'] if 'n_trees' in species_data else np.ones_like(biomass_c)

    soil_years = soil_data['year']
    total_soil_c = soil_data['a0c0'] + soil_data['ac0']

    # Total ecosystem biomass
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(sp_years, biomass_c, linewidth=2, color='#2E7D32')
    ax1.fill_between(sp_years, biomass_c, alpha=0.2, color='#4CAF50')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Total Biomass Carbon (kg/m²)')
    ax1.set_title('Total Ecosystem Biomass', fontweight='bold')
    ax1.grid(True, alpha=0.2)

    # Final biomass fraction (just show total for single species data)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.text(0.5, 0.5, f'Final Biomass\n{biomass_c[-1]:.2f} kg C/m²',
             ha='center', va='center', transform=ax2.transAxes,
             fontsize=14, fontweight='bold')
    ax2.set_title(f'Biomass (Year {int(sp_years[-1])})', fontweight='bold')
    ax2.axis('off')

    # Total tree count
    ax3 = fig.add_subplot(gs[0, 3])
    ax3.plot(sp_years, n_trees, linewidth=2, color='#7B1FA2')
    ax3.fill_between(sp_years, n_trees, alpha=0.2, color='#9C27B0')
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Total Trees')
    ax3.set_title('Tree Population', fontweight='bold')
    ax3.grid(True, alpha=0.2)

    # Soil carbon accumulation
    ax4 = fig.add_subplot(gs[1, :2])
    ax4.plot(soil_years, total_soil_c, linewidth=2, color='#6D4C41')
    ax4.fill_between(soil_years, total_soil_c, alpha=0.2, color='#8D6E63')
    ax4.set_xlabel('Year')
    ax4.set_ylabel('Total Soil Carbon (kg C/m²)')
    ax4.set_title('Soil Carbon Accumulation', fontweight='bold')
    ax4.grid(True, alpha=0.2)

    # Environmental stress
    ax5 = fig.add_subplot(gs[1, 2:])
    if 'dryd_upper' in site_data:
        ax5.plot(site_data['year'], site_data['dryd_upper'],
                linewidth=2, label='Drought Days', color='#DC143C')
    if 'flood_d' in site_data:
        ax5.plot(site_data['year'], site_data['flood_d'],
                linewidth=2, label='Flood Days', color='#1E90FF')
    ax5.set_xlabel('Year')
    ax5.set_ylabel('Stress Indicator')
    ax5.set_title('Environmental Stress', fontweight='bold')
    if 'dryd_upper' in site_data or 'flood_d' in site_data:
        ax5.legend(frameon=False)
    ax5.grid(True, alpha=0.2)

    # Forest productivity
    ax6 = fig.add_subplot(gs[2, :2])
    productivity = biomass_c / np.maximum(n_trees, 1)
    ax6.plot(sp_years, productivity, linewidth=2, color='#F57C00')
    ax6.fill_between(sp_years, productivity, alpha=0.2, color='#FF9800')
    ax6.set_xlabel('Year')
    ax6.set_ylabel('Biomass per Tree (kg C/tree)')
    ax6.set_title('Forest Productivity', fontweight='bold')
    ax6.grid(True, alpha=0.2)

    # Climate summary
    ax7 = fig.add_subplot(gs[2, 2:])
    ax7.plot(site_data['year'], site_data['rain'], linewidth=2, color='#1E90FF', label='Rainfall')
    ax7.set_xlabel('Year')
    ax7.set_ylabel('Rainfall (m/year)')
    ax7.set_title('Climate', fontweight='bold')
    ax7.legend(frameon=False)
    ax7.grid(True, alpha=0.2)

    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {os.path.basename(save_path)}")


def plot_size_distribution(species_data, save_path):
    """Plot number of trees by diameter size bins over time."""
    size_bins = ['0-8', '8-28', '-48', '-68', '-88', '>88']
    bin_labels = ['0-8', '8-28', '28-48', '48-68', '68-88', '>88']
    bin_colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728', '#9467bd', '#8c564b']

    # Aggregate size bins across all species per year
    years, bins_by_class = {}, {}
    for col, label in zip(size_bins, bin_labels):
        if col in species_data:
            yr, by_sp = pivot_species_data(species_data, col)
            # Sum across all species
            total = np.zeros(len(yr))
            for vals in by_sp.values():
                total += vals
            bins_by_class[label] = total
            years = yr

    if not bins_by_class:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Tree Size Distribution Over Time', fontsize=14, fontweight='bold')

    # Plot 1: Stacked area of size classes over time
    ax1 = axes[0]
    labels_present = [l for l in bin_labels if l in bins_by_class]
    stack_arrays = [bins_by_class[l] for l in labels_present]
    colors = [bin_colors[bin_labels.index(l)] for l in labels_present]
    ax1.stackplot(years, *stack_arrays, labels=[f'{l} cm' for l in labels_present],
                  colors=colors, alpha=0.7)
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Number of Trees')
    ax1.set_title('Size Class Distribution')
    ax1.legend(loc='upper left', fontsize=8, frameon=False)
    ax1.grid(True, alpha=0.2)

    # Plot 2: Bar chart at final year
    ax2 = axes[1]
    final_idx = -1
    final_counts = [bins_by_class[l][final_idx] for l in labels_present]
    bars = ax2.bar([f'{l} cm' for l in labels_present], final_counts, color=colors, alpha=0.7,
                   edgecolor='black')
    ax2.set_xlabel('Diameter Class')
    ax2.set_ylabel('Number of Trees')
    ax2.set_title(f'Size Distribution (Year {int(years[final_idx])})')
    ax2.grid(True, alpha=0.2, axis='y')
    for bar, count in zip(bars, final_counts):
        if count > 0:
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f'{int(count)}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {os.path.basename(save_path)}")


def main():
    if len(sys.argv) < 3:
        print("Usage: python plot_conus_site.py <site_dir> <output_plots_dir>")
        sys.exit(1)

    site_dir = Path(sys.argv[1])
    plots_dir = Path(sys.argv[2])
    plots_dir.mkdir(exist_ok=True, parents=True)

    print(f"Loading data from: {site_dir}")

    # Load data files
    species_data = read_csv(site_dir / "species_data.csv")
    soil_data = read_csv(site_dir / "soil_data.csv")
    site_data = read_csv(site_dir / "site_data.csv")

    print(f"Creating plots...")

    # Create all plots
    plot_forest_dynamics(species_data, site_data, plots_dir / "forest_dynamics.png")
    plot_soil_biogeochemistry(soil_data, plots_dir / "soil_biogeochemistry.png")
    plot_environmental_conditions(site_data, plots_dir / "environmental_conditions.png")
    plot_size_distribution(species_data, plots_dir / "size_distribution.png")
    plot_summary_dashboard(species_data, soil_data, site_data, plots_dir / "summary_dashboard.png")

    print(f"\nAll plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()
