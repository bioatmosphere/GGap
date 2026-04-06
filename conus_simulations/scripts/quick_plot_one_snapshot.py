#!/usr/bin/env python3
"""
Quick script to extract and plot data from one snapshot for one site.
No CSV writing, just direct plotting from numpy data.

Usage: python quick_plot_one_snapshot.py <snapshot.npz> <metadata.pkl> <site_id> <output_dir>
"""

import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    if len(sys.argv) < 5:
        print("Usage: python quick_plot_one_snapshot.py <snapshot.npz> <metadata.pkl> <site_id> <output_dir>")
        sys.exit(1)

    snapshot_file = sys.argv[1]
    metadata_file = sys.argv[2]
    target_site_id = int(sys.argv[3])
    output_dir = Path(sys.argv[4])
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"Loading metadata from {metadata_file}...")
    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)

    local_sites = metadata['local_sites']
    tree_to_gap = metadata['tree_to_gap']
    tree_ids = metadata['tree_ids']
    species_by_id = metadata['species_by_id']

    # Find target site
    target_site = None
    site_idx = None
    for idx, site in enumerate(local_sites):
        if site['site_id'] == target_site_id:
            target_site = site
            site_idx = idx
            break

    if target_site is None:
        print(f"ERROR: Site {target_site_id} not found in this rank")
        print(f"Available sites: {[s['site_id'] for s in local_sites]}")
        sys.exit(1)

    print(f"Found site {target_site_id} at index {site_idx}")
    print(f"Site has {len(target_site['gaps'])} gaps")

    print(f"\nLoading snapshot from {snapshot_file}...")
    data = np.load(snapshot_file)
    year = int(data['year'])

    # Extract site data
    site_params = data['site_params'][site_idx]
    site_states = data['site_states'][site_idx]

    # Filter trees for this site
    site_gap_ids = set(target_site['gaps'])
    tree_mask = np.array([
        tree_to_gap.get(int(tid), -1) in site_gap_ids
        for tid in tree_ids
    ], dtype=bool)

    site_tree_params = data['tree_params'][tree_mask]
    site_tree_states = data['tree_states'][tree_mask]
    site_tree_ids = np.array(tree_ids)[tree_mask]

    # Filter to living trees
    TreeS_IS_ALIVE = 0
    TreeS_DIAM = 1
    TreeS_HEIGHT = 2
    TreeS_SPECIES_ID = 3
    TreeS_CANOPY_HT = 4

    TreeP_BIOMC = 0
    TreeP_BIOMN = 1
    TreeP_LEAF_BM = 2
    TreeP_AGE = 3

    alive_mask = site_tree_states[:, TreeS_IS_ALIVE] > 0.5
    n_alive = alive_mask.sum()

    print(f"\nYear {year} Summary:")
    print(f"  Total trees in site: {len(site_tree_ids)}")
    print(f"  Living trees: {n_alive}")

    if n_alive == 0:
        print("No living trees, skipping plot")
        return

    alive_states = site_tree_states[alive_mask]
    alive_params = site_tree_params[alive_mask]

    diameters = alive_states[:, TreeS_DIAM]
    heights = alive_states[:, TreeS_HEIGHT]
    biomass_c = alive_params[:, TreeP_BIOMC]
    ages = alive_params[:, TreeP_AGE]

    print(f"  Mean diameter: {diameters.mean():.2f} cm")
    print(f"  Mean height: {heights.mean():.2f} m")
    print(f"  Total biomass C: {biomass_c.sum():.2f} kg C")
    print(f"  Mean age: {ages.mean():.1f} years")

    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Site {target_site_id} - Year {year}', fontsize=14, fontweight='bold')

    # Diameter distribution
    ax1 = axes[0, 0]
    ax1.hist(diameters, bins=30, color='#2E7D32', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Diameter (cm)')
    ax1.set_ylabel('Number of Trees')
    ax1.set_title(f'Diameter Distribution (n={n_alive})')
    ax1.grid(True, alpha=0.3)

    # Height distribution
    ax2 = axes[0, 1]
    ax2.hist(heights, bins=30, color='#1976D2', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Height (m)')
    ax2.set_ylabel('Number of Trees')
    ax2.set_title('Height Distribution')
    ax2.grid(True, alpha=0.3)

    # Diameter vs Height
    ax3 = axes[1, 0]
    ax3.scatter(diameters, heights, alpha=0.5, s=20, c=biomass_c, cmap='viridis')
    ax3.set_xlabel('Diameter (cm)')
    ax3.set_ylabel('Height (m)')
    ax3.set_title('Diameter vs Height (colored by biomass)')
    ax3.grid(True, alpha=0.3)
    cbar = plt.colorbar(ax3.collections[0], ax=ax3, label='Biomass C (kg)')

    # Age distribution
    ax4 = axes[1, 1]
    ax4.hist(ages, bins=30, color='#D32F2F', alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Age (years)')
    ax4.set_ylabel('Number of Trees')
    ax4.set_title('Age Distribution')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = output_dir / f"site_{target_site_id}_year_{year}.png"
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"\nPlot saved to: {output_file}")

if __name__ == "__main__":
    main()
