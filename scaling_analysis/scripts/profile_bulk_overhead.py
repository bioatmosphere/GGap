"""
Profile the overhead in bulk initialization to understand what's slow.

Usage:
    python profile_bulk_overhead.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from gap.gap_model import GAPModel


def profile_bulk_initialization():
    """Profile where time is spent in bulk initialization."""
    print("="*70)
    print("Profiling Bulk Initialization Overhead")
    print("="*70)

    # Setup
    model = GAPModel()
    model.load_globals(prefix='CONUS')
    all_site_ids = sorted(model._site_id_to_slot.keys())
    site_ids = [sid for sid in all_site_ids if sid in model._climate_rows][:1]
    model.partition_sites(site_ids)

    num_gaps = 500
    maxtrees = 1000
    site_id = site_ids[0]

    print(f"Testing 1 site: {site_id}")
    print(f"Config: {num_gaps} gaps, {maxtrees} trees per gap")
    print()

    # Get site data (mimicking what bulk method does)
    site_params, site_states, site_info = model._build_site_params_states(
        site_id, "input_data", "CONUS"
    )

    # Get species
    import csv
    base_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "input_data"
    )
    range_file = os.path.join(base_path, "CONUS_rangelist.csv")

    with open(range_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row.get('site', -1)) == site_id:
                species_present = set()
                for col, val in row.items():
                    if col not in ('site', 'latitude', 'longitude'):
                        if val == '1':
                            species_present.add(col)
                site_species = sorted(
                    [model.unique_species[sp_code] for sp_code in species_present
                     if sp_code in model.unique_species],
                    key=lambda sp: sp['global_id']
                )
                break

    deg_days = site_info['deg_days']
    dry_days = site_info['dry_days']

    print(f"Site has {len(site_species)} species")
    print()

    # ========================================================================
    # Time: Building gap params arrays
    # ========================================================================
    print("Building gap params arrays...")
    t_start = time.time()
    gap_params_list = []
    gap_states_list = []
    for gap_idx in range(num_gaps):
        gap_id = gap_idx
        gap_params, gap_states = model._build_gap_params_states(gap_id, deg_days, dry_days)
        gap_params_list.append(gap_params)
        gap_states_list.append(gap_states)
    t_gap_build = time.time() - t_start
    print(f"  Time: {t_gap_build:.3f}s")

    # ========================================================================
    # Time: Converting to numpy arrays
    # ========================================================================
    print("Converting to numpy arrays...")
    t_start = time.time()
    gap_params_array = np.array(gap_params_list)
    gap_states_array = np.array(gap_states_list)
    t_gap_convert = time.time() - t_start
    print(f"  Time: {t_gap_convert:.3f}s")

    # ========================================================================
    # Time: Building tree params arrays
    # ========================================================================
    trees_per_gap = len(site_species) + maxtrees
    total_trees = num_gaps * trees_per_gap

    print(f"Building tree params arrays ({total_trees:,} trees)...")
    t_start = time.time()
    tree_params_list = []
    tree_states_list = []

    for gap_idx in range(num_gaps):
        gap_agent_id = gap_idx  # Dummy ID

        # Template trees
        for species_info in site_species:
            params, states = model._build_tree_params_states(
                gap_agent_id, species_info, diam=0.0, age=0.0, is_alive=-1.0
            )
            tree_params_list.append(params)
            tree_states_list.append(states)

        # Free slots
        placeholder_species = site_species[0]
        for _ in range(maxtrees):
            params, states = model._build_tree_params_states(
                gap_agent_id, placeholder_species, diam=0.0, age=0.0, is_alive=0.0
            )
            tree_params_list.append(params)
            tree_states_list.append(states)

    t_tree_build = time.time() - t_start
    print(f"  Time: {t_tree_build:.3f}s")

    # ========================================================================
    # Time: Converting tree arrays
    # ========================================================================
    print("Converting tree arrays to numpy...")
    t_start = time.time()
    tree_params_array = np.array(tree_params_list)
    tree_states_array = np.array(tree_states_list)
    t_tree_convert = time.time() - t_start
    print(f"  Time: {t_tree_convert:.3f}s")

    # ========================================================================
    # Summary
    # ========================================================================
    print()
    print("="*70)
    print("OVERHEAD BREAKDOWN")
    print("="*70)
    total_overhead = t_gap_build + t_gap_convert + t_tree_build + t_tree_convert
    print(f"  Gap params building:     {t_gap_build:.3f}s")
    print(f"  Gap array conversion:    {t_gap_convert:.3f}s")
    print(f"  Tree params building:    {t_tree_build:.3f}s  ({total_trees:,} trees)")
    print(f"  Tree array conversion:   {t_tree_convert:.3f}s")
    print(f"  TOTAL overhead:          {total_overhead:.3f}s")
    print()
    print(f"This is BEFORE calling create_agents_bulk()!")
    print(f"Expected per-site overhead in bulk method: ~{total_overhead:.1f}s")
    print()


if __name__ == '__main__':
    profile_bulk_initialization()
