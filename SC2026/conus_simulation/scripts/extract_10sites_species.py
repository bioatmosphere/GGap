#!/usr/bin/env python3
"""
Extract data for 10 representative CONUS sites from simulation output.

Processes ALL snapshots (not just the last) to produce full time-series
species_data.csv files needed by the conus_10sites_with_bars figure.

Produces:
  results/10sites/10sites_species.csv         — summary table (species counts at final year)
  results/10sites/site_NNNN/species_data.csv  — full time-series species detail (all years)

Usage:
    python extract_10sites_species.py
    python extract_10sites_species.py --results_dir ../results/simulation --output_dir ../results/10sites
"""

import argparse
import csv
import os
import sys
import glob
import pickle
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
from gap.output_utils import OutputWriter
from gap.constants import TreeP, TreeS

# 10 representative sites (must match conus_10sites_with_bars.py)
SITES = [
    (1,  "Pacific NW",         83, 47.25, -122.25),
    (2,  "N. Rockies",        124, 46.75, -114.75),
    (3,  "Sierra Nevada",     690, 39.25, -120.75),
    (4,  "S. Rockies",        913, 36.25, -106.75),
    (5,  "Upper Midwest",     236, 45.75,  -88.75),
    (6,  "Northeast",         349, 44.75,  -72.75),
    (7,  "Central Appal.",    784, 38.25,  -80.25),
    (8,  "S. Appalachia",     978, 35.75,  -82.75),
    (9,  "SE Coastal",       1274, 31.75,  -81.75),
    (10, "Gulf Coast",       1313, 30.75,  -92.75),
]

TARGET_SITE_IDS = {s[2] for s in SITES}


def find_all_snapshots(snapshot_dir, rank_id):
    """Find all snapshot files for a rank, sorted by year."""
    pattern = os.path.join(snapshot_dir, f"year_*_rank_{rank_id:03d}.npz")
    return sorted(glob.glob(pattern))


def build_tree_site_index(tree_ids, tree_to_gap, local_sites):
    gap_to_site_idx = {}
    for site_idx, site in enumerate(local_sites):
        for gid in site['gaps']:
            gap_to_site_idx[gid] = site_idx

    gap_ids = np.array([tree_to_gap.get(int(tid), -1) for tid in tree_ids], dtype=np.int64)
    unique_gaps = np.array(sorted(gap_to_site_idx.keys()), dtype=np.int64)
    gap_site_values = np.array([gap_to_site_idx[g] for g in unique_gaps], dtype=np.int32)

    insert_idx = np.searchsorted(unique_gaps, gap_ids)
    insert_idx = np.clip(insert_idx, 0, len(unique_gaps) - 1)
    matched = unique_gaps[insert_idx] == gap_ids
    tree_site_idx = np.where(matched, gap_site_values[insert_idx], -1)

    site_indices = {}
    for site_idx, site in enumerate(local_sites):
        site_indices[site['site_id']] = np.where(tree_site_idx == site_idx)[0]

    return site_indices


def extract_tree_data(all_tree_params, all_tree_states, idx, tree_ids_arr, tree_to_gap, species_by_id):
    """Extract tree_data dict for one site from snapshot arrays."""
    site_tree_params = all_tree_params[idx]
    site_tree_states = all_tree_states[idx]
    site_tree_ids = tree_ids_arr[idx]

    alive_mask = site_tree_states[:, TreeS.IS_ALIVE] > 0.5
    alive_params = site_tree_params[alive_mask]
    alive_states = site_tree_states[alive_mask]
    alive_ids = site_tree_ids[alive_mask].astype(np.int32)

    gap_ids = np.array([tree_to_gap[int(a)] for a in alive_ids], dtype=np.int32)
    species_ids = alive_states[:, TreeS.SPECIES_ID].astype(np.int32)
    evergreen = np.array([
        species_by_id.get(int(spid), {}).get('evergreen', 0) > 0.5
        for spid in species_ids
    ], dtype=bool)

    return {
        'count': int(alive_mask.sum()),
        'gap_agent_id': gap_ids,
        'species_id': species_ids,
        'diam': alive_states[:, TreeS.DIAM],
        'height': alive_states[:, TreeS.HEIGHT],
        'biomC': alive_params[:, TreeP.BIOMC],
        'biomN': alive_params[:, TreeP.BIOMN],
        'leaf_bm': alive_params[:, TreeP.LEAF_BM],
        'age': alive_params[:, TreeP.AGE],
        'canopy_ht': alive_states[:, TreeS.CANOPY_HT],
        'evergreen': evergreen,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract data for 10 representative CONUS sites (all years)"
    )
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--results_dir", type=str,
                        default=os.path.join(script_dir, "..", "results", "simulation"),
                        help="Simulation results directory (with .pkl and snapshots/)")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(script_dir, "..", "results", "10sites"),
                        help="Output directory (default: ../results/10sites)")
    args = parser.parse_args()

    results_dir = args.results_dir
    output_dir = args.output_dir
    snapshot_dir = os.path.join(results_dir, "snapshots")
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Scan rank metadata to find which ranks contain target sites
    metadata_files = sorted(glob.glob(os.path.join(results_dir, "rank_*_sites.pkl")))
    if not metadata_files:
        print(f"ERROR: No rank metadata files found in {results_dir}")
        return 1

    print(f"Scanning {len(metadata_files)} rank metadata files...")

    site_to_rank = {}
    rank_metadata_cache = {}

    for mf in metadata_files:
        rank_id = int(os.path.basename(mf).split('_')[1])
        with open(mf, 'rb') as f:
            metadata = pickle.load(f)
        for site in metadata['local_sites']:
            if site['site_id'] in TARGET_SITE_IDS:
                site_to_rank[site['site_id']] = rank_id
                rank_metadata_cache[rank_id] = metadata

    missing = TARGET_SITE_IDS - set(site_to_rank.keys())
    if missing:
        print(f"WARNING: Sites not found in any rank: {missing}")

    print(f"Found 10 sites across {len(rank_metadata_cache)} ranks")

    # Step 2: For each relevant rank, open OutputWriters and process ALL snapshots
    # Group target sites by rank
    rank_to_sites = defaultdict(list)
    for sid, rank_id in site_to_rank.items():
        rank_to_sites[rank_id].append(sid)

    species_counts = {}  # final-year species counts for summary
    last_year = None

    for rank_id in sorted(rank_to_sites.keys()):
        metadata = rank_metadata_cache[rank_id]
        local_sites = metadata['local_sites']
        species_by_id = metadata['species_by_id']
        tree_to_gap = metadata['tree_to_gap']
        tree_ids = metadata['tree_ids']

        site_indices = build_tree_site_index(tree_ids, tree_to_gap, local_sites)
        tree_ids_arr = np.array(tree_ids, dtype=np.int64)

        # Build site info lookup
        site_info = {}
        for site in local_sites:
            if site['site_id'] in TARGET_SITE_IDS:
                site_info[site['site_id']] = site

        # Open OutputWriters for each target site on this rank
        writers = {}
        for sid in rank_to_sites[rank_id]:
            site_output_dir = os.path.join(output_dir, f"site_{sid:04d}")
            w = OutputWriter(site_output_dir, site_id=sid)
            w.open(species_by_id, len(site_info[sid]['gaps']))
            writers[sid] = w

        # Process all snapshots for this rank
        snapshot_files = find_all_snapshots(snapshot_dir, rank_id)
        print(f"\n  Rank {rank_id}: {len(snapshot_files)} snapshots, "
              f"sites {sorted(rank_to_sites[rank_id])}")

        for snap_idx, snapshot_file in enumerate(snapshot_files):
            data = np.load(snapshot_file)
            year = int(data['year'])
            all_tree_params = data['tree_params']
            all_tree_states = data['tree_states']

            for sid in rank_to_sites[rank_id]:
                idx = site_indices[sid]
                site_gap_ids = list(site_info[sid]['gaps'])
                tree_data = extract_tree_data(
                    all_tree_params, all_tree_states, idx,
                    tree_ids_arr, tree_to_gap, species_by_id
                )
                writers[sid].write_species_data(year, tree_data, site_gap_ids)

            if snap_idx % 20 == 0 or snap_idx == len(snapshot_files) - 1:
                print(f"    Year {year} ({snap_idx+1}/{len(snapshot_files)})")

        # Close writers
        for w in writers.values():
            w.close()

        # Compute final-year species counts from last snapshot
        last_year = year
        last_data = np.load(snapshot_files[-1])
        all_tree_params = last_data['tree_params']
        all_tree_states = last_data['tree_states']

        for sid in rank_to_sites[rank_id]:
            idx = site_indices[sid]
            site_tree_params = all_tree_params[idx]
            site_tree_states = all_tree_states[idx]

            alive_mask = site_tree_states[:, TreeS.IS_ALIVE] > 0.5
            alive_params = site_tree_params[alive_mask]
            alive_states = site_tree_states[alive_mask]
            species_ids = alive_states[:, TreeS.SPECIES_ID].astype(np.int32)

            species_biomass = defaultdict(float)
            for sp_id, bc in zip(species_ids, alive_params[:, TreeP.BIOMC]):
                species_biomass[int(sp_id)] += float(bc)

            species_names = []
            for sp_id, bc in sorted(species_biomass.items(), key=lambda x: -x[1]):
                if bc > 0:
                    sp_info = species_by_id.get(sp_id, {})
                    species_names.append(sp_info.get('species_code', f'sp_{sp_id}'))

            species_counts[sid] = {
                'count': len(species_names),
                'year': last_year,
                'alive_trees': int(alive_mask.sum()),
                'species_names': species_names,
            }

    # Step 3: Print summary table
    print(f"\n{'='*70}")
    print(f"Species counts at 10 representative sites (year {last_year})")
    print(f"{'='*70}")
    print(f"{'#':>2}  {'Region':<18} {'Site':>6}  {'Spp':>4}  {'Trees':>8}")
    print(f"{'-'*70}")

    for num, region, sid, lat, lon in SITES:
        if sid in species_counts:
            sc = species_counts[sid]
            print(f"{num:>2}  {region:<18} {sid:>6}  {sc['count']:>4}  {sc['alive_trees']:>8,}")
        else:
            print(f"{num:>2}  {region:<18} {sid:>6}  {'N/A':>4}  {'N/A':>8}")

    print(f"{'='*70}")

    # Step 4: Write summary CSV
    summary_csv = os.path.join(output_dir, "10sites_species.csv")
    with open(summary_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['site_num', 'region', 'site_id', 'lat', 'lon', 'year',
                         'species_count', 'alive_trees', 'species_list'])
        for num, region, sid, lat, lon in SITES:
            if sid in species_counts:
                sc = species_counts[sid]
                writer.writerow([num, region, sid, lat, lon, sc['year'],
                                 sc['count'], sc['alive_trees'],
                                 ';'.join(sc['species_names'])])

    print(f"\nOutputs saved to {output_dir}/")
    print(f"  10sites_species.csv              — summary table")
    print(f"  site_NNNN/species_data.csv       — per-site species detail (all 100 years, 10 sites)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
