#!/usr/bin/env python3
"""
Extract species_data.csv for all sites from all ranks using only the last snapshot.

Much faster than process_snapshots.py since it reads only 1 snapshot per rank
and only writes species_data.csv.

Usage:
    python extract_last_species_data.py --snapshot_dir ../results/simulation/snapshots \
        --metadata_dir ../results/simulation --output_dir ../results/simulation/last_year_species

    # Process specific ranks
    python extract_last_species_data.py --snapshot_dir ../results/simulation/snapshots \
        --metadata_dir ../results/simulation --output_dir ../results/simulation/last_year_species \
        --ranks 0,1,2
"""

import argparse
import os
import sys
import glob
import csv
import pickle
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from gap.output_utils import OutputWriter
from gap.constants import TreeP, TreeS


def build_tree_site_index(tree_ids, tree_to_gap, local_sites):
    """Build per-site index arrays for fast tree filtering."""
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


def find_last_snapshot(snapshot_dir, rank_id):
    """Find the last snapshot file for a given rank."""
    pattern = os.path.join(snapshot_dir, f"year_*_rank_{rank_id:03d}.npz")
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def process_rank(rank_id, snapshot_file, metadata_file, output_dir, species_by_id_override=None):
    """Process the last snapshot for one rank, writing species_data.csv per site."""
    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)

    local_sites = metadata['local_sites']
    species_by_id = metadata['species_by_id']
    tree_to_gap = metadata['tree_to_gap']
    tree_ids = metadata['tree_ids']

    print(f"  Rank {rank_id}: {len(local_sites)} sites, {len(tree_ids):,} trees")

    # Build index
    site_indices = build_tree_site_index(tree_ids, tree_to_gap, local_sites)
    tree_ids_arr = np.array(tree_ids, dtype=np.int64)

    # Load snapshot
    data = np.load(snapshot_file)
    year = int(data['year'])
    all_tree_params = data['tree_params']
    all_tree_states = data['tree_states']

    print(f"  Snapshot year: {year}")

    site_count = 0
    for site_idx, site in enumerate(local_sites):
        sid = site['site_id']
        site_gap_ids = set(site['gaps'])
        idx = site_indices[sid]

        site_tree_params = all_tree_params[idx]
        site_tree_states = all_tree_states[idx]
        site_tree_ids = tree_ids_arr[idx]

        # Filter to living trees
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

        tree_data = {
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

        # Write species_data.csv using OutputWriter
        site_output_dir = os.path.join(output_dir, f"site_{sid:04d}")
        w = OutputWriter(site_output_dir, site_id=sid)
        w.open(species_by_id, len(site['gaps']))
        w.write_species_data(year, tree_data, list(site_gap_ids))
        w.close()
        site_count += 1

    return site_count, year


def main():
    parser = argparse.ArgumentParser(
        description="Extract species_data.csv from the last snapshot for all sites"
    )
    parser.add_argument("--snapshot_dir", type=str, required=True,
                       help="Directory containing .npz snapshot files")
    parser.add_argument("--metadata_dir", type=str, required=True,
                       help="Directory containing rank_NNN_sites.pkl files")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for species_data.csv files")
    parser.add_argument("--ranks", type=str, default=None,
                       help="Comma-separated rank IDs (e.g., '0,1,2'). Default: all ranks")

    args = parser.parse_args()

    # Find all rank metadata files
    metadata_files = sorted(glob.glob(os.path.join(args.metadata_dir, "rank_*_sites.pkl")))
    if not metadata_files:
        print(f"ERROR: No rank metadata files found in {args.metadata_dir}")
        return 1

    # Determine ranks to process
    if args.ranks:
        ranks_to_process = [int(r) for r in args.ranks.split(',')]
    else:
        ranks_to_process = []
        for mf in metadata_files:
            rank_str = os.path.basename(mf).split('_')[1]
            ranks_to_process.append(int(rank_str))

    print(f"Found {len(metadata_files)} rank metadata files")
    print(f"Processing {len(ranks_to_process)} ranks")
    print(f"Output directory: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    total_sites = 0
    snapshot_year = None

    for rank_id in ranks_to_process:
        metadata_file = os.path.join(args.metadata_dir, f"rank_{rank_id:03d}_sites.pkl")
        if not os.path.exists(metadata_file):
            print(f"  WARNING: Metadata file not found for rank {rank_id}")
            continue

        snapshot_file = find_last_snapshot(args.snapshot_dir, rank_id)
        if snapshot_file is None:
            print(f"  WARNING: No snapshots found for rank {rank_id}")
            continue

        n_sites, year = process_rank(rank_id, snapshot_file, metadata_file, args.output_dir)
        total_sites += n_sites
        snapshot_year = year

    print(f"\nDone! Extracted species_data.csv for {total_sites} sites (year {snapshot_year})")
    print(f"Output: {args.output_dir}/site_NNNN/species_data.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
