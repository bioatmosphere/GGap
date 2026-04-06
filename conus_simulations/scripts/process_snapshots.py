"""
Post-processing script to generate CSVs from raw simulation snapshots.

This script reads .npz snapshot files created during simulation and generates
the full set of CSV outputs (site_data, soil_data, species_data, genus_data, tree_data).

Usage:
    python process_snapshots.py --snapshot_dir ../results/simulation/snapshots --output_dir ../results/simulation
    python process_snapshots.py --snapshot_dir ../results/simulation/snapshots --output_dir ../results/simulation --no_tree_data
"""

import argparse
import os
import sys
import glob
import pickle
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from gap.output_utils import OutputWriter
from gap.constants import SiteP, SiteS, TreeP, TreeS


def build_tree_site_index(tree_ids, tree_to_gap, local_sites):
    """
    Build an array mapping each tree index to its site index.

    Instead of per-site boolean masks (N_trees × N_sites), this builds a single
    int array of length N_trees where each entry is the site index (0..N_sites-1),
    plus pre-sorted index arrays for fast slicing per site.

    Returns:
        Dict mapping site_id → sorted integer index array for fancy indexing
    """
    # Build gap_id → site_idx lookup
    gap_to_site_idx = {}
    for site_idx, site in enumerate(local_sites):
        for gid in site['gaps']:
            gap_to_site_idx[gid] = site_idx

    # Build gap_id array for all trees (one Python loop, unavoidable for dict)
    gap_ids = np.array([tree_to_gap.get(int(tid), -1) for tid in tree_ids], dtype=np.int64)

    # Vectorized lookup: gap_id → site_idx using numpy searchsorted
    unique_gaps = np.array(sorted(gap_to_site_idx.keys()), dtype=np.int64)
    gap_site_values = np.array([gap_to_site_idx[g] for g in unique_gaps], dtype=np.int32)

    insert_idx = np.searchsorted(unique_gaps, gap_ids)
    insert_idx = np.clip(insert_idx, 0, len(unique_gaps) - 1)
    matched = unique_gaps[insert_idx] == gap_ids
    tree_site_idx = np.where(matched, gap_site_values[insert_idx], -1)

    # Build per-site index arrays (sorted indices for each site)
    site_indices = {}
    for site_idx, site in enumerate(local_sites):
        sid = site['site_id']
        site_indices[sid] = np.where(tree_site_idx == site_idx)[0]

    return site_indices


def process_rank_snapshots(rank_id, snapshot_dir, output_dir, metadata_file, no_tree_data=False, rank_local=False):
    """
    Process all snapshots for a single rank and generate CSV outputs.

    Args:
        rank_id: MPI rank number
        snapshot_dir: Directory containing .npz snapshot files
        output_dir: Base output directory
        metadata_file: Path to rank_XXX_sites.pkl file
        no_tree_data: Skip writing tree_data.csv if True
        rank_local: If True, write to rank_NNN/site_NNN/ instead of flat site_NNNN/
    """
    # Load metadata
    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)

    local_sites = metadata['local_sites']
    species_by_id = metadata['species_by_id']
    tree_to_gap = metadata['tree_to_gap']
    tree_ids = metadata['tree_ids']  # Agent IDs in array order

    print(f"Rank {rank_id}: {len(local_sites)} sites")

    # Find all snapshot files for this rank
    snapshot_pattern = os.path.join(snapshot_dir, f"year_*_rank_{rank_id:03d}.npz")
    snapshot_files = sorted(glob.glob(snapshot_pattern))

    if not snapshot_files:
        print(f"  WARNING: No snapshot files found matching {snapshot_pattern}")
        return

    print(f"  Found {len(snapshot_files)} snapshots")

    # Pre-build tree-to-site index arrays
    print(f"  Building tree-to-site index arrays...")
    site_indices = build_tree_site_index(tree_ids, tree_to_gap, local_sites)
    print(f"  Indices built for {len(site_indices)} sites")

    # Create output writers for each site
    writers = {}
    for site in local_sites:
        sid = site['site_id']
        if rank_local:
            site_output_dir = os.path.join(output_dir, f"rank_{rank_id:03d}", f"site_{sid}")
        else:
            site_output_dir = os.path.join(output_dir, f"site_{sid:04d}")
        writer = OutputWriter(site_output_dir, site_id=sid)
        writer.open(species_by_id, len(site['gaps']))
        writers[sid] = writer

    # Pre-convert tree_ids to numpy array
    tree_ids_arr = np.array(tree_ids, dtype=np.int64)

    # Process each snapshot
    for snap_idx, snapshot_file in enumerate(snapshot_files):
        data = np.load(snapshot_file)
        year = int(data['year'])
        all_site_params = data['site_params']
        all_site_states = data['site_states']
        all_tree_params = data['tree_params']
        all_tree_states = data['tree_states']
        # tree_ids come from metadata, not snapshot (array indices match tree_ids order)

        # Progress indicator (every 10 snapshots)
        if snap_idx % 10 == 0 or snap_idx == len(snapshot_files) - 1:
            print(f"  Processing year {year}... ({snap_idx+1}/{len(snapshot_files)})")

        # Process each site
        for site_idx, site in enumerate(local_sites):
            sid = site['site_id']
            w = writers[sid]

            # Site data (one row per site)
            site_params = all_site_params[site_idx]
            site_states = all_site_states[site_idx]

            # Filter trees using pre-built index arrays
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
                species_by_id.get(int(sid), {}).get('evergreen', 0) > 0.5
                for sid in species_ids
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

            # Write CSVs
            w.write_site_data(
                year,
                site_params[SiteP.ANNUAL_RAIN],
                site_params[SiteP.POT_EVAP],
                site_params[SiteP.ACT_EVAP],
                site_params[SiteP.GROW_DAYS],
                site_states[SiteS.DEG_DAYS],
                site_states[SiteS.DRY_DAYS],
                site_states[SiteS.DRY_DAYS_BASE],
                site_states[SiteS.FLOOD_DAYS],
            )
            w.write_soil_data(
                year,
                site_params[SiteP.A0_C], site_params[SiteP.A_C],
                site_params[SiteP.A0_N], site_params[SiteP.A_N],
                site_params[SiteP.BL_C], site_params[SiteP.BL_N],
                site_states[SiteS.AVAIL_N],
                soilresp=site_params[SiteP.SOIL_RESP],
                c_into_a0=site_params[SiteP.C_INTO_A0],
                n_into_a0=site_params[SiteP.N_INTO_A0],
                net_n_into_a0=site_params[SiteP.NET_N_INTO_A0],
            )
            w.write_species_data(year, tree_data, list(site_gap_ids))
            w.write_genus_data(year, tree_data, list(site_gap_ids))
            if not no_tree_data:
                w.write_tree_data(year, tree_data, list(site_gap_ids))

    # Close all writers
    for w in writers.values():
        w.close()

    print(f"  Rank {rank_id} processing complete")


def main():
    parser = argparse.ArgumentParser(
        description="Process simulation snapshots to generate CSV outputs"
    )
    parser.add_argument("--snapshot_dir", type=str, required=True,
                       help="Directory containing .npz snapshot files")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Base output directory (same as simulation output_dir)")
    parser.add_argument("--ranks", type=str, default=None,
                       help="Comma-separated rank IDs to process (e.g., '0,1,2'). Default: all ranks")
    parser.add_argument("--no_tree_data", action="store_true",
                       help="Skip writing tree_data.csv (can be very large)")
    parser.add_argument("--rank_local", action="store_true",
                       help="Write to rank_NNN/site_NNN/ instead of flat site_NNNN/")

    args = parser.parse_args()

    # Find all rank metadata files
    metadata_pattern = os.path.join(args.output_dir, "rank_*_sites.pkl")
    metadata_files = sorted(glob.glob(metadata_pattern))

    if not metadata_files:
        print(f"ERROR: No rank metadata files found in {args.output_dir}")
        print(f"  Expected pattern: rank_XXX_sites.pkl")
        return 1

    print(f"Found {len(metadata_files)} rank metadata files")

    # Determine which ranks to process
    if args.ranks:
        ranks_to_process = [int(r) for r in args.ranks.split(',')]
    else:
        # Extract rank numbers from filenames
        ranks_to_process = []
        for mf in metadata_files:
            basename = os.path.basename(mf)
            # rank_000_sites.pkl → 0
            rank_str = basename.split('_')[1]
            ranks_to_process.append(int(rank_str))

    print(f"Processing ranks: {ranks_to_process}")
    print()

    # Process each rank
    for rank_id in ranks_to_process:
        metadata_file = os.path.join(args.output_dir, f"rank_{rank_id:03d}_sites.pkl")
        if not os.path.exists(metadata_file):
            print(f"WARNING: Metadata file not found: {metadata_file}")
            continue

        process_rank_snapshots(
            rank_id=rank_id,
            snapshot_dir=args.snapshot_dir,
            output_dir=args.output_dir,
            metadata_file=metadata_file,
            no_tree_data=args.no_tree_data,
            rank_local=args.rank_local,
        )

    print()
    print("All ranks processed successfully!")
    if args.rank_local:
        print(f"CSV outputs written to: {args.output_dir}/rank_NNN/site_NNN/")
    else:
        print(f"CSV outputs written to: {args.output_dir}/site_XXXX/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
