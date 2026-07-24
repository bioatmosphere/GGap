#!/usr/bin/env python3
"""
Extract a per-site species_data.csv TIME SERIES (all snapshot years) for chosen sites.

`extract.py` reads only the final snapshot (one year). To plot species composition over
time you need every snapshot. This loops all `year_*_rank_NNN.npz` snapshots and appends a
per-year row-block to each site's species_data.csv (same format the paper figure expects).

Usage (defaults reproduce the TN paper-comparison site):
    python extract_timeseries.py                 # site 978 (S. Appalachia), all years
    python extract_timeseries.py --sites 978 349 # several sites
"""
import argparse
import glob
import os
import pickle
import sys

import numpy as np

# Make the GGap `gap` package importable (tn_example -> SC2026 -> GGap root), then reuse
# the exact per-site tree-index logic from extract.py so both extractors stay in sync.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from extract import build_tree_site_index          # noqa: E402  (reused, identical logic)
from gap.output_utils import OutputWriter           # noqa: E402
from gap.constants import TreeP, TreeS              # noqa: E402


def find_all_snapshots(snapshot_dir, rank_id):
    """All snapshots for a rank, sorted ascending by year."""
    pattern = os.path.join(snapshot_dir, f"year_*_rank_{rank_id:03d}.npz")
    return sorted(glob.glob(pattern))


def build_tree_data(site, idx, tree_ids_arr, tree_to_gap, species_by_id,
                    all_tree_params, all_tree_states):
    """Build the OutputWriter tree_data dict for one site from one snapshot.

    Mirrors extract.process_rank's per-site block (living-tree filter + field pack).
    """
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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description="Extract per-site species_data.csv time series (all years).")
    p.add_argument("--sites", type=int, nargs="+", default=[978],
                   help="Site IDs to extract (default: 978 = S. Appalachia, one of the paper's 10).")
    p.add_argument("--snapshot_dir", type=str, default=os.path.join(script_dir, "results", "snapshots"))
    p.add_argument("--metadata_dir", type=str, default=os.path.join(script_dir, "results"))
    p.add_argument("--output_dir", type=str, default=os.path.join(script_dir, "results", "species_timeseries"))
    args = p.parse_args()

    wanted = set(args.sites)
    metadata_files = sorted(glob.glob(os.path.join(args.metadata_dir, "rank_*_sites.pkl")))
    if not metadata_files:
        print(f"ERROR: no rank_*_sites.pkl in {args.metadata_dir}")
        return 1
    os.makedirs(args.output_dir, exist_ok=True)

    remaining = set(wanted)
    for mf in metadata_files:
        rank_id = int(os.path.basename(mf).split('_')[1])
        with open(mf, 'rb') as f:
            meta = pickle.load(f)
        local_sites = meta['local_sites']
        species_by_id = meta['species_by_id']
        tree_to_gap = meta['tree_to_gap']
        tree_ids = meta['tree_ids']

        targets = [s for s in local_sites if s['site_id'] in wanted]
        if not targets:
            continue

        site_indices = build_tree_site_index(tree_ids, tree_to_gap, local_sites)
        tree_ids_arr = np.array(tree_ids, dtype=np.int64)
        snaps = find_all_snapshots(args.snapshot_dir, rank_id)
        if not snaps:
            print(f"  rank {rank_id}: no snapshots found, skipping")
            continue

        # One writer per target site, opened ONCE so each year appends a block.
        writers = {}
        for site in targets:
            sid = site['site_id']
            w = OutputWriter(os.path.join(args.output_dir, f"site_{sid:04d}"), site_id=sid)
            w.open(species_by_id, len(site['gaps']))
            writers[sid] = (w, site)
            remaining.discard(sid)

        print(f"  rank {rank_id}: {len(targets)} target site(s), {len(snaps)} snapshots")
        for snap in snaps:
            data = np.load(snap)
            year = int(data['year'])
            atp = data['tree_params']
            ats = data['tree_states']
            for sid, (w, site) in writers.items():
                td = build_tree_data(site, site_indices[sid], tree_ids_arr, tree_to_gap,
                                     species_by_id, atp, ats)
                w.write_species_data(year, td, list(site['gaps']))

        for w, _ in writers.values():
            w.close()

    if remaining:
        print(f"WARNING: requested sites not found in any rank: {sorted(remaining)}")
    done = sorted(wanted - remaining)
    print(f"\nDone! Time series for site(s) {done} -> {args.output_dir}/site_NNNN/species_data.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
