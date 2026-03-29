"""
Run script for multi-site simulation with inter-site seed dispersal.
Uses Site -> Gap(s) -> Trees agent hierarchy with soil biogeochemistry.
Sites communicate via SAGESim ghost agents for dispersal data exchange.

Usage:
    python run_multi_site.py --site_ids 0,1 --num_gaps 10 --years 50
    mpirun -n 2 python run_multi_site.py --site_ids 0,1 --num_gaps 10 --years 50
"""

import argparse
import sys
import os
import time
import numpy as np

# Add parent directory to path for imports
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Try installed SAGESim first, fall back to submodule
try:
    import sagesim  # noqa: F401
except ImportError:
    _sagesim_path = os.path.join(_parent_dir, "SAGESim")
    if _sagesim_path not in sys.path:
        sys.path.insert(0, _sagesim_path)

from mpi4py import MPI
from gap.gap_model import (
    GAPModel,
    SITE_P_A0_C, SITE_P_A0_N,
    SITE_P_A_C, SITE_P_A_N,
    SITE_P_BL_C, SITE_P_BL_N,
    SITE_P_ANNUAL_RAIN, SITE_P_GROW_DAYS, SITE_P_POT_EVAP, SITE_P_ACT_EVAP,
    SITE_P_SOIL_RESP, SITE_P_C_INTO_A0, SITE_P_N_INTO_A0, SITE_P_NET_N_INTO_A0,
    SITE_S_AVAIL_N, SITE_S_DEG_DAYS, SITE_S_DRY_DAYS, SITE_S_DRY_DAYS_BASE, SITE_S_FLOOD_DAYS,
    TREE_P_BIOMC, TREE_P_BIOMN, TREE_P_LEAF_BM, TREE_P_AGE,
    TREE_S_IS_ALIVE, TREE_S_DIAM, TREE_S_HEIGHT, TREE_S_CANOPY_HT, TREE_S_SPECIES_ID,
)
from gap.output_utils import OutputWriter

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_workers = comm.Get_size()


def collect_per_site_data(model, sites):
    """Collect site params/states and tree data per site.

    Returns list of (site_params, site_states, tree_data, gap_agents) per site
    on rank 0, None on other ranks.
    """
    # Bulk download all breeds (returns None on non-root ranks)
    all_site_params = model.get_breed_data("Site", "params")
    all_site_states = model.get_breed_data("Site", "states")
    all_tree_params = model.get_breed_data("Tree", "params")
    all_tree_states = model.get_breed_data("Tree", "states")
    all_tree_ids = model.get_breed_agent_ids("Tree")

    if all_site_params is None:
        return None

    results = []
    for site_idx, site in enumerate(sites):
        # Site data (one row per site agent in breed order)
        site_params = all_site_params[site_idx]
        site_states = all_site_states[site_idx]

        # Filter trees belonging to this site's gaps
        site_gap_ids = set(site['gaps'])
        tree_mask = np.array([
            model.tree_to_gap.get(int(tid), -1) in site_gap_ids
            for tid in all_tree_ids
        ], dtype=bool)

        site_tree_params = all_tree_params[tree_mask]
        site_tree_states = all_tree_states[tree_mask]
        site_tree_ids = all_tree_ids[tree_mask]

        # Filter to living trees
        alive_mask = site_tree_states[:, TREE_S_IS_ALIVE] > 0.5
        alive_params = site_tree_params[alive_mask]
        alive_states = site_tree_states[alive_mask]
        alive_ids = site_tree_ids[alive_mask].astype(np.int32)

        gap_ids = np.array([model.tree_to_gap[int(a)] for a in alive_ids], dtype=np.int32)
        species_ids = alive_states[:, TREE_S_SPECIES_ID].astype(np.int32)
        evergreen = np.array([
            model.species_by_id.get(int(sid), {}).get('evergreen', 0) > 0.5
            for sid in species_ids
        ], dtype=bool)

        tree_data = {
            'count': int(alive_mask.sum()),
            'gap_agent_id': gap_ids,
            'species_id': species_ids,
            'diam': alive_states[:, TREE_S_DIAM],
            'height': alive_states[:, TREE_S_HEIGHT],
            'biomC': alive_params[:, TREE_P_BIOMC],
            'biomN': alive_params[:, TREE_P_BIOMN],
            'leaf_bm': alive_params[:, TREE_P_LEAF_BM],
            'age': alive_params[:, TREE_P_AGE],
            'canopy_ht': alive_states[:, TREE_S_CANOPY_HT],
            'evergreen': evergreen,
        }

        results.append((site_params, site_states, tree_data, list(site_gap_ids)))

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run multi-site simulation with inter-site seed dispersal"
    )
    parser.add_argument(
        "--site_ids",
        type=str,
        default=None,
        help="Comma-separated site IDs (e.g., '0,1'). Default: all sites from CSV."
    )
    parser.add_argument(
        "--num_gaps",
        type=int,
        default=200,
        help="Number of gaps per site (default: 200)"
    )
    parser.add_argument(
        "--maxtrees",
        type=int,
        default=1000,
        help="Max tree slots per gap (default: 1000)"
    )
    parser.add_argument(
        "--years",
        type=int,
        default=500,
        help="Number of years to simulate (default: 500)"
    )
    parser.add_argument(
        "--report_interval",
        type=int,
        default=10,
        help="Years between progress reports and CSV output (default: 10)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="input_data",
        help="Directory containing UVAFME CSV files"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="UVAFME2012",
        help="File prefix for UVAFME CSV files (default: UVAFME2012)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(_parent_dir, "output_data"),
        help="Base directory for CSV output files"
    )
    parser.add_argument(
        "--no_tree_data",
        action="store_true",
        help="Skip writing tree_data.csv (can be very large)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: random)"
    )

    args = parser.parse_args()

    if args.site_ids is not None:
        site_ids = [int(x.strip()) for x in args.site_ids.split(",")]
    else:
        # Auto-detect all site IDs from CSV
        import csv
        site_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            args.data_dir, f"{args.prefix}_site.csv"
        )
        with open(site_file, 'r') as f:
            reader = csv.DictReader(f)
            site_ids = [int(row['site']) for row in reader]

    if rank == 0:
        print("=" * 60)
        print("GGap Multi-Site Simulation with Seed Dispersal")
        print("=" * 60)
        print(f"\n  Sites: {site_ids}")
        print(f"  Workers: {num_workers}")
        print(f"  Gaps per site: {args.num_gaps}")
        print(f"  Max trees per gap: {args.maxtrees}")
        print(f"  Years: {args.years}")
        print(f"  Seed: {args.seed if args.seed is not None else 'random'}")
        print()

    t_total_start = time.time()

    # Create model and load globals
    model = GAPModel()
    if args.seed is not None:
        model.set_seed(args.seed)

    if rank == 0:
        print("Loading globals (species traits + site configs)...")
    model.load_globals(data_dir=args.data_dir, prefix=args.prefix)

    if rank == 0:
        print(f"  {model.get_species_count()} species loaded")

    # Partition sites across workers, then initialize
    model.partition_sites(site_ids)

    sites = []
    for sid in site_ids:
        if rank == 0:
            print(f"\nInitializing site {sid} → rank {model._site_partition[sid]}...")
        site = model.initialize_site_with_gaps(
            sid, args.num_gaps, args.maxtrees,
            data_dir=args.data_dir, prefix=args.prefix,
        )
        sites.append(site)
        if rank == 0:
            print(f"  {site['site_name']} ({site['latitude']:.2f}°N, {site['longitude']:.2f}°W)")
            print(f"  {len(site['species'])} species, {args.num_gaps} gaps, deg_days: {site['deg_days']:.0f}")

    # Connect sites for dispersal
    if rank == 0:
        print("\nConnecting sites for dispersal...")
    model.connect_sites()

    t_init_end = time.time()

    if rank == 0:
        total_agents = len(model.site_agents) + len(model.gap_agents) + len(model.tree_ids)
        print(f"\nTotal agents: {total_agents}")
        print(f"  {len(model.site_agents)} sites, {len(model.gap_agents)} gaps, {len(model.tree_ids)} trees")

    # Initialize per-site output writers (rank 0 only)
    writers = {}
    if rank == 0:
        for site in sites:
            sid = site['site_id']
            site_output_dir = os.path.join(args.output_dir, f"site_{sid}")
            writer = OutputWriter(site_output_dir, site_id=sid)
            writer.open(model.species_by_id, args.num_gaps)
            writers[sid] = writer

        print(f"\nOutput directories:")
        for sid in writers:
            print(f"  site_{sid}/: genus_data.csv, species_data.csv, site_data.csv, soil_data.csv"
                  + ("" if args.no_tree_data else ", tree_data.csv"))

    # Register breed-local arrays and setup GPU kernels
    model.register_breed_local_arrays()
    if rank == 0:
        print("\nSetting up GPU kernels...")
    model.setup(use_gpu=True)

    t_setup_end = time.time()
    if rank == 0:
        print(f"  Init: {t_init_end - t_total_start:.2f}s, GPU setup: {t_setup_end - t_init_end:.2f}s")
        print("\nStarting simulation...")
        print()

    # Run simulation
    t_sim_start = time.time()
    for year_batch in range(0, args.years, args.report_interval):
        years_to_run = min(args.report_interval, args.years - year_batch)

        t_batch_start = time.time()
        model.simulate(ticks=years_to_run, sync_workers_every_n_ticks=1)
        t_sim_end = time.time()

        current_year = year_batch + years_to_run

        # Collect per-site data (returns None on non-root ranks)
        per_site = collect_per_site_data(model, sites)
        t_collect_end = time.time()

        if rank == 0 and per_site is not None:
            # Write CSV outputs for each site
            for site_idx, (site_params, site_states, tree_data, gap_agents) in enumerate(per_site):
                sid = sites[site_idx]['site_id']
                w = writers[sid]

                w.write_site_data(
                    current_year,
                    site_params[SITE_P_ANNUAL_RAIN],
                    site_params[SITE_P_POT_EVAP],
                    site_params[SITE_P_ACT_EVAP],
                    site_params[SITE_P_GROW_DAYS],
                    site_states[SITE_S_DEG_DAYS],
                    site_states[SITE_S_DRY_DAYS],
                    site_states[SITE_S_DRY_DAYS_BASE],
                    site_states[SITE_S_FLOOD_DAYS],
                )
                w.write_soil_data(
                    current_year,
                    site_params[SITE_P_A0_C], site_params[SITE_P_A_C],
                    site_params[SITE_P_A0_N], site_params[SITE_P_A_N],
                    site_params[SITE_P_BL_C], site_params[SITE_P_BL_N],
                    site_states[SITE_S_AVAIL_N],
                    soilresp=site_params[SITE_P_SOIL_RESP],
                    c_into_a0=site_params[SITE_P_C_INTO_A0],
                    n_into_a0=site_params[SITE_P_N_INTO_A0],
                    net_n_into_a0=site_params[SITE_P_NET_N_INTO_A0],
                )
                w.write_species_data(current_year, tree_data, gap_agents)
                w.write_genus_data(current_year, tree_data, gap_agents)
                if not args.no_tree_data:
                    w.write_tree_data(current_year, tree_data, gap_agents)

            t_csv_end = time.time()

            # Console summary
            elapsed = time.time() - t_sim_start
            site_summaries = []
            for site_idx, (_, site_states, tree_data, _) in enumerate(per_site):
                sid = sites[site_idx]['site_id']
                alive = tree_data['count']
                biomass = float(tree_data['biomC'].sum()) if alive > 0 else 0.0
                site_summaries.append(f"S{sid}:{alive}trees/{biomass:.0f}kg")

            print(f"  Year {current_year:>4}/{args.years}  "
                  f"sim:{t_sim_end - t_batch_start:.2f}s  "
                  f"csv:{t_csv_end - t_collect_end:.2f}s  "
                  f"({elapsed:.1f}s total)  "
                  f"{'  '.join(site_summaries)}")

    t_total_end = time.time()

    # Close all writers
    for w in writers.values():
        w.close()

    if rank == 0:
        print(f"\nSimulation complete: {t_total_end - t_total_start:.2f}s total")
        print(f"Output written to {args.output_dir}/site_*/")


if __name__ == "__main__":
    main()
