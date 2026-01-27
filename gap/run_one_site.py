"""
Run script for a single site simulation.
Uses Site -> Gap(s) -> Trees agent hierarchy with soil biogeochemistry.
"""

import argparse
import sys
import os

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
    # Site params indices (soil pools are private)
    SITE_P_A0_C, SITE_P_A0_N,
    SITE_P_A_C, SITE_P_A_N,
    SITE_P_BL_C, SITE_P_BL_N,
    # Site states indices (public: climate + avail_n)
    SITE_S_AVAIL_N,
)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def main():
    parser = argparse.ArgumentParser(
        description="Run single site simulation with soil biogeochemistry"
    )
    parser.add_argument(
        "--num_gaps",
        type=int,
        default=1,
        help="Number of gaps per site (default: 1)"
    )
    parser.add_argument(
        "--trees_per_gap",
        type=int,
        default=100,
        help="Number of trees per gap (default: 100)"
    )
    parser.add_argument(
        "--years",
        type=int,
        default=50,
        help="Number of years to simulate (default: 50)"
    )
    parser.add_argument(
        "--base_mortality",
        type=float,
        default=0.02,
        help="Base annual mortality rate (default: 0.02)"
    )
    parser.add_argument(
        "--report_interval",
        type=int,
        default=10,
        help="Years between progress reports (default: 10)"
    )
    parser.add_argument(
        "--site_id",
        type=int,
        default=0,
        help="Site ID from UVAFME CSV files (default: 0)"
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

    args = parser.parse_args()

    if rank == 0:
        print("=" * 60)
        print("GGap Single Site Simulation (with Soil)")
        print("=" * 60)
        print(f"\nSimulation Parameters:")
        print(f"  Number of gaps: {args.num_gaps}")
        print(f"  Trees per gap: {args.trees_per_gap}")
        print(f"  Total trees: {args.num_gaps * args.trees_per_gap}")
        print(f"  Simulation duration: {args.years} years")
        print(f"  Base mortality rate: {args.base_mortality}")
        print(f"  Data directory: {args.data_dir}")
        print(f"  File prefix: {args.prefix}")
        print(f"  Site ID: {args.site_id}")
        print()

    # Create model
    model = GAPModel()

    if rank == 0:
        print("Initializing site from UVAFME CSV files...")

    # Initialize site (loads from UVAFME CSV files)
    site = model.initialize_site(
        site_id=args.site_id,
        data_dir=args.data_dir,
        prefix=args.prefix,
        base_mortality_rate=args.base_mortality,
    )

    if rank == 0:
        print(f"Site: {site['site_name']} ({site['latitude']:.2f}°N, {site['longitude']:.2f}°W)")
        print(f"Loaded {len(site['species'])} species for site")
        print(f"Calculated deg_days: {site['deg_days']:.0f}, dry_days: {site['dry_days']:.1f}")
        print(f"Site agent ID: {site['site_agent_id']}")
        print(f"\nInitializing {args.num_gaps} gap(s) with trees...")

    # Initialize gaps and trees - creates gap agents and trees, connects them
    total_trees = 0
    total_alive = 0
    for gap_num in range(args.num_gaps):
        tree_ids, initial_alive = model.initialize_trees(
            site=site,
            maxtrees=args.trees_per_gap,
            initial_size_range=(3.0, 25.0),
        )
        total_trees += len(tree_ids)
        total_alive += initial_alive
        if rank == 0:
            print(f"  Gap {gap_num + 1}: {initial_alive} alive / {len(tree_ids)} slots")

    if rank == 0:
        print(f"\nTotal agents: {len(model.site_agents)} site, {len(model.gap_agents)} gaps, {len(model.tree_ids)} tree slots")
        print(f"  Initial alive: {total_alive}, Dormant slots: {total_trees - total_alive}")
        stats = model.get_statistics()
        print(f"\nInitial state:")
        print(f"  Living trees: {stats['living_trees']}")
        print(f"  Total biomass: {stats['total_biomass']:.1f} kg C")

        # Print initial soil state (soil pools in params, avail_n in states)
        site_agent_id = site['site_agent_id']
        params = model.get_agent_property_value(site_agent_id, "params")
        states = model.get_agent_property_value(site_agent_id, "states")
        print(f"\nInitial soil state:")
        print(f"  A0 layer C: {params[SITE_P_A0_C]:.2f} tn/ha")
        print(f"  A layer C: {params[SITE_P_A_C]:.2f} tn/ha")
        print(f"  Base layer C: {params[SITE_P_BL_C]:.2f} tn/ha")
        print(f"  Available N: {states[SITE_S_AVAIL_N]:.4f} tn/ha")
        print("\nSetting up GPU kernels...")

    # Setup model (generates GPU kernels)
    model.setup(use_gpu=True)

    if rank == 0:
        print("Starting simulation...")
        print()
        print(f"{'Year':<8} {'Alive':<8} {'Dead':<8} {'Biomass':<12} {'Avail_N':<12}")
        print("-" * 50)

    # Run simulation
    site_agent_id = site['site_agent_id']
    for year_batch in range(0, args.years, args.report_interval):
        years_to_run = min(args.report_interval, args.years - year_batch)

        # Simulate
        model.simulate(ticks=years_to_run, sync_workers_every_n_ticks=1)

        # Print statistics
        if rank == 0:
            current_year = year_batch + years_to_run
            stats = model.get_statistics()
            states = model.get_agent_property_value(site_agent_id, "states")
            avail_n = states[SITE_S_AVAIL_N] if isinstance(states, list) else states
            print(f"{current_year:<8} {stats['living_trees']:<8} {stats['dead_trees']:<8} {stats['total_biomass']:<12.1f} {avail_n:<12.4f}")

    if rank == 0:
        print("\n" + "=" * 60)
        print("Simulation Complete")
        print("=" * 60)
        stats = model.get_statistics()
        print(f"\nFinal Site State:")
        print(f"  Living trees: {stats['living_trees']}")
        print(f"  Dead trees: {stats['dead_trees']}")
        print(f"  Total biomass: {stats['total_biomass']:.1f} kg C")

        # Final soil state (soil pools in params, avail_n in states)
        params = model.get_agent_property_value(site_agent_id, "params")
        states = model.get_agent_property_value(site_agent_id, "states")
        avail_n = states[SITE_S_AVAIL_N] if isinstance(states, list) else states
        print(f"\nFinal soil state:")
        print(f"  A0 layer C: {params[SITE_P_A0_C]:.2f} tn/ha, N: {params[SITE_P_A0_N]:.4f} tn/ha")
        print(f"  A layer C: {params[SITE_P_A_C]:.2f} tn/ha, N: {params[SITE_P_A_N]:.4f} tn/ha")
        print(f"  Base layer C: {params[SITE_P_BL_C]:.2f} tn/ha, N: {params[SITE_P_BL_N]:.4f} tn/ha")
        print(f"  Available N: {avail_n:.4f} tn/ha/yr")


if __name__ == "__main__":
    main()
