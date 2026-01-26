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
    # Index constants for accessing consolidated properties
    S_AVAIL_N,
    O_SOIL_RESP,
    SOIL_A0_C, SOIL_A0_N,
    SOIL_A_C, SOIL_A_N,
    SOIL_BL_C, SOIL_BL_N,
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
        "--deg_days",
        type=float,
        default=2500.0,
        help="Annual degree days (default: 2500.0)"
    )
    parser.add_argument(
        "--dry_days",
        type=float,
        default=30.0,
        help="Annual drought days (default: 30.0)"
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
        "--site_csv",
        type=str,
        default="input_data/UVAFME2012_specieslist.csv",
        help="Path to site species CSV file"
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
        print(f"  Degree days: {args.deg_days}")
        print(f"  Drought days: {args.dry_days}")
        print(f"  Base mortality rate: {args.base_mortality}")
        print(f"  Site CSV: {args.site_csv}")
        print()

    # Create model
    model = GAPModel()

    if rank == 0:
        print("Loading site and species data...")

    # Load site (creates site agent with soil, loads species from CSV)
    site = model.load_site(
        site_csv=args.site_csv,
        deg_days=args.deg_days,
        dry_days=args.dry_days,
        base_mortality_rate=args.base_mortality,
    )

    if rank == 0:
        print(f"Loaded {len(site['species'])} species for site")
        print(f"Site agent ID: {site['site_agent_id']}")
        print(f"\nInitializing {args.num_gaps} gap(s) with trees...")

    # Initialize gaps - creates gap agents and trees, connects them
    total_trees = 0
    for gap_num in range(args.num_gaps):
        tree_ids = model.initialize_gap(
            site=site,
            maxtrees=args.trees_per_gap,
            age_range=(5, 50),
            size_range=(3.0, 25.0),
        )
        total_trees += len(tree_ids)
        if rank == 0:
            print(f"  Gap {gap_num + 1}: {len(tree_ids)} trees")

    if rank == 0:
        print(f"\nTotal agents: {len(model.site_agents)} site, {len(model.gap_agents)} gaps, {len(model.tree_ids)} trees")
        stats = model.get_statistics()
        print(f"\nInitial state:")
        print(f"  Living trees: {stats['living_trees']}")
        print(f"  Total biomass: {stats['total_biomass']:.1f} kg C")

        # Print initial soil state using consolidated properties
        site_agent_id = site['site_agent_id']
        soil = model.get_agent_property_value(site_agent_id, "soil")
        state = model.get_agent_property_value(site_agent_id, "state")
        print(f"\nInitial soil state:")
        print(f"  A0 layer C: {soil[SOIL_A0_C]:.2f} tn/ha")
        print(f"  A layer C: {soil[SOIL_A_C]:.2f} tn/ha")
        print(f"  Base layer C: {soil[SOIL_BL_C]:.2f} tn/ha")
        print(f"  Available N: {state[S_AVAIL_N]:.4f} tn/ha")
        print("\nSetting up GPU kernels...")

    # Setup model (generates GPU kernels)
    model.setup(use_gpu=True)

    if rank == 0:
        print("Starting simulation...")
        print()
        print(f"{'Year':<8} {'Alive':<8} {'Dead':<8} {'Biomass':<12} {'Avail_N':<12} {'Soil_Resp':<12}")
        print("-" * 60)

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
            state = model.get_agent_property_value(site_agent_id, "state")
            output = model.get_agent_property_value(site_agent_id, "output")
            avail_n = state[S_AVAIL_N] if isinstance(state, list) else state
            soil_resp = output[O_SOIL_RESP] if isinstance(output, list) else output
            print(f"{current_year:<8} {stats['living_trees']:<8} {stats['dead_trees']:<8} {stats['total_biomass']:<12.1f} {avail_n:<12.4f} {soil_resp:<12.4f}")

    if rank == 0:
        print("\n" + "=" * 60)
        print("Simulation Complete")
        print("=" * 60)
        stats = model.get_statistics()
        print(f"\nFinal Site State:")
        print(f"  Living trees: {stats['living_trees']}")
        print(f"  Dead trees: {stats['dead_trees']}")
        print(f"  Total biomass: {stats['total_biomass']:.1f} kg C")

        # Final soil state using consolidated properties
        soil = model.get_agent_property_value(site_agent_id, "soil")
        state = model.get_agent_property_value(site_agent_id, "state")
        output = model.get_agent_property_value(site_agent_id, "output")
        avail_n = state[S_AVAIL_N] if isinstance(state, list) else state
        soil_resp = output[O_SOIL_RESP] if isinstance(output, list) else output
        print(f"\nFinal soil state:")
        print(f"  A0 layer C: {soil[SOIL_A0_C]:.2f} tn/ha, N: {soil[SOIL_A0_N]:.4f} tn/ha")
        print(f"  A layer C: {soil[SOIL_A_C]:.2f} tn/ha, N: {soil[SOIL_A_N]:.4f} tn/ha")
        print(f"  Base layer C: {soil[SOIL_BL_C]:.2f} tn/ha, N: {soil[SOIL_BL_N]:.4f} tn/ha")
        print(f"  Available N: {avail_n:.4f} tn/ha/yr")
        print(f"  Soil respiration: {soil_resp:.4f} tn C/ha/yr")


if __name__ == "__main__":
    main()
