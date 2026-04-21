"""
Run script for multi-gap site simulation.
Uses initialize_site() to create multiple independent gaps.
Trees only interact within their own gap (no cross-gap connections).
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
from gap.gap_model import GAPModel

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_workers = comm.Get_size()


def main():
    parser = argparse.ArgumentParser(
        description="Run multi-gap site simulation"
    )
    parser.add_argument(
        "--num_gaps",
        type=int,
        default=10,
        help="Number of independent gaps in the site (default: 10)"
    )
    parser.add_argument(
        "--trees_per_gap",
        type=int,
        default=50,
        help="Number of trees per gap (default: 50)"
    )
    parser.add_argument(
        "--maxheight",
        type=int,
        default=60,
        help="Maximum tree height in meters (default: 60)"
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
        "--specieslist",
        type=str,
        default="input_data/UVAFME2012_specieslist.csv",
        help="Path to species list CSV file"
    )
    parser.add_argument(
        "--show_gap_details",
        action="store_true",
        help="Show per-gap statistics in reports"
    )

    args = parser.parse_args()

    total_trees = args.num_gaps * args.trees_per_gap

    if rank == 0:
        print("=" * 60)
        print("GGap Multi-Gap Site Simulation")
        print("=" * 60)
        print(f"\nSite Structure:")
        print(f"  Number of gaps: {args.num_gaps}")
        print(f"  Trees per gap: {args.trees_per_gap}")
        print(f"  Total trees: {total_trees}")
        print(f"\nSimulation Parameters:")
        print(f"  Maximum height: {args.maxheight}m")
        print(f"  Simulation duration: {args.years} years")
        print(f"  Degree days: {args.deg_days}")
        print(f"  Drought days: {args.dry_days}")
        print(f"  Base mortality rate: {args.base_mortality}")
        print(f"  Species list: {args.specieslist}")
        print(f"\nExecution:")
        print(f"  MPI workers: {num_workers}")
        print()

    # Create model
    model = GAPModel(
        deg_days=args.deg_days,
        dry_days=args.dry_days,
        base_mortality_rate=args.base_mortality,
    )

    if rank == 0:
        print("Initializing site with multiple gaps...")

    # Initialize site - creates multiple gaps with trees
    # Trees within each gap are fully connected
    # Trees across gaps are NOT connected (independent gaps)
    site_info = model.initialize_site(
        num_gaps=args.num_gaps,
        trees_per_gap=args.trees_per_gap,
        specieslist_file=args.specieslist,
        maxheight=args.maxheight,
        age_range=(5, 50),
        size_range=(3.0, 25.0),
    )

    if rank == 0:
        print(f"Created {site_info['num_gaps']} gaps with "
              f"{len(site_info['all_trees'])} total trees")
        print("(Trees only interact within their own gap)")
        model.print_statistics(tick=0, by_gap=args.show_gap_details)
        print("\nSetting up GPU kernels...")

    # Register breed-local arrays and setup model (generates GPU kernels)
    model.register_breed_local_arrays()
    model.setup(use_gpu=True)

    if rank == 0:
        print("Starting simulation...")
        print()

    # Run simulation
    for year_batch in range(0, args.years, args.report_interval):
        years_to_run = min(args.report_interval, args.years - year_batch)

        # Simulate
        model.simulate(ticks=years_to_run, sync_workers_every_n_ticks=1)

        # Print statistics
        if rank == 0:
            current_year = year_batch + years_to_run
            model.print_statistics(tick=current_year, by_gap=args.show_gap_details)

    if rank == 0:
        print("\n" + "=" * 60)
        print("Simulation Complete")
        print("=" * 60)
        print("\nFinal Site State:")
        model.print_statistics(by_gap=True)


if __name__ == "__main__":
    main()
