"""
Run script for a single gap simulation.
Uses initialize_plot() to create and connect tree agents in one gap.
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


def main():
    parser = argparse.ArgumentParser(
        description="Run single gap simulation"
    )
    parser.add_argument(
        "--maxtrees",
        type=int,
        default=100,
        help="Number of trees in the gap (default: 100)"
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

    args = parser.parse_args()

    if rank == 0:
        print("=" * 60)
        print("GGap Single Gap Simulation")
        print("=" * 60)
        print(f"\nSimulation Parameters:")
        print(f"  Number of trees (maxtrees): {args.maxtrees}")
        print(f"  Maximum height: {args.maxheight}m")
        print(f"  Simulation duration: {args.years} years")
        print(f"  Degree days: {args.deg_days}")
        print(f"  Drought days: {args.dry_days}")
        print(f"  Base mortality rate: {args.base_mortality}")
        print(f"  Species list: {args.specieslist}")
        print()

    # Create model
    model = GAPModel(
        deg_days=args.deg_days,
        dry_days=args.dry_days,
        base_mortality_rate=args.base_mortality,
    )

    if rank == 0:
        print("Initializing plot with trees...")

    # Initialize plot - creates trees and connects them (fully connected network)
    tree_ids = model.initialize_gap(
        specieslist_file=args.specieslist,
        maxtrees=args.maxtrees,
        maxheight=args.maxheight,
        age_range=(5, 50),
        size_range=(3.0, 25.0),
    )

    if rank == 0:
        print(f"Created {len(tree_ids)} trees (all mutually connected)")
        model.print_statistics(tick=0)
        print("\nSetting up GPU kernels...")

    # Setup model (generates GPU kernels)
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
            model.print_statistics(tick=current_year)

    if rank == 0:
        print("\n" + "=" * 60)
        print("Simulation Complete")
        print("=" * 60)
        print("\nFinal Gap State:")
        model.print_statistics()


if __name__ == "__main__":
    main()
