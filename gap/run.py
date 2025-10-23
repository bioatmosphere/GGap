"""
Run script for GGap forest gap dynamics model.
Demonstrates UVAFME-based forest simulation using SAGESim framework.
"""

import argparse
import sys
import os

# Add parent directory and SAGESim to path for imports
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
_sagesim_path = os.path.join(_parent_dir, "SAGESim")

if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)
if _sagesim_path not in sys.path:
    sys.path.insert(0, _sagesim_path)

from mpi4py import MPI
from gap.gap_model import GAPModel

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def main():
    parser = argparse.ArgumentParser(
        description="Run GGap forest gap dynamics model"
    )
    parser.add_argument(
        "--num_trees",
        type=int,
        default=100,
        help="Number of trees in forest (default: 100)"
    )
    parser.add_argument(
        "--forest_size",
        type=float,
        default=100.0,
        help="Size of square forest in meters (default: 100.0)"
    )
    parser.add_argument(
        "--years",
        type=int,
        default=50,
        help="Number of years to simulate (default: 50)"
    )
    parser.add_argument(
        "--neighborhood_radius",
        type=float,
        default=10.0,
        help="Radius for tree interactions in meters (default: 10.0)"
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
        "--species_dist",
        type=str,
        default="equal",
        help="Species distribution: 'equal', 'mixed', or 'single:ID' (default: equal)"
    )

    args = parser.parse_args()

    if rank == 0:
        print("=" * 60)
        print("GGap Forest Gap Dynamics Model")
        print("GPU-Accelerated UVAFME-based Simulation")
        print("=" * 60)
        print(f"\nSimulation Parameters:")
        print(f"  Number of trees: {args.num_trees}")
        print(f"  Forest size: {args.forest_size}m x {args.forest_size}m")
        print(f"  Simulation duration: {args.years} years")
        print(f"  Neighborhood radius: {args.neighborhood_radius}m")
        print(f"  Degree days: {args.deg_days}")
        print(f"  Drought days: {args.dry_days}")
        print(f"  Base mortality rate: {args.base_mortality}")
        print(f"  Report interval: {args.report_interval} years")
        print()

    # Create model
    model = GAPModel(
        neighborhood_radius=args.neighborhood_radius,
        deg_days=args.deg_days,
        dry_days=args.dry_days,
        base_mortality_rate=args.base_mortality,
    )

    # Parse species distribution
    species_distribution = None
    if args.species_dist == "equal":
        species_distribution = None  # Default equal distribution
    elif args.species_dist == "mixed":
        # Mixed forest: more mid-successional species
        species_distribution = {
            1: 0.15,  # Red Maple
            2: 0.20,  # Loblolly Pine
            3: 0.30,  # White Oak (dominant)
            4: 0.20,  # Sweetgum
            5: 0.10,  # Eastern Hemlock
            6: 0.05,  # Tulip Poplar
        }
    elif args.species_dist.startswith("single:"):
        species_id = int(args.species_dist.split(":")[1])
        species_distribution = {species_id: 1.0}

    if rank == 0:
        print("Creating initial forest...")

    # Create forest
    model.create_forest(
        num_trees=args.num_trees,
        forest_size=args.forest_size,
        species_distribution=species_distribution,
        age_range=(5, 50),
        size_range=(3.0, 25.0),
    )

    if rank == 0:
        model.print_forest_statistics(tick=0)
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
            model.print_forest_statistics(tick=current_year)

    if rank == 0:
        print("\n" + "=" * 60)
        print("Simulation Complete")
        print("=" * 60)
        print("\nFinal Forest State:")
        model.print_forest_statistics()


if __name__ == "__main__":
    main()
