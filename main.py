"""
GGap - Main entry point for forest gap dynamics simulation.
"""

import sys
import os

# Add SAGESim to path
_sagesim_path = os.path.join(os.path.dirname(__file__), "SAGESim")
if _sagesim_path not in sys.path:
    sys.path.insert(0, _sagesim_path)

def main():
    print("=" * 60)
    print("GGap Forest Gap Dynamics Model")
    print("GPU-Accelerated UVAFME-based Simulation")
    print("=" * 60)
    print()
    print("This is a quick demo showing tree species data.")
    print()
    print("For full MPI-enabled GPU simulation, use:")
    print("  cd gap")
    print("  mpirun -n 4 python run.py --num_trees 200 --years 100")
    print()
    print("=" * 60)
    print()

    # Import species data (doesn't require cupy)
    from gap.tree_species_data import SPECIES_DATA, get_species_params

    print("Available Tree Species:")
    print()

    for species_id, species in SPECIES_DATA.items():
        print(f"{species_id}. {species.common_name} ({species.genus_name})")
        print(f"   Type: {'Conifer' if species.conifer else 'Deciduous'}")
        print(f"   Max age: {species.max_age:.0f} years")
        print(f"   Max height: {species.max_ht:.1f} m")
        print(f"   Max diameter: {species.max_diam:.1f} cm")
        print(f"   Shade tolerance: {species.shade_tol}/5")
        print(f"   Drought tolerance: {species.drought_tol}/5")
        print(f"   Growth rate (g): {species.g:.1f}")
        print()

    print("=" * 60)
    print()
    print("Model Architecture:")
    print()
    print("  gap/tree_species_data.py - Species parameters (6 Eastern US species)")
    print("  gap/tree_breed.py        - TreeBreed agent definition (15 properties)")
    print("  gap/tree_step_func.py    - GPU kernels (light, growth, mortality)")
    print("  gap/gap_model.py         - GAPModel simulation class")
    print("  gap/run.py               - MPI runner script")
    print()
    print("=" * 60)
    print()
    print("Quick Test (no GPU required):")
    print()
    print("  python -c \"from gap.tree_species_data import get_species_params;")
    print("            sp = get_species_params(3); print(sp.common_name, sp.max_ht)\"")
    print()
    print("Full GPU Simulation:")
    print()
    print("  cd gap")
    print("  mpirun -n 4 python run.py --num_trees 200 --years 100 --species_dist mixed")
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
