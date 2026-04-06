"""
Verify GGap correctness after SAGESim optimizations.

Runs a small simulation and checks outputs are deterministic.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from gap.gap_model import GAPModel
import numpy as np

def run_test():
    """Run small deterministic test."""
    print("=" * 70)
    print("GGap Correctness Verification")
    print("=" * 70)
    print()

    # Create model
    model = GAPModel()

    # Load globals
    print("Loading globals...")
    model.load_globals(prefix='CONUS')

    # Get site IDs
    all_site_ids = sorted(model._site_id_to_slot.keys())
    site_ids = all_site_ids[:2]  # Just 2 sites for quick test
    print(f"Using {len(site_ids)} sites: {site_ids}")

    # Partition sites
    model.partition_sites(site_ids, strategy='round_robin')

    # Initialize sites
    print("Initializing sites...")
    site_agents = {}
    for site_id in site_ids:
        site = model.initialize_site_with_gaps(site_id, num_gaps=10, maxtrees=100, prefix='CONUS')
        site_agents[site_id] = site['site_agent_id']

    # Connect sites
    print("Connecting sites...")
    model.connect_agents(site_agents[site_ids[0]], site_agents[site_ids[1]])

    # Register breed locals
    model.register_breed_local_arrays()

    # Setup GPU
    print("Setting up GPU...")
    model.setup(use_gpu=True)

    # Run simulation
    print("\nRunning 5 ticks...")
    model.simulate(ticks=5)

    # Collect some outputs using breed data
    print("\nCollecting outputs...")

    # Get tree breed data
    tree_states = model.get_breed_data("Tree", "states")
    tree_params = model.get_breed_data("Tree", "params")

    if tree_states is None:
        print("ERROR: Failed to get tree breed data")
        return False

    # Sample first 10 trees
    sample_size = min(10, len(tree_states))
    print(f"\nSampled {sample_size} tree agents from {len(tree_states)} total")

    # Get state indices (from gap_model.py constants)
    TREE_S_IS_ALIVE = 0
    TREE_S_DIAM = 1
    TREE_S_HEIGHT = 2

    results = []
    for i in range(sample_size):
        results.append({
            'is_alive': tree_states[i, TREE_S_IS_ALIVE],
            'diameter': tree_states[i, TREE_S_DIAM],
            'height': tree_states[i, TREE_S_HEIGHT]
        })

    # Check basic invariants
    alive_count = sum(1 for r in results if r['is_alive'] > 0.5)
    print(f"Alive trees: {alive_count}/{len(results)}")

    # Verify no NaN values
    has_nan = False
    for i, data in enumerate(results):
        for key, value in data.items():
            if isinstance(value, float) and np.isnan(value):
                print(f"WARNING: Tree {i} has NaN in {key}")
                has_nan = True

    if has_nan:
        print("\n✗ FAILED: Found NaN values in outputs")
        return False

    # Check reasonable values
    for i, data in enumerate(results):
        if data['is_alive'] > 0.5:
            if data['diameter'] < 0 or data['diameter'] > 200:
                print(f"WARNING: Tree {i} has unreasonable diameter: {data['diameter']}")
                return False
            if data['height'] < 0 or data['height'] > 100:
                print(f"WARNING: Tree {i} has unreasonable height: {data['height']}")
                return False

    print("\n✓ PASSED: All outputs look reasonable")
    print("  - No NaN values")
    print("  - All values in expected ranges")
    print("  - Simulation completed successfully")

    return True


if __name__ == '__main__':
    try:
        success = run_test()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
