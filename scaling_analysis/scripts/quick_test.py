"""
Quick test with 5 sites for debugging on login node.

Usage:
    python quick_test.py
"""

import sys
import os
import time
import random

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from gap.gap_model import GAPModel

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    print("WARNING: CuPy not available")


def main():
    print("="*70)
    print("GGap Quick Test - 5 Sites")
    print("="*70)

    num_sites = 5
    num_gaps = 500
    maxtrees = 1000
    neighbors_per_site = 10

    try:
        # Create model
        print("\n1. Creating GAPModel...")
        model = GAPModel()

        # Load CONUS globals (species + site data)
        print("2. Loading CONUS globals...")
        model.load_globals(prefix='CONUS')

        # Get actual site IDs from CONUS (handles sparse site IDs)
        print("3. Getting site IDs...")
        all_site_ids = sorted(model._site_id_to_slot.keys())
        print(f"   Total available sites: {len(all_site_ids)}")
        print(f"   First 20 site IDs: {all_site_ids[:20]}")
        print(f"   First 20 climate_rows keys: {sorted(model._climate_rows.keys())[:20]}")

        site_ids = all_site_ids[:num_sites]
        print(f"   Using {num_sites} sites: {site_ids}")

        # Initialize sites
        print(f"4. Initializing {num_sites} sites with {num_gaps} gaps each...")
        print(f"   Total species loaded: {len(model.unique_species)}")
        print(f"   First 5 species codes: {list(model.unique_species.keys())[:5]}")
        site_agents = {}
        for i, site_id in enumerate(site_ids):
            print(f"   Initializing site {site_id} ({i+1}/{num_sites})...")
            site = model.initialize_site_with_gaps(site_id, num_gaps, maxtrees, prefix='CONUS')
            site_agents[site_id] = site['site_agent_id']
        print(f"   ✓ All {num_sites} sites initialized")

        # Random 10 neighbors per site
        random.seed(42)
        print(f"5. Connecting sites ({neighbors_per_site} random neighbors each)...")
        edges_added = 0
        for site_id in site_ids:
            other_sites = [s for s in site_ids if s != site_id]
            if len(other_sites) == 0:
                continue
            neighbors = random.sample(other_sites, min(neighbors_per_site, len(other_sites)))

            for neighbor_id in neighbors:
                if site_id < neighbor_id:  # Avoid duplicates
                    model.connect_agents(site_agents[site_id], site_agents[neighbor_id])
                    edges_added += 1

        avg_degree = 2 * edges_added / num_sites if num_sites > 0 else 0
        print(f"   ✓ Site edges: {edges_added} (avg {avg_degree:.1f} neighbors/site)")

        # Register breed locals
        print("6. Registering breed local arrays...")
        model.register_breed_local_arrays()
        print("   ✓ Breed locals registered")

        # Setup GPU
        print("7. Setting up GPU...")
        model.setup(use_gpu=True)
        print("   ✓ GPU setup complete")

        # Memory after setup
        if HAS_CUPY:
            mem_pool = cp.get_default_memory_pool()
            mem_bytes = mem_pool.used_bytes()
            mem_gb = mem_bytes / 1e9
            print(f"   GPU memory: {mem_gb:.2f} GB")

        # Run 1 tick
        print("8. Running 1 tick...")
        t_start = time.time()
        model.simulate(ticks=1, sync_workers_every_n_ticks=1)
        t_tick = time.time() - t_start
        print(f"   ✓ Tick completed in {t_tick:.3f} s")

        # Memory after tick
        if HAS_CUPY:
            mem_bytes_after = mem_pool.used_bytes()
            mem_gb = max(mem_gb, mem_bytes_after / 1e9)
            print(f"   GPU memory: {mem_gb:.2f} GB")

        total_agents = num_sites * (num_gaps * maxtrees + num_gaps + 1)
        print("\n" + "="*70)
        print("SUCCESS!")
        print("="*70)
        print(f"Sites: {num_sites}")
        print(f"Total agents: {total_agents:,}")
        print(f"Edges: {edges_added}")
        print(f"GPU memory: {mem_gb:.2f} GB")
        print(f"Time per tick: {t_tick:.3f} s")
        print("="*70)

    except Exception as e:
        print("\n" + "="*70)
        print("ERROR!")
        print("="*70)
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        print("="*70)
        sys.exit(1)


if __name__ == '__main__':
    main()
