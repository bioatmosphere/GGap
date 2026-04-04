"""
Find maximum sites per GPU before OOM.

Config:
- Single GPU test (no MPI)
- 500 gaps × 1000 trees per site
- 10 random neighbors per site
- CONUS species data

Usage:
    srun -N1 -n1 --gpus-per-node=1 python -u find_memory_limit.py
"""

import sys
import os
import time
import random

# Force unbuffered output for immediate logging
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from gap.gap_model import GAPModel

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    print("WARNING: CuPy not available")


def test_sites(num_sites, num_gaps=500, maxtrees=1000, neighbors_per_site=10):
    """
    Test if num_sites fit in GPU memory.

    Args:
        num_sites: Number of sites to test
        num_gaps: Gaps per site (default: 500)
        maxtrees: Max trees per gap (default: 1000)
        neighbors_per_site: Random neighbors per site (default: 10)

    Returns:
        (success, mem_gb, timings, edges) or (False, None, None, 0) on OOM
        where timings is a dict with: globals_load, site_init, connection, breed_reg, gpu_setup, simulation
    """
    timings = {}
    try:
        # Create model
        t_start = time.time()
        model = GAPModel()

        # Load CONUS globals (species + site data)
        print(f"  Loading CONUS globals...", flush=True)
        model.load_globals(prefix='CONUS')
        timings['globals_load'] = time.time() - t_start
        print(f"    Loaded in {timings['globals_load']:.2f}s", flush=True)

        # Get actual site IDs from CONUS (handles sparse site IDs)
        all_site_ids = sorted(model._site_id_to_slot.keys())
        site_ids = all_site_ids[:num_sites]

        # Initialize sites using bulk method
        t_start = time.time()
        print(f"  Bulk creating agents for {num_sites} sites...", flush=True)
        site_agents = model.initialize_sites_bulk(site_ids, num_gaps, maxtrees, prefix='CONUS')
        timings['site_init'] = time.time() - t_start
        print(f"    Bulk creation completed in {timings['site_init']:.2f}s", flush=True)

        # Random neighbors per site
        t_start = time.time()
        random.seed(42)
        print(f"  Connecting sites ({neighbors_per_site} random neighbors each)...", flush=True)
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
        timings['connection'] = time.time() - t_start
        print(f"    {edges_added} edges added in {timings['connection']:.2f}s (avg {avg_degree:.1f} neighbors/site)", flush=True)

        # Register breed locals
        t_start = time.time()
        print(f"  Registering breed local arrays...", flush=True)
        model.register_breed_local_arrays()
        timings['breed_reg'] = time.time() - t_start
        print(f"    Registered in {timings['breed_reg']:.2f}s", flush=True)

        # Setup GPU
        t_start = time.time()
        print(f"  Setting up GPU...", flush=True)
        model.setup(use_gpu=True)
        timings['gpu_setup'] = time.time() - t_start
        print(f"    GPU setup in {timings['gpu_setup']:.2f}s", flush=True)

        # Memory after setup
        if HAS_CUPY:
            mem_pool = cp.get_default_memory_pool()
            mem_bytes = mem_pool.used_bytes()
            mem_gb = mem_bytes / 1e9
        else:
            mem_gb = 0.0

        # Run 1 tick
        print(f"  Running 1 tick...", flush=True)
        t_start = time.time()
        model.simulate(ticks=1, sync_workers_every_n_ticks=1)
        timings['simulation'] = time.time() - t_start
        print(f"    Tick completed in {timings['simulation']:.2f}s", flush=True)

        # Memory after tick
        if HAS_CUPY:
            mem_bytes_after = mem_pool.used_bytes()
            mem_gb = max(mem_gb, mem_bytes_after / 1e9)

        return True, mem_gb, timings, edges_added

    except Exception as e:
        error_str = str(e).lower()
        is_oom = any(kw in error_str for kw in ['memory', 'alloc', 'oom'])

        if is_oom:
            return False, None, {}, 0
        else:
            print(f"  ERROR (not OOM): {type(e).__name__}: {str(e)[:200]}")
            raise


def main():
    print("="*70)
    print("GGap Memory Capacity Test")
    print("="*70)
    print("Config:")
    print("  - Single GPU (no MPI)")
    print("  - 500 gaps × 1000 trees per site")
    print("  - 10 random neighbors per site")
    print("  - CONUS species (235 species)")
    print("="*70)

    # Test points - start at 10, then 25, 50, 100, 150, 200, 250
    test_sizes = [10, 25, 50, 100, 150, 200, 250]

    results = []
    last_success = 0

    for num_sites in test_sizes:
        print(f"\n{'='*70}", flush=True)
        print(f"Testing {num_sites} sites", flush=True)
        print(f"{'='*70}", flush=True)

        success, mem_gb, timings, edges = test_sites(num_sites)

        if success:
            total_agents = num_sites * 500501
            total_time = sum(timings.values())
            print(f"✓ SUCCESS", flush=True)
            print(f"  Total agents: {total_agents:,}", flush=True)
            print(f"  GPU memory: {mem_gb:.2f} GB / 64 GB ({100*mem_gb/64:.1f}%)", flush=True)
            print(f"  Timing breakdown:", flush=True)
            print(f"    Globals load:  {timings.get('globals_load', 0):.2f}s", flush=True)
            print(f"    Site init:     {timings.get('site_init', 0):.2f}s", flush=True)
            print(f"    Connection:    {timings.get('connection', 0):.2f}s", flush=True)
            print(f"    Breed reg:     {timings.get('breed_reg', 0):.2f}s", flush=True)
            print(f"    GPU setup:     {timings.get('gpu_setup', 0):.2f}s", flush=True)
            print(f"    Simulation:    {timings.get('simulation', 0):.2f}s", flush=True)
            print(f"    TOTAL:         {total_time:.2f}s", flush=True)

            results.append({
                'sites': num_sites,
                'agents': total_agents,
                'edges': edges,
                'mem_gb': mem_gb,
                **{f'time_{k}': v for k, v in timings.items()},
                'time_total': total_time
            })
            last_success = num_sites
        else:
            print(f"✗ OUT OF MEMORY", flush=True)
            break

    # Summary
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")

    if results:
        last = results[-1]
        print(f"\nMaximum: {last_success} sites")
        print(f"  Total agents: {last['agents']:,}")
        print(f"  GPU memory: {last['mem_gb']:.2f} GB / 64 GB")
        print(f"  Memory per site: {last['mem_gb']*1000/last_success:.1f} MB")

        # Recommend for weak scaling (80% of max)
        safe_limit = int(last_success * 0.8)
        print(f"\nRECOMMENDED FOR WEAK SCALING:")
        print(f"  {safe_limit} sites per GPU")
        print(f"  (80% of max for safety margin)")

        print(f"\nExample scaling configurations:")
        for nodes in [1, 2, 5, 10, 20]:
            gpus = nodes * 8
            total_sites = gpus * safe_limit
            print(f"  {nodes:2d} nodes ({gpus:3d} GPUs) = {total_sites:5d} sites")
    else:
        print("\nNo successful tests!")

    print(f"{'='*70}")


if __name__ == '__main__':
    main()
