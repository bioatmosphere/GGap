"""
Quick occupancy test with 5 sites and 4 SM values for debugging.

Tests that max_blocks_per_sm parameter works correctly before submitting full job.

Usage:
    python quick_test_occupancy.py
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


def test_one_sm_value(num_sites, max_blocks_per_sm, num_ticks=3):
    """Test a single max_blocks_per_sm value."""
    try:
        print(f"\n{'='*70}")
        print(f"Testing max_blocks_per_sm = {max_blocks_per_sm}")
        print(f"{'='*70}")

        # Create model
        print("  Creating GAPModel...", flush=True)
        model = GAPModel()

        # Load globals
        print("  Loading CONUS globals...", flush=True)
        model.load_globals(prefix='CONUS')
        print("    Globals loaded", flush=True)

        # Get site IDs
        all_site_ids = sorted(model._site_id_to_slot.keys())
        site_ids = all_site_ids[:num_sites]
        print(f"  Using {num_sites} sites: {site_ids}")

        # Partition sites (required before initialization)
        print("  Partitioning sites...", flush=True)
        model.partition_sites(site_ids, strategy='round_robin')
        print("    Partitioned", flush=True)

        # Initialize sites (loop method)
        print(f"  Initializing {num_sites} sites...", flush=True)
        site_agents = {}
        for i, site_id in enumerate(site_ids):
            print(f"    Site {i+1}/{num_sites} (ID: {site_id})...", flush=True)
            site = model.initialize_site_with_gaps(site_id, num_gaps=500, maxtrees=1000, prefix='CONUS')
            site_agents[site_id] = site['site_agent_id']
        print("    All sites initialized", flush=True)

        # Connect sites (random neighbors)
        print("  Connecting sites...", flush=True)
        random.seed(42)
        edges_added = 0
        for site_id in site_ids:
            other_sites = [s for s in site_ids if s != site_id]
            if len(other_sites) == 0:
                continue
            neighbors = random.sample(other_sites, min(10, len(other_sites)))
            for neighbor_id in neighbors:
                if site_id < neighbor_id:
                    model.connect_agents(site_agents[site_id], site_agents[neighbor_id])
                    edges_added += 1

        # Register breed locals
        print("  Registering breed local arrays...", flush=True)
        model.register_breed_local_arrays()
        print("    Registered", flush=True)

        # SET OCCUPANCY PARAMETER BEFORE SETUP
        print(f"  Setting max_blocks_per_sm = {max_blocks_per_sm}", flush=True)
        model._max_blocks_per_sm = max_blocks_per_sm

        # Setup GPU
        print("  Setting up GPU...", flush=True)
        t_start = time.time()
        model.setup()
        setup_time = time.time() - t_start
        print(f"    Setup time: {setup_time:.2f}s")

        # Memory
        if HAS_CUPY:
            mem_pool = cp.get_default_memory_pool()
            mem_gb = mem_pool.used_bytes() / 1e9
            print(f"    GPU memory: {mem_gb:.2f} GB")

        # Run ticks
        print(f"  Running {num_ticks} ticks...")
        tick_times = []
        for tick in range(num_ticks):
            t_tick = time.time()
            model.simulate(ticks=1, sync_workers_every_n_ticks=1)
            tick_time = time.time() - t_tick
            tick_times.append(tick_time)
            print(f"    Tick {tick+1}: {tick_time:.3f}s")

        # Calculate metrics
        import numpy as np
        mean_tick = np.mean(tick_times)

        # GPU configuration
        CUs = 110
        threads_per_block = 128
        max_concurrent_blocks = max_blocks_per_sm * CUs
        total_threads = max_concurrent_blocks * threads_per_block
        total_agents = num_sites * 500501
        agents_per_thread = total_agents / total_threads if total_threads > 0 else 0

        print(f"\n  ✓ SUCCESS")
        print(f"    GPU config: {max_concurrent_blocks:,} blocks, {total_threads:,} threads")
        print(f"    Agents/thread: {agents_per_thread:.1f}")
        print(f"    Mean tick time: {mean_tick:.3f}s")

        return {
            'max_blocks_per_sm': max_blocks_per_sm,
            'mean_tick_time': mean_tick,
            'mem_gb': mem_gb if HAS_CUPY else 0,
            'success': True
        }

    except Exception as e:
        print(f"\n  ✗ FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'max_blocks_per_sm': max_blocks_per_sm,
            'success': False,
            'error': str(e)
        }


def main():
    print("="*70)
    print("GGap Quick Occupancy Test")
    print("="*70)
    print("Configuration:")
    print("  Sites: 5")
    print("  Gaps per site: 500")
    print("  Trees per gap: 1000")
    print("  Ticks per test: 3")
    print("  Testing max_blocks_per_sm: [4, 8, 16, 32]")
    print("="*70)

    num_sites = 5
    sm_values = [4, 8, 16, 32]
    num_ticks = 3

    results = []

    for sm_value in sm_values:
        result = test_one_sm_value(num_sites, sm_value, num_ticks)
        results.append(result)

        # Small delay between tests
        time.sleep(1)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]

    if successful:
        print(f"\n✓ Successful tests: {len(successful)}/{len(results)}")
        print(f"\n{'SM Value':<15} {'Mean Tick (s)':<15} {'Memory (GB)':<15}")
        print("-" * 45)

        for r in successful:
            print(f"{r['max_blocks_per_sm']:<15} {r['mean_tick_time']:<15.3f} {r['mem_gb']:<15.2f}")

        # Find fastest
        fastest = min(successful, key=lambda r: r['mean_tick_time'])
        slowest = max(successful, key=lambda r: r['mean_tick_time'])

        print("-" * 45)
        print(f"\nFastest: max_blocks_per_sm={fastest['max_blocks_per_sm']} ({fastest['mean_tick_time']:.3f}s)")
        if slowest['max_blocks_per_sm'] != fastest['max_blocks_per_sm']:
            speedup = slowest['mean_tick_time'] / fastest['mean_tick_time']
            print(f"Speedup: {speedup:.2f}× vs max_blocks_per_sm={slowest['max_blocks_per_sm']}")

    if failed:
        print(f"\n✗ Failed tests: {len(failed)}/{len(results)}")
        for r in failed:
            print(f"  max_blocks_per_sm={r['max_blocks_per_sm']}: {r.get('error', 'Unknown error')}")

    print("\n" + "="*70)

    if len(successful) == len(results):
        print("✓ ALL TESTS PASSED - Ready to submit full job!")
        print("\nNext step:")
        print("  cd /lustre/orion/lrn088/proj-shared/objective3/xxz/GGap/scaling_analysis/scripts")
        print("  sbatch tune_occupancy.sh")
    else:
        print("⚠️  SOME TESTS FAILED - Fix errors before submitting job")

    print("="*70)


if __name__ == '__main__':
    main()
