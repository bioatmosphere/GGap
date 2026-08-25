"""
GPU Occupancy Tuning for GGap (Experiment 0b)

Find optimal max_blocks_per_sm for AMD MI250X GPUs.

Background:
- AMD MI250X GCD has 110 compute units (CUs)
- max_grid_blocks = max_blocks_per_sm × 110
- Default (2): 220 blocks → may underutilize GPU
- Higher values: More parallelism, but risk register pressure
- This parameter affects runtime performance, NOT memory allocation

Usage:
    # Single test
    python tune_occupancy.py --sites 50 --max_blocks_per_sm 16 --ticks 20

    # Sweep multiple SM values at fixed site count
    python tune_occupancy.py --sites 50 --max_blocks_per_sm 4,8,16,32 --ticks 20

    # Test site scaling at fixed SM
    python tune_occupancy.py --sites 10,20,30,40,50 --max_blocks_per_sm 16 --ticks 20
"""

import sys
import os
import time
import random
import argparse

# Force unbuffered output
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


def test_configuration(num_sites, max_blocks_per_sm, num_ticks=20, num_gaps=500,
                       maxtrees=1000, neighbors_per_site=10, prefix='CONUS'):
    """
    Test performance at a given configuration.

    Args:
        num_sites: Number of sites to test
        max_blocks_per_sm: GPU occupancy parameter
        num_ticks: Number of ticks to run (default: 20)
        num_gaps: Gaps per site (default: 500)
        maxtrees: Max trees per gap (default: 1000)
        neighbors_per_site: Random neighbors per site (default: 10)
        prefix: Data prefix (default: 'CONUS')

    Returns:
        dict with results or None on failure
    """
    try:
        # Create model
        model = GAPModel()

        # Load globals
        t_start = time.time()
        model.load_globals(prefix=prefix)
        load_time = time.time() - t_start

        # Get site IDs
        all_site_ids = sorted(model._site_id_to_slot.keys())
        site_ids = all_site_ids[:num_sites]

        # Partition sites (required before initialization)
        model.partition_sites(site_ids, strategy='round_robin')

        # Initialize sites (loop method)
        t_start = time.time()
        site_agents = {}
        for site_id in site_ids:
            site = model.initialize_site_with_gaps(site_id, num_gaps, maxtrees, prefix=prefix)
            site_agents[site_id] = site['site_agent_id']
        init_time = time.time() - t_start

        # Connect sites (random neighbors)
        t_start = time.time()
        random.seed(42)
        edges_added = 0
        for site_id in site_ids:
            other_sites = [s for s in site_ids if s != site_id]
            if len(other_sites) == 0:
                continue
            neighbors = random.sample(other_sites, min(neighbors_per_site, len(other_sites)))
            for neighbor_id in neighbors:
                if site_id < neighbor_id:
                    model.connect_agents(site_agents[site_id], site_agents[neighbor_id])
                    edges_added += 1
        connect_time = time.time() - t_start

        # Register breed locals
        t_start = time.time()
        model.register_breed_local_arrays()
        register_time = time.time() - t_start

        # SET OCCUPANCY PARAMETER BEFORE SETUP
        model._max_blocks_per_sm = max_blocks_per_sm
        model._verbose_timing = True  # Enable detailed timing output

        # GPU setup
        print(f"[DIAGNOSTIC] Starting GPU setup...", flush=True)
        t_start = time.time()
        model.setup()
        setup_time = time.time() - t_start
        print(f"[DIAGNOSTIC] GPU setup completed in {setup_time:.2f}s", flush=True)

        # Memory after setup
        if HAS_CUPY:
            mem_pool = cp.get_default_memory_pool()
            mem_bytes = mem_pool.used_bytes()
            mem_gb = mem_bytes / 1e9
        else:
            mem_gb = 0.0

        # Calculate actual kernel launch config
        import math
        num_agents = total_agents = num_sites * 500501
        threadsperblock = 128
        blockspergrid = int(math.ceil(num_agents / threadsperblock))
        num_sms = 110
        max_grid_blocks = max_blocks_per_sm * num_sms
        effective_blocks = min(blockspergrid, max_grid_blocks)

        print(f"[DIAGNOSTIC] Kernel launch config:", flush=True)
        print(f"  Total agents: {num_agents:,}", flush=True)
        print(f"  Blocks needed (blockspergrid): {blockspergrid:,}", flush=True)
        print(f"  Max blocks allowed (max_grid_blocks): {max_grid_blocks:,}", flush=True)
        print(f"  Effective blocks (min): {effective_blocks:,}", flush=True)
        print(f"  Threads per block: {threadsperblock}", flush=True)
        print(f"  Total threads: {effective_blocks * threadsperblock:,}", flush=True)
        print(f"  Agents per thread: {num_agents / (effective_blocks * threadsperblock):.1f}", flush=True)
        print(f"  Blocks per CU: {effective_blocks / num_sms:.1f}", flush=True)
        print(f"[DIAGNOSTIC] Starting simulation with {num_ticks} ticks...", flush=True)
        print(f"[DIAGNOSTIC] WARNING: If this hangs, it's a grid barrier deadlock!", flush=True)

        # Run all ticks in one fused call (single-worker optimization)
        t_sim_start = time.time()
        model.simulate(ticks=num_ticks)
        total_sim_time = time.time() - t_sim_start
        print(f"[DIAGNOSTIC] Simulation completed in {total_sim_time:.2f}s", flush=True)

        # Get timing data from SAGESim
        tick_times = []
        if hasattr(model, '_tick_timings') and model._tick_timings:
            tick_times = [t['total'] for t in model._tick_timings if 'total' in t]

        # Memory after simulation
        if HAS_CUPY:
            mem_bytes_after = mem_pool.used_bytes()
            mem_gb = max(mem_gb, mem_bytes_after / 1e9)

        # Calculate metrics
        import numpy as np
        mean_tick = np.mean(tick_times) if tick_times else total_sim_time
        min_tick = np.min(tick_times) if tick_times else total_sim_time
        max_tick = np.max(tick_times) if tick_times else total_sim_time
        std_tick = np.std(tick_times) if tick_times else 0

        # Extract timing breakdown from SAGESim
        timing_breakdown = {}
        if hasattr(model, '_tick_timings') and model._tick_timings:
            # Average timings across all ticks (or just first tick for one-time costs)
            timing_breakdown['agent_ids_gen'] = model._tick_timings[0].get('agent_ids_generation', 0)
            timing_breakdown['gpu_config'] = model._tick_timings[0].get('gpu_config', 0)

            # First tick data prep (includes buffer build)
            timing_breakdown['data_prep_first'] = model._tick_timings[0].get('data_prep', 0)

            # Average kernel execution across all ticks
            gpu_compute_times = [t.get('gpu_compute', 0) for t in model._tick_timings]
            timing_breakdown['gpu_compute_avg'] = np.mean(gpu_compute_times) if gpu_compute_times else 0
            timing_breakdown['gpu_compute_total'] = np.sum(gpu_compute_times) if gpu_compute_times else 0

        # Total agents (1 site + num_gaps + num_gaps × (species + maxtrees))
        # For CONUS avg ~1 species: 1 + 500 + 500×(1+1000) = 500,501 per site
        agents_per_site = 500501
        total_agents = num_sites * agents_per_site

        # GPU configuration
        CUs = 110
        threads_per_block = 128  # Hardcoded in SAGESim
        max_concurrent_blocks = max_blocks_per_sm * CUs
        total_threads = max_concurrent_blocks * threads_per_block
        agents_per_thread = total_agents / total_threads if total_threads > 0 else 0

        # Throughput: tree-years per second
        throughput = (total_agents * num_ticks) / total_sim_time if total_sim_time > 0 else 0

        return {
            'max_blocks_per_sm': max_blocks_per_sm,
            'sites': num_sites,
            'total_agents': total_agents,
            'max_concurrent_blocks': max_concurrent_blocks,
            'threads_per_block': threads_per_block,
            'total_threads': total_threads,
            'agents_per_thread': agents_per_thread,
            'mem_gb': mem_gb,
            'load_time': load_time,
            'init_time': init_time,
            'connect_time': connect_time,
            'register_time': register_time,
            'setup_time': setup_time,
            'mean_tick_time': mean_tick,
            'min_tick_time': min_tick,
            'max_tick_time': max_tick,
            'std_tick_time': std_tick,
            'total_sim_time': total_sim_time,
            'throughput_tree_years_per_sec': throughput,
            'tick_times': tick_times,
            'timing_breakdown': timing_breakdown,
            'success': True
        }

    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        return None


def print_result(result):
    """Print formatted result."""
    if result is None:
        print("✗ FAILED")
        return

    print("="*70)
    print(f"GPU Configuration:")
    print(f"  AMD MI250X: 110 CUs")
    print(f"  max_blocks_per_sm: {result['max_blocks_per_sm']}")
    print(f"  Max concurrent blocks: {result['max_concurrent_blocks']:,} ({result['max_blocks_per_sm']} × 110)")
    print(f"  Threads per block: {result['threads_per_block']}")
    print(f"  Total GPU threads: {result['total_threads']:,}")
    print(f"  Total agents: {result['total_agents']:,} ({result['sites']} sites × 500,501)")
    print(f"  Agents per thread: {result['agents_per_thread']:.1f}")
    print()
    print(f"Performance:")
    print(f"  Mean tick time: {result['mean_tick_time']:.4f}s ± {result['std_tick_time']:.4f}s")
    print(f"  Min tick time:  {result['min_tick_time']:.4f}s")
    print(f"  Max tick time:  {result['max_tick_time']:.4f}s")
    print(f"  Throughput:     {result['throughput_tree_years_per_sec']:,.0f} tree-years/sec")
    print()
    print(f"Memory & Timing:")
    print(f"  GPU memory:    {result['mem_gb']:.2f} GB / 64 GB ({100*result['mem_gb']/64:.1f}%)")
    print(f"  Load time:     {result['load_time']:.2f}s")
    print(f"  Init time:     {result['init_time']:.2f}s")
    print(f"  Setup time:    {result['setup_time']:.2f}s")
    print(f"  Simulation:    {result['total_sim_time']:.2f}s")

    # Timing breakdown
    if result.get('timing_breakdown'):
        tb = result['timing_breakdown']
        print()
        print(f"Detailed Timing Breakdown:")
        print(f"  Agent IDs generation:   {tb.get('agent_ids_gen', 0):.4f}s")
        print(f"  GPU config (cached):    {tb.get('gpu_config', 0):.4f}s")
        print(f"  Data prep (first tick): {tb.get('data_prep_first', 0):.4f}s")
        print(f"  GPU compute (avg/tick): {tb.get('gpu_compute_avg', 0):.4f}s")
        print(f"  GPU compute (total):    {tb.get('gpu_compute_total', 0):.4f}s")

        # Calculate percentages
        total = result['total_sim_time']
        print()
        print(f"Time Distribution:")
        print(f"  Initialization overhead: {100*result['init_time']/total:.1f}%")
        print(f"  GPU compute:            {100*tb.get('gpu_compute_total', 0)/total:.1f}%")
        print(f"  Other overhead:         {100*(total - result['init_time'] - tb.get('gpu_compute_total', 0))/total:.1f}%")

    print("="*70)


def write_csv(results, csv_path):
    """Write results to CSV."""
    import csv

    with open(csv_path, 'w', newline='') as f:
        fieldnames = [
            'max_blocks_per_sm', 'sites', 'total_agents', 'max_concurrent_blocks',
            'threads_per_block', 'total_threads', 'agents_per_thread', 'mem_gb',
            'setup_time', 'mean_tick_time', 'min_tick_time', 'max_tick_time',
            'std_tick_time', 'throughput_tree_years_per_sec'
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            if result is None:
                continue
            row = {k: result[k] for k in fieldnames}
            writer.writerow(row)

    print(f"\n✓ Results written to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='GPU Occupancy Tuning for GGap')
    parser.add_argument('--sites', type=str, required=True,
                       help='Site count(s) to test (single or comma-separated, e.g., "50" or "10,20,30,40,50")')
    parser.add_argument('--max_blocks_per_sm', type=str, required=True,
                       help='max_blocks_per_sm value(s) to test (single or comma-separated, e.g., "16" or "4,8,16,32")')
    parser.add_argument('--num-gaps', type=int, default=500,
                       help='Gaps per site (default: 500)')
    parser.add_argument('--maxtrees', type=int, default=1000,
                       help='Max trees per gap (default: 1000)')
    parser.add_argument('--ticks', type=int, default=20,
                       help='Number of ticks per test (default: 20)')
    parser.add_argument('--neighbors', type=int, default=10,
                       help='Random neighbors per site (default: 10)')
    parser.add_argument('--prefix', type=str, default='CONUS',
                       help='Data prefix (default: CONUS)')
    parser.add_argument('--csv', type=str, required=True,
                       help='Output CSV file path')

    args = parser.parse_args()

    # Parse site counts
    site_counts = [int(x.strip()) for x in args.sites.split(',')]

    # Parse max_blocks_per_sm values
    sm_values = [int(x.strip()) for x in args.max_blocks_per_sm.split(',')]

    print("="*70)
    print("GGap GPU Occupancy Tuning (Experiment 0b)")
    print("="*70)
    print(f"Configuration:")
    print(f"  Site counts: {site_counts}")
    print(f"  max_blocks_per_sm values: {sm_values}")
    print(f"  Gaps per site: {args.num_gaps}")
    print(f"  Trees per gap: {args.maxtrees}")
    print(f"  Ticks per test: {args.ticks}")
    print(f"  Neighbors per site: {args.neighbors}")
    print(f"  Data prefix: {args.prefix}")
    print("="*70)
    print()

    # Run tests
    results = []

    for num_sites in site_counts:
        for max_blocks_per_sm in sm_values:
            print(f"\n{'='*70}")
            print(f"Testing: {num_sites} sites, max_blocks_per_sm={max_blocks_per_sm}")
            print(f"{'='*70}")

            result = test_configuration(
                num_sites=num_sites,
                max_blocks_per_sm=max_blocks_per_sm,
                num_ticks=args.ticks,
                num_gaps=args.num_gaps,
                maxtrees=args.maxtrees,
                neighbors_per_site=args.neighbors,
                prefix=args.prefix
            )

            if result:
                print_result(result)
                results.append(result)
            else:
                print("✗ Test failed")

            print()

    # Write CSV
    if results:
        write_csv(results, args.csv)

        # Summary
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        print(f"Completed {len(results)} successful tests")

        if len(sm_values) > 1 and len(site_counts) == 1:
            # Find optimal SM value
            best_idx = min(range(len(results)), key=lambda i: results[i]['mean_tick_time'])
            best = results[best_idx]
            baseline = results[0]  # Assuming first is lowest SM value

            print(f"\n🏆 BEST PERFORMANCE (at {best['sites']} sites):")
            print(f"  max_blocks_per_sm: {best['max_blocks_per_sm']}")
            print(f"  Mean tick time: {best['mean_tick_time']:.4f}s")
            print(f"  Throughput: {best['throughput_tree_years_per_sec']:,.0f} tree-years/sec")
            print(f"  Speedup vs {baseline['max_blocks_per_sm']}: {baseline['mean_tick_time']/best['mean_tick_time']:.2f}×")

        print(f"{'='*70}")
    else:
        print("\n⚠️  No successful tests!")


if __name__ == '__main__':
    main()
