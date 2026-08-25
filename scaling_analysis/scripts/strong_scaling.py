"""
Strong Scaling Test

Fixed problem size, increase GPU count.
Each rank gets fewer sites as GPUs increase.

Setup: grid_height=4, width=total_sites/grid_height
  1 node (8 GPUs):   64 sites/GPU, block=4x16
  2 nodes (16 GPUs):  32 sites/GPU, block=4x8
  4 nodes (32 GPUs):  16 sites/GPU, block=4x4
  8 nodes (64 GPUs):   8 sites/GPU, block=4x2
  16 nodes (128 GPUs):  4 sites/GPU, block=4x1

Usage:
    srun -N1 -n8 --gpus-per-node=8 python -u strong_scaling.py --total-sites 512
    srun -N16 -n128 --gpus-per-node=8 python -u strong_scaling.py --total-sites 512
"""

import argparse
import csv
import sys
import os
import time
import numpy as np
from pathlib import Path

# Add GGap to path
ggap_root = Path(__file__).resolve().parent.parent.parent
if str(ggap_root) not in sys.path:
    sys.path.insert(0, str(ggap_root))

from gap.gap_model import GAPModel

# MPI
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_workers = comm.Get_size()


def slab_partition(width, height, num_gpus):
    """1D column-slab partition: site (i,j) -> rank j // block_width."""
    block_width = width // num_gpus
    return {i * width + j: j // block_width
            for i in range(height) for j in range(width)}


def run_strong_scaling_test(args):
    """Run strong scaling test with fixed total sites."""
    result = {}

    # Configuration
    total_sites = args.total_sites
    grid_height = args.grid_height

    assert total_sites % grid_height == 0, \
        f"total_sites ({total_sites}) must be divisible by grid_height ({grid_height})"

    width = total_sites // grid_height

    assert total_sites % num_workers == 0, \
        f"total_sites ({total_sites}) must be divisible by num_workers ({num_workers})"
    assert width % num_workers == 0, \
        f"width ({width}) must be divisible by num_workers ({num_workers})"

    sites_per_gpu = total_sites // num_workers
    block_width = width // num_workers

    result['test_name'] = 'strong_scaling'
    result['num_gpus'] = num_workers
    result['sites_per_gpu'] = sites_per_gpu
    result['grid_height'] = grid_height
    result['grid_width'] = width
    result['total_sites'] = total_sites
    result['num_gaps'] = args.num_gaps
    result['maxtrees'] = args.maxtrees
    result['max_blocks_per_sm'] = args.max_blocks_per_sm
    result['num_ticks'] = args.ticks

    if rank == 0:
        print("=" * 70)
        print("Strong Scaling Test")
        print("=" * 70)
        print(f"  GPUs: {num_workers}")
        print(f"  Total sites: {total_sites} (FIXED)")
        print(f"  Sites per GPU: {sites_per_gpu}")
        print(f"  Grid: {grid_height} x {width} (1D slab, block={grid_height}x{block_width})")
        print(f"  Gaps per site: {args.num_gaps}")
        print(f"  Trees per gap: {args.maxtrees}")
        print(f"  Ticks: {args.ticks}")
        print(f"  max_blocks_per_sm: {args.max_blocks_per_sm}")
        print("=" * 70)

    # Phase 1: Model Creation
    t0 = time.time()
    model = GAPModel(verbose_timing=True)
    result['model_creation_time'] = time.time() - t0

    if rank == 0:
        print(f"\n[Phase 1] Model creation: {result['model_creation_time']:.4f}s")

    # Phase 2: Load Globals
    t0 = time.time()
    data_dir = os.path.join(ggap_root, "scaling_analysis", "synthetic_data")
    model.load_globals(data_dir=data_dir, prefix="SYNTHETIC")
    result['load_globals_time'] = time.time() - t0

    if rank == 0:
        num_species = len(model.unique_species)
        print(f"[Phase 2] Load globals: {result['load_globals_time']:.4f}s ({num_species} species)")

    # Phase 3: 1D Slab Partitioning
    t0 = time.time()
    site_partition = slab_partition(width, grid_height, num_workers)
    model._site_partition = site_partition

    local_site_ids = [sid for sid in range(total_sites) if site_partition[sid] == rank]
    result['num_local_sites'] = len(local_site_ids)
    result['partitioning_time'] = time.time() - t0

    if rank == 0:
        print(f"[Phase 3] Partitioning: {result['partitioning_time']:.4f}s ({len(local_site_ids)} sites/rank)")

    # Phase 4: Site Initialization
    t0 = time.time()
    local_site_agents = {}
    local_sites = []
    for site_id in local_site_ids:
        site = model.initialize_site_with_gaps(
            site_id=site_id,
            num_gaps=args.num_gaps,
            maxtrees=args.maxtrees,
            data_dir=data_dir,
            prefix="SYNTHETIC",
            rank=rank,
        )
        local_site_agents[site_id] = site['site_agent_id']
        local_sites.append(site)

    result['site_init_time'] = time.time() - t0
    if rank == 0:
        print(f"[Phase 4] Site init: {result['site_init_time']:.4f}s "
              f"({len(model.site_agents)} sites, {len(model.gap_agents)} gaps, "
              f"{len(model.tree_ids)} trees)")

    # Phase 5: Torus Connectivity
    t0 = time.time()

    all_site_agents = comm.allgather(local_site_agents)
    all_site_agents_dict = {}
    for d in all_site_agents:
        all_site_agents_dict.update(d)

    total_edges = model.connect_torus_sites(width, grid_height, all_site_agents_dict)
    result['connectivity_time'] = time.time() - t0
    result['total_edges'] = total_edges

    if rank == 0:
        print(f"[Phase 5] Connectivity: {result['connectivity_time']:.4f}s "
              f"({total_edges} edges, {total_edges/total_sites:.0f}/site)")

    # Phase 6: Register Arrays + GPU Setup
    t0 = time.time()
    model.register_breed_local_arrays()
    result['register_arrays_time'] = time.time() - t0

    if rank == 0:
        print(f"[Phase 6a] Register arrays: {result['register_arrays_time']:.4f}s")

    if hasattr(model, '_max_blocks_per_sm'):
        model._max_blocks_per_sm = args.max_blocks_per_sm

    t0 = time.time()
    model.setup()
    result['gpu_setup_time'] = time.time() - t0

    if rank == 0:
        print(f"[Phase 6b] GPU setup: {result['gpu_setup_time']:.4f}s")

    # Phase 7: Simulation
    if rank == 0:
        print(f"\n[Phase 7] Running simulation ({args.ticks} ticks)...")
        print("-" * 70)

    t0 = time.time()
    model.simulate(ticks=args.ticks, sync_workers_every_n_ticks=1)
    total_sim_time = time.time() - t0

    # Extract first-tick vs steady-state from _tick_timings
    timings = model._tick_timings
    result['first_tick_time'] = timings[0]['total'] if timings else 0.0

    if len(timings) > 1:
        steady_timings = timings[1:]
        result['steady_state_time'] = sum(t['total'] for t in steady_timings)
        result['mean_tick_time'] = result['steady_state_time'] / len(steady_timings)
    else:
        result['steady_state_time'] = 0.0
        result['mean_tick_time'] = result['first_tick_time']

    result['simulation_time'] = total_sim_time

    # Aggregate per-tick breakdown from steady-state ticks
    if len(timings) > 1:
        steady = timings[1:]
        timing_keys = ['gpu_compute', 'gpu_sync', 'data_prep', 'kernel_args_build',
                        'write_back', 'mpi_gpu_pack', 'mpi_gpu_sync_pack',
                        'mpi_exchange', 'mpi_wait_time', 'mpi_gpu_unpack', 'total']
        for k in timing_keys:
            vals = [t.get(k, 0.0) for t in steady]
            result[f'mean_{k}'] = sum(vals) / len(vals)

        # Derived metrics
        # gpu_compute = kernel launch overhead (async, ~0.5ms)
        # gpu_sync = CPU waiting for kernel to finish (~35ms)
        # gpu_execution = actual GPU work time (launch + execution)
        result['mean_gpu_execution'] = (result.get('mean_gpu_compute', 0.0) +
                                         result.get('mean_gpu_sync', 0.0))
        result['mean_mpi_total'] = (result.get('mean_mpi_gpu_pack', 0.0) +
                                     result.get('mean_mpi_exchange', 0.0) +
                                     result.get('mean_mpi_gpu_unpack', 0.0))
        result['gpu_execution_fraction'] = (result['mean_gpu_execution'] / result['mean_total']
                                             if result.get('mean_total', 0) > 0 else 0.0)
        result['mpi_fraction'] = (result['mean_mpi_total'] / result['mean_total']
                                   if result.get('mean_total', 0) > 0 else 0.0)

    if rank == 0:
        remaining_ticks = args.ticks - 1
        print(f"  First tick (buffer build): {result['first_tick_time']:.4f}s")
        if remaining_ticks > 0:
            print(f"  Steady state ({remaining_ticks} ticks): {result['steady_state_time']:.4f}s "
                  f"({result['mean_tick_time']:.4f}s/tick)")
            if 'gpu_execution_fraction' in result:
                print(f"  GPU execution: {result.get('mean_gpu_execution', 0)*1000:.2f}ms/tick "
                      f"({result['gpu_execution_fraction']*100:.1f}%)")
                print(f"  MPI comm: {result.get('mean_mpi_total', 0)*1000:.2f}ms/tick "
                      f"({result['mpi_fraction']*100:.1f}%)")
        print(f"  Total simulation: {result['simulation_time']:.4f}s")
        print("-" * 70)

    return result


def write_csv(result, csv_file):
    """Append result row to CSV (rank 0 only). Writes header if file is new."""
    if rank != 0:
        return

    os.makedirs(os.path.dirname(os.path.abspath(csv_file)), exist_ok=True)

    fieldnames = [
        'num_gpus', 'sites_per_gpu', 'total_sites', 'grid_height', 'grid_width',
        'num_gaps', 'maxtrees', 'num_ticks',
        'model_creation_time', 'load_globals_time', 'partitioning_time',
        'site_init_time', 'connectivity_time', 'register_arrays_time',
        'gpu_setup_time', 'first_tick_time', 'steady_state_time',
        'mean_tick_time', 'simulation_time',
        'total_edges',
        # Per-tick timing breakdown (averaged over steady-state ticks)
        'mean_gpu_compute', 'mean_gpu_sync', 'mean_data_prep',
        'mean_kernel_args_build', 'mean_write_back',
        'mean_mpi_gpu_pack', 'mean_mpi_gpu_sync_pack', 'mean_mpi_exchange',
        'mean_mpi_wait_time', 'mean_mpi_gpu_unpack',
        'mean_gpu_execution', 'mean_mpi_total', 'mean_total',
        'gpu_execution_fraction', 'mpi_fraction',
    ]

    write_header = not os.path.exists(csv_file)

    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        if write_header:
            writer.writeheader()
        writer.writerow(result)

    print(f"Results appended to: {csv_file}")


def main():
    parser = argparse.ArgumentParser(description="Strong scaling test")
    parser.add_argument('--total-sites', type=int, default=512,
                       help='Fixed total sites (default: 512)')
    parser.add_argument('--grid-height', type=int, default=4,
                       help='Grid height (default: 4)')
    parser.add_argument('--num-gaps', type=int, default=500,
                       help='Gaps per site (default: 500)')
    parser.add_argument('--maxtrees', type=int, default=1000,
                       help='Max trees per gap (default: 1000)')
    parser.add_argument('--ticks', type=int, default=1000,
                       help='Number of simulation ticks (default: 1000)')
    parser.add_argument('--max-blocks-per-sm', type=int, default=8,
                       help='Max GPU blocks per SM (default: 8)')
    parser.add_argument('--csv', type=str, default=None,
                       help='Output CSV file for results')

    args = parser.parse_args()

    try:
        result = run_strong_scaling_test(args)

        if rank == 0:
            print("\n" + "=" * 70)
            print("Test Complete")
            print("=" * 70)
            print(f"  GPUs: {result['num_gpus']}")
            print(f"  Total sites: {result['total_sites']} (fixed)")
            print(f"  Sites/GPU: {result['sites_per_gpu']}")
            print(f"  Grid: {result['grid_height']}x{result['grid_width']}")
            print(f"  First tick: {result['first_tick_time']:.4f}s")
            print(f"  Mean tick (steady): {result['mean_tick_time']:.4f}s")
            print(f"  Total simulation: {result['simulation_time']:.4f}s")
            if 'gpu_execution_fraction' in result:
                print(f"  GPU execution fraction: {result['gpu_execution_fraction']*100:.1f}%")
                print(f"  MPI fraction: {result['mpi_fraction']*100:.1f}%")
            print("=" * 70)

        if args.csv:
            write_csv(result, args.csv)

    except Exception as e:
        print(f"[Rank {rank}] ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        comm.Abort(1)


if __name__ == "__main__":
    main()
