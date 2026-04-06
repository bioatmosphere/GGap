"""
Weak Scaling Test (Single Node: 1-8 GPUs)

Tests weak scaling with synthetic torus topology using 1D slab decomposition.
- Fix grid_height=4, each rank owns a 4x1 column slab
- Width scales with GPU count: width = num_gpus
- Cross-rank edges per rank = 24 (constant for 2+ GPUs)
- All ranks do identical work (4 sites each)

Usage:
    # 1 GPU
    srun -N1 -n1 --gpus-per-node=1 python test_weak_scaling_single_node.py

    # 4 GPUs
    srun -N1 -n4 --gpus-per-node=4 python test_weak_scaling_single_node.py

    # 8 GPUs with CSV output
    srun -N1 -n8 --gpus-per-node=8 python test_weak_scaling_single_node.py --csv ../results/weak_8gpus.csv
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
    """1D column-slab partition: site (i,j) -> rank j.

    Each rank owns a height x 1 column of the torus grid.
    """
    block_width = width // num_gpus
    return {i * width + j: j // block_width
            for i in range(height) for j in range(width)}


def run_weak_scaling_test(args):
    """Run weak scaling test."""
    result = {}

    # Configuration
    grid_height = args.grid_height
    sites_per_gpu = args.sites_per_gpu
    block_width = sites_per_gpu // grid_height

    assert block_width * grid_height == sites_per_gpu, \
        f"sites_per_gpu ({sites_per_gpu}) must be divisible by grid_height ({grid_height})"

    width = block_width * num_workers
    height = grid_height
    total_sites = width * height

    result['test_name'] = 'weak_scaling_single_node'
    result['num_gpus'] = num_workers
    result['sites_per_gpu'] = sites_per_gpu
    result['grid_height'] = height
    result['grid_width'] = width
    result['total_sites'] = total_sites
    result['num_gaps'] = args.num_gaps
    result['maxtrees'] = args.maxtrees
    result['max_blocks_per_sm'] = args.max_blocks_per_sm
    result['num_ticks'] = args.ticks

    if rank == 0:
        print("=" * 70)
        print("Weak Scaling Test: Single Node")
        print("=" * 70)
        print(f"  GPUs: {num_workers}")
        print(f"  Sites per GPU: {sites_per_gpu}")
        print(f"  Total sites: {total_sites}")
        print(f"  Grid: {height} x {width} (1D slab, block={height}x{block_width})")
        print(f"  Gaps per site: {args.num_gaps}")
        print(f"  Trees per gap: {args.maxtrees}")
        print(f"  Ticks: {args.ticks}")
        print(f"  max_blocks_per_sm: {args.max_blocks_per_sm}")
        print("=" * 70)

    # ========================================
    # Phase 1: Model Creation
    # ========================================
    t0 = time.time()
    model = GAPModel()
    result['model_creation_time'] = time.time() - t0

    if rank == 0:
        print(f"\n[Phase 1] Model creation: {result['model_creation_time']:.4f}s")

    # ========================================
    # Phase 2: Load Globals
    # ========================================
    t0 = time.time()
    data_dir = os.path.join(ggap_root, "scaling_analysis", "synthetic_data")
    model.load_globals(data_dir=data_dir, prefix="SYNTHETIC")
    result['load_globals_time'] = time.time() - t0

    if rank == 0:
        num_species = len(model.unique_species)
        print(f"[Phase 2] Load globals: {result['load_globals_time']:.4f}s ({num_species} species)")

    # ========================================
    # Phase 3: 1D Slab Partitioning
    # ========================================
    t0 = time.time()
    site_partition = slab_partition(width, height, num_workers)
    model._site_partition = site_partition

    local_site_ids = [sid for sid in range(total_sites) if site_partition[sid] == rank]
    result['num_local_sites'] = len(local_site_ids)
    result['partitioning_time'] = time.time() - t0

    if rank == 0:
        print(f"[Phase 3] Partitioning: {result['partitioning_time']:.4f}s ({len(local_site_ids)} sites/rank)")

    # ========================================
    # Phase 4: Site Initialization
    # ========================================
    t0 = time.time()
    local_site_agents = {}  # site_id -> site_agent_id (local only)
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

    # ========================================
    # Phase 5: Torus Connectivity
    # ========================================
    t0 = time.time()

    # Gather site_id -> agent_id mapping from all ranks
    all_site_agents = comm.allgather(local_site_agents)
    all_site_agents_dict = {}
    for d in all_site_agents:
        all_site_agents_dict.update(d)

    total_edges = model.connect_torus_sites(width, height, all_site_agents_dict)
    result['connectivity_time'] = time.time() - t0
    result['total_edges'] = total_edges

    if rank == 0:
        print(f"[Phase 5] Connectivity: {result['connectivity_time']:.4f}s "
              f"({total_edges} edges, {total_edges/total_sites:.0f}/site)")

    # ========================================
    # Phase 6: Register Arrays + GPU Setup
    # ========================================
    t0 = time.time()
    model.register_breed_local_arrays()
    result['register_arrays_time'] = time.time() - t0

    if rank == 0:
        print(f"[Phase 6a] Register arrays: {result['register_arrays_time']:.4f}s")

    if hasattr(model, '_max_blocks_per_sm'):
        model._max_blocks_per_sm = args.max_blocks_per_sm

    t0 = time.time()
    model.setup(use_gpu=True)
    result['gpu_setup_time'] = time.time() - t0

    if rank == 0:
        print(f"[Phase 6b] GPU setup: {result['gpu_setup_time']:.4f}s")

    # ========================================
    # Phase 7: Simulation
    # ========================================
    if rank == 0:
        print(f"\n[Phase 7] Running simulation ({args.ticks} ticks)...")
        print("-" * 70)

    # Warm-up tick (first tick builds buffers)
    t0 = time.time()
    model.simulate(ticks=1, sync_workers_every_n_ticks=1)
    result['first_tick_time'] = time.time() - t0

    if rank == 0:
        print(f"  First tick (buffer build): {result['first_tick_time']:.4f}s")

    # Steady-state ticks
    remaining_ticks = args.ticks - 1
    if remaining_ticks > 0:
        t0 = time.time()
        model.simulate(ticks=remaining_ticks, sync_workers_every_n_ticks=1)
        steady_time = time.time() - t0
        result['steady_state_time'] = steady_time
        result['mean_tick_time'] = steady_time / remaining_ticks
    else:
        result['steady_state_time'] = 0.0
        result['mean_tick_time'] = result['first_tick_time']

    result['simulation_time'] = result['first_tick_time'] + result['steady_state_time']

    if rank == 0:
        if remaining_ticks > 0:
            print(f"  Steady state ({remaining_ticks} ticks): {result['steady_state_time']:.4f}s "
                  f"({result['mean_tick_time']:.4f}s/tick)")
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
    ]

    write_header = not os.path.exists(csv_file)

    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        if write_header:
            writer.writeheader()
        writer.writerow(result)

    print(f"Results appended to: {csv_file}")


def main():
    parser = argparse.ArgumentParser(description="Weak scaling test (single node)")
    parser.add_argument('--sites-per-gpu', type=int, default=10,
                       help='Sites per GPU (default: 10)')
    parser.add_argument('--grid-height', type=int, default=5,
                       help='Fixed grid height (default: 5)')
    parser.add_argument('--num-gaps', type=int, default=500,
                       help='Gaps per site (default: 500)')
    parser.add_argument('--maxtrees', type=int, default=1000,
                       help='Max trees per gap (default: 1000)')
    parser.add_argument('--ticks', type=int, default=20,
                       help='Number of simulation ticks (default: 20)')
    parser.add_argument('--max-blocks-per-sm', type=int, default=8,
                       help='Max GPU blocks per SM (default: 8)')
    parser.add_argument('--csv', type=str, default=None,
                       help='Output CSV file for results')

    args = parser.parse_args()

    try:
        result = run_weak_scaling_test(args)

        if rank == 0:
            print("\n" + "=" * 70)
            print("Test Complete")
            print("=" * 70)
            print(f"  GPUs: {result['num_gpus']}")
            print(f"  Sites/GPU: {result['sites_per_gpu']}")
            print(f"  Total sites: {result['total_sites']}")
            print(f"  Grid: {result['grid_height']}x{result['grid_width']}")
            print(f"  First tick: {result['first_tick_time']:.4f}s")
            print(f"  Mean tick (steady): {result['mean_tick_time']:.4f}s")
            print(f"  Total simulation: {result['simulation_time']:.4f}s")
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
