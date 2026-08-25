"""
Find maximum sites per GPU with synthetic data (100 species).

Tests: 10, 25, 50, 64, 80, 100 sites per GPU.
Uses torus connectivity (grid_height=1) matching strong scaling config.

Usage:
    srun -N1 -n1 --gpus-per-node=1 python -u test_memory_synthetic.py
"""

import sys
import os
import time

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from gap.gap_model import GAPModel

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


def test_sites(num_sites, data_dir, prefix="SYNTHETIC"):
    """Test if num_sites fit on 1 GPU. Returns (success, mem_gb, timings)."""
    timings = {}
    try:
        model = GAPModel()

        t0 = time.time()
        model.load_globals(data_dir=data_dir, prefix=prefix)
        timings['load_globals'] = time.time() - t0
        num_species = len(model.unique_species)

        site_ids = list(range(num_sites))
        model._site_partition = {sid: 0 for sid in site_ids}

        t0 = time.time()
        site_agents_dict = {}
        for sid in site_ids:
            site = model.initialize_site_with_gaps(
                sid, num_gaps=500, maxtrees=1000,
                data_dir=data_dir, prefix=prefix, rank=0)
            site_agents_dict[sid] = site['site_agent_id']
        timings['site_init'] = time.time() - t0

        t0 = time.time()
        model.connect_torus_sites(num_sites, 1, site_agents_dict)
        timings['connect'] = time.time() - t0

        t0 = time.time()
        model.register_breed_local_arrays()
        timings['register'] = time.time() - t0

        t0 = time.time()
        model.setup()
        timings['gpu_setup'] = time.time() - t0

        if HAS_CUPY:
            mem_gb = cp.get_default_memory_pool().used_bytes() / 1e9
        else:
            mem_gb = 0.0

        t0 = time.time()
        model.simulate(ticks=1, sync_workers_every_n_ticks=1)
        timings['first_tick'] = time.time() - t0

        if HAS_CUPY:
            mem_gb = max(mem_gb, cp.get_default_memory_pool().used_bytes() / 1e9)

        total_agents = len(model.site_agents) + len(model.gap_agents) + len(model.tree_ids)
        return True, mem_gb, timings, total_agents, num_species

    except Exception as e:
        error_str = str(e).lower()
        if any(kw in error_str for kw in ['memory', 'alloc', 'oom']):
            return False, None, {}, 0, 0
        else:
            print(f"  ERROR (not OOM): {type(e).__name__}: {str(e)[:200]}")
            raise


def main():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'synthetic_data'))

    print("=" * 70)
    print("Memory Capacity Test (Synthetic Data)")
    print("=" * 70)
    print(f"  Data: {data_dir}")
    print(f"  Config: 500 gaps x 1000 trees, torus connectivity")
    print("=" * 70)

    test_sizes = [256]
    results = []

    for num_sites in test_sizes:
        print(f"\n--- Testing {num_sites} sites ---", flush=True)

        success, mem_gb, timings, total_agents, num_species = test_sites(num_sites, data_dir)

        if success:
            total_time = sum(timings.values())
            print(f"  OK: {total_agents:,} agents, {mem_gb:.2f} GB ({mem_gb/64*100:.1f}%), "
                  f"{num_species} species, {total_time:.1f}s total", flush=True)
            print(f"    init={timings['site_init']:.1f}s setup={timings['gpu_setup']:.1f}s "
                  f"tick={timings['first_tick']:.1f}s", flush=True)
            results.append((num_sites, total_agents, mem_gb))
        else:
            print(f"  OOM at {num_sites} sites", flush=True)
            break

    print(f"\n{'='*70}")
    print("Summary:")
    print(f"{'Sites':>6} {'Agents':>12} {'Memory':>8} {'%GPU':>6}")
    print("-" * 36)
    for sites, agents, mem in results:
        print(f"{sites:>6} {agents:>12,} {mem:>7.2f}G {mem/64*100:>5.1f}%")

    if results:
        max_sites = results[-1][0]
        print(f"\nMax tested: {max_sites} sites/GPU")
        print(f"Strong scaling baseline (64 sites/GPU): {'OK' if max_sites >= 64 else 'NOT TESTED'}")
    print("=" * 70)


if __name__ == '__main__':
    main()
