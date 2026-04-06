"""
Test synthetic torus model: grid validation + GPU simulation.

Part A — Grid/Partition Validation (no GPU):
  For 1, 2, 4, 8 GPU configurations, verify 1D slab decomposition
  produces correct grid, exact site counts, and constant cross-rank edges.

Part B — Full GPU Simulation (1 GPU):
  Run the complete GAPModel pipeline on a 4x1 torus grid (4 sites).

1D Slab Decomposition for Weak Scaling:
  Fix grid_height=4. Each rank owns a 4x1 column slab.
  Width scales with GPU count: width = num_gpus.
  Cross-rank edges per rank = height * 3 * 2 = 24 (constant for 2+ GPUs).

  GPUs  Grid   Sites/GPU  Cross edges/rank
  1     4x1    4          0
  2     4x2    4          24
  4     4x4    4          24
  8     4x8    4          24

Usage (on compute node with GPU):
    python test_synthetic_model_local.py
    python test_synthetic_model_local.py --num_gaps 5 --maxtrees 50 --ticks 5
"""

import argparse
import sys
import os
import time
import numpy as np
from pathlib import Path

# Add GGap to path
ggap_root = Path(__file__).resolve().parent.parent.parent
if str(ggap_root) not in sys.path:
    sys.path.insert(0, str(ggap_root))


# --- 1D slab decomposition utilities ---

def slab_partition(width, height, num_gpus):
    """1D column-slab partition: site (i,j) -> rank j // block_width.

    Each rank owns a height x block_width slab of the torus grid.
    """
    block_width = width // num_gpus
    return {i * width + j: j // block_width
            for i in range(height) for j in range(width)}


def count_cross_rank_edges(width, height, site_partition):
    """Count torus Moore-neighbor edges that cross partition boundaries.

    Returns (cross_per_rank dict, total_edges).
    Each site has 8 Moore neighbors on the torus.  An edge is cross-rank
    when the site and its neighbor belong to different ranks.
    """
    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1),           (0, 1),
                 (1, -1),  (1, 0),  (1, 1)]
    cross_per_rank = {}
    total_edges = 0
    for i in range(height):
        for j in range(width):
            sid = i * width + j
            r = site_partition[sid]
            for di, dj in neighbors:
                ni = (i + di) % height
                nj = (j + dj) % width
                nid = ni * width + nj
                total_edges += 1
                if r != site_partition[nid]:
                    cross_per_rank[r] = cross_per_rank.get(r, 0) + 1
    return cross_per_rank, total_edges


# ======================================================================
# Part A: Grid/Partition Validation (no GPU)
# ======================================================================

def run_grid_tests(sites_per_gpu, grid_height):
    """Validate 1D slab decomposition for 1, 2, 4, 8 GPU configs."""
    print("=" * 70)
    print("Part A: Grid / Partition Validation (no GPU)")
    print("=" * 70)
    print(f"  sites_per_gpu={sites_per_gpu}, grid_height={grid_height}")
    print(f"  block shape per rank: {grid_height} x {sites_per_gpu // grid_height}")
    print()

    block_width = sites_per_gpu // grid_height
    assert block_width * grid_height == sites_per_gpu, \
        f"sites_per_gpu ({sites_per_gpu}) must be divisible by grid_height ({grid_height})"

    header = f"{'GPUs':>4}  {'Grid':>8}  {'Sites':>5}  {'Sites/GPU':>9}  {'Cross/rank':>10}  {'Total edges':>11}  {'Status'}"
    print(header)
    print("-" * len(header))

    for num_gpus in [1, 2, 4, 8]:
        width = block_width * num_gpus
        height = grid_height
        total_sites = width * height

        part = slab_partition(width, height, num_gpus)
        cross_per_rank, total_edges = count_cross_rank_edges(width, height, part)

        # Verify each rank gets exactly sites_per_gpu sites
        sites_per_rank = {}
        for sid, r in part.items():
            sites_per_rank[r] = sites_per_rank.get(r, 0) + 1

        # Expected cross-rank edges: 0 for 1 GPU, height*6 for 2+ GPUs
        if num_gpus == 1:
            expected_cross = 0
        else:
            expected_cross = height * 3 * 2  # 3 neighbors x 2 boundaries

        cross_vals = sorted(cross_per_rank.values()) if cross_per_rank else [0]
        actual_cross = cross_vals[0]
        all_equal = len(set(cross_vals)) <= 1

        # Checks
        ok = True
        errors = []

        if len(sites_per_rank) != num_gpus:
            ok = False
            errors.append(f"only {len(sites_per_rank)}/{num_gpus} ranks have sites")

        for r in range(num_gpus):
            if sites_per_rank.get(r, 0) != sites_per_gpu:
                ok = False
                errors.append(f"rank {r} has {sites_per_rank.get(r,0)} sites, expected {sites_per_gpu}")
                break

        if actual_cross != expected_cross:
            ok = False
            errors.append(f"cross/rank={actual_cross}, expected {expected_cross}")

        if not all_equal:
            ok = False
            errors.append(f"cross-rank edges not equal across ranks: {cross_vals}")

        status = "OK" if ok else f"FAIL: {'; '.join(errors)}"
        print(f"{num_gpus:>4}  {height}x{width:>5}  {total_sites:>5}  {sites_per_gpu:>9}  "
              f"{actual_cross:>10}  {total_edges:>11}  {status}")

        if not ok:
            print(f"\n  FATAL: {status}")
            sys.exit(1)

    print()
    print("Part A: all grid configurations validated.")
    return True


# ======================================================================
# Part B: Full GPU Simulation (1 GPU)
# ======================================================================

def run_gpu_simulation(sites_per_gpu, grid_height, num_gaps, maxtrees, ticks):
    """Run full GPU simulation on the 1-GPU case."""
    from gap.gap_model import GAPModel
    from gap.constants import SiteP, TreeS

    block_width = sites_per_gpu // grid_height
    width = block_width * 1  # 1 GPU
    height = grid_height
    total_sites = width * height

    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'synthetic_data'))
    prefix = "SYNTHETIC"

    print()
    print("=" * 70)
    print("Part B: GPU Simulation (1 GPU)")
    print("=" * 70)
    print(f"  Grid: {height}x{width} = {total_sites} sites")
    print(f"  Gaps/site: {num_gaps}, Trees/gap: {maxtrees}")
    print(f"  Ticks: {ticks}")
    print(f"  Data: {data_dir}")
    print()

    timings = {}

    # B.1 Create model
    print("[B.1] Creating model...")
    t0 = time.time()
    model = GAPModel()
    timings['model_create'] = time.time() - t0
    print(f"  OK ({timings['model_create']:.3f}s)")

    # B.2 Load globals
    print("[B.2] Loading globals...")
    t0 = time.time()
    model.load_globals(data_dir=data_dir, prefix=prefix)
    timings['load_globals'] = time.time() - t0
    num_species = len(model.unique_species)
    print(f"  OK - {num_species} species, {model._num_sites_in_globals} sites in globals ({timings['load_globals']:.3f}s)")

    # B.3 Partition (all on rank 0)
    print("[B.3] Partitioning (all sites -> rank 0)...")
    site_ids = list(range(total_sites))
    model._site_partition = {sid: 0 for sid in site_ids}
    print(f"  OK - {total_sites} sites on rank 0")

    # B.4 Initialize sites
    print(f"[B.4] Initializing {total_sites} sites ({num_gaps} gaps x {maxtrees} trees)...")
    t0 = time.time()
    site_agents_dict = {}
    local_sites = []
    for site_id in site_ids:
        site = model.initialize_site_with_gaps(
            site_id=site_id,
            num_gaps=num_gaps,
            maxtrees=maxtrees,
            data_dir=data_dir,
            prefix=prefix,
            rank=0,
        )
        site_agents_dict[site_id] = site['site_agent_id']
        local_sites.append(site)
    timings['site_init'] = time.time() - t0

    num_species_per_site = len(local_sites[0]['species'])
    trees_per_gap = num_species_per_site + maxtrees
    expected_gaps = total_sites * num_gaps
    expected_trees = total_sites * num_gaps * trees_per_gap
    total_agents = total_sites + expected_gaps + expected_trees

    assert len(model.site_agents) == total_sites, \
        f"Sites: {len(model.site_agents)} != {total_sites}"
    assert len(model.gap_agents) == expected_gaps, \
        f"Gaps: {len(model.gap_agents)} != {expected_gaps}"
    assert len(model.tree_ids) == expected_trees, \
        f"Trees: {len(model.tree_ids)} != {expected_trees}"

    print(f"  OK ({timings['site_init']:.2f}s)")
    print(f"    Sites={total_sites}, Gaps={expected_gaps}, Trees={expected_trees} ({num_species_per_site} templates + {maxtrees} free/gap)")
    print(f"    Total agents: {total_agents}")

    # B.5 Connect torus
    print(f"[B.5] Connecting torus ({height}x{width})...")
    t0 = time.time()
    total_edges = model.connect_torus_sites(width, height, site_agents_dict)
    timings['connect'] = time.time() - t0
    print(f"  OK - {total_edges} edges ({total_edges // total_sites}/site) ({timings['connect']:.3f}s)")

    # B.6 Register breed-local arrays
    print("[B.6] Registering breed-local arrays...")
    t0 = time.time()
    model.register_breed_local_arrays()
    timings['register_arrays'] = time.time() - t0
    print(f"  OK ({timings['register_arrays']:.3f}s)")

    # B.7 GPU setup
    print("[B.7] GPU setup...")
    t0 = time.time()
    model.setup(use_gpu=True)
    timings['gpu_setup'] = time.time() - t0
    print(f"  OK ({timings['gpu_setup']:.2f}s)")

    # B.8 Simulate
    print(f"[B.8] Simulating ({ticks} ticks)...")
    t0 = time.time()
    model.simulate(ticks=ticks, sync_workers_every_n_ticks=1)
    timings['simulate'] = time.time() - t0
    print(f"  OK ({timings['simulate']:.2f}s, {timings['simulate']/ticks:.3f}s/tick)")

    # B.9 Retrieve data from GPU
    print("[B.9] Retrieving data from GPU...")
    t0 = time.time()
    site_params = model.get_breed_data("Site", "params")
    site_states = model.get_breed_data("Site", "states")
    tree_params = model.get_breed_data("Tree", "params")
    tree_states = model.get_breed_data("Tree", "states")
    timings['gpu_download'] = time.time() - t0

    print(f"  OK ({timings['gpu_download']:.3f}s)")
    print(f"    Site params: {site_params.shape}, states: {site_states.shape}")
    print(f"    Tree params: {tree_params.shape}, states: {tree_states.shape}")

    # Sanity checks
    a0_c = site_params[:, int(SiteP.A0_C)]
    assert not np.any(np.isnan(a0_c)), "NaN in soil A0 carbon!"
    print(f"    Soil A0 C: min={a0_c.min():.3f}, max={a0_c.max():.3f}, mean={a0_c.mean():.3f}")

    is_alive = tree_states[:, int(TreeS.IS_ALIVE)]
    alive = int((is_alive > 0.5).sum())
    templates = int((is_alive < -0.5).sum())
    free = int(((is_alive > -0.5) & (is_alive < 0.5)).sum())
    print(f"    Trees: alive={alive}, templates={templates}, free={free}")

    # Summary
    print()
    print("-" * 70)
    print("Timing breakdown:")
    print(f"  Model creation:      {timings['model_create']:.3f}s")
    print(f"  Load globals:        {timings['load_globals']:.3f}s")
    print(f"  Site initialization: {timings['site_init']:.2f}s")
    print(f"  Torus connectivity:  {timings['connect']:.3f}s")
    print(f"  Register arrays:     {timings['register_arrays']:.3f}s")
    print(f"  GPU setup:           {timings['gpu_setup']:.2f}s")
    print(f"  Simulation:          {timings['simulate']:.2f}s ({ticks} ticks, {timings['simulate']/ticks:.3f}s/tick)")
    print(f"  GPU download:        {timings['gpu_download']:.3f}s")
    print()
    print(f"Agents: {total_agents} ({total_sites} sites, {expected_gaps} gaps, {expected_trees} trees)")
    print(f"Alive trees after {ticks} ticks: {alive}")

    return timings


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Test synthetic torus model: grid validation + GPU simulation"
    )
    parser.add_argument("--sites_per_gpu", type=int, default=4,
                       help="Sites per GPU (default: 4, block=4x1)")
    parser.add_argument("--grid_height", type=int, default=4,
                       help="Fixed grid height (default: 4)")
    parser.add_argument("--num_gaps", type=int, default=10,
                       help="Gaps per site (default: 10)")
    parser.add_argument("--maxtrees", type=int, default=100,
                       help="Max tree slots per gap (default: 100)")
    parser.add_argument("--ticks", type=int, default=10,
                       help="Simulation ticks (default: 10)")
    args = parser.parse_args()

    assert args.sites_per_gpu % args.grid_height == 0, \
        f"sites_per_gpu ({args.sites_per_gpu}) must be divisible by grid_height ({args.grid_height})"

    # Part A: validate grids for 1, 2, 4, 8 GPUs
    run_grid_tests(
        sites_per_gpu=args.sites_per_gpu,
        grid_height=args.grid_height,
    )

    # Part B: full GPU simulation on 1-GPU case
    run_gpu_simulation(
        sites_per_gpu=args.sites_per_gpu,
        grid_height=args.grid_height,
        num_gaps=args.num_gaps,
        maxtrees=args.maxtrees,
        ticks=args.ticks,
    )

    print()
    print("=" * 70)
    print("ALL TESTS PASSED - Ready to submit scaling jobs!")
    print("=" * 70)


if __name__ == "__main__":
    main()
