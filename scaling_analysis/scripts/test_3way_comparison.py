"""
3-way performance comparison: original loop vs create_agents_bulk vs preallocate+populate.

Compares three initialization approaches:
1. Original loop (job 4334624): create_agent_of_breed() one at a time
2. Bulk with arrays: create_agents_bulk() with numpy arrays + list.extend()
3. Preallocate+populate: SuperNeuroABM style with populate_agent_at_index()

Usage:
    python test_3way_comparison.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from gap.gap_model import GAPModel


def test_3way_comparison(num_sites=2, num_gaps=500, maxtrees=1000):
    """Compare all three initialization methods."""
    print("="*80)
    print("3-Way Initialization Performance Comparison")
    print("="*80)
    print(f"Config: {num_sites} sites, {num_gaps} gaps, {maxtrees} trees per gap")
    print()

    # Get test sites
    temp_model = GAPModel()
    temp_model.load_globals(prefix='CONUS')
    all_site_ids = sorted(temp_model._site_id_to_slot.keys())
    site_ids = [sid for sid in all_site_ids if sid in temp_model._climate_rows][:num_sites]
    print(f"Testing with sites: {site_ids}")
    print()

    results = {}

    # ========================================================================
    # Method 1: Original LOOP (job 4334624)
    # ========================================================================
    print("="*80)
    print("Method 1: ORIGINAL LOOP (initialize_site_with_gaps)")
    print("  - Calls create_agent_of_breed() one agent at a time")
    print("  - This was used in job 4334624 (~10s per site)")
    print("="*80)

    model_loop = GAPModel()
    model_loop.load_globals(prefix='CONUS')
    model_loop.partition_sites(site_ids)

    t_start = time.time()
    for site_id in site_ids:
        model_loop.initialize_site_with_gaps(site_id, num_gaps, maxtrees, prefix='CONUS')
    t_loop = time.time() - t_start

    print(f"  Time: {t_loop:.3f}s")
    print(f"  Agents: {model_loop._agent_factory.num_agents:,}")
    print(f"  Sites: {len(model_loop.site_agents)}")
    print(f"  Gaps: {len(model_loop.gap_agents)}")
    print(f"  Trees: {len(model_loop.tree_ids)}")
    print()

    results['loop'] = {
        'time': t_loop,
        'agents': model_loop._agent_factory.num_agents,
        'sites': len(model_loop.site_agents),
        'gaps': len(model_loop.gap_agents),
        'trees': len(model_loop.tree_ids)
    }

    # ========================================================================
    # Method 2: BULK with numpy arrays (initialize_sites_bulk)
    # ========================================================================
    print("="*80)
    print("Method 2: BULK with numpy arrays (initialize_sites_bulk)")
    print("  - Calls create_agents_bulk() with numpy arrays")
    print("  - Uses list.extend() for efficient property tensor updates")
    print("  - 3 bulk calls per site: 1 site + N gaps + N*M trees")
    print("="*80)

    model_bulk = GAPModel()
    model_bulk.load_globals(prefix='CONUS')
    model_bulk.partition_sites(site_ids)

    t_start = time.time()
    site_agents = model_bulk.initialize_sites_bulk(site_ids, num_gaps, maxtrees, prefix='CONUS')
    t_bulk = time.time() - t_start

    print(f"  Time: {t_bulk:.3f}s")
    print(f"  Agents: {model_bulk._agent_factory.num_agents:,}")
    print(f"  Sites: {len(model_bulk.site_agents)}")
    print(f"  Gaps: {len(model_bulk.gap_agents)}")
    print(f"  Trees: {len(model_bulk.tree_ids)}")
    print()

    results['bulk'] = {
        'time': t_bulk,
        'agents': model_bulk._agent_factory.num_agents,
        'sites': len(model_bulk.site_agents),
        'gaps': len(model_bulk.gap_agents),
        'trees': len(model_bulk.tree_ids)
    }

    # ========================================================================
    # Method 3: PREALLOCATE+POPULATE (initialize_sites_preallocate)
    # ========================================================================
    print("="*80)
    print("Method 3: PREALLOCATE+POPULATE (initialize_sites_preallocate)")
    print("  - Pre-allocates all agent metadata globally")
    print("  - Calls populate_agent_at_index() for each agent individually")
    print("  - SuperNeuroABM-style approach with direct index writes")
    print("="*80)

    model_preallocate = GAPModel()
    model_preallocate.load_globals(prefix='CONUS')
    model_preallocate.partition_sites(site_ids)

    t_start = time.time()
    site_agents_pre = model_preallocate.initialize_sites_preallocate(
        site_ids, num_gaps, maxtrees, prefix='CONUS'
    )
    t_preallocate = time.time() - t_start

    print(f"  Time: {t_preallocate:.3f}s")
    print(f"  Agents: {model_preallocate._agent_factory.num_agents:,}")
    print(f"  Sites: {len(model_preallocate.site_agents)}")
    print(f"  Gaps: {len(model_preallocate.gap_agents)}")
    print(f"  Trees: {len(model_preallocate.tree_ids)}")
    print()

    results['preallocate'] = {
        'time': t_preallocate,
        'agents': model_preallocate._agent_factory.num_agents,
        'sites': len(model_preallocate.site_agents),
        'gaps': len(model_preallocate.gap_agents),
        'trees': len(model_preallocate.tree_ids)
    }

    # ========================================================================
    # RESULTS COMPARISON
    # ========================================================================
    print("="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print()

    # Verify agent counts match
    print("Agent Count Verification:")
    loop_agents = results['loop']['agents']
    bulk_agents = results['bulk']['agents']
    pre_agents = results['preallocate']['agents']

    if loop_agents == bulk_agents == pre_agents:
        print(f"  ✅ All methods create identical agent counts: {loop_agents:,}")
    else:
        print(f"  ❌ Agent count mismatch!")
        print(f"     Loop: {loop_agents:,}")
        print(f"     Bulk: {bulk_agents:,}")
        print(f"     Preallocate: {pre_agents:,}")
    print()

    # Performance comparison
    print("Performance Comparison:")
    print(f"  Method 1 (Loop):         {t_loop:.3f}s  [baseline]")
    print(f"  Method 2 (Bulk):         {t_bulk:.3f}s  [{t_loop/t_bulk:.2f}x speedup]")
    print(f"  Method 3 (Preallocate):  {t_preallocate:.3f}s  [{t_loop/t_preallocate:.2f}x speedup]")
    print()

    # Determine winner
    fastest_time = min(t_loop, t_bulk, t_preallocate)
    if fastest_time == t_bulk:
        winner = "Bulk with arrays"
        print(f"  🏆 WINNER: {winner}")
        print(f"     {fastest_time:.3f}s ({t_loop/fastest_time:.2f}x faster than original loop)")
    elif fastest_time == t_preallocate:
        winner = "Preallocate+populate"
        print(f"  🏆 WINNER: {winner}")
        print(f"     {fastest_time:.3f}s ({t_loop/fastest_time:.2f}x faster than original loop)")
    else:
        print(f"  ⚠️  Original loop is still fastest (bulk methods have overhead)")

    print()

    # Analysis
    print("="*80)
    print("ANALYSIS")
    print("="*80)
    print()
    print(f"Per-site initialization time:")
    print(f"  Loop:         {t_loop/num_sites:.3f}s per site")
    print(f"  Bulk:         {t_bulk/num_sites:.3f}s per site")
    print(f"  Preallocate:  {t_preallocate/num_sites:.3f}s per site")
    print()

    agents_per_site = loop_agents // num_sites
    print(f"Agents per site: {agents_per_site:,}")
    print(f"  - 1 site agent")
    print(f"  - {num_gaps} gap agents")
    print(f"  - ~{agents_per_site - num_gaps - 1:,} tree agents")
    print()

    print("Expected scaling to 50 sites (job 4334624 config):")
    print(f"  Loop:         {(t_loop/num_sites) * 50:.1f}s")
    print(f"  Bulk:         {(t_bulk/num_sites) * 50:.1f}s")
    print(f"  Preallocate:  {(t_preallocate/num_sites) * 50:.1f}s")
    print()

    print("="*80)


if __name__ == '__main__':
    test_3way_comparison()
