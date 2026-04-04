"""
Quick performance test comparing loop vs preallocate+populate initialization.

Usage:
    python test_bulk_performance.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from gap.gap_model import GAPModel


def test_initialization_performance(num_sites=2, num_gaps=500, maxtrees=1000):
    """Compare initialization time between loop and bulk methods."""
    print("="*70)
    print(f"Initialization Performance Test")
    print("="*70)
    print(f"Config: {num_sites} sites, {num_gaps} gaps, {maxtrees} trees per gap")
    print()

    # Get test sites
    temp_model = GAPModel()
    temp_model.load_globals(prefix='CONUS')
    all_site_ids = sorted(temp_model._site_id_to_slot.keys())
    site_ids = [sid for sid in all_site_ids if sid in temp_model._climate_rows][:num_sites]

    # Test LOOP method
    print("Testing LOOP method...")
    model_loop = GAPModel()
    model_loop.load_globals(prefix='CONUS')
    model_loop.partition_sites(site_ids)

    t_start = time.time()
    for site_id in site_ids:
        model_loop.initialize_site_with_gaps(site_id, num_gaps, maxtrees, prefix='CONUS')
    t_loop = time.time() - t_start

    print(f"  Time: {t_loop:.3f}s")
    print(f"  Agents: {model_loop._agent_factory.num_agents}")
    print()

    # Test BULK method (preallocate+populate)
    print("Testing BULK method (preallocate+populate)...")
    model_bulk = GAPModel()
    model_bulk.load_globals(prefix='CONUS')
    model_bulk.partition_sites(site_ids)

    t_start = time.time()
    site_agents = model_bulk.initialize_sites_bulk(site_ids, num_gaps, maxtrees, prefix='CONUS')
    t_bulk = time.time() - t_start

    print(f"  Time: {t_bulk:.3f}s")
    print(f"  Agents: {model_bulk._agent_factory.num_agents}")
    print()

    # Compare
    print("="*70)
    print("RESULTS")
    print("="*70)
    print(f"Loop method:  {t_loop:.3f}s")
    print(f"Bulk method:  {t_bulk:.3f}s")
    print(f"Speedup:      {t_loop/t_bulk:.2f}x")
    print()

    if model_loop._agent_factory.num_agents == model_bulk._agent_factory.num_agents:
        print("✅ Agent counts match!")
    else:
        print(f"❌ Agent count mismatch: {model_loop._agent_factory.num_agents} vs {model_bulk._agent_factory.num_agents}")

    print("="*70)


if __name__ == '__main__':
    test_initialization_performance()
