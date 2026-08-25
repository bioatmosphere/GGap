"""
Test AMD GPU detection in SAGESim.

Verifies that SAGESim correctly detects 110 CUs on AMD MI250X.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    print("ERROR: CuPy not available")
    sys.exit(1)

from gap.gap_model import GAPModel

print("="*70)
print("Testing AMD GPU Detection")
print("="*70)

# Test 1: Check what CuPy reports directly
print("\n1. CuPy device query:")
dev = cp.cuda.Device()
attrs = dev.attributes
cupy_sms = attrs['MultiProcessorCount']
print(f"   MultiProcessorCount: {cupy_sms}")

try:
    props = cp.cuda.runtime.getDeviceProperties(dev.id)
    gpu_name = props['name'].decode('utf-8') if isinstance(props['name'], bytes) else str(props['name'])
    print(f"   GPU name: {gpu_name}")
except Exception as e:
    print(f"   WARNING: Could not get device properties: {e}")

# Test 2: Create minimal GAPModel and check SM detection during kernel launch
print("\n2. SAGESim GPU detection:")
print("   Creating minimal GAPModel...")

model = GAPModel()
model.load_globals(prefix='CONUS')

# Get one site
all_site_ids = sorted(model._site_id_to_slot.keys())
site_id = all_site_ids[0]

print(f"   Initializing site {site_id}...")
model.partition_sites([site_id], strategy='round_robin')
model.initialize_site_with_gaps(site_id, num_gaps=10, maxtrees=100, prefix='CONUS')

# Set small SM value for quick test
model._max_blocks_per_sm = 4

print("   Running setup...")
model.setup()

print("\n   Running 1 tick (AMD detection happens at kernel launch)...")
model.simulate(ticks=1, sync_workers_every_n_ticks=1)

print("\n" + "="*70)
print("GPU detection test complete!")
print("="*70)
print("\nExpected output:")
print("  - CuPy MultiProcessorCount: 1 (incorrect)")
print("  - SAGESim detected: AMD MI210/MI250X")
print("  - Kernel config should show: X blocks (4/SM × 110 CUs)")
print("="*70)
