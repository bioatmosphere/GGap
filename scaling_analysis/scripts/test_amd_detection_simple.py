"""Simple test of AMD GPU detection logic."""

try:
    import cupy as cp
except ImportError:
    print("ERROR: CuPy not available")
    exit(1)

import os

print("="*70)
print("AMD GPU Detection Test")
print("="*70)

dev = cp.cuda.Device()
attrs = dev.attributes
num_sms = attrs['MultiProcessorCount']

print(f"\n1. CuPy reports MultiProcessorCount: {num_sms}")

# Check for manual override
if 'SAGESIM_NUM_SMS' in os.environ:
    num_sms = int(os.environ['SAGESIM_NUM_SMS'])
    print(f"2. Using SAGESIM_NUM_SMS override: {num_sms}")
elif num_sms == 1:
    # CuPy reports 1 SM on AMD - detect GPU architecture
    try:
        props = cp.cuda.runtime.getDeviceProperties(dev.id)
        gpu_name = props['name'].decode('utf-8') if isinstance(props['name'], bytes) else str(props['name'])
        print(f"2. Detected GPU name: {gpu_name}")

        # AMD MI250X and MI210 have 110 CUs per GCD (gfx90a architecture)
        if 'gfx90a' in gpu_name.lower() or 'mi250x' in gpu_name.lower() or 'mi210' in gpu_name.lower():
            num_sms = 110
            print(f"3. AMD MI250X/MI210 detected, using {num_sms} CUs")
        else:
            print(f"3. WARNING: Unknown AMD GPU, using reported value: {num_sms}")
    except Exception as e:
        print(f"2. WARNING: Could not detect GPU type: {e}")
        print(f"3. Using reported MultiProcessorCount: {num_sms}")
else:
    print(f"2. NVIDIA GPU detected (MultiProcessorCount = {num_sms})")

print(f"\nFinal result: num_sms = {num_sms}")
print("="*70)
