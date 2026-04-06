"""
Verify AMD MI250X GPU hardware specifications on Frontier compute node.

This script queries GPU properties via multiple methods to confirm:
1. GPU model (MI250X vs MI210)
2. Number of Compute Units (should be 110)
3. Memory capacity
4. What CuPy reports vs actual hardware

Usage:
    srun -N1 -n1 --gpus-per-node=1 python verify_gpu_hardware.py
"""

import sys
import subprocess
import os

print("="*70)
print("AMD GPU Hardware Verification")
print("="*70)
print()

# Method 1: rocm-smi
print("Method 1: rocm-smi")
print("-" * 70)
try:
    result = subprocess.run(['rocm-smi', '--showproductname'],
                          capture_output=True, text=True, timeout=10)
    if result.returncode == 0:
        print(result.stdout)
    else:
        print(f"Error: {result.stderr}")
except Exception as e:
    print(f"rocm-smi not available: {e}")
print()

# Method 2: rocminfo
print("Method 2: rocminfo (Compute Units)")
print("-" * 70)
try:
    result = subprocess.run(['rocminfo'], capture_output=True, text=True, timeout=10)
    if result.returncode == 0:
        # Parse for compute units
        lines = result.stdout.split('\n')
        in_gpu_section = False
        for line in lines:
            if 'Device Type:' in line and 'GPU' in line:
                in_gpu_section = True
            if in_gpu_section:
                if 'Name:' in line or 'Compute Unit:' in line or 'Max Clock Freq' in line:
                    print(line.strip())
                if 'Compute Unit:' in line:
                    in_gpu_section = False
    else:
        print(f"Error: {result.stderr}")
except Exception as e:
    print(f"rocminfo not available: {e}")
print()

# Method 3: CuPy device query
print("Method 3: CuPy Device Properties")
print("-" * 70)
try:
    import cupy as cp

    dev = cp.cuda.Device()
    attrs = dev.attributes

    print(f"CuPy MultiProcessorCount: {attrs['MultiProcessorCount']}")

    # Try to get device properties
    try:
        props = cp.cuda.runtime.getDeviceProperties(dev.id)
        gpu_name = props['name'].decode('utf-8') if isinstance(props['name'], bytes) else str(props['name'])
        print(f"GPU Name: {gpu_name}")
        print(f"Total Memory: {props['totalGlobalMem'] / 1e9:.2f} GB")
        print(f"Max Threads Per Block: {props['maxThreadsPerBlock']}")
        print(f"Max Grid Size: {props['maxGridSize']}")
    except Exception as e:
        print(f"Could not get detailed properties: {e}")

except ImportError:
    print("CuPy not available")
except Exception as e:
    print(f"CuPy error: {e}")
print()

# Method 4: Check environment
print("Method 4: Environment Variables")
print("-" * 70)
gpu_vars = ['ROCR_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES']
for var in gpu_vars:
    val = os.environ.get(var, 'not set')
    print(f"{var}: {val}")
print()

print("="*70)
print("Expected for AMD MI250X on Frontier:")
print("  - GPU Name: AMD Instinct MI250X (or gfx90a)")
print("  - Compute Units: 110 per GCD")
print("  - Total Memory: ~64 GB per GCD")
print("  - CuPy MultiProcessorCount: 1 (BUG - should be 110)")
print("="*70)
