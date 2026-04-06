"""
Continuous GPU monitoring for occupancy tuning experiments.

Samples GPU utilization and memory usage every N seconds via rocm-smi.
Writes timestamped CSV for correlation with max_blocks_per_sm tests.

Usage:
    python monitor_gpu_usage.py --interval 5 --output gpu_monitor.csv &
    GPU_MONITOR_PID=$!

    # Run your experiment...

    kill $GPU_MONITOR_PID

The output CSV has columns:
    timestamp,elapsed_sec,gpu_util_pct,mem_used_gb,mem_total_gb
"""

import subprocess
import time
import json
import csv
import argparse
import sys
import signal
import os

# Global flag for graceful shutdown
shutdown_flag = False

def signal_handler(sig, frame):
    global shutdown_flag
    print("\n[GPU Monitor] Shutdown signal received, stopping...", flush=True)
    shutdown_flag = True

def query_gpu_stats():
    """Query GPU stats via rocm-smi JSON output."""
    try:
        result = subprocess.run(
            ['rocm-smi', '--showuse', '--showmemuse', '--json'],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode != 0:
            return None

        data = json.loads(result.stdout)

        # Extract stats from first GPU (card0)
        # MI250X node typically has multiple GCDs, we'll track card0
        if 'card0' not in data:
            return None

        card0 = data['card0']

        # GPU utilization percentage
        gpu_util_str = card0.get('GPU use (%)', '0')
        gpu_util = float(gpu_util_str) if gpu_util_str != 'N/A' else 0.0

        # Memory usage - parse from "X / Y" format if available
        # Otherwise use VRAM%
        mem_str = card0.get('GPU Memory Allocated (VRAM%)', '0')
        mem_pct = float(mem_str) if mem_str != 'N/A' else 0.0

        return {
            'gpu_util_pct': gpu_util,
            'mem_pct': mem_pct
        }

    except subprocess.TimeoutExpired:
        print("[GPU Monitor] rocm-smi timeout", flush=True)
        return None
    except json.JSONDecodeError as e:
        print(f"[GPU Monitor] JSON parse error: {e}", flush=True)
        return None
    except Exception as e:
        print(f"[GPU Monitor] Error querying GPU: {e}", flush=True)
        return None


def main():
    global shutdown_flag

    parser = argparse.ArgumentParser(description='Continuous GPU monitoring')
    parser.add_argument('--interval', type=int, default=5,
                       help='Sampling interval in seconds (default: 5)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output CSV file path')
    parser.add_argument('--verbose', action='store_true',
                       help='Print samples to stdout')

    args = parser.parse_args()

    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print(f"[GPU Monitor] Starting GPU monitoring", flush=True)
    print(f"[GPU Monitor] Interval: {args.interval}s", flush=True)
    print(f"[GPU Monitor] Output: {args.output}", flush=True)
    print(f"[GPU Monitor] PID: {os.getpid()}", flush=True)
    print(f"[GPU Monitor] Use 'kill {os.getpid()}' to stop", flush=True)

    # Open CSV file
    csv_file = open(args.output, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['timestamp', 'elapsed_sec', 'gpu_util_pct', 'mem_pct'])
    csv_file.flush()

    start_time = time.time()
    sample_count = 0

    try:
        while not shutdown_flag:
            current_time = time.time()
            elapsed = current_time - start_time

            stats = query_gpu_stats()

            if stats is not None:
                csv_writer.writerow([
                    time.strftime('%Y-%m-%d %H:%M:%S'),
                    f'{elapsed:.1f}',
                    f'{stats["gpu_util_pct"]:.1f}',
                    f'{stats["mem_pct"]:.1f}'
                ])
                csv_file.flush()
                sample_count += 1

                if args.verbose:
                    print(f"[GPU Monitor] Sample {sample_count}: "
                          f"GPU={stats['gpu_util_pct']:.1f}% "
                          f"Mem={stats['mem_pct']:.1f}%",
                          flush=True)

            # Sleep for interval
            time.sleep(args.interval)

    except KeyboardInterrupt:
        pass
    finally:
        csv_file.close()
        print(f"\n[GPU Monitor] Stopped after {sample_count} samples", flush=True)
        print(f"[GPU Monitor] Data written to: {args.output}", flush=True)


if __name__ == '__main__':
    main()
