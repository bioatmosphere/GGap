"""
Analyze GPU occupancy tuning results and generate publication plots.

Usage:
    python analyze_occupancy.py ../results/exp0b_phase1_coarse_sweep.csv \\
        [--output-dir ../plots] \\
        [--gpu-monitor ../logs/gpu_monitor_12345.csv]
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def load_results(csv_path):
    """Load results from CSV."""
    df = pd.read_csv(csv_path)
    return df


def load_gpu_monitor(csv_path):
    """Load GPU monitoring data from CSV."""
    if not csv_path or not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    return df


def plot_occupancy_performance(df, output_dir):
    """Plot performance vs max_blocks_per_sm."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('GPU Occupancy Tuning Results', fontsize=16, fontweight='bold')

    # Plot 1: Time per tick vs occupancy
    ax = axes[0, 0]
    ax.errorbar(df['max_blocks_per_sm'], df['mean_tick_time'],
                yerr=df['std_tick_time'], marker='o', capsize=5, linewidth=2, markersize=8)
    ax.set_xlabel('max_blocks_per_sm', fontsize=12)
    ax.set_ylabel('Time per tick (s)', fontsize=12)
    ax.set_title('Performance vs GPU Occupancy', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Highlight optimal
    optimal_idx = df['mean_tick_time'].idxmin()
    optimal_sm = df.loc[optimal_idx, 'max_blocks_per_sm']
    optimal_time = df.loc[optimal_idx, 'mean_tick_time']
    ax.plot(optimal_sm, optimal_time, 'r*', markersize=20, label=f'Optimal: {optimal_sm}')
    ax.legend()

    # Plot 2: Throughput vs occupancy
    ax = axes[0, 1]
    ax.plot(df['max_blocks_per_sm'], df['throughput_tree_years_per_sec'] / 1e6,
            marker='o', linewidth=2, markersize=8)
    ax.set_xlabel('max_blocks_per_sm', fontsize=12)
    ax.set_ylabel('Throughput (M tree-years/s)', fontsize=12)
    ax.set_title('Throughput vs GPU Occupancy', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Plot 3: Memory usage (sanity check)
    ax = axes[1, 0]
    ax.plot(df['max_blocks_per_sm'], df['mem_gb'], marker='o', linewidth=2, markersize=8)
    ax.axhline(y=64, color='r', linestyle='--', linewidth=2, label='64 GB limit')
    ax.set_xlabel('max_blocks_per_sm', fontsize=12)
    ax.set_ylabel('GPU Memory (GB)', fontsize=12)
    ax.set_title('Memory Usage vs Occupancy', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Speedup vs baseline
    ax = axes[1, 1]
    baseline_time = df.loc[0, 'mean_tick_time']  # First entry (lowest SM)
    speedup = baseline_time / df['mean_tick_time']
    ax.plot(df['max_blocks_per_sm'], speedup, marker='o', linewidth=2, markersize=8)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('max_blocks_per_sm', fontsize=12)
    ax.set_ylabel('Speedup vs baseline', fontsize=12)
    ax.set_title(f'Speedup vs max_blocks_per_sm={df.loc[0, "max_blocks_per_sm"]}',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'occupancy_tuning.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot: {output_path}")
    plt.close()


def plot_gpu_utilization(gpu_df, output_dir):
    """Plot GPU utilization time series."""
    if gpu_df is None or len(gpu_df) == 0:
        print("⚠️  Skipping GPU utilization plot (no monitoring data)")
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('GPU Utilization During Occupancy Tuning', fontsize=16, fontweight='bold')

    # Plot 1: GPU utilization over time
    ax = axes[0]
    ax.plot(gpu_df['elapsed_sec'], gpu_df['gpu_util_pct'], linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('GPU Utilization (%)', fontsize=12)
    ax.set_title('GPU Utilization % Over Time', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    # Add mean line
    mean_util = gpu_df['gpu_util_pct'].mean()
    ax.axhline(y=mean_util, color='r', linestyle='--', linewidth=2,
              label=f'Mean: {mean_util:.1f}%')
    ax.legend()

    # Plot 2: Memory utilization over time
    ax = axes[1]
    ax.plot(gpu_df['elapsed_sec'], gpu_df['mem_pct'], linewidth=1.5, alpha=0.8, color='orange')
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Memory Utilization (%)', fontsize=12)
    ax.set_title('GPU Memory Utilization % Over Time', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add mean line
    mean_mem = gpu_df['mem_pct'].mean()
    ax.axhline(y=mean_mem, color='r', linestyle='--', linewidth=2,
              label=f'Mean: {mean_mem:.1f}%')
    ax.legend()

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'gpu_utilization_timeseries.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot: {output_path}")
    plt.close()


def plot_site_scaling(df, output_dir):
    """Plot time per tick vs number of sites (for Phase 3)."""
    if 'sites' not in df.columns or len(df['sites'].unique()) <= 1:
        print("⚠️  Skipping site scaling plot (need multiple site counts)")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by max_blocks_per_sm if multiple values
    for sm_value in df['max_blocks_per_sm'].unique():
        subset = df[df['max_blocks_per_sm'] == sm_value]
        ax.plot(subset['sites'], subset['mean_tick_time'], marker='o',
                linewidth=2, markersize=8, label=f'max_blocks_per_sm={sm_value}')

    ax.set_xlabel('Number of sites', fontsize=12)
    ax.set_ylabel('Time per tick (s)', fontsize=12)
    ax.set_title('Runtime Scaling with Site Count', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'site_count_scaling.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot: {output_path}")
    plt.close()


def print_summary(df):
    """Print summary statistics and recommendations."""
    print("\n" + "="*70)
    print("ANALYSIS SUMMARY")
    print("="*70)

    # Find optimal
    optimal_idx = df['mean_tick_time'].idxmin()
    optimal = df.loc[optimal_idx]
    baseline = df.loc[0]  # Assuming first entry is lowest SM

    print(f"\n📊 Performance Statistics:")
    print(f"{'max_blocks_per_sm':<20} {'Mean Time (s)':<15} {'Throughput (M/s)':<20} {'Speedup':<10}")
    print("-" * 70)

    for _, row in df.iterrows():
        speedup = baseline['mean_tick_time'] / row['mean_tick_time']
        throughput = row['throughput_tree_years_per_sec'] / 1e6
        marker = " 🏆" if row.name == optimal_idx else ""
        print(f"{row['max_blocks_per_sm']:<20} {row['mean_tick_time']:<15.4f} "
              f"{throughput:<20.2f} {speedup:<10.2f}{marker}")

    print("-" * 70)

    print(f"\n🏆 OPTIMAL CONFIGURATION:")
    print(f"  max_blocks_per_sm: {optimal['max_blocks_per_sm']}")
    print(f"  Mean tick time: {optimal['mean_tick_time']:.4f}s ± {optimal['std_tick_time']:.4f}s")
    print(f"  Throughput: {optimal['throughput_tree_years_per_sec']:,.0f} tree-years/sec")
    print(f"  GPU memory: {optimal['mem_gb']:.2f} GB / 64 GB ({100*optimal['mem_gb']/64:.1f}%)")
    print(f"  Speedup vs {baseline['max_blocks_per_sm']}: {baseline['mean_tick_time']/optimal['mean_tick_time']:.2f}×")

    print(f"\n⚙️  GPU Configuration:")
    print(f"  Max concurrent blocks: {optimal['max_concurrent_blocks']:,.0f}")
    print(f"  Total threads: {optimal['total_threads']:,.0f}")
    print(f"  Agents per thread: {optimal['agents_per_thread']:.1f}")

    print(f"\n💡 RECOMMENDATION:")
    print(f"  Use max_blocks_per_sm={optimal['max_blocks_per_sm']} for all subsequent experiments")
    print(f"  Expected {baseline['mean_tick_time']/optimal['mean_tick_time']:.1f}× speedup over default")

    # Check if site scaling data available
    if len(df['sites'].unique()) > 1:
        print(f"\n📈 Site Scaling:")
        for sm_value in sorted(df['max_blocks_per_sm'].unique()):
            subset = df[df['max_blocks_per_sm'] == sm_value].sort_values('sites')
            if len(subset) > 1:
                # Check if linear
                sites_ratio = subset['sites'].iloc[-1] / subset['sites'].iloc[0]
                time_ratio = subset['mean_tick_time'].iloc[-1] / subset['mean_tick_time'].iloc[0]
                scaling = "linear" if 0.9 < time_ratio / sites_ratio < 1.1 else "non-linear"
                print(f"  max_blocks_per_sm={sm_value}: {scaling} scaling")
                print(f"    {subset['sites'].min()}-{subset['sites'].max()} sites: "
                      f"{time_ratio:.2f}× time for {sites_ratio:.0f}× sites")

    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Analyze GPU occupancy tuning results')
    parser.add_argument('csv', type=str, help='Path to results CSV file')
    parser.add_argument('--output-dir', type=str, default='../plots',
                       help='Output directory for plots (default: ../plots)')
    parser.add_argument('--gpu-monitor', type=str, default=None,
                       help='Path to GPU monitoring CSV file (optional)')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load results
    print(f"Loading results from: {args.csv}")
    df = load_results(args.csv)
    print(f"Loaded {len(df)} results")

    # Load GPU monitoring data if available
    gpu_df = None
    if args.gpu_monitor:
        print(f"Loading GPU monitoring data from: {args.gpu_monitor}")
        gpu_df = load_gpu_monitor(args.gpu_monitor)
        if gpu_df is not None:
            print(f"Loaded {len(gpu_df)} GPU samples")
            print(f"  Mean GPU utilization: {gpu_df['gpu_util_pct'].mean():.1f}%")
            print(f"  Mean memory utilization: {gpu_df['mem_pct'].mean():.1f}%")

    # Generate plots
    print("\nGenerating plots...")
    plot_occupancy_performance(df, args.output_dir)
    plot_site_scaling(df, args.output_dir)
    if gpu_df is not None:
        plot_gpu_utilization(gpu_df, args.output_dir)

    # Print summary
    print_summary(df)

    print(f"\n✓ Analysis complete! Plots saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
