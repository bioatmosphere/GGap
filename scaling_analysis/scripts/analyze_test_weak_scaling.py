"""
Analyze Weak Scaling Test Results

Reads CSV files from single-node weak scaling tests and generates:
- Timing breakdown plots
- Parallel efficiency curves
- MPI overhead analysis
- Summary tables

Usage:
    python analyze_test_weak_scaling.py \\
        --results-dir ../results \\
        --plots-dir ../plots \\
        --format png
"""

import os
import sys
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# Use non-interactive backend for cluster
mpl.use('Agg')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze weak scaling test results")
    parser.add_argument("--results-dir", type=str, default="../results",
                       help="Directory containing CSV results")
    parser.add_argument("--plots-dir", type=str, default="../plots",
                       help="Output directory for plots")
    parser.add_argument("--format", type=str, default="png",
                       choices=["png", "pdf", "svg"],
                       help="Plot format")
    parser.add_argument("--dpi", type=int, default=150,
                       help="Plot DPI")
    return parser.parse_args()


def load_results(results_dir):
    """Load all CSV files from results directory."""
    pattern = os.path.join(results_dir, "test_weak_scaling_*gpus.csv")
    csv_files = sorted(glob.glob(pattern))

    if not csv_files:
        print(f"ERROR: No CSV files found in {results_dir}", flush=True)
        sys.exit(1)

    print(f"Found {len(csv_files)} result files:", flush=True)
    for f in csv_files:
        print(f"  {os.path.basename(f)}", flush=True)

    # Read and combine all CSVs
    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    # Sort by num_gpus
    combined = combined.sort_values('num_gpus')

    return combined


def plot_time_per_tick(df, output_path, fmt='png', dpi=150):
    """Plot time per tick vs GPUs (should be flat for ideal weak scaling)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    gpus = df['num_gpus'].values
    mean_tick = df['mean_tick_time'].values
    std_tick = df['std_tick_time'].values

    ax.errorbar(gpus, mean_tick, yerr=std_tick, marker='o', linewidth=2,
                markersize=8, capsize=5, label='Measured')

    # Ideal line (constant time)
    if len(mean_tick) > 0:
        ideal = np.full_like(gpus, mean_tick[0], dtype=float)
        ax.plot(gpus, ideal, 'k--', linewidth=1.5, label='Ideal (constant)')

    ax.set_xlabel('Number of GPUs', fontsize=12)
    ax.set_ylabel('Time per Tick (s)', fontsize=12)
    ax.set_title('Weak Scaling: Time per Tick vs GPUs', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_xticks(gpus)

    plt.tight_layout()
    plt.savefig(output_path, format=fmt, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"  Created: {output_path}", flush=True)


def plot_parallel_efficiency(df, output_path, fmt='png', dpi=150):
    """Plot parallel efficiency vs GPUs."""
    fig, ax = plt.subplots(figsize=(10, 6))

    gpus = df['num_gpus'].values
    mean_tick = df['mean_tick_time'].values

    # Calculate efficiency: E(N) = T(1) / T(N)
    if len(mean_tick) > 0:
        baseline = mean_tick[0]
        efficiency = (baseline / mean_tick) * 100  # Percentage

        ax.plot(gpus, efficiency, marker='o', linewidth=2, markersize=8,
                label='Measured', color='tab:blue')

        # Ideal line (100%)
        ax.plot(gpus, np.full_like(gpus, 100.0, dtype=float), 'k--',
                linewidth=1.5, label='Ideal (100%)')

        ax.set_xlabel('Number of GPUs', fontsize=12)
        ax.set_ylabel('Parallel Efficiency (%)', fontsize=12)
        ax.set_title('Weak Scaling Efficiency', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 105])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        ax.set_xticks(gpus)

        # Add text annotations
        for g, e in zip(gpus, efficiency):
            ax.text(g, e + 2, f'{e:.1f}%', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, format=fmt, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"  Created: {output_path}", flush=True)


def plot_timing_breakdown(df, output_path, fmt='png', dpi=150):
    """Plot stacked bar chart of timing breakdown."""
    fig, ax = plt.subplots(figsize=(12, 6))

    gpus = df['num_gpus'].values

    # Components (steady state)
    gpu_compute = df['mean_gpu_compute'].values
    data_prep = df['mean_data_prep'].values
    mpi_total = df['mean_mpi_total'].values
    other = df['mean_tick_time'].values - (gpu_compute + data_prep + mpi_total)

    # Stack bars
    width = 0.6
    ax.bar(gpus, gpu_compute, width, label='GPU Compute', color='tab:blue')
    ax.bar(gpus, data_prep, width, bottom=gpu_compute,
           label='Data Prep', color='tab:orange')
    ax.bar(gpus, mpi_total, width, bottom=gpu_compute + data_prep,
           label='MPI Communication', color='tab:green')
    ax.bar(gpus, other, width, bottom=gpu_compute + data_prep + mpi_total,
           label='Other', color='tab:gray')

    ax.set_xlabel('Number of GPUs', fontsize=12)
    ax.set_ylabel('Time per Tick (s)', fontsize=12)
    ax.set_title('Weak Scaling: Timing Breakdown (Steady State)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(gpus)

    plt.tight_layout()
    plt.savefig(output_path, format=fmt, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"  Created: {output_path}", flush=True)


def plot_mpi_overhead(df, output_path, fmt='png', dpi=150):
    """Plot MPI communication overhead vs GPUs."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Filter multi-GPU runs
    df_multi = df[df['num_gpus'] > 1].copy()

    if len(df_multi) > 0:
        gpus = df_multi['num_gpus'].values
        mpi_fraction = df_multi['mpi_fraction'].values * 100  # Percentage

        ax.plot(gpus, mpi_fraction, marker='o', linewidth=2, markersize=8,
                color='tab:red')

        ax.set_xlabel('Number of GPUs', fontsize=12)
        ax.set_ylabel('MPI Overhead (%)', fontsize=12)
        ax.set_title('MPI Communication Overhead', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(gpus)

        # Add text annotations
        for g, frac in zip(gpus, mpi_fraction):
            ax.text(g, frac + 1, f'{frac:.1f}%', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, format=fmt, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"  Created: {output_path}", flush=True)


def plot_first_vs_steady(df, output_path, fmt='png', dpi=150):
    """Plot first tick vs steady state comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))

    gpus = df['num_gpus'].values
    first_tick = df['first_tick_total'].values
    mean_tick = df['mean_tick_time'].values

    x = np.arange(len(gpus))
    width = 0.35

    ax.bar(x - width/2, first_tick, width, label='First Tick', color='tab:blue')
    ax.bar(x + width/2, mean_tick, width, label='Steady State', color='tab:orange')

    ax.set_xlabel('Number of GPUs', fontsize=12)
    ax.set_ylabel('Time per Tick (s)', fontsize=12)
    ax.set_title('First Tick vs Steady State', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(gpus)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, format=fmt, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"  Created: {output_path}", flush=True)


def print_summary_table(df):
    """Print summary table to console."""
    print("\n" + "=" * 80, flush=True)
    print("WEAK SCALING SUMMARY", flush=True)
    print("=" * 80, flush=True)

    print(f"\n{'GPUs':<6} {'Sites':<8} {'Agents':<12} {'Tick (s)':<12} {'Efficiency':<12} {'MPI %':<10}", flush=True)
    print("-" * 80, flush=True)

    for _, row in df.iterrows():
        gpus = int(row['num_gpus'])
        sites = int(row['total_sites'])
        agents = int(row['total_agents'])
        tick_time = row['mean_tick_time']

        # Calculate efficiency
        baseline = df[df['num_gpus'] == df['num_gpus'].min()]['mean_tick_time'].values[0]
        efficiency = (baseline / tick_time) * 100 if tick_time > 0 else 0.0

        mpi_pct = row['mpi_fraction'] * 100 if gpus > 1 else 0.0

        print(f"{gpus:<6} {sites:<8} {agents:<12} {tick_time:<12.4f} {efficiency:<12.1f} {mpi_pct:<10.1f}", flush=True)

    print("=" * 80, flush=True)
    print(f"\nSetup Time Breakdown (1 GPU):", flush=True)
    first_row = df.iloc[0]
    print(f"  Model creation:      {first_row['model_creation_time']:8.4f}s", flush=True)
    print(f"  Site init:           {first_row['site_init_time']:8.4f}s", flush=True)
    print(f"  Connectivity:        {first_row['connectivity_time']:8.4f}s", flush=True)
    print(f"  Partitioning:        {first_row['partitioning_time']:8.4f}s", flush=True)
    print(f"  GPU setup:           {first_row['gpu_setup_time']:8.4f}s", flush=True)
    print(f"  Total:               {first_row['total_setup_time']:8.4f}s", flush=True)

    print(f"\nFirst Tick Breakdown (1 GPU):", flush=True)
    print(f"  Total:               {first_row['first_tick_total']:8.4f}s", flush=True)
    print(f"  Buffer build:        {first_row.get('first_tick_buffer_build', 0.0):8.4f}s", flush=True)
    print(f"  GPU compute:         {first_row.get('first_tick_gpu_compute', 0.0):8.4f}s", flush=True)
    print(f"  Data prep:           {first_row.get('first_tick_data_prep', 0.0):8.4f}s", flush=True)

    print(f"\nThroughput:", flush=True)
    for _, row in df.iterrows():
        print(f"  {int(row['num_gpus'])} GPU(s): {row['throughput_tree_years_per_sec']:.2e} tree-years/sec", flush=True)

    print("=" * 80, flush=True)


def main():
    """Main entry point."""
    args = parse_args()

    print("=" * 60, flush=True)
    print("Weak Scaling Analysis", flush=True)
    print("=" * 60, flush=True)

    # Load results
    df = load_results(args.results_dir)

    print(f"\nLoaded {len(df)} results", flush=True)

    # Print summary table
    print_summary_table(df)

    # Create plots directory
    os.makedirs(args.plots_dir, exist_ok=True)

    print(f"\nGenerating plots...", flush=True)

    # Generate plots
    plot_time_per_tick(df,
                      os.path.join(args.plots_dir, f'time_per_tick.{args.format}'),
                      fmt=args.format, dpi=args.dpi)

    plot_parallel_efficiency(df,
                            os.path.join(args.plots_dir, f'parallel_efficiency.{args.format}'),
                            fmt=args.format, dpi=args.dpi)

    plot_timing_breakdown(df,
                         os.path.join(args.plots_dir, f'timing_breakdown.{args.format}'),
                         fmt=args.format, dpi=args.dpi)

    plot_mpi_overhead(df,
                     os.path.join(args.plots_dir, f'mpi_overhead.{args.format}'),
                     fmt=args.format, dpi=args.dpi)

    plot_first_vs_steady(df,
                        os.path.join(args.plots_dir, f'first_vs_steady.{args.format}'),
                        fmt=args.format, dpi=args.dpi)

    print(f"\nAll plots saved to: {args.plots_dir}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
