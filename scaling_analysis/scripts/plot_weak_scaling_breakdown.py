#!/usr/bin/env python3
"""Plot timing breakdown for weak scaling results."""

import csv
import matplotlib.pyplot as plt
import numpy as np

# Read CSV manually
data = {}
with open("../results/weak_scaling.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        g = int(row["num_gpus"])
        if g not in data:
            data[g] = []
        data[g].append({k: float(v) for k, v in row.items()})

# Average duplicate runs
gpus_sorted = sorted(data.keys())
avg = {}
for g in gpus_sorted:
    runs = data[g]
    avg[g] = {}
    for key in runs[0]:
        avg[g][key] = sum(r[key] for r in runs) / len(runs)

gpus = np.array(gpus_sorted)
gpu_labels = [str(g) for g in gpus]

site_init = np.array([avg[g]["site_init_time"] for g in gpus])
gpu_setup = np.array([avg[g]["gpu_setup_time"] for g in gpus])
connectivity = np.array([avg[g]["connectivity_time"] for g in gpus])
first_tick = np.array([avg[g]["first_tick_time"] for g in gpus])
steady_state = np.array([avg[g]["steady_state_time"] for g in gpus])
load_globals = np.array([avg[g]["load_globals_time"] for g in gpus])
sim_time = np.array([avg[g]["simulation_time"] for g in gpus])
mean_tick = np.array([avg[g]["mean_tick_time"] for g in gpus])

# --- Figure: 3 panels ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Panel 1: Stacked bar total wall time
ax = axes[0]
bottom = np.zeros(len(gpus))
colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0', '#607D8B']
labels = ['Site Init', 'GPU Setup', 'First Tick', 'Steady State (ticks 2-1000)', 'Load Globals', 'Connectivity']
data_arrays = [site_init, gpu_setup, first_tick, steady_state, load_globals, connectivity]

for d, c, l in zip(data_arrays, colors, labels):
    ax.bar(gpu_labels, d, bottom=bottom, color=c, label=l, edgecolor='white', linewidth=0.5)
    bottom += d

ax.set_xlabel("Number of GPUs", fontsize=12)
ax.set_ylabel("Time (seconds)", fontsize=12)
ax.set_title("Total Wall Time Breakdown", fontsize=14)
ax.legend(loc='upper left', fontsize=9)
ax.grid(axis='y', alpha=0.3)

# Panel 2: Per-phase line plot
ax = axes[1]
ax.plot(gpus, site_init, 'o-', color='#2196F3', label='Site Init', linewidth=2, markersize=6)
ax.plot(gpus, gpu_setup, 's-', color='#4CAF50', label='GPU Setup', linewidth=2, markersize=6)
ax.plot(gpus, first_tick, '^-', color='#FF9800', label='First Tick', linewidth=2, markersize=6)
ax.plot(gpus, steady_state, 'D-', color='#F44336', label='Steady State', linewidth=2, markersize=6)
ax.plot(gpus, connectivity, 'v-', color='#607D8B', label='Connectivity', linewidth=2, markersize=6)

ax.set_xlabel("Number of GPUs", fontsize=12)
ax.set_ylabel("Time (seconds)", fontsize=12)
ax.set_title("Per-Phase Timing", fontsize=14)
ax.set_xscale('log', base=2)
ax.set_xticks(gpus)
ax.set_xticklabels(gpu_labels)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 3: Parallel efficiency + mean tick
ax = axes[2]
baseline_sim = sim_time[0]  # 8 GPUs
efficiency = (baseline_sim / sim_time) * 100

ax.plot(gpus, efficiency, 'o-', color='#2196F3', label='Parallel Efficiency', linewidth=2, markersize=8)
ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='Ideal (100%)')
ax.axhline(y=95, color='red', linestyle=':', alpha=0.5, label='95% threshold')

ax.set_xlabel("Number of GPUs", fontsize=12)
ax.set_ylabel("Parallel Efficiency (%)", fontsize=12)
ax.set_title("Weak Scaling Efficiency", fontsize=14)
ax.set_xscale('log', base=2)
ax.set_xticks(gpus)
ax.set_xticklabels(gpu_labels)
ax.set_ylim([85, 105])
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

ax2 = ax.twinx()
ax2.plot(gpus, mean_tick * 1000, 's--', color='#FF9800', label='Mean Tick (ms)', linewidth=1.5, markersize=6, alpha=0.7)
ax2.set_ylabel("Mean Tick Time (ms)", fontsize=12, color='#FF9800')
ax2.tick_params(axis='y', labelcolor='#FF9800')
ax2.legend(loc='lower left', fontsize=10)

plt.tight_layout()
plt.savefig("../results/weak_scaling_breakdown.png", dpi=150, bbox_inches='tight')
print("Saved: ../results/weak_scaling_breakdown.png")

# Print summary
print("\n=== Weak Scaling Summary ===")
print(f"{'GPUs':>6} {'Sites':>7} {'Init(s)':>8} {'GPU(s)':>7} {'1stTick':>8} {'Steady':>8} {'SimTotal':>9} {'Tick(ms)':>9} {'Eff%':>6}")
print("-" * 75)
for i, g in enumerate(gpus):
    eff = baseline_sim / sim_time[i] * 100
    total_sites = int(avg[g]["total_sites"])
    print(f"{g:>6} {total_sites:>7} {site_init[i]:>8.1f} {gpu_setup[i]:>7.1f} {first_tick[i]:>8.1f} {steady_state[i]:>8.1f} {sim_time[i]:>9.1f} {mean_tick[i]*1000:>9.2f} {eff:>5.1f}%")
