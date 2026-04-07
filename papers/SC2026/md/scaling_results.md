# GGap Scaling Results — SC2026 Application Track

This document is the source of truth for all numbers in the scaling section of the SC2026 paper. Figures referenced live in `../figs/` (each plot is a standalone PDF for inclusion as a LaTeX subfigure). Regenerate by running:

```bash
# Each plot is its own script:
python papers/SC2026/src/weak_scaling_a_efficiency.py
python papers/SC2026/src/weak_scaling_a_throughput.py
# ... etc.
# And the markdown:
python papers/SC2026/src/generate_results_md.py
```

## 1. Executive Summary

### Headline Numbers

| Metric | Weak A (10 sites/GPU) | Weak B (100 sites/GPU) | Strong (2,048 sites) |
|---|---|---|---|
| Peak GPUs | 2048 | 128 | 512 |
| Peak total sites | 20,480 | 12,800 | 2,048 |
| Peak total agents | 10,250,260,480 | 1,282,572,800 | 205,211,648 |
| Peak throughput | 243.57 B upd/s | 27.59 B upd/s | 10.83 B upd/s |
| Mean tick at peak | 42.08 ms | 46.48 ms | 18.95 ms |
| Parallel efficiency at peak | 98.7% | 94.3% | 19.3% |
| Speedup at peak | — | — | 12.4× |

**Key claims for the paper:**

- Weak Scaling A demonstrates **99% parallel efficiency at 2048 GPUs** (10,250,260,480 agents) under a stress-test communication regime (37.5% cross-rank edges).
- Weak Scaling B reaches **1,282,572,800 agents** on 128 GPUs at 94% efficiency — runs at 256/512 GPUs are still queued and will extend the curve to multi-billion agents.
- Strong Scaling delivers a **12.4× speedup** from 8 → 512 GPUs (19% parallel efficiency at the largest scale). The efficiency drop is correlated with the per-GPU partition shrinking from 256 → 4 sites: at the smallest partitions each GPU kernel underutilizes the MI250X GCD (kernel launch overhead becomes a larger fraction of GPU time), and the cross-rank edge fraction climbs from 1.2% → 75%.
- Across all three experiments, GPU execution is ≥99% of per-tick wall time and MPI ghost-exchange consumes <0.005% of per-tick time, demonstrating that GPU-aware MPI on Slingshot-11 is effectively overlapped with computation.

## 2. Hardware & Software Stack

- **Platform:** OLCF Frontier (Oak Ridge National Laboratory)
- **Per node:** AMD EPYC 7A53 (64 cores) + 4× AMD MI250X GPUs (8 GCDs/node)
- **Interconnect:** HPE Slingshot-11, 4× 25 GB/s injection bandwidth/node
- **GPU-aware MPI:** Cray MPICH with `MPICH_GPU_SUPPORT_ENABLED=1`
- **Software:** PrgEnv-gnu 8.6.0, ROCm 6.4.1, Python 3.13, CuPy (ROCm), mpi4py
- **Per-rank affinity:** 7 CPU cores/task, `--gpu-bind=closest`
- **SAGESim config:** `SAGESIM_NUM_SMS=110` (matches MI250X GCD CU count)

## 3. Experiment Matrix

| Experiment | Sites/GPU | Gaps/site | Trees/gap | Grid H | Cross-rank % | GPU Range |
|---|---:|---:|---:|---:|---:|---|
| Weak A (comm-heavy)        |  10   | 500 | 1,000 |  5 | 37.5%       | 8–2,048 |
| Weak B (compute-heavy)     | 100   | 200 |   500 | 10 |  7.5%       | 8–512   |
| Strong (fixed 2,048 sites) | 256→4 | 200 |   500 |  4 | 1.2%→75.0%  | 8–512   |

## 4. Weak Scaling A Results

### Weak Scaling A — 10 sites/GPU

Source: `weak_scaling.csv`

| GPUs | Total Sites | Total Agents | Sim Time (s) | First Tick (s) | Steady (s) | Mean Tick (ms) | Throughput (B upd/s) | Efficiency (%) | MPI % |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 80 | 40,040,080 | 94.6 | 53.1 | 41.5 | 41.52 | 0.96 | 100.0 | 0.002 |
| 16 | 160 | 80,080,160 | 93.8 | 53.1 | 40.7 | 40.71 | 1.97 | 102.0 | 0.003 |
| 32 | 320 | 160,160,320 | 94.3 | 53.3 | 40.9 | 40.99 | 3.91 | 101.3 | 0.002 |
| 64 | 640 | 320,320,640 | 92.9 | 52.1 | 40.8 | 40.85 | 7.84 | 101.6 | 0.002 |
| 128 | 1,280 | 640,641,280 | 94.6 | 53.8 | 40.8 | 40.84 | 15.69 | 101.7 | 0.002 |
| 256 | 2,560 | 1,281,282,560 | 94.7 | 53.4 | 41.3 | 41.35 | 30.99 | 100.4 | 0.002 |
| 512 | 5,120 | 2,562,565,120 | 94.5 | 52.1 | 42.4 | 42.41 | 60.42 | 97.9 | 0.002 |
| 1024 | 10,240 | 5,125,130,240 | 98.4 | 56.5 | 41.9 | 41.91 | 122.28 | 99.1 | 0.002 |
| 2048 | 20,480 | 10,250,260,480 | 98.3 | 56.3 | 42.0 | 42.08 | 243.57 | 98.7 | 0.002 |

## 5. Weak Scaling B Results

> **Status:** 5/7 GPU configurations complete. Missing: 256, 512 GPUs (still queued on Frontier).

### Weak Scaling B — 100 sites/GPU

Source: `weak_scaling_b.csv`

| GPUs | Total Sites | Total Agents | Sim Time (s) | First Tick (s) | Steady (s) | Mean Tick (ms) | Throughput (B upd/s) | Efficiency (%) | MPI % |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 800 | 80,160,800 | 155.0 | 111.2 | 43.8 | 43.84 | 1.83 | 100.0 | 0.002 |
| 16 | 1,600 | 160,321,600 | 155.5 | 111.1 | 44.4 | 44.46 | 3.61 | 98.6 | 0.002 |
| 32 | 3,200 | 320,643,200 | 156.1 | 110.8 | 45.4 | 45.40 | 7.06 | 96.6 | 0.002 |
| 64 | 6,400 | 641,286,400 | 158.0 | 111.6 | 46.3 | 46.38 | 13.83 | 94.5 | 0.002 |
| 128 | 12,800 | 1,282,572,800 | 157.3 | 110.8 | 46.4 | 46.48 | 27.59 | 94.3 | 0.002 |
| 256 | — | — | — | — | — | — | — | — | — |
| 512 | — | — | — | — | — | — | — | — | — |

## 6. Strong Scaling Results

### Strong Scaling — 2,048 sites fixed

Source: `strong_scaling_b.csv`

| GPUs | Sites/GPU | Sim Time (s) | Mean Tick (ms) | Speedup | Efficiency (%) | Cross-rank % | MPI % |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 256 | 377.9 | 95.21 | 1.00× | 100.0 | 1.2 | 0.001 |
| 16 | 128 | 195.8 | 56.21 | 1.93× | 96.5 | 2.3 | 0.002 |
| 32 | 64 | 107.7 | 33.84 | 3.51× | 87.7 | 4.7 | 0.003 |
| 64 | 32 | 69.5 | 26.26 | 5.43× | 67.9 | 9.4 | 0.003 |
| 128 | 16 | 44.9 | 22.32 | 8.42× | 52.6 | 18.8 | 0.005 |
| 256 | 8 | 35.1 | 20.15 | 10.77× | 33.7 | 37.5 | 0.005 |
| 512 | 4 | 30.5 | 18.95 | 12.37× | 19.3 | 75.0 | 0.005 |

## 7. Per-Tick Time Decomposition

All values in microseconds (μs). `GPU Exec` = `mean_gpu_compute + mean_gpu_sync` (kernel launch overhead + actual GPU work). MPI columns are the GPU-aware MPI ghost-cell exchange. `Total` is the average per-tick wall time.

### Weak A — Per-tick Time Breakdown (μs)

| GPUs | GPU Exec | MPI Pack | MPI Exchg | MPI Unpack | Data Prep | Write Back | Kernel Args | Total |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 41,445.5 | 0.53 | 0.00 | 0.45 | 5.15 | 15.34 | 13.51 | 41,480.5 |
| 16 | 40,633.0 | 0.56 | 0.00 | 0.47 | 5.21 | 14.92 | 13.70 | 40,667.9 |
| 32 | 40,911.8 | 0.54 | 0.00 | 0.45 | 5.13 | 15.69 | 13.81 | 40,947.4 |
| 64 | 40,773.3 | 0.51 | 0.00 | 0.43 | 5.08 | 15.44 | 13.88 | 40,808.6 |
| 128 | 40,759.2 | 0.54 | 0.00 | 0.45 | 5.33 | 16.32 | 14.05 | 40,795.9 |
| 256 | 41,276.4 | 0.49 | 0.00 | 0.42 | 5.03 | 15.26 | 13.85 | 41,311.5 |
| 512 | 42,328.9 | 0.52 | 0.00 | 0.46 | 5.35 | 16.47 | 14.34 | 42,366.0 |
| 1024 | 41,835.1 | 0.53 | 0.00 | 0.44 | 5.35 | 16.32 | 14.34 | 41,872.1 |
| 2048 | 42,002.9 | 0.55 | 0.00 | 0.44 | 5.30 | 16.51 | 14.45 | 42,040.1 |

### Weak B — Per-tick Time Breakdown (μs)

| GPUs | GPU Exec | MPI Pack | MPI Exchg | MPI Unpack | Data Prep | Write Back | Kernel Args | Total |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 43,764.7 | 0.56 | 0.00 | 0.49 | 5.35 | 15.45 | 13.88 | 43,800.4 |
| 16 | 44,384.0 | 0.56 | 0.00 | 0.51 | 5.53 | 15.70 | 14.06 | 44,420.3 |
| 32 | 45,326.4 | 0.53 | 0.00 | 0.44 | 5.15 | 15.36 | 13.90 | 45,361.8 |
| 64 | 46,300.3 | 0.54 | 0.00 | 0.45 | 5.32 | 15.83 | 14.15 | 46,336.6 |
| 128 | 46,406.3 | 0.53 | 0.00 | 0.45 | 5.23 | 15.98 | 13.99 | 46,442.5 |

### Strong — Per-tick Time Breakdown (μs)

| GPUs | GPU Exec | MPI Pack | MPI Exchg | MPI Unpack | Data Prep | Write Back | Kernel Args | Total |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 95,130.9 | 0.59 | 0.00 | 0.49 | 5.59 | 15.78 | 14.07 | 95,167.4 |
| 16 | 56,129.5 | 0.52 | 0.00 | 0.45 | 5.35 | 16.01 | 14.11 | 56,166.0 |
| 32 | 33,757.7 | 0.55 | 0.00 | 0.44 | 5.38 | 16.41 | 13.92 | 33,794.4 |
| 64 | 26,184.2 | 0.47 | 0.00 | 0.43 | 4.87 | 15.00 | 13.65 | 26,218.6 |
| 128 | 22,241.7 | 0.56 | 0.00 | 0.46 | 5.35 | 16.66 | 14.41 | 22,279.1 |
| 256 | 20,076.7 | 0.50 | 0.00 | 0.43 | 4.99 | 15.10 | 13.58 | 20,111.3 |
| 512 | 18,867.1 | 0.56 | 0.00 | 0.48 | 5.73 | 16.91 | 14.85 | 18,905.7 |

## 8. Setup-Phase Breakdown (Weak B representative)

Setup phases are one-time costs amortized over the simulation run. Setup is dominated by Python-side site initialization and the first-tick buffer build (ghost topology discovery + GPU buffer allocation). At Weak B steady-state cost (~46 ms/tick) the setup fraction reaches the 5% threshold around ~91k ticks — short benchmarking runs are setup-dominated, long-horizon production runs are not. See `setup_amortization.{pdf,png}` for the curve.

### Weak B — End-to-End Phase Breakdown (s)

| GPUs | Model Create | Load Globals | Site Init | Connectivity | GPU Setup | First Tick | Steady State | Total Sim |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.032 | 1.59 | 90.9 | 0.45 | 10.47 | 111.2 | 43.8 | 155.0 |
| 16 | 0.028 | 1.61 | 92.2 | 0.59 | 10.51 | 111.1 | 44.4 | 155.5 |
| 32 | 0.028 | 1.60 | 91.9 | 1.71 | 10.49 | 110.8 | 45.4 | 156.1 |
| 64 | 0.036 | 1.60 | 92.9 | 0.56 | 10.46 | 111.6 | 46.3 | 158.0 |
| 128 | 0.023 | 1.60 | 95.9 | 0.38 | 10.49 | 110.8 | 46.4 | 157.3 |

## 9. Figures (standalone subfigures for LaTeX `subcaption`)

All figures are saved as paired PDF (paper) + PNG (preview) at 600 DPI in `../figs/` (sibling of this `md/` folder). Each panel is its own file so LaTeX can compose them via `\begin{subfigure}` / `\subfloat`.

**Weak Scaling A (10 sites/GPU, comm-heavy)** — `weak_scaling_a_*.{pdf,png}`
- `weak_scaling_a_efficiency` — parallel efficiency vs. GPUs
- `weak_scaling_a_throughput` — billion agent-updates/s vs. GPUs
- `weak_scaling_a_breakdown` — per-tick time decomposition

**Weak Scaling B (100 sites/GPU, compute-heavy)** — `weak_scaling_b_*.{pdf,png}`
- `weak_scaling_b_efficiency` — parallel efficiency vs. GPUs
- `weak_scaling_b_throughput` — billion agent-updates/s vs. GPUs
- `weak_scaling_b_breakdown` — per-tick time decomposition

**Strong Scaling (2,048 fixed sites)** — `strong_scaling_*.{pdf,png}`
- `strong_scaling_speedup` — speedup vs. GPUs (log-log) with ideal line
- `strong_scaling_efficiency` — efficiency overlaid with cross-rank fraction
- `strong_scaling_breakdown` — per-tick time decomposition

**Setup vs. Simulation** — `setup_*.{pdf,png}`
- `setup_phase_breakdown` — end-to-end phase breakdown for Weak B
- `setup_amortization` — setup fraction vs. simulation length

## 10. Narrative for the Paper

### 10.1 Weak scaling — communication stress test (Configuration A)

We evaluate weak scaling on OLCF Frontier across two complementary configurations that span the compute-vs-communication spectrum. **Configuration A** places 10 sites per GPU with full-fidelity per-site workload (500 gaps × 1,000 trees each). With grid height 5 and 1D column-slab partitioning, each rank exchanges 30 of its 80 site-edges across rank boundaries — a **37.5% cross-rank fraction** that stress-tests the communication subsystem. From 8 to 2048 GPUs (a 256× scale-up to 10,250,260,480 agents), parallel efficiency stays at **98.7%** (`weak_scaling_a_efficiency`), with mean tick time within +1.4% of the single-node baseline. The per-tick decomposition (`weak_scaling_a_breakdown`) confirms GPU execution dominates (99.8% of total tick time), while MPI ghost-exchange remains under 0.002% across the entire range, demonstrating that GPU-aware MPI on Slingshot-11 is effectively overlapped with computation.

### 10.2 Weak scaling — high site density (Configuration B)

**Configuration B** increases per-GPU workload by 10× (100 sites/GPU, 200 gaps, 500 trees) while reducing cross-rank density to 7.5%, exercising the high-site-density regime relevant to continental-scale forest simulations. At 128 GPUs the model handles 1,282,572,800 agents at 94.3% parallel efficiency (`weak_scaling_b_efficiency`). Runs at 256 and 512 GPUs are queued on Frontier; once those complete, the curve will extend to multi-billion-agent simulations covering the full southeastern-US continental forest extent.

### 10.3 Strong scaling — time-to-solution

Strong scaling on a fixed problem of 2,048 sites (~205,211,648 agents). Going from 8 → 512 GPUs achieves a **12.4× speedup** (19.3% parallel efficiency at the end of the curve, `strong_scaling_speedup`). The per-tick decomposition (`strong_scaling_breakdown`) shows mean tick time falls from 95.2 ms at 8 GPUs to 18.9 ms at 512 GPUs, with GPU execution remaining >99% of total per-tick wall time across the entire range — MPI ghost-exchange is <0.005% throughout. The efficiency drop is therefore not communication-bound. Rather, it reflects **GPU underutilization** at small per-GPU partitions: as sites/GPU shrinks from 256 to 4, per-agent compute cost rises ~13× as kernel launch overhead and intra-GPU parallelism limits become a larger fraction of the work. `strong_scaling_efficiency` overlays the cross-rank edge fraction (climbing from 1.2% to 75% as the partition shrinks) on the efficiency curve to visually correlate the partition-size effect, but the bottleneck mechanism is GPU kernel granularity, not network. At the most extreme partition (4 sites/GPU, single-column slab), each GPU still updates ~400K agents per tick in 18.95 ms, showing the runtime retains usable throughput even at minimum granularity.

### 10.4 Setup vs. simulation

Setup costs — model construction, global tensor load, agent creation, GPU kernel compilation, and first-tick buffer build — are one-time and amortized across the simulation run. `setup_phase_breakdown` decomposes the end-to-end wall time for Weak Scaling B across GPU counts. At 128 GPUs, setup totals roughly 219 s, dominated by **site initialization** (96 s, Python agent-object construction on the host) and the **first-tick buffer build** (111 s, ghost-cell topology discovery and GPU buffer allocation). At a steady-state cost of 46 ms/tick, `setup_amortization` shows the setup fraction crosses the 5% threshold around **89,613 ticks**. Short benchmarking runs (≤1,000 ticks) are therefore dominated by setup, but long-horizon production simulations — e.g., continental forest projections at decadal-to-centennial resolution — easily exceed this threshold and make setup negligible. Reducing site-init and first-tick costs is a clear engineering target for future work.

## 11. Methodology Notes

**Parallel efficiency definitions**:

- *Weak scaling*: `eff(N) = T_baseline_tick / T_N_tick × 100`. Ideal = 100%.
- *Strong scaling*: `speedup(N) = T_baseline_sim / T_N_sim`; `eff(N) = speedup(N) / (N / N_baseline) × 100`. Ideal = 100%.

**Throughput**: `total_agents / mean_tick_time` in agent-updates/second.

**Total agents**: `total_sites × (1 + num_gaps × (1 + maxtrees))` (1 site agent + N gap agents + N×M tree agents per site).

**Cross-rank fraction**: `(grid_height × 6) / (sites_per_gpu × 8)` — derived from the 8-neighbor Moore connectivity and 1D column-slab partition.

**Duplicates**: where the CSV contains repeated runs for the same GPU count, all numeric columns are averaged before plotting (same convention as the existing `plot_weak_scaling_breakdown.py` script).

**Mean tick window**: SAGESim's `verbose_timing` separates tick 1 (buffer build, ghost topology discovery, communication map build) from ticks 2..N. `mean_tick_time` is the average of ticks 2..N (steady state) and is the figure used in all efficiency and throughput calculations.

**`gpu_execution`**: equals `mean_gpu_compute + mean_gpu_sync`. The GPU kernel launches asynchronously, so `gpu_compute` only captures CPU-side launch overhead (~0.5 ms) — actual kernel execution completes during `gpu_sync` when the CPU calls `stream.synchronize()`.

## 12. Data Provenance

| Source CSV | Used by | GPU counts present | Missing |
|---|---|---|---|
| `weak_scaling.csv` | weak_scaling_a_*, §4 | [8, 16, 32, 64, 128, 256, 512, 1024, 2048] | none |
| `weak_scaling_b.csv` | weak_scaling_b_*, setup_*, §5, §8 | [8, 16, 32, 64, 128] | [256, 512] |
| `strong_scaling_b.csv` | strong_scaling_*, §6 | [8, 16, 32, 64, 128, 256, 512] | none |

**To refresh:** drop new rows into the relevant CSV and re-run the affected plot scripts in `papers/SC2026/src/`, then re-run `python papers/SC2026/src/generate_results_md.py` to regenerate this document.
