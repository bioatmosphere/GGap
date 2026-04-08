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

| Metric | **Weak A (10 sites/GPU)** | **Strong (2,048 sites)** | Weak B (100 sites/GPU, §S.1) |
|---|---|---|---|
| Peak GPUs | 2048 | 512 | 128 |
| Peak total sites | 20,480 | 2,048 | 12,800 |
| Peak total agents | 10,250,260,480 | 205,211,648 | 1,282,572,800 |
| Peak total throughput | 243.57 B upd/s | 10.83 B upd/s | 27.59 B upd/s |
| Sustained per-GPU throughput | 119 M upd/s/GPU | — | 216 M upd/s/GPU |
| Sustained steady-state per-tick (avg) | ~41 ms | — | ~45 ms |
| Mean tick at peak GPU count | 42.08 ms | 18.95 ms | 46.48 ms |
| Parallel efficiency at peak | 96.2% | 19.3% | 98.6% |
| Speedup at peak | — | 12.4× | — |

**Key claims for the paper:**

- Weak Scaling A (main weak-scaling experiment, headline) demonstrates **96% parallel efficiency at 2048 GPUs** (10,250,260,480 agents = 20,480 sites at 10 sites/GPU under full UVAFME per-site fidelity), sustaining ~119 M agent-updates/sec/GPU under a stress-test communication regime (37.5% cross-rank edges).
- Strong Scaling delivers a **12.4× speedup** from 8 → 512 GPUs (19% parallel efficiency at the largest scale). The efficiency drop is correlated with the per-GPU partition shrinking from 256 → 4 sites: at the smallest partitions each GPU kernel underutilizes the MI250X GCD (kernel launch overhead becomes a larger fraction of GPU time), and the cross-rank edge fraction climbs from 1.2% → 75%.
- Weak Scaling B (supplementary, §S.1) cross-validates Weak A at a different point in the design space (100 sites/GPU, 200 gaps × 500 trees per site — matches the Strong scaling experiment's per-site fidelity) and reaches **1,282,572,800 agents on 128 GPUs at 99% efficiency**, sustaining ~216 M agent-updates/sec/GPU. *(Runs above 128 GPUs were queued on Frontier but did not complete before the SC2026 deadline.)*
- Across all experiments, GPU execution is ≥99% of per-tick wall time and MPI ghost-exchange consumes <0.005% of per-tick time, demonstrating that GPU-aware MPI on Slingshot-11 is effectively overlapped with computation.

## 2. Hardware & Software Stack

- **Platform:** OLCF Frontier (Oak Ridge National Laboratory)
- **Per node:** AMD EPYC 7A53 (64 cores) + 4× AMD MI250X GPUs (8 GCDs/node)
- **Interconnect:** HPE Slingshot-11, 4× 25 GB/s injection bandwidth/node
- **GPU-aware MPI:** Cray MPICH with `MPICH_GPU_SUPPORT_ENABLED=1`
- **Software:** PrgEnv-gnu 8.6.0, ROCm 6.4.1, Python 3.13, CuPy (ROCm), mpi4py
- **Per-rank affinity:** 7 CPU cores/task, `--gpu-bind=closest`
- **SAGESim config:** `SAGESIM_NUM_SMS=110` (matches MI250X GCD CU count)

## 3. Experiment Matrix

| Experiment | Sites/GPU | Gaps/site | Trees/gap | Grid H | Cross-rank % | GPU Range (complete) | Paper section |
|---|---:|---:|---:|---:|---:|---|---|
| Weak A (comm-heavy)        |  10   | 500 | 1,000 |  5 | 37.5%       | 8–2,048 (9/9) | **Main (§4, §9.1)** |
| Strong (fixed 2,048 sites) | 256→4 | 200 |   500 |  4 | 1.2%→75.0%  | 8–512 (7/7)   | **Main (§5, §9.2)** |
| Weak B (compute-heavy)     | 100   | 200 |   500 | 10 |  7.5%       | 8–128 (5/9, runs above 128 GPUs did not complete before deadline) | Supplementary (§S.1) |

## 4. Weak Scaling A — Main Result

Full UVAFME per-site fidelity (500 gaps × 1,000 trees), 10 sites/GPU, 37.5% cross-rank density. This is the headline weak-scaling configuration for the main paper, with all 9 rank counts (8 → 2,048 GPUs) complete. *(A complementary high-site-density configuration — Weak B — is reported in §S.1 as supplementary cross-validation; its runs above 128 GPUs were queued on Frontier but did not complete before the SC2026 deadline.)*

### Weak Scaling A — 10 sites/GPU

Source: `weak_scaling.csv`

| GPUs | Total Sites | Total Agents | Sim Time (s) | First Tick (s) | Steady (s) | Mean Tick (ms) | Throughput (B upd/s) | Efficiency (%) | MPI % |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 80 | 40,040,080 | 94.6 | 53.1 | 41.5 | 41.52 | 0.96 | 100.0 | 0.002 |
| 16 | 160 | 80,080,160 | 93.8 | 53.1 | 40.7 | 40.71 | 1.97 | 100.9 | 0.003 |
| 32 | 320 | 160,160,320 | 94.3 | 53.3 | 40.9 | 40.99 | 3.91 | 100.3 | 0.002 |
| 64 | 640 | 320,320,640 | 92.9 | 52.1 | 40.8 | 40.85 | 7.84 | 101.8 | 0.002 |
| 128 | 1,280 | 640,641,280 | 94.6 | 53.8 | 40.8 | 40.84 | 15.69 | 100.0 | 0.002 |
| 256 | 2,560 | 1,281,282,560 | 94.7 | 53.4 | 41.3 | 41.35 | 30.99 | 99.9 | 0.002 |
| 512 | 5,120 | 2,562,565,120 | 94.5 | 52.1 | 42.4 | 42.41 | 60.42 | 100.1 | 0.002 |
| 1024 | 10,240 | 5,125,130,240 | 98.4 | 56.5 | 41.9 | 41.91 | 122.28 | 96.1 | 0.002 |
| 2048 | 20,480 | 10,250,260,480 | 98.3 | 56.3 | 42.0 | 42.08 | 243.57 | 96.2 | 0.002 |

## 5. Strong Scaling Results

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

## 6. Per-Tick Time Decomposition

All values in microseconds (μs). `GPU Exec` = `mean_gpu_compute + mean_gpu_sync` (kernel launch overhead + actual GPU work). MPI columns are the GPU-aware MPI ghost-cell exchange. `Total` is the average per-tick wall time. Decomposition tables for both main (Weak A, Strong) and supplementary (Weak B) experiments are co-located here for reference.

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

### Weak B (supplementary) — Per-tick Time Breakdown (μs)

| GPUs | GPU Exec | MPI Pack | MPI Exchg | MPI Unpack | Data Prep | Write Back | Kernel Args | Total |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 43,764.7 | 0.56 | 0.00 | 0.49 | 5.35 | 15.45 | 13.88 | 43,800.4 |
| 16 | 44,384.0 | 0.56 | 0.00 | 0.51 | 5.53 | 15.70 | 14.06 | 44,420.3 |
| 32 | 45,326.4 | 0.53 | 0.00 | 0.44 | 5.15 | 15.36 | 13.90 | 45,361.8 |
| 64 | 46,300.3 | 0.54 | 0.00 | 0.45 | 5.32 | 15.83 | 14.15 | 46,336.6 |
| 128 | 46,406.3 | 0.53 | 0.00 | 0.45 | 5.23 | 15.98 | 13.99 | 46,442.5 |

## 7. Setup-Phase Breakdown

Setup phases are one-time costs amortized over the simulation run. Setup is dominated by Python-side site initialization and the first-tick buffer build (ghost topology discovery + GPU buffer allocation). Three figures use the setup-phase data: (1) Fig.~`setup_amortization` plots setup-fraction-of-total vs. simulation length using the **largest Weak A 2,048-GPU configuration** as the representative steady-state cost — see §9.3 for the temporal-capability story. (2) Fig.~`weak_scaling_a_phase_breakdown` shows a single representative bar averaged across all 9 Weak A rank counts (8 → 2,048 GPUs), decomposed into the four phases (Initialize / GPU Setup / First Tick / Steady State) — the variance across the sweep is <5%, which is the direct visual proof that **per-rank cost is constant in N under weak scaling**. (3) Fig.~`strong_scaling_phase_breakdown` shows 7 stacked bars across the strong-scaling sweep (sites/GPU = 256 → 4, one per rank count from 8 → 512 GPUs), demonstrating how each phase **scales with per-rank workload** as the partition shrinks. The two breakdown figures together characterize setup cost on both axes: constant in N (Weak A) and scales with per-rank work (Strong) — see §9.1 and §9.2.

### Weak A (representative for amortization) — End-to-End Phase Breakdown (s)

| GPUs | Model Create | Load Globals | Site Init | Connectivity | GPU Setup | First Tick | Steady State | Total Sim |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.028 | 1.60 | 39.9 | 0.14 | 4.75 | 53.1 | 41.5 | 94.6 |
| 16 | 0.028 | 1.62 | 41.0 | 0.02 | 4.77 | 53.1 | 40.7 | 93.8 |
| 32 | 0.028 | 1.59 | 40.0 | 0.68 | 4.83 | 53.3 | 40.9 | 94.3 |
| 64 | 0.031 | 1.61 | 39.9 | 0.98 | 4.86 | 52.1 | 40.8 | 92.9 |
| 128 | 0.035 | 1.61 | 40.2 | 0.33 | 4.99 | 53.8 | 40.8 | 94.6 |
| 256 | 0.038 | 1.61 | 39.9 | 0.62 | 4.96 | 53.4 | 41.3 | 94.7 |
| 512 | 0.031 | 1.59 | 39.8 | 0.60 | 4.90 | 52.1 | 42.4 | 94.5 |
| 1024 | 0.070 | 1.71 | 39.9 | 1.68 | 4.98 | 56.5 | 41.9 | 98.4 |
| 2048 | 0.100 | 1.70 | 40.3 | 1.26 | 5.07 | 56.3 | 42.0 | 98.3 |

## 8. Figures (standalone subfigures for LaTeX `subcaption`)

All figures are saved as paired PDF (paper) + PNG (preview) at 600 DPI in `../figs/` (sibling of this `md/` folder). Each panel is its own file so LaTeX can compose them via `\begin{subfigure}` / `\subfloat`.

### 8.1 Main paper figures

**Weak Scaling A (10 sites/GPU, comm-heavy) — subfigure pair**, composed in LaTeX as a double-column figure with two side-by-side subfigures:
- (a) `weak_scaling_a_efficiency` — combined plot: stacked-bar of first-tick + steady-state simulation time, with ideal-baseline reference line and parallel efficiency annotated above each bar. Headline weak-scaling figure.
- (b) `weak_scaling_a_phase_breakdown` — **single representative bar** (averaged across all 9 Weak A rank counts 8 → 2,048 GPUs) showing the 4-phase decomposition (Initialize / GPU Setup / First Tick / Steady State), with a variance annotation `var <5%` that visually IS the **constant-in-N** claim of weak scaling — applied end-to-end across the full setup phase, not just steady-state per-tick cost.

**Strong Scaling (2,048 fixed sites) — subfigure pair**, composed in LaTeX as a double-column figure with two side-by-side subfigures:
- (a) `strong_scaling_speedup` — combined plot: log-log speedup curve with ideal-linear reference (left axis), cross-rank edge fraction on right twin axis, and a charcoal pill annotating peak speedup + efficiency.
- (b) `strong_scaling_phase_breakdown` — **7 stacked bars** across the strong-scaling sweep (sites/GPU = 4 / 8 / 16 / 32 / 64 / 128 / 256, one per rank count from 8 → 512 GPUs), each decomposed into the same 4 phases. Bars shrink dramatically left-to-right as per-rank work decreases, with composition shifting from setup-dominated (8 GPUs, ~85% setup) to steady-state-dominated (512 GPUs, ~51% steady state). Visually shows the kernel-granularity bound and the per-phase scaling-with-per-rank-work claim.

**Setup vs. Simulation** — `setup_amortization.{pdf,png}`
- `setup_amortization` — setup fraction of total wall time vs. simulation length, computed from the largest **Weak A** configuration (2,048 GPUs, the main weak-scaling experiment); pure computational, no use-case markers — production use cases are referenced in §9.3 text.

### 8.2 Supplementary figures

**Weak Scaling A — extra plot** (numbers redundant with §8.1's efficiency figure):
- `weak_scaling_a_throughput` — billion agent-updates/s vs. GPUs (essentially linear; redundant with the efficiency plot)

**Weak Scaling B (100 sites/GPU, compute-heavy)** — `weak_scaling_b_*.{pdf,png}` (supplementary cross-validation, see §S.1 narrative)
- `weak_scaling_b_efficiency` — combined plot: same stacked-bar design as Weak A
- `weak_scaling_b_throughput` — billion agent-updates/s vs. GPUs

## 9. Narrative for the Paper

### 9.0 Why scaling matters for GGap

Today's GGap CONUS production run uses ~1,400 sites at 1-year temporal resolution for ~1,000 simulated years (~1,000 ticks). This is small in both dimensions: spatially because 1,400 sites barely capture 25 km grid coverage over the continental US, and temporally because annual time stepping cannot resolve sub-year ecosystem dynamics (drought stress, fire-weather coupling, phenology timing). The scaling analysis below characterizes whether GGap can support **higher spatial resolution** (more sites at the same fidelity) and **higher temporal resolution** (much shorter time steps over comparable simulated horizons), or both.

The main-paper figures answer four independent questions. The Weak A subfigure pair — Fig.~`weak_scaling_a_efficiency` (a) paired with Fig.~`weak_scaling_a_phase_breakdown` (b) — proves the framework scales to ~14× more sites than the current CONUS baseline at near-perfect efficiency, and the (b) panel's single representative bar with a `var <5%` annotation directly demonstrates that per-rank cost is **constant in N** across the 8 → 2,048 GPU sweep. The Strong subfigure pair — Fig.~`strong_scaling_speedup` (a) paired with Fig.~`strong_scaling_phase_breakdown` (b) — proves wall time can be shrunk for a fixed problem AND shows how each setup phase **scales with per-rank workload** as the partition shrinks from 256 → 4 sites/GPU. Fig.~`setup_amortization` proves that one-time setup costs become negligible for long-horizon production runs at any temporal resolution finer than yearly. The two breakdown panels together characterize setup cost on both axes (constant in N + scales with per-rank work), completing the 2D characterization without needing to merge experiments. Together the four results argue that GGap's scaling unlocks production-quality high-resolution forest simulation that was previously infeasible.

### 9.1 Weak scaling — full-fidelity, communication-heavy (Weak A)

**Spatial capability vs. CONUS baseline.** Today's CONUS production run uses ~1,400 sites; the framework sustains **96% parallel efficiency at 2048 GPUs** (20,480 sites = 10,250,260,480 agents, ~15× more sites than current CONUS), demonstrating that spatial resolution is not capacity-limited at full per-site fidelity. The methodology and per-tick numbers behind this claim are detailed below.

We evaluate weak scaling on OLCF Frontier with the **full-fidelity, communication-heavy configuration**: 10 sites per GPU with 500 gaps × 1,000 trees per site (matching production UVAFME runs). With grid height 5 and 1D column-slab partitioning, each rank exchanges 30 of its 80 site-edges across rank boundaries — a **37.5% cross-rank fraction** that stress-tests the communication subsystem. From 8 to 2048 GPUs (a 256× scale-up to 10,250,260,480 agents), parallel efficiency stays at **96.2%** (Fig.~`weak_scaling_a_efficiency`). The model sustains a steady-state per-tick wall time of **~41 ms/tick** across the entire 256× scale-up (variation < 5%; full range 40.7–42.4 ms), and each GPU sustains **~119 M agent-updates/sec**, totaling **244 B agent-updates/sec at 2048 GPUs**. Fig.~`weak_scaling_a_efficiency` decomposes the 1,000-tick simulation time into the one-time first-tick warmup (orange) and the 999 steady-state ticks (green); the dashed reference line marks the 8-GPU baseline. The first 7 bars (8–512 GPUs) sit at the baseline (~100% efficiency); the 1024- and 2048-GPU bars exceed it slightly because the first-tick segment grows from ~53 s to ~56 s — the steady-state green segment stays flat at ~41 ms × 999 ticks. Steady-state per-tick MPI ghost-exchange remains constant at ~1 μs/rank across the entire range — see §9.3 for the first-tick anatomy and §S.1 for the Weak B cross-validation at a different point in the design space.

**Constant in N: the per-phase view.** Fig.~`weak_scaling_a_phase_breakdown` (the (b) panel of the Weak A subfigure pair) decomposes the wall-clock time into four phases — Initialize (host-side Python agent construction), GPU Setup (write-buffer allocation), First Tick (buffer build + ghost topology + first kernel JIT), and Steady State (the ~41 ms/tick × 999 repeating ticks) — averaged across all 9 Weak A rank counts. The total bar height varies by **<5%** across the 8 → 2,048 GPU sweep, which is the direct visual proof that **per-rank cost is independent of rank count under weak scaling**. This is what *weak scaling* means by definition, but here we demonstrate it end-to-end across the full setup phase, not just the steady-state per-tick cost. The orthogonal question — how does each phase scale with per-rank workload? — is answered by Fig.~`strong_scaling_phase_breakdown` in §9.2.

### 9.2 Strong scaling — time-to-solution

Strong scaling on a fixed problem of 2,048 sites (~205,211,648 agents). Fig.~`strong_scaling_speedup` shows the combined speedup-and-cross-rank picture: going from 8 → 512 GPUs achieves a **12.4× speedup** (19.3% parallel efficiency at the end of the curve), with the vertical gap between the measured (green) and ideal-linear (gray dashed) curves directly visualizing the efficiency loss; the orange right-axis line tracks the cross-rank edge fraction climbing from 1.2% to 75% as the per-GPU partition shrinks. Mean tick time falls from 95.2 ms at 8 GPUs to 18.9 ms at 512 GPUs, with GPU execution remaining >99% of total per-tick wall time across the entire range. At the most extreme partition (4 sites/GPU, single-column slab), each GPU still updates ~400K agents per tick in 18.95 ms, showing the runtime retains usable throughput even at minimum granularity.

**Per-tick non-GPU overhead is flat across the entire sweep.** Direct instrumentation shows that the per-tick cost outside the fused GPU kernel — MPI ghost-exchange (the SAGESim `data_prep` wrapper around `exchange_ghost_data()`, covering GPU pack, GPU-aware Slingshot Isend/Irecv/Waitall, and GPU unpack), CuPy kernel-arg dispatch, and double-buffer write-back — is constant at **~5 μs + ~14 μs + ~16 μs ≈ 35 μs** across all seven configurations. This is **<0.2% of even the smallest tick (19 ms at 512 GPUs)**, and confirms the runtime imposes no rank-count tax on overhead. The strong-scaling efficiency drop is therefore **GPU-kernel-granularity bound**, not framework-bound: as sites/GPU shrinks from 256 to 4, per-agent compute cost rises ~13× because CuPy kernel launch overhead and intra-GCD parallelism limits become a larger fraction of the per-GPU work.

**Why MPI cost stays constant despite the rising cross-rank fraction.** Fig.~`strong_scaling_speedup`'s right axis shows the cross-rank edge fraction climbing from 1.2% to 75% as the per-GPU partition shrinks. This rise is purely a *relative* effect — boundary divided by interior. The **absolute** cross-rank edge count per rank is fixed at **24** (= `grid_height × 3 × 2` = `4 × 3 × 2` for the 1D column-slab partition with constant `grid_height = 4`), and each rank exchanges only with 2 peers (left/right neighbor with periodic wraparound) regardless of total rank count. Every rank therefore packs and exchanges a constant ~960 B per tick across the entire 8 → 512 GPU sweep, costing the constant ~5 μs MPI time reported above. The rising cross-rank fraction is a relative indicator of partition shrinkage, not a driver of communication cost — and the constancy of measured MPI time confirms this directly.

**Scope and limitations.** This strong-scaling experiment characterizes time-to-solution for the GGap model's natural 8-neighbor Moore connectivity under 1D column-slab partitioning with `grid_height = 4`. As GPU count grows, per-rank compute work shrinks while per-rank cross-rank edges remain fixed, so the experiment isolates GPU kernel-granularity effects rather than network effects. Other partition topologies (2D decomposition with O(√N) peers per rank, k-NN connectivity, larger ghost halos) would induce different cross-rank scaling and are left to future work.

**Scales with per-rank work: the per-phase view.** Fig.~`strong_scaling_phase_breakdown` (the (b) panel of the Strong subfigure pair) decomposes total wall time across the strong-scaling sweep into four merged segments (Initialize / GPU Setup / First Tick / Steady State). Initialize (host-side Python agent construction, dominated by site init) and GPU Setup (write-buffer allocation) **scale near-linearly with the per-rank agent count** — as sites/GPU shrinks 64× (256 → 4), Initialize falls from ~233 s to ~5 s and GPU Setup from ~26 s to ~1 s. First Tick scales **sub-linearly** (~24× shrinkage from ~283 s to ~12 s) because it bundles per-rank buffer construction with smaller per-rank components for kernel JIT and the MPI communication-map handshake. Steady State scales **least** (~5× shrinkage from ~95 s to ~19 s), reflecting the GPU kernel-granularity bound discussed above. Bars shrink from ~636 s at 8 GPUs (~85% setup-dominated) to ~37 s at 512 GPUs (~51% steady-state-dominated), and the crossover where steady state catches up with setup is directly visible. Combined with the **constant-in-N** proof from Fig.~`weak_scaling_a_phase_breakdown` (§9.1), the two breakdown panels complete the 2D characterization: **setup time depends only on per-rank work, with no rank-count tax**.

### 9.3 Setup amortization — temporal capability for long-horizon production

Today's CONUS run uses 1,000 yearly ticks. At higher temporal resolutions the tick count grows substantially: a centennial run at daily resolution is 36,500 ticks; a millennial run at daily resolution is 365,000 ticks; an hourly-resolution centennial run is 876,000 ticks. Fig.~`setup_amortization` shows that the one-time setup cost (~105 s for the representative Weak A 2048-GPU configuration, dominated by site initialization (40 s of host-side Python agent construction) and the first-tick buffer build (56 s of ghost-cell topology discovery + GPU buffer allocation)) drops below 5% of total wall time at **~47,266 ticks**. Reading the curve at typical use cases:

- **1,000 ticks (current CONUS, yearly)** — setup is ~71% of total wall time; our benchmark is heavily setup-dominated.
- **36,500 ticks (centennial daily)** — setup falls to ~6%.
- **365,000 ticks (millennial daily)** — setup falls to ~0.7%.
- **876,000 ticks (centennial hourly)** — setup is ~0.3% (essentially zero).

**Any production simulation at daily resolution or finer for centennial-or-longer horizons makes setup negligible.** This is the key temporal-capability claim of the paper: GGap is *not* setup-bound for the science use cases that motivate it; the setup-bound regime only contains the short benchmarking runs reported in this very section. Reducing site-init and first-tick costs remains a clear engineering target for future work, but is not on the critical path for production science use cases.

**Anatomy of the first tick — why setup costs what it does.** The first-tick cost consists of four one-time SAGESim operations that are cached for steady-state ticks and never repeated (see `sagesim/model.py:1340-1398`):

1. **GPU buffer construction** (`sagesim/model.py:1014-1254`) — host→device transfer of every agent property tensor (breed IDs, ragged neighbor CSR, scalars, vectors), allocation of double-buffering write buffers, creation of agent-id ↔ local-index hash maps, and one MPI allreduce to synchronize property widths across ranks. Cost is **O(N_local × num_properties)** — for Weak A that is ~5 M agents × ~20 properties per GPU.
2. **Ghost topology discovery** (`sagesim/gpu_kernels.py:31`, called from `sagesim/model.py:1349`) — each rank scans its CSR neighbor lists to identify the unique remote agent IDs it needs as ghosts. CPU-side vectorized scan, no MPI yet. **O(local_edges)**.
3. **Communication map build** (`sagesim/gpu_kernels.py:517-846`, called from `sagesim/model.py:1377`) — two-phase MPI handshake: an Alltoall to exchange request counts (`gpu_kernels.py:626`), then Isend/Irecv pairs to exchange the actual remote-agent-id lists (`:629-647`). The rank then builds per-peer pack/unpack index arrays and pre-allocates persistent send/recv GPU buffers. **O(P × peers_per_rank)** for the handshake and **O(boundary_size)** for index construction. SAGESim's 1D column-slab partition keeps `peers_per_rank` constant at 2 regardless of total rank count, so this scales gracefully.
4. **First ghost exchange + first kernel launch** — pack/MPI/unpack runs once to fill ghost slots, and CuPy NVRTC compiles the fused step function from PTX to binary on the first `@jit.rawkernel` launch (`sagesim/model.py:1466`). Subsequent ticks reuse the cached binary.

In steady state (`sagesim/model.py:1386-1398`) only step 4 repeats — buffers, hash maps, comm maps, and the compiled kernel are all cached. That is why steady ticks are ~41 ms vs ~53 s for tick 1 in Weak A. The mild growth of first_tick from ~53 s → ~56 s at 1024+ GPUs (visible as the orange-segment growth in `weak_scaling_a_efficiency`) is the source of the small efficiency drop; we attribute it to the O(P) Alltoall handshake in step 3 dominating the otherwise constant per-rank work as P grows.

**Steady-state MPI is constant in weak scaling** (verified, not just predicted). Per-rank cross-rank edges (30) and MPI peer count (2, by 1D column-slab partitioning with periodic wraparound) are independent of total rank count, and SAGESim's per-tick exchange uses non-blocking Isend/Irecv only (no per-tick collective operations). Measured ghost-exchange wall time — the full SAGESim `data_prep` wrapper around `exchange_ghost_data()`, covering GPU pack, GPU-aware Slingshot Isend/Irecv/Waitall, and GPU unpack — stays at **~5 μs/rank ± 10%** across the full 8 → 2,048 GPU range, confirming the GPU-aware Slingshot-11 path scales as the methodology predicts. The small efficiency drop at 1024+ GPUs is therefore attributable to first-tick growth, not steady-state communication.

### 9.4 Synthesis: what GGap unlocks

Combining the three results: the **spatial capability** (Fig.~`weak_scaling_a_efficiency`, ~15× more sites than current CONUS at 96% efficiency under full UVAFME per-site fidelity), the **temporal capability** (Fig.~`setup_amortization`, setup amortized for daily-or-finer resolution at centennial horizons), and the **time-to-solution capability** (Fig.~`strong_scaling_speedup`, 12.4× wall-time reduction for fixed problems) collectively argue that GGap's scaling unlocks a class of high-resolution, long-horizon, large-domain forest simulations that were previously infeasible with the current 1,400-site / 1-year-per-tick CONUS baseline. Fig.~`strong_scaling_phase_breakdown` (and its companion Fig.~`weak_scaling_a_phase_breakdown`) provide the per-phase characterization that lets us project these capabilities forward: with setup time = f(per-rank work) and steady-state time pinned at ~41 ms/tick in the weak-scaling-A regime, scaling to ~20,480 sites at sub-yearly resolution for centennial horizons is a straightforward extension of the measurements reported here.

## 10. Methodology Notes

**Parallel efficiency definitions** (both weak and strong are computed from `simulation_time = first_tick_time + steady_state_time`, the full wall-clock cost of the `simulate()` call including the one-time first-tick warmup amortized over the run; this is more honest than reporting steady-state-only because `first_tick` grows mildly with rank count and is a real per-simulation cost):

- *Weak scaling*: `eff(N) = T_baseline_sim / T_N_sim × 100`. Ideal = 100%.
- *Strong scaling*: `speedup(N) = T_baseline_sim / T_N_sim`; `eff(N) = speedup(N) / (N / N_baseline) × 100`. Ideal = 100%.

**Throughput**: `total_agents / mean_tick_time` in agent-updates/second. Throughput uses the steady-state per-tick time (not the amortized one) because it measures *sustained* per-tick capability — a separate concept from full-run wall-clock efficiency.

**Total agents**: `total_sites × (1 + num_gaps × (1 + maxtrees))` (1 site agent + N gap agents + N×M tree agents per site).

**Cross-rank fraction**: `(grid_height × 6) / (sites_per_gpu × 8)` — derived from the 8-neighbor Moore connectivity and 1D column-slab partition.

**Duplicates**: where the CSV contains repeated runs for the same GPU count, all numeric columns are averaged before plotting (same convention as the existing `plot_weak_scaling_breakdown.py` script).

**Mean tick window**: SAGESim's `verbose_timing` separates tick 1 (buffer build, ghost topology discovery, communication map build) from ticks 2..N. `mean_tick_time` is the average of ticks 2..N (steady state) and is the figure used in all efficiency and throughput calculations.

**`gpu_execution`**: equals `mean_gpu_compute + mean_gpu_sync`. The GPU kernel launches asynchronously, so `gpu_compute` only captures CPU-side launch overhead (~0.5 ms) — actual kernel execution completes during `gpu_sync` when the CPU calls `stream.synchronize()`.

## 11. Data Provenance

| Source CSV | Used by | GPU counts present | Missing |
|---|---|---|---|
| `weak_scaling.csv` | weak_scaling_a_*, weak_scaling_a_phase_breakdown, setup_amortization, §4, §9.1, §9.3 | [8, 16, 32, 64, 128, 256, 512, 1024, 2048] | none |
| `strong_scaling_b.csv` | strong_scaling_*, strong_scaling_phase_breakdown, §5, §9.2 | [8, 16, 32, 64, 128, 256, 512] | none |
| `weak_scaling_b.csv` | weak_scaling_b_*, §S.1 | [8, 16, 32, 64, 128] | [256, 512, 1024, 2048] |

**To refresh:** drop new rows into the relevant CSV and re-run the affected plot scripts in `papers/SC2026/src/`, then re-run `python papers/SC2026/src/generate_results_md.py` to regenerate this document.

---

## §S.1 Supplementary: Weak Scaling B (high site density)

This supplementary section reports the results of the **Weak Scaling B** configuration, which complements the main paper's Weak Scaling A (§4, §9.1) by exercising a different point in the design space:

- **100 sites per GPU** (10× more than Weak A) — high site density
- **200 gaps × 500 trees** per site (reduced fidelity, deliberately matching the Strong scaling experiment's per-site workload — see §5 — so that Weak B and Strong live on the same `f(per-rank work)` curve and per-rank costs are directly comparable)
- **7.5% cross-rank edges** — a low-communication-density regime where MPI is amortized aggressively over local compute
- Originally targeted **8–2,048 GPUs** (matching Weak A), but the 256 / 512 / 1024 / 2048 GPU runs were queued on Frontier and **did not complete before the SC2026 submission deadline**. We report the 5/9 measured rank counts (8/16/32/64/128) below.

Together with Weak A, these two configurations span the compute-vs-communication spectrum: Weak A is the comm-stress / full-fidelity end, Weak B is the high-site-density / large-total-agents end. The two configurations also have different per-site fidelity (Weak A: 500 gaps × 1,000 trees; Weak B: 200 gaps × 500 trees), and Weak B's choice was made specifically so it shares fidelity with the Strong scaling experiment, allowing direct per-rank-cost comparison.

> **Status:** 5/9 GPU configurations complete. Missing: 256, 512, 1024, 2048 GPUs (queued on Frontier; runs did not complete before the SC2026 deadline).

### Weak Scaling B — 100 sites/GPU

Source: `weak_scaling_b.csv`

| GPUs | Total Sites | Total Agents | Sim Time (s) | First Tick (s) | Steady (s) | Mean Tick (ms) | Throughput (B upd/s) | Efficiency (%) | MPI % |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 800 | 80,160,800 | 155.0 | 111.2 | 43.8 | 43.84 | 1.83 | 100.0 | 0.002 |
| 16 | 1,600 | 160,321,600 | 155.5 | 111.1 | 44.4 | 44.46 | 3.61 | 99.6 | 0.002 |
| 32 | 3,200 | 320,643,200 | 156.1 | 110.8 | 45.4 | 45.40 | 7.06 | 99.3 | 0.002 |
| 64 | 6,400 | 641,286,400 | 158.0 | 111.6 | 46.3 | 46.38 | 13.83 | 98.1 | 0.002 |
| 128 | 12,800 | 1,282,572,800 | 157.3 | 110.8 | 46.4 | 46.48 | 27.59 | 98.6 | 0.002 |
| 256 | — | — | — | — | — | — | — | — | — |
| 512 | — | — | — | — | — | — | — | — | — |
| 1024 | — | — | — | — | — | — | — | — | — |
| 2048 | — | — | — | — | — | — | — | — | — |

At 128 GPUs the model handles 1,282,572,800 agents at 98.6% parallel efficiency (Fig.~`weak_scaling_b_efficiency` — same combined design as Weak A: stacked first-tick + steady-state simulation time with ideal-baseline reference and efficiency annotations). The sustained per-GPU throughput is **~216 M agent-updates/sec** — ~1.8× the Weak A per-GPU rate, indicating the runtime is GPU-occupancy-bound rather than algorithm-bound (Weak B's ~10 M agents per GPU and lighter per-site compute saturate the MI250X GCD more effectively than Weak A's ~5 M agents per GPU with full-fidelity per-site workload). The pattern matches Weak A: GPU execution dominates and MPI ghost-exchange is in the noise (<0.002%).

Supplementary figures: `../figs/weak_scaling_b_efficiency.png` and `../figs/weak_scaling_b_throughput.png`.
