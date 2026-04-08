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

| Metric | **Weak B (100 sites/GPU)** | **Strong (2,048 sites)** | Weak A (10 sites/GPU, §S.1) |
|---|---|---|---|
| Peak GPUs | 128 | 512 | 2048 |
| Peak total sites | 12,800 | 2,048 | 20,480 |
| Peak total agents | 1,282,572,800 | 205,211,648 | 10,250,260,480 |
| Peak total throughput | 27.59 B upd/s | 10.83 B upd/s | 243.57 B upd/s |
| Sustained per-GPU throughput | 216 M upd/s/GPU | — | 119 M upd/s/GPU |
| Sustained steady-state per-tick (avg) | ~45 ms | — | ~41 ms |
| Mean tick at peak GPU count | 46.48 ms | 18.95 ms | 42.08 ms |
| Parallel efficiency at peak | 98.6% | 19.3% | 96.2% |
| Speedup at peak | — | 12.4× | — |

**Key claims for the paper:**

- Weak Scaling B (main weak-scaling experiment, headline) reaches **1,282,572,800 agents on 128 GPUs** at **99% parallel efficiency**, sustaining ~216 M agent-updates/sec/GPU. The queued 256, 512, 1024, 2048 GPU runs are expected to extend the sweep to **~20.5 B agents at 2,048 GPUs** while holding per-rank work fixed.
- Strong Scaling delivers a **12.4× speedup** from 8 → 512 GPUs (19% parallel efficiency at the largest scale). The efficiency drop is correlated with the per-GPU partition shrinking from 256 → 4 sites: at the smallest partitions each GPU kernel underutilizes the MI250X GCD (kernel launch overhead becomes a larger fraction of GPU time), and the cross-rank edge fraction climbs from 1.2% → 75%.
- Strong and Weak B share the same per-site fidelity (200 gaps × 500 trees), so `setup_phase_breakdown` can plot both on a single per-rank-workload axis — Weak B's hatched bar lies precisely on the strong-scaling curve, **directly proving setup time depends only on per-rank work, not on rank count**.
- Weak Scaling A (supplementary, §S.1) cross-validates Weak B at a different point in the design space (10 sites/GPU, 500 gaps × 1,000 trees per site, 37.5% cross-rank stress) and reaches **96% parallel efficiency at 2048 GPUs** (10,250,260,480 agents).
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

| Experiment | Sites/GPU | Gaps/site | Trees/gap | Grid H | Cross-rank % | GPU Range | Paper section |
|---|---:|---:|---:|---:|---:|---|---|
| Weak B (compute-heavy)     | 100   | 200 |   500 | 10 |  7.5%       | 8–2,048 | **Main (§4, §9.1)** |
| Strong (fixed 2,048 sites) | 256→4 | 200 |   500 |  4 | 1.2%→75.0%  | 8–512   | **Main (§5, §9.2)** |
| Weak A (comm-heavy)        |  10   | 500 | 1,000 |  5 | 37.5%       | 8–2,048 | Supplementary (§S.1) |

## 4. Weak Scaling B — Main Result

High-density compute-heavy configuration: 100 sites/GPU with 200 gaps × 500 trees per site (matching the Strong scaling experiment's per-site fidelity for direct cross-comparison — see §5). 7.5% cross-rank density, 1D column-slab partitioning with grid height 10. This is the headline weak-scaling configuration for the main paper. *(A complementary configuration with 5× higher per-site fidelity but 10× fewer sites per GPU — Weak A — is reported in §S.1 as supplementary cross-validation.)*

> **Status:** 5/9 GPU configurations complete. Missing: 256, 512, 1024, 2048 GPUs (queued on Frontier; will fill in before submission). All numbers below auto-update when the new CSV rows land.

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

All values in microseconds (μs). `GPU Exec` = `mean_gpu_compute + mean_gpu_sync` (kernel launch overhead + actual GPU work). MPI columns are the GPU-aware MPI ghost-cell exchange. `Total` is the average per-tick wall time. Decomposition tables for both main (Weak B, Strong) and supplementary (Weak A) experiments are co-located here for reference.

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

### Weak A (supplementary) — Per-tick Time Breakdown (μs)

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

## 7. Setup-Phase Breakdown

Setup phases are one-time costs amortized over the simulation run. Setup is dominated by Python-side site initialization and the first-tick buffer build (ghost topology discovery + GPU buffer allocation). Fig.~`setup_phase_breakdown` merges **strong scaling and Weak B** on a single x-axis (sites/GPU = per-rank workload) — both experiments share the same per-site fidelity (200 gaps × 500 trees), so they live on the same curve. Strong contributes 7 bars at sites/GPU = 256, 128, 64, 32, 16, 8, 4 (each a different rank count, 8 → 512 GPUs); Weak B contributes 1 hatched bar at sites/GPU = 100, **averaged across the Weak B rank sweep (8 → 2,048 GPUs)** — currently 5/9 rank counts complete (8/16/32/64/128) with var <3% across them; the queued 256/512/1024/2048 GPU runs are expected to extend the sweep without changing the bar height because per-rank work is fixed in weak scaling. The Weak B bar lands precisely between Strong's 64 and 128 bars on every segment, visually demonstrating that **setup time = f(per-rank work) independent of rank count**. (Weak A is excluded from this plot because its per-site fidelity is 5× higher and would land on a different curve — see §S.1 for Weak A's separate efficiency story; see §9.2 for the merged-figure narrative and §9.3 for the amortization story.) The end-to-end Weak B breakdown table below is kept as the data source for `setup_amortization.{pdf,png}`, which uses the largest measured Weak B configuration as the representative steady-state cost.

### Weak B (representative) — End-to-End Phase Breakdown (s)

| GPUs | Model Create | Load Globals | Site Init | Connectivity | GPU Setup | First Tick | Steady State | Total Sim |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.032 | 1.59 | 90.9 | 0.45 | 10.47 | 111.2 | 43.8 | 155.0 |
| 16 | 0.028 | 1.61 | 92.2 | 0.59 | 10.51 | 111.1 | 44.4 | 155.5 |
| 32 | 0.028 | 1.60 | 91.9 | 1.71 | 10.49 | 110.8 | 45.4 | 156.1 |
| 64 | 0.036 | 1.60 | 92.9 | 0.56 | 10.46 | 111.6 | 46.3 | 158.0 |
| 128 | 0.023 | 1.60 | 95.9 | 0.38 | 10.49 | 110.8 | 46.4 | 157.3 |

## 8. Figures (standalone subfigures for LaTeX `subcaption`)

All figures are saved as paired PDF (paper) + PNG (preview) at 600 DPI in `../figs/` (sibling of this `md/` folder). Each panel is its own file so LaTeX can compose them via `\begin{subfigure}` / `\subfloat`.

### 8.1 Main paper figures

**Weak Scaling B (100 sites/GPU, compute-heavy)** — `weak_scaling_b_*.{pdf,png}`
- `weak_scaling_b_efficiency` — combined plot: stacked-bar of first-tick + steady-state simulation time, with ideal-baseline reference line and parallel efficiency annotated above each bar. Headline weak-scaling figure.

**Strong Scaling (2,048 fixed sites)** — `strong_scaling_speedup.{pdf,png}`
- `strong_scaling_speedup` — combined plot: log-log speedup curve with ideal-linear reference (left axis), cross-rank edge fraction on right twin axis, and a charcoal pill annotating peak speedup + efficiency. The vertical gap between the measured and ideal curves visually IS the parallel-efficiency loss; the explicit % is in the annotation. Replaces the earlier separate `strong_scaling_speedup` + `strong_scaling_efficiency` pair which were mathematically equivalent.

**Setup vs. Simulation** — `setup_*.{pdf,png}`
- `setup_phase_breakdown` — end-to-end phase breakdown plotted against **sites/GPU** (per-rank workload), merging **strong scaling** (7 bars at sites/GPU = 4 / 8 / 16 / 32 / 64 / 128 / 256, one per rank count) and **Weak B** (1 hatched bar at sites/GPU = 100, averaged across the Weak B rank sweep 8 → 2,048 GPUs; 5/9 currently complete with var <3%). Weak B lies exactly on Strong's curve between sites/GPU = 64 and 128, visually proving that setup time depends only on per-rank work and not on rank count (the 2D characterization, in one figure).
- `setup_amortization` — setup fraction of total wall time vs. simulation length (uses Weak B 128-GPU representative steady-state cost; pure computational, no use-case markers — production use cases are referenced in §9.3 text)

### 8.2 Supplementary figures

**Weak Scaling A (10 sites/GPU, comm-heavy)** — `weak_scaling_a_*.{pdf,png}` (supplementary cross-validation, see §S.1 narrative)
- `weak_scaling_a_efficiency` — combined plot: stacked-bar of first-tick + steady-state simulation time with ideal-baseline reference and per-bar efficiency annotations. Same design as `weak_scaling_b_efficiency` for visual consistency.
- `weak_scaling_a_throughput` — billion agent-updates/s vs. GPUs (essentially linear; redundant with the efficiency plot)

**Weak Scaling B — extra plot** (numbers redundant with §8.1's efficiency figure):
- `weak_scaling_b_throughput` — billion agent-updates/s vs. GPUs

## 9. Narrative for the Paper

### 9.0 Why scaling matters for GGap

Today's GGap CONUS production run uses ~1,400 sites at 1-year temporal resolution for ~1,000 simulated years (~1,000 ticks). This is small in both dimensions: spatially because 1,400 sites barely capture 25 km grid coverage over the continental US, and temporally because annual time stepping cannot resolve sub-year ecosystem dynamics (drought stress, fire-weather coupling, phenology timing). The scaling analysis below characterizes whether GGap can support **higher spatial resolution** (more sites at the same fidelity) and **higher temporal resolution** (much shorter time steps over comparable simulated horizons), or both.

The four figures of this section answer four independent questions. Fig.~`weak_scaling_b_efficiency` proves the framework scales to ~146× more sites than the current CONUS baseline at near-perfect efficiency. Fig.~`strong_scaling_speedup` proves wall time can be shrunk for a fixed problem so per-scenario turnaround drops. Fig.~`setup_amortization` proves that one-time setup costs become negligible for long-horizon production runs at any temporal resolution finer than yearly. Fig.~`setup_phase_breakdown` (merging Weak B with the strong-scaling sweep on a single per-rank-workload axis) shows how each phase scales with per-rank workload AND that rank count alone has no effect, completing the 2D characterization in one figure. Together they argue that GGap's scaling unlocks production-quality high-resolution forest simulation that was previously infeasible.

### 9.1 Weak scaling — high-density compute-heavy (Weak B)

**Spatial capability vs. CONUS baseline.** Today's CONUS production run uses ~1,400 sites. Weak Scaling B's full target sweep extends to **2048 GPUs at 100 sites/GPU = 204,800 sites (~146× more sites than current CONUS) and ~20.5 B agents** at production-relevant per-site density, demonstrating that spatial resolution is not capacity-limited.
  *(Currently 5/9 rank counts complete: the measured peak is 12,800 sites = 1,282,572,800 agents at 128 GPUs / 98.6% efficiency. The queued 256, 512, 1024, 2048 GPU runs are expected to extend the sweep without changing the per-rank cost because per-rank work is fixed by construction in weak scaling.)*

We evaluate weak scaling on OLCF Frontier with the **high-density compute-heavy configuration**: 100 sites per GPU with 200 gaps × 500 trees per site (deliberately matching the Strong scaling experiment's per-site fidelity, see §5, so the two experiments live on the same `f(per-rank work)` curve and can be merged in `setup_phase_breakdown`). With grid height 10 and 1D column-slab partitioning, each rank exchanges 60 of its 800 site-edges across rank boundaries — a **7.5% cross-rank fraction**, a low-communication-density regime where MPI is amortized aggressively over local compute. From 8 to 128 GPUs (currently a 16× scale-up; target is 256× to 20,521,164,800 agents), parallel efficiency stays at **98.6%** (Fig.~`weak_scaling_b_efficiency`). The model sustains a steady-state per-tick wall time of **~45 ms/tick** across the measured range (variation < 5%; full range 43.8–46.5 ms), and each GPU sustains **~216 M agent-updates/sec** — the highest sustained per-GPU rate of any GGap configuration tested, indicating that 10 M agents per GPU saturate the MI250X GCD effectively. At the eventual 2048-GPU target, total throughput projects to **~441 B agent-updates/sec**.

Fig.~`weak_scaling_b_efficiency` decomposes the 1,000-tick simulation time into the one-time first-tick warmup (orange) and the 999 steady-state ticks (green); the dashed reference line marks the 8-GPU baseline. The steady-state green segment stays flat at ~45 ms × 999 ticks across the measured sweep — direct visual proof that per-rank work alone determines per-tick cost in this regime. Steady-state per-tick MPI ghost-exchange remains in the noise (<0.005% of per-tick wall time) across the entire range — see §9.3 for the first-tick anatomy and the constant-MPI verification, and §S.1 for the Weak A cross-validation at a different point in the design space.

### 9.2 Strong scaling — time-to-solution

Strong scaling on a fixed problem of 2,048 sites (~205,211,648 agents). Fig.~`strong_scaling_speedup` shows the combined speedup-and-cross-rank picture: going from 8 → 512 GPUs achieves a **12.4× speedup** (19.3% parallel efficiency at the end of the curve), with the vertical gap between the measured (green) and ideal-linear (gray dashed) curves directly visualizing the efficiency loss; the orange right-axis line tracks the cross-rank edge fraction climbing from 1.2% to 75% as the per-GPU partition shrinks. Mean tick time falls from 95.2 ms at 8 GPUs to 18.9 ms at 512 GPUs, with GPU execution remaining >99% of total per-tick wall time across the entire range. At the most extreme partition (4 sites/GPU, single-column slab), each GPU still updates ~400K agents per tick in 18.95 ms, showing the runtime retains usable throughput even at minimum granularity.

**Per-tick non-GPU overhead is flat across the entire sweep.** Direct instrumentation shows that the per-tick cost outside the fused GPU kernel — MPI ghost-exchange (the SAGESim `data_prep` wrapper around `exchange_ghost_data()`, covering GPU pack, GPU-aware Slingshot Isend/Irecv/Waitall, and GPU unpack), CuPy kernel-arg dispatch, and double-buffer write-back — is constant at **~5 μs + ~14 μs + ~16 μs ≈ 35 μs** across all seven configurations. This is **<0.2% of even the smallest tick (19 ms at 512 GPUs)**, and confirms the runtime imposes no rank-count tax on overhead. The strong-scaling efficiency drop is therefore **GPU-kernel-granularity bound**, not framework-bound: as sites/GPU shrinks from 256 to 4, per-agent compute cost rises ~13× because CuPy kernel launch overhead and intra-GCD parallelism limits become a larger fraction of the per-GPU work.

**Why MPI cost stays constant despite the rising cross-rank fraction.** Fig.~`strong_scaling_speedup`'s right axis shows the cross-rank edge fraction climbing from 1.2% to 75% as the per-GPU partition shrinks. This rise is purely a *relative* effect — boundary divided by interior. The **absolute** cross-rank edge count per rank is fixed at **24** (= `grid_height × 3 × 2` = `4 × 3 × 2` for the 1D column-slab partition with constant `grid_height = 4`), and each rank exchanges only with 2 peers (left/right neighbor with periodic wraparound) regardless of total rank count. Every rank therefore packs and exchanges a constant ~960 B per tick across the entire 8 → 512 GPU sweep, costing the constant ~5 μs MPI time reported above. The rising cross-rank fraction is a relative indicator of partition shrinkage, not a driver of communication cost — and the constancy of measured MPI time confirms this directly.

**Scope and limitations.** This strong-scaling experiment characterizes time-to-solution for the GGap model's natural 8-neighbor Moore connectivity under 1D column-slab partitioning with `grid_height = 4`. As GPU count grows, per-rank compute work shrinks while per-rank cross-rank edges remain fixed, so the experiment isolates GPU kernel-granularity effects rather than network effects. Other partition topologies (2D decomposition with O(√N) peers per rank, k-NN connectivity, larger ghost halos) would induce different cross-rank scaling and are left to future work.

**How each phase scales with per-rank work — and why it's the same regardless of rank count.** Fig.~`setup_phase_breakdown` decomposes total wall time on a single **sites/GPU** axis, merging strong scaling (7 solid bars, one per rank count from 8–512 GPUs at sites/GPU = 256 → 4) with Weak B (1 hatched bar at sites/GPU = 100, averaged across the Weak B rank sweep 8 → 2,048 GPUs). Reading the strong-scaling sweep first: Initialize (host-side Python agent construction, dominated by site init) and GPU Setup (write-buffer allocation) **scale near-linearly with the per-rank agent count** — as sites/GPU shrinks 64× (256 → 4), Initialize falls from ~233 s to ~5 s and GPU Setup from ~26 s to ~1 s. First Tick scales **sub-linearly** (~24× shrinkage from ~283 s to ~12 s) because it bundles per-rank buffer construction with smaller per-rank components for kernel JIT and the MPI communication-map handshake. Steady State scales **least** (~5× shrinkage from ~95 s to ~19 s), reflecting the GPU kernel-granularity bound discussed above. Bars shrink from ~636 s at 8 GPUs (~85% setup-dominated) to ~37 s at 512 GPUs (~51% steady-state-dominated), and the crossover where steady state catches up with setup is directly visible.

**The Weak B bar (hatched, sites/GPU = 100) closes the 2D characterization.** Averaged across the Weak B rank sweep (8 → 2,048 GPUs target; 5/9 currently complete), it sits at ~95 s Initialize, ~10.5 s GPU Setup, ~111 s First Tick, ~45 s Steady State — landing **precisely between Strong's sites/GPU = 64 and 128 bars** on every segment (linear interpolation across the same fidelity curve). The variation across the currently-complete rank counts is **<3% on the total bar height**, annotated above the bar; the queued 256/512/1024/2048 GPU runs are expected to keep this flat because per-rank work is fixed by construction in weak scaling. That flatness is the direct visual proof that **setup time depends only on per-rank work, with no rank-count tax**. Strong scaling sweeps the per-rank workload axis (varying x at varying N); Weak B holds the per-rank workload fixed and varies N (one x with collapsed N). They agree, so the curve is f(per-rank work) alone.

### 9.3 Setup amortization — temporal capability for long-horizon production

Today's CONUS run uses 1,000 yearly ticks. At higher temporal resolutions the tick count grows substantially: a centennial run at daily resolution is 36,500 ticks; a millennial run at daily resolution is 365,000 ticks; an hourly-resolution centennial run is 876,000 ticks. Fig.~`setup_amortization` shows that the one-time setup cost (~219 s for the representative Weak B 128-GPU configuration, dominated by site initialization (96 s of host-side Python agent construction) and the first-tick buffer build (111 s of ghost-cell topology discovery + GPU buffer allocation)) drops below 5% of total wall time at **~89,613 ticks**. Reading the curve at typical use cases:

- **1,000 ticks (current CONUS, yearly)** — setup is ~83% of total wall time; our benchmark is heavily setup-dominated.
- **36,500 ticks (centennial daily)** — setup falls to ~11%.
- **365,000 ticks (millennial daily)** — setup falls to ~1.3%.
- **876,000 ticks (centennial hourly)** — setup is ~0.5% (essentially zero).

**Any production simulation at daily resolution or finer for centennial-or-longer horizons makes setup negligible.** This is the key temporal-capability claim of the paper: GGap is *not* setup-bound for the science use cases that motivate it; the setup-bound regime only contains the short benchmarking runs reported in this very section. Reducing site-init and first-tick costs remains a clear engineering target for future work, but is not on the critical path for production science use cases.

**Anatomy of the first tick — why setup costs what it does.** The first-tick cost consists of four one-time SAGESim operations that are cached for steady-state ticks and never repeated (see `sagesim/model.py:1340-1398`):

1. **GPU buffer construction** (`sagesim/model.py:1014-1254`) — host→device transfer of every agent property tensor (breed IDs, ragged neighbor CSR, scalars, vectors), allocation of double-buffering write buffers, creation of agent-id ↔ local-index hash maps, and one MPI allreduce to synchronize property widths across ranks. Cost is **O(N_local × num_properties)** — for Weak A that is ~5 M agents × ~20 properties per GPU.
2. **Ghost topology discovery** (`sagesim/gpu_kernels.py:31`, called from `sagesim/model.py:1349`) — each rank scans its CSR neighbor lists to identify the unique remote agent IDs it needs as ghosts. CPU-side vectorized scan, no MPI yet. **O(local_edges)**.
3. **Communication map build** (`sagesim/gpu_kernels.py:517-846`, called from `sagesim/model.py:1377`) — two-phase MPI handshake: an Alltoall to exchange request counts (`gpu_kernels.py:626`), then Isend/Irecv pairs to exchange the actual remote-agent-id lists (`:629-647`). The rank then builds per-peer pack/unpack index arrays and pre-allocates persistent send/recv GPU buffers. **O(P × peers_per_rank)** for the handshake and **O(boundary_size)** for index construction. SAGESim's 1D column-slab partition keeps `peers_per_rank` constant at 2 regardless of total rank count, so this scales gracefully.
4. **First ghost exchange + first kernel launch** — pack/MPI/unpack runs once to fill ghost slots, and CuPy NVRTC compiles the fused step function from PTX to binary on the first `@jit.rawkernel` launch (`sagesim/model.py:1466`). Subsequent ticks reuse the cached binary.

In steady state (`sagesim/model.py:1386-1398`) only step 4 repeats — buffers, hash maps, comm maps, and the compiled kernel are all cached. That is why steady ticks are ~41 ms vs ~53 s for tick 1 in Weak A. The mild growth of first_tick from ~53 s → ~56 s at 1024+ GPUs (visible as the orange-segment growth in `weak_scaling_a_efficiency`) is the source of the small efficiency drop; we attribute it to the O(P) Alltoall handshake in step 3 dominating the otherwise constant per-rank work as P grows.

**Steady-state MPI is constant in weak scaling** (verified, not just predicted). Per-rank cross-rank edges (30) and MPI peer count (2, by 1D column-slab partitioning with periodic wraparound) are independent of total rank count, and SAGESim's per-tick exchange uses non-blocking Isend/Irecv only (no per-tick collective operations). Measured ghost-exchange wall time — the full SAGESim `data_prep` wrapper around `exchange_ghost_data()`, covering GPU pack, GPU-aware Slingshot Isend/Irecv/Waitall, and GPU unpack — stays at **~5 μs/rank ± 10%** across the full 8 → 2,048 GPU range, confirming the GPU-aware Slingshot-11 path scales as the methodology predicts. The small efficiency drop at 1024+ GPUs is therefore attributable to first-tick growth, not steady-state communication.

### 9.4 Synthesis: what GGap unlocks

Combining the three results: the **spatial capability** (Fig.~`weak_scaling_b_efficiency`, ~146× more sites than current CONUS at ~99% efficiency), the **temporal capability** (Fig.~`setup_amortization`, setup amortized for daily-or-finer resolution at centennial horizons), and the **time-to-solution capability** (Fig.~`strong_scaling_speedup`, 12.4× wall-time reduction for fixed problems) collectively argue that GGap's scaling unlocks a class of high-resolution, long-horizon, large-domain forest simulations that were previously infeasible with the current 1,400-site / 1-year-per-tick CONUS baseline. Fig.~`setup_phase_breakdown` provides the per-phase characterization that lets us project these capabilities forward: with setup time = f(per-rank work) and steady-state time pinned at ~45 ms/tick in the weak-scaling-B regime, scaling to ~204,800 sites at sub-yearly resolution for centennial horizons is a straightforward extension of the measurements reported here.

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
| `weak_scaling_b.csv` | weak_scaling_b_*, setup_*, §4, §9.1 | [8, 16, 32, 64, 128] | [256, 512, 1024, 2048] |
| `strong_scaling_b.csv` | strong_scaling_*, §5, §9.2 | [8, 16, 32, 64, 128, 256, 512] | none |
| `weak_scaling.csv` | weak_scaling_a_*, §S.1 | [8, 16, 32, 64, 128, 256, 512, 1024, 2048] | none |

**To refresh:** drop new rows into the relevant CSV and re-run the affected plot scripts in `papers/SC2026/src/`, then re-run `python papers/SC2026/src/generate_results_md.py` to regenerate this document.

---

## §S.1 Supplementary: Weak Scaling A (full-fidelity, communication-heavy)

This supplementary section reports the results of the **Weak Scaling A** configuration, which complements the main paper's Weak Scaling B (§4, §9.1) by exercising a different point in the design space:

- **10 sites per GPU** (10× fewer than Weak B) — low site density per rank
- **500 gaps × 1,000 trees** per site (5× higher per-site fidelity than Weak B / Strong) — matches canonical full-fidelity UVAFME production runs
- **37.5% cross-rank edges** — a high-communication-density regime that stress-tests the GPU-aware Slingshot-11 path (5× the cross-rank fraction of Weak B)
- Range: **8–2,048 GPUs** (all 9 rank counts complete)

Together with Weak B, these two configurations span the compute-vs-communication spectrum: Weak B is the high-site-density / large-total-agents / compute-bound end, Weak A is the comm-stress / full-fidelity / communication-bound end. They are deliberately paired so that no reviewer can claim the main result depends on a single design point.

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

At 2048 GPUs the model handles 10,250,260,480 agents at 96.2% parallel efficiency under the 37.5% cross-rank stress regime (Fig.~`weak_scaling_a_efficiency` — same combined design as `weak_scaling_b_efficiency`: stacked first-tick + steady-state simulation time with ideal-baseline reference and efficiency annotations). The sustained per-GPU throughput is **~119 M agent-updates/sec** — about half the Weak B per-GPU rate (~216 M upd/s/GPU), which is consistent with Weak A having half as many agents per GPU (5 M vs 10 M) under heavier per-site compute. Steady-state per-tick MPI ghost-exchange remains constant at ~5 μs/rank across the entire 8 → 2,048 GPU sweep, confirming that the GPU-aware Slingshot-11 path scales as the methodology predicts even at the high cross-rank fraction.

**Why Weak A is excluded from `setup_phase_breakdown`.** Weak A's per-site fidelity (500 gaps × 1,000 trees = ~500,001 agents/site) is 5× higher than Weak B / Strong (200 × 500 = ~100,001 agents/site). Setup time is dominated by per-rank Python agent construction, which is roughly linear in *total agents per rank*. Plotting Weak A on the same `sites/GPU` axis as Weak B + Strong would put its bar on a different curve (~5× higher at the same sites/GPU), breaking the visual claim that *all bars lie on one curve*. The agreement between Weak B and Strong on the merged figure is a clean per-fidelity result; Weak A's existence as a separate consistent measurement at a different fidelity is documented here for cross-validation.

Supplementary figures: `../figs/weak_scaling_a_efficiency.png` and `../figs/weak_scaling_a_throughput.png`.
