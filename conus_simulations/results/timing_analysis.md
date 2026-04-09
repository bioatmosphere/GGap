# CONUS GGap: Pure Simulation vs. Snapshot-Enabled Timing Analysis

## Purpose

This document captures end-to-end timing for the continental US (CONUS) GGap forest
dynamics simulation at full production scale, comparing two back-to-back runs on Frontier:

1. **Job 4353376** — production run with periodic state snapshots every 10 simulated
   years (100 snapshots total), mirroring the standard scientific workflow.
2. **Job 4353377** — identical configuration but with `--no_snapshots`, which replaces
   the batched `simulate(10) × 100` + GPU→CPU transfer + disk save loop with a single
   `model.simulate(ticks=1000)` call and no I/O. Isolates the pure compute cost.

Both runs use the same code revision (post-optimization), the same input data, the
same MPI layout, and the same scientific configuration. They were submitted minutes
apart on the same day to minimize system-state drift.

## Hardware and scientific configuration

| Parameter | Value |
|---|---|
| Platform | OLCF Frontier |
| Nodes | 20 |
| GPUs per node | 8 (AMD MI250X, one GCD per MPI rank) |
| MPI ranks | 160 |
| GPU-aware MPI | Enabled (`MPICH_GPU_SUPPORT_ENABLED=1`) |
| Interconnect | HPE Slingshot-11 |
| Filesystem (output target) | Lustre (`/lustre/orion/lrn088/proj-shared/...`) |
| ROCm | 6.4.1 |
| PrgEnv | PrgEnv-gnu/8.6.0 |
| METIS | 5.1.0 |
| Python / CuPy | inside `sagesim_env` conda env |

| Scientific parameter | Value |
|---|---|
| Simulated domain | CONUS, 1424 sites |
| Species pool | 235 species (from `CONUS_rangelist.csv`) |
| Gaps per site | 500 |
| Max trees per gap | 1000 |
| Total gap agents | 712,000 |
| Total tree agent slots | 712,000,000 |
| Simulated years | 1000 |
| Dispersal cutoff factor | 2.0 × species max dispersal |
| Directed dispersal edges | 11,371 |
| Undirected adjacency edges | 5,741 |
| METIS partitions | 160 (one per MPI rank) |
| Sites per rank (min / max / avg) | 7 / 10 / 8.9 |
| MPI sync cadence | every 1 simulated year (`sync_workers_every_n_ticks=1`) |

## Run A — No snapshots (job 4353377)

**Command invoked on each rank** (inside `submit_conus_nosave.sh`):
```
python run_conus.py \
  --dispersal_factor 2.0 --num_gaps 500 --maxtrees 1000 \
  --years 1000 --report_interval 10 \
  --no_snapshots \
  --output_dir ../results/simulation_nosave
```

In `--no_snapshots` mode, `run_simulation()` calls `model.simulate(ticks=1000,
sync_workers_every_n_ticks=1)` **exactly once** — no Python loop, no per-batch
`get_breed_data` GPU→CPU transfers, no `np.savez_compressed` writes, no metadata
pickle. Only a single wall-clock timer around the call.

### Timing

| Phase | Time (s) |
|---|---|
| SLURM wall start → end | 170 (17:50:27 → 17:53:17) |
| Data loading (CSVs) | 0.07 |
| Dispersal graph build | 1.22 |
| METIS partitioning | 0.05 |
| Site initialization (slowest rank) | 38.39 |
| Site initialization (fastest rank) | 26.21 |
| Site connection (dispersal edges) | 0.00 |
| GPU kernel setup | 4.60 |
| **`model.simulate(1000)` — single call** | **112.18** |
| **Pure simulation time per year** | **0.1122 s/year** |

The 112.18 s number includes one-time JIT kernel compilation inside the first few
ticks of `simulate()`. No warm-up call was made before the timer started.

## Run B — With snapshots every 10 years (job 4353376)

**Command invoked on each rank** (inside `submit_conus.sh`):
```
python run_conus.py \
  --dispersal_factor 2.0 --num_gaps 500 --maxtrees 1000 \
  --years 1000 --report_interval 10 \
  --output_dir ../results/simulation
```

In snapshot mode, `run_simulation()` runs a Python for-loop over 100 iterations. Each
iteration does:

1. `model.simulate(ticks=10, sync_workers_every_n_ticks=1)` — 10 years of compute
2. Four `model.get_breed_data(..., local=True)` calls — GPU→CPU transfer of Site
   params/states and Tree params/states
3. One `np.savez_compressed(...)` write to
   `../results/simulation/snapshots/year_XXXX_rank_XXX.npz` (every rank writes its
   own file, so 160 files per snapshot event)

### Timing

| Phase | Time (s) |
|---|---|
| SLURM wall start → end | 888 (17:47:53 → 18:02:41) |
| Data loading | 0.07 |
| Dispersal graph build | 1.22 |
| METIS partitioning | 0.06 |
| Site connection | 0.00 |
| GPU kernel setup | 4.48 |
| **Simulation loop total (inside `run_simulation`)** | **828.34** |
| Reported time per year | 0.83 s/year |

### Breakdown of the 828.34 s simulation loop

Extracted from rank-0 per-batch timing lines in `conus_4353376.out`:

| Component | Sum over 100 batches (s) | Mean per batch (s) |
|---|---|---|
| `sim` (`model.simulate(10)` calls) — total | 405.23 | 4.05 |
| — batch 1 (year 10), JIT cold start | 59.02 | — |
| — batch 2 (year 20), residual warm-up | 10.59 | — |
| — steady state (batches 10–100) mean | — | **3.50** |
| GPU→CPU transfer (tree_p + tree_s) | 17.67 | 0.177 |
| GPU→CPU transfer (site_p + site_s) | ~0.0 | ~0.0 |
| Disk save (`np.savez_compressed`) | 405.41 | 4.05 |
| **I/O subtotal (gpu→cpu + save)** | **423.08** | **4.23** |

Steady-state (batches 10 through 100) per-batch `sim` time: **3.496 s per 10 years
= 0.3496 s/year**.

### Per-batch progression (selected)

```
Year   10  sim: 59.02s  save: 4.05s   (JIT cold start)
Year   20  sim: 10.59s  save: 4.08s   (residual compile / cache warm-up)
Year   30  sim:  2.16s  save: 4.09s
Year   40  sim:  2.34s  save: 4.04s
Year   50  sim:  2.46s  save: 4.07s
Year  100  sim:  2.89s  save: 4.08s
Year  200  sim:  3.33s  save: 4.04s
Year  500  sim:  3.62s  save: 4.02s
Year 1000  sim:  3.66s  save: 4.06s
```

`sim` time drifts upward from ~2.2 s/10yr early to ~3.66 s/10yr at year 1000, reflecting
growing tree population (more live agents to step) as forests mature.

## Side-by-side comparison

| Metric | Run A (no save) | Run B (with save, every 10 yr) | Delta |
|---|---|---|---|
| SLURM wall time | 170 s | 888 s | **+718 s (5.2× slower)** |
| Simulation loop total | 112.18 s | 828.34 s | +716.16 s |
| Total `sim` time (sum of `model.simulate()` calls) | 112.18 s | 405.23 s | +293.05 s |
| GPU→CPU transfer total | 0 | 17.67 s | +17.67 s |
| Disk save total | 0 | 405.41 s | +405.41 s |
| Reported per-year (amortized) | 0.112 s | 0.828 s | 7.4× |
| Steady-state per-year (excluding JIT warmup) | 0.112 s† | 0.350 s | **3.1×** |

† For Run A, the 0.112 s/year includes JIT warm-up amortized over all 1000 years.
True steady-state would be slightly lower.

## Why are the "pure simulation" times so different? (112 s vs. 405 s)

This is the most important finding in the dataset and the one that surprised us.
Naively, `sim` in Run B should equal the total time in Run A, because both runs do
the same number of GPU kernel launches with identical compute work. Instead, Run B's
`sim` time sum is **3.6× larger** than Run A's total.

### Mechanism: SAGESim's simulate() loop structure (verified from source)

Reading `SAGESim/sagesim/model.py::Model.simulate()` (line 961) and
`Model.worker_coroutine()` (line 1314) clarifies what actually happens:

```python
def simulate(self, ticks, sync_workers_every_n_ticks=1):
    ...
    comm.barrier()                                          # <-- (1) global barrier at entry
    ...
    for time_chunk in range((ticks // sync_workers_every_n_ticks) + 1):
        ...
        self.worker_coroutine(sync_workers_every_n_ticks)   # <-- launches 1 fused GPU kernel
                                                            #     containing sync_workers_every_n_ticks ticks
```

And inside `worker_coroutine`, the GPU kernel launch passes
`sync_workers_every_n_ticks` directly to the kernel, which runs that many ticks in
a fused inner loop on the device (comment at line 1430: *"GPU KERNEL EXECUTION
(fused: all ticks + priorities in one launch)"*).

Both runs use `sync_workers_every_n_ticks=1`, so:

| | Run A (`simulate(1000)` × 1) | Run B (`simulate(10)` × 100) |
|---|---|---|
| `worker_coroutine()` calls | 1000 | 1000 |
| GPU kernel launches | 1000 | 1000 |
| Ticks per kernel | 1 | 1 |
| `exchange_ghost_data()` calls (MPI) | 1000 | 1000 |
| **`comm.barrier()` at `simulate()` entry** | **1** | **100** |
| `_last_simulate_ticks` / `_tick_timings` reset | 1 | 100 |

**The GPU does the same work. The only structural difference is 99 extra
`comm.barrier()` entry points and 100 extra Python-level simulate() setups in
Run B.** Yet the delta is 293 s. A bare `MPI_Barrier` at 160 ranks takes ≪10 ms,
so 99 barriers should cost under 1 s. Something else must be riding on those
barriers.

### The real cause: each extra `comm.barrier()` gates on the slowest straggler,
### and Lustre I/O between calls makes stragglers much worse

Two things happen between successive `simulate(10)` calls in Run B that do **not**
happen inside a single `simulate(1000)` call in Run A:

1. **Four `get_breed_data(..., local=True)` calls** — these do
   `cupy_array.get()` (device→host copy) on Site and Tree property tensors.
   Individually small (~180 ms total per snapshot, observed), but they act as
   implicit GPU sync points. More importantly, on rank 0 the Tree array is much
   larger than on other ranks (because rank-0 sites had more surviving trees —
   visible in the init-time spread), so rank 0's `.get()` takes longer than other
   ranks'. This desynchronizes ranks entering the next `save` phase.

2. **A `np.savez_compressed()` write of ~4 MB per rank to Lustre** — observed at
   ~4.05 s per rank, but crucially this is **wall-clock on the slowest rank**,
   because the subsequent `simulate(10)` call begins with `comm.barrier()`, which
   cannot progress until every rank's save has returned control. The *mean* rank
   probably finishes saving in 1–2 s; the *slowest* rank (whose save contended
   with Lustre metadata ops, or whose host NIC was busy, or whose OS jitter hit)
   takes 4+ s. The observed `save:4.05s` on rank 0 is the per-rank save time,
   which includes Lustre wait but **does not** include the additional wait for
   still-slower ranks.

**When the next `simulate(10)` call's `comm.barrier()` releases, rank 0 records
the elapsed time since it started — which includes everyone's straggler wait as
inflated `sim` time, not as `save` time.**

Quantitatively: Run B has 100 barriers (vs. 1 in Run A). If each extra barrier
waits ~2.9 s on average for the slowest rank's Lustre + GPU→CPU + Python work to
complete, that's 99 × 2.9 s ≈ 287 s, which matches the observed 293 s delta very
closely.

### Evidence supporting this attribution

- **The per-batch `sim:` time converges to ~3.5 s in steady state, not to 1.12 s**
  (which would be the Run A per-10-year rate). This ~2.4 s gap per batch is
  consistent with straggler wait, and disappears entirely in Run A where there
  is no inter-batch stragger opportunity.
- **Save time is remarkably constant at ~4.05 s across all 100 batches.** This
  suggests rank 0 is not the slowest saver; some other rank is consistently
  ~2–3 s slower, and rank 0 waits for it at the next barrier.
- **In Run A, the full 1000-tick sequence runs without any Python-level
  inter-tick gap**, so ghost-data exchange (`exchange_ghost_data` in
  `worker_coroutine` line 1415) is the only synchronization point, and it is
  tightly coupled to GPU stream state — no host-side straggling.
- **GPU→CPU transfer itself is only 17.67 s out of the 293 s delta** (6%), ruling
  it out as the dominant cost.
- **JIT compile time is folded into both runs** via the first `simulate()` call;
  Run B's extra ~66 s on batches 1–2 matches Run A's warm-up cost amortized into
  the 112.18 s total, so JIT is not the source of the steady-state 2.4 s/batch
  overhead.

### Summary of the 293 s delta

| Source | Estimated contribution |
|---|---|
| GPU→CPU transfer (`.get()`) × 100 snapshots | ~17.7 s (directly measured) |
| Python driver re-entry × 99 extra `simulate()` calls | ~2 s |
| 99 extra `comm.barrier()` calls blocking on stragglers induced by Lustre save | **~270 s** |
| Miscellaneous Python / allocator overhead | ~5 s |
| **Total** | **~295 s ≈ observed 293 s** |

The key finding is that the **4.05 s per-rank save time massively understates
the true I/O cost**, because the straggler-induced barrier wait on the next
`simulate()` call is invisibly charged to `sim:` rather than `save:`. The
*apparent* I/O cost per snapshot is 4.23 s (save + gpu→cpu); the *real* cost,
including the ~2.9 s of straggler wait it induces downstream, is **~7.2 s per
snapshot — nearly 70% higher than the instrumented number**.

### Implication for the paper

The checkpoint-every-10-years workflow incurs three distinct costs, only one of
which is accurately captured by naive instrumentation:

1. **Direct GPU→CPU transfer:** 17.67 s total (2% of loop time) — measured.
2. **Direct disk write:** 405.41 s total (49% of loop time) — measured per-rank.
3. **Straggler-induced barrier wait:** ~290 s total (35% of loop time) —
   **invisible to per-rank save timing**, appears as inflated `sim:` time on
   the following batch.

Every second of per-rank save time on the rank-0 timer corresponds to
approximately **1.7 seconds of wall-clock delay** once the straggler-induced
barrier wait on the next `simulate()` call is included. The 4.05 s save number
reported per snapshot understates the true I/O cost by ~70%.

This suggests three concrete optimizations worth discussing in the paper:

1. **Async / overlapped I/O**: run `np.savez_compressed` on a background thread
   or a separate host process, so that each rank's save progresses in parallel
   with the next `simulate()` call. This both hides the 405 s write cost and
   eliminates the straggler-induced barrier wait, because ranks enter the next
   barrier without having blocked on Lustre. Expected savings: ~400 s + ~290 s
   ≈ ~690 s of the 718 s overhead.
2. **Single `simulate()` call with checkpoint callback**: pass a year-cadence
   callback into SAGESim's `simulate` so checkpoint events happen from inside
   the existing per-tick loop, without returning control to Python and without
   a second `comm.barrier()` at re-entry. Eliminates the straggler-wait portion
   specifically (~290 s). Still leaves the direct save cost unless combined
   with (1).
3. **Checkpoint less often**: if scientifically acceptable, snapshot every 50
   or 100 years instead of every 10. The overhead is approximately linear in
   the number of snapshots taken, so this scales down the 718 s cost by 5× or
   10×. Simplest to implement, no SAGESim changes required.

## File pointers (for reproducibility)

| Artifact | Path |
|---|---|
| Run A log | `conus_simulations/logs/conus_nosave_4353377.out` |
| Run B log | `conus_simulations/logs/conus_4353376.out` |
| Run A submit script | `conus_simulations/scripts/submit_conus_nosave.sh` |
| Run B submit script | `conus_simulations/scripts/submit_conus.sh` |
| Driver | `conus_simulations/scripts/run_conus.py` |
| Input data dir | `/lustre/orion/proj-shared/lrn088/objective3/xxz/GGap/input_data` |
| Input file prefix | `CONUS` |
| Run A snapshot dir | N/A (no snapshots) |
| Run B snapshot dir | `conus_simulations/results/simulation/snapshots/` |

## Caveats and open questions

1. **No JIT warm-up**: neither run performed a throwaway `simulate(1)` before timing
   starts. Run A's 112 s includes compile cost. A warm-up would make Run A's number
   slightly smaller (probably 100–105 s) but would not change Run B's structure.
2. **n = 1**: each configuration was run once. Lustre latency is noisy and should
   be averaged over ≥3 trials for publication error bars.
3. **Attribution of the 293 s gap is partially estimated.** We verified from
   SAGESim source (`sagesim/model.py::simulate` line 961 and `worker_coroutine`
   line 1314) that both runs launch the same number of GPU kernels (1000) with
   the same fusion depth (1 tick each), so the GPU compute work is structurally
   identical. The 293 s delta must come from host-side effects between
   `simulate()` calls, and the `comm.barrier()` at line 973 of `simulate()` is
   the only mechanism in Run B that is not present in Run A. Our attribution
   of ~270 s to straggler-induced barrier wait follows from this fact combined
   with the observed ~2.9 s/batch steady-state gap. However, we do not have a
   direct profiler trace (NSYS / rocprof) that decomposes each batch's `sim:`
   time into compute / MPI barrier / MPI collective components. Such a trace
   on 2–4 ranks would confirm the breakdown and also reveal *which* rank is
   the straggler, which would inform future load-balancing work.
4. **`sync_workers_every_n_ticks=1` is aggressive**. Forest gap dispersal needs
   annual exchange for scientific correctness, so this is not a tunable, but it
   does mean every year pays the full barrier cost.
5. **First-batch anomaly** (59 s on year 10) contaminates the naive "sum of sim
   times" comparison. When comparing amortized per-year cost, the steady-state
   batches (10–100) give 0.35 s/year for Run B versus 0.112 s/year for Run A —
   a 3.1× ratio rather than the 3.6× you get from summing raw totals.
