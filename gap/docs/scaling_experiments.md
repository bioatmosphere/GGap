# GGap Scaling Experiments for SC2026 Paper

## Overview

Scaling evidence for the claim: *GGap achieves weak scaling to thousands of GPUs on Frontier, enabling billion-tree CONUS simulations.*

**Key framing**: GGap's parallelism is **weak scaling by design**. Each site is an atomic work unit colocated on one GPU (colocation invariant). Strong scaling is not meaningful — you can't subdivide a site across GPUs. The paper should say: *"adding GPUs lets us simulate more of the Earth's surface at constant per-GPU cost."*

## System Facts

- **Dataset**: CONUS low-res = 1,424 sites, 235 species. High-res (planned) = ~50K sites.
- **Per-site config**: 500 gaps x 1000 tree slots = 500,501 agents/site
- **Frontier**: AMD MI250X, 8 GCDs per node (64 GB HBM each), Cray MPICH with GPU-aware MPI
- **Allocation**: LRN088, >200K node-hours
- **MPI traffic**: Only Site states (8 floats) + site_avail_spec (235 floats) cross ranks. Scales with sites, not trees.
- **Kernel**: Single fused kernel per tick, 11 grid-barrier-separated priorities, all ticks batched via `sync_workers_every_n_ticks`

### Memory Budget (per site, 500 gaps x 1000 trees, 235 species)

| Component | Size |
|-----------|------|
| params tensor (500,501 x 21 x 4B) | 42 MB |
| states tensor (500,501 x 16 x 4B) | 32 MB |
| Write buffers | ~74 MB |
| CSR neighbor structure (~1M edges) | 6 MB |
| Breed-local arrays | ~1.2 MB |
| **Per-site total** | **~155 MB** |
| Shared globals | ~10 MB |

MI250X GCD (64 GB): theoretical max ~400 sites/GCD. Practical limit TBD by Exp 0.

### Scale Targets

| Scale | Sites | Trees | GPUs (at S/GPU) |
|-------|-------|-------|-----------------|
| Low-res CONUS | 1,424 | 712M | 1424/S |
| Billion-tree | 2,000 | 1.0B | 2000/S |
| High-res CONUS | 50,000 | 25B | 50000/S |

---

## Experiment 0: Memory Capacity Test (RUN FIRST)

Determines max sites `S_max` per GCD. Sets `S` for all other experiments.

```bash
# Single GPU, incrementally add sites, 10 years each, stop at OOM
# Need to build: gap/run_memory_test.py (see Implementation below)
srun -N1 -n1 --gpus-per-node=1 --gpu-bind=closest \
    python gap/run_memory_test.py --prefix CONUS --num_gaps 500 --maxtrees 1000
```

Test points: S = 1, 5, 10, 25, 50, 100, 200, 300

Measure: `cupy.get_default_memory_pool().used_bytes()`, wall time per tick, OOM point.

---

## Experiment 1: Single-GPU Baseline

Baseline time-per-tick vs number of sites on one GPU.

```bash
# For each S in {1, 5, 10, S_max}:
srun -N1 -n1 --gpus-per-node=1 --gpu-bind=closest \
    python gap/run_scaling_test.py \
        --prefix CONUS --sites_per_gpu $S \
        --num_gaps 500 --maxtrees 1000 --years 100 \
        --no_output --timing_output results/single_gpu_S${S}.json
```

Measure: wall time/tick, per-priority breakdown (`_verbose_timing`), GPU memory, throughput (tree-years/sec).

---

## Experiment 2: Weak Scaling (CORE RESULT)

Fixed S sites per GPU, increase GPU count.

```bash
# For each N_GPUS in {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096}:
NODES=$(( (N_GPUS + 7) / 8 ))
srun -N${NODES} -n${N_GPUS} --gpus-per-node=8 --gpu-bind=closest \
    python gap/run_scaling_test.py \
        --prefix CONUS --sites_per_gpu $S \
        --num_gaps 500 --maxtrees 1000 --years 100 \
        --no_output --timing_output results/weak_N${N_GPUS}.json
```

Note: 1424 CONUS sites caps GPUs at 1424/S. Beyond that, replicate sites with offset IDs or use 50K high-res data.

Measure:
- Time per tick: `T_tick(N)`
- Weak scaling efficiency: `E = T_tick(1) / T_tick(N)`
- Breakdown: GPU kernel time vs MPI ghost exchange time vs data-prep

Plots:
1. Time-per-tick vs GPUs (log-log) with ideal horizontal line
2. Stacked bar: compute vs communication at each GPU count
3. Efficiency curve

---

## Experiment 3: Communication Analysis

Quantify MPI overhead. Show colocation keeps communication negligible.

### 3a: Sync frequency sweep (64 GPUs, 1 site/GPU)

```bash
# For each SYNC in {1, 2, 5, 10}:
srun -N8 -n64 --gpus-per-node=8 --gpu-bind=closest \
    python gap/run_scaling_test.py \
        --prefix CONUS --sites_per_gpu 1 \
        --num_gaps 500 --maxtrees 1000 --years 50 \
        --sync_every $SYNC --no_output \
        --timing_output results/comm_sync${SYNC}.json
```

### 3b: Ghost data volume

Per connected site pair per tick: Site states (32B) + site_avail_spec (940B) = **~1 KB**. Report total bytes vs GPU count.

---

## Experiment 4: CONUS Demonstration (HEADLINE)

### 4a: Low-res CONUS (712M trees)
```bash
srun -N$((1424/S/8 + 1)) -n$((1424/S)) --gpus-per-node=8 --gpu-bind=closest \
    python gap/run_scaling_test.py \
        --prefix CONUS --all_sites \
        --num_gaps 500 --maxtrees 1000 --years 500 \
        --report_interval 50 --no_tree_data \
        --timing_output results/conus_lowres.json
```

### 4b: Billion-tree run (2000 sites, replicated)
Replicate 576 CONUS sites with offset IDs to reach 2000 total.

### 4c: High-res CONUS (50K sites, 25B trees)
Use high-res dataset when ready. ~50000/S GPUs.

Measure: total wall-clock for 500 years, time/tick, memory/GPU, forest composition maps at years 100/250/500.

---

## Experiment 5: Per-Priority Profiling

Identify bottleneck priorities (expect P1 soil ~40%, P3/P7 tree growth ~35%).

```bash
# 1 GPU, 1 site, verbose timing
srun -N1 -n1 --gpus-per-node=1 --gpu-bind=closest \
    python gap/run_scaling_test.py \
        --prefix CONUS --sites_per_gpu 1 \
        --num_gaps 500 --maxtrees 1000 --years 50 \
        --no_output --verbose_timing \
        --timing_output results/profile_1site.json
```

Also run with `rocprof` for per-priority GPU metrics if `_verbose_timing` lacks priority-level detail.

---

## Implementation: Scripts to Build

### 1. `gap/run_memory_test.py`

Single GPU, incrementally adds sites until OOM.

```python
"""
Memory capacity test for GGap scaling.
Incrementally loads sites on 1 GPU, reports memory and timing.
Catches OOM gracefully.

Usage:
    srun -N1 -n1 python gap/run_memory_test.py --prefix CONUS --num_gaps 500 --maxtrees 1000
"""
# Key logic:
# for S in [1, 5, 10, 25, 50, 100, 200, 300, 400]:
#     try:
#         model = GAPModel()
#         model.load_globals(prefix=args.prefix)
#         model.partition_sites(site_ids[:S])
#         for sid in site_ids[:S]:
#             model.initialize_site_with_gaps(sid, args.num_gaps, args.maxtrees)
#         model.connect_sites()
#         model.register_breed_local_arrays()
#         model.setup(use_gpu=True)
#         mem_after_setup = cupy.get_default_memory_pool().used_bytes()
#         model.simulate(ticks=1, sync_workers_every_n_ticks=1)
#         mem_after_tick = cupy.get_default_memory_pool().used_bytes()
#         time_per_tick = ...
#         print(f"S={S}: {total_agents} agents, setup={mem_after_setup/1e9:.2f}GB, "
#               f"tick={mem_after_tick/1e9:.2f}GB, time={time_per_tick:.3f}s")
#     except cupy.cuda.memory.OutOfMemoryError:
#         print(f"S={S}: OOM! Max feasible S = {prev_S}")
#         break
```

### 2. `gap/run_scaling_test.py`

Benchmark runner wrapping run_multi_site.py logic.

CLI args:
- `--prefix CONUS` — dataset prefix
- `--sites_per_gpu S` — sites assigned to each rank
- `--all_sites` — use all sites in CSV (alternative to sites_per_gpu)
- `--num_gaps 500`, `--maxtrees 1000`
- `--years 100`
- `--sync_every N` — sync_workers_every_n_ticks (default 1)
- `--no_output` — skip CSV writing
- `--verbose_timing` — enable SAGESim verbose timing
- `--timing_output PATH` — write per-tick JSON timing
- `--report_interval 10`
- `--seed 42`

Key logic:
- Auto-select site IDs: rank gets sites `[rank*S, rank*S+1, ..., rank*S+S-1]` from CONUS CSV
- After simulation: dump JSON with per-tick `{tick, kernel_time, mpi_time, total_time}` + summary stats
- Report GPU memory usage
- Print summary table: avg/min/max time per tick, total agents, memory

### 3. `gap/run_multi_site.py` modifications

- Add `--no_output` flag: skip all CSV writing and collection
- Add `--timing_json PATH`: dump per-batch timing as JSON array
- Add `--verbose_timing`: set `model._verbose_timing = True`
- Add `--sync_every N`: control sync_workers_every_n_ticks

### 4. `scaling_jobs/` — Frontier Slurm templates

Common header:
```bash
#!/bin/bash
#SBATCH -A LRN088
#SBATCH -p batch
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

module load PrgEnv-gnu rocm craype-accel-amd-gfx90a
export MPICH_GPU_SUPPORT_ENABLED=1

cd $SLURM_SUBMIT_DIR
```

Create: `memory_test.sbatch`, `weak_scaling.sbatch` (parameterized), `comm_analysis.sbatch`, `conus_demo.sbatch`

### 5. `gap/plot_scaling_results.py`

Reads JSON timing files, generates publication plots:
- Weak scaling efficiency curve (log-log)
- Compute vs communication stacked bar
- Per-priority time breakdown (pie/bar)
- Memory vs sites-per-GPU
- CONUS composition maps (from genus_data.csv output)

---

## Execution Order

1. **Exp 0** — memory test → determines S
2. **Exp 5** — profiling → find if optimization needed
3. **Exp 1** — single-GPU baseline → needed for efficiency calc
4. **Exp 2** — weak scaling → core paper result
5. **Exp 3** — communication analysis → supplementary
6. **Exp 4** — CONUS demo → headline figure

## Verification Checklist

- [ ] Memory test completes, S_max determined
- [ ] Determinism: 1-GPU vs 8-GPU same 8 sites with same seed → identical output
- [ ] Timing sanity: ~0.5-1.0s/tick for 1 site, 500 gaps x 1000 trees
- [ ] Weak scaling efficiency >95% at 8 GPUs (intra-node)
- [ ] CONUS demo completes 500 years without crash
