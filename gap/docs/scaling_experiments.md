# GGap Scaling Experiments for SC2026 Paper

## Overview

Scaling evidence for the claim: *GGap achieves weak scaling to thousands of GPUs on Frontier, enabling billion-tree CONUS simulations.*

**Key framing**: GGap's parallelism is **weak scaling by design**. Each site is an atomic work unit colocated on one GPU (colocation invariant). Strong scaling is not meaningful — you can't subdivide a site across GPUs. The paper should say: *"adding GPUs lets us simulate more of the Earth's surface at constant per-GPU cost."*

## Development Phases

### Phase 1: Small-Scale Tests (CURRENT - Debug Queue, <20 nodes, <160 GPUs)
- **Goal:** Rapid iteration on scaling infrastructure
- **Sites:** ~500-1000 synthetic sites with realistic connectivity
- **Connectivity:** Fixed average degree (e.g., 6 neighbors/site) for weak scaling
- **Partition:** METIS on synthetic connectivity graph
- **Focus:** Memory limits, weak scaling baseline, MPI overhead measurement

### Phase 2: Medium-Scale Tests (Batch Queue, 20-100 nodes)
- **Goal:** Validate scaling to ~800 GPUs
- **Sites:** Up to 1,424 real CONUS sites
- **Connectivity:** Real CONUS site connections based on species dispersal
- **Focus:** Realistic workload validation

### Phase 3: Large-Scale Tests (FUTURE - Production Runs)
- **Goal:** 50k sites, thousands of GPUs
- **Sites:** High-resolution dataset (50k sites)
- **Connectivity:** Streaming/chunked graph building (can't fit 50k×50k in memory)
- **Partition:** Distributed METIS or ParMETIS
- **TODO:**
  - Implement chunked distance matrix computation
  - Use ParMETIS for distributed graph partitioning
  - Optimize ghost data exchange for high-degree graphs

## Critical Implementation Requirements

### Spatial Partitioning (REQUIRED for Scaling)

**Problem:** Current `partition_sites()` uses round-robin (`sid % num_workers`), which distributes spatially adjacent sites across different ranks. This maximizes MPI cross-rank communication for seed dispersal.

**Solution:** Spatial partitioning to minimize edge cuts in the site connectivity graph.

**Partitioning Strategies:**
1. **METIS graph partitioning** (Recommended) - Minimizes edge cuts using site connectivity graph from lat/lon + species dispersal distances
2. **KD-tree spatial decomposition** (Fallback) - Recursive median splits on lat/lon
3. **Hilbert space-filling curve** (Fallback) - Maps 2D lat/lon to 1D curve preserving locality

**Implementation Location:** `gap/gap_model.py:partition_sites()` - Add `strategy='spatial_metis'` parameter

**Site Connectivity for Weak Scaling:**
- **Key principle:** Fix average connections per site (NOT total connections)
- Each site connects to ~6 neighbors (estimated from CONUS)
- As we add more GPUs, total edges grow linearly: `edges = sites × 6 / 2`
- METIS partition keeps most connections intra-rank (minimize cross-rank MPI)
- For Phase 1 tests: Generate synthetic sites with exactly K neighbors each (K=6)

### Profiler Selection (Frontier)

**Available tools:** rocprofv3, pat_run+pat_report, HPCToolkit, ROCm Systems Profiler, Score-P+Vampir

**Recommended:** **rocprofv3 only** - Per-kernel timing, minimal overhead, native AMD MI250X support

**NOT needed:** pat_run (too heavyweight), HPCToolkit (overkill), Score-P (trace-based unnecessary), ROCm Systems Profiler (system-wide not kernel-specific)

**Usage:** Use SAGESim `_verbose_timing` for comprehensive timing in weak scaling tests (Exp 2).

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
srun -N1 -n1 --gpus-per-node=1 --gpu-bind=closest \
    python gap/find_memory_limit.py --prefix CONUS --num_gaps 500 --maxtrees 1000
```

Test points: S = 1, 5, 10, 25, 50, 100, 200, 300

Measure: `cupy.get_default_memory_pool().used_bytes()`, wall time per tick, OOM point.

**Script:** `gap/find_memory_limit.py` - Following superneuroabm pattern, progressively tests larger site counts until OOM.

---

## Experiment 0b: GPU Occupancy Tuning (After Exp 0)

**Goal:** Find optimal `max_blocks_per_sm` for best performance at chosen site count.

**Dependencies:** Requires `S` (sites per GPU) from Experiment 0. Use S = 80% of S_max for safety margin.

**Background:** SAGESim limits concurrent GPU blocks via `max_blocks_per_sm` parameter (default: 2):
- AMD MI250X GCD: 110 compute units (CUs)
- `max_grid_blocks = max_blocks_per_sm * 110`
- Default (2): Only 220 blocks concurrent → 28,160 threads
- Higher values: More parallelism, but risk register pressure/occupancy limits
- **NOTE:** This parameter affects runtime performance, NOT memory allocation

**Method:**
```bash
# Test different max_blocks_per_sm values: 2, 4, 8, 16, 32
# Use S = 80% of S_max from Exp 0

for BLOCKS_PER_SM in 2 4 8 16 32; do
  srun -N1 -n1 --gpus-per-node=1 --gpu-bind=closest \
    python gap/tune_occupancy.py \
      --sites $S --num_gaps 500 --maxtrees 1000 \
      --max_blocks_per_sm $BLOCKS_PER_SM --years 10 \
      --csv results/occupancy_bsm${BLOCKS_PER_SM}.csv
done
```

**Script:** `gap/tune_occupancy.py` - Similar to find_memory_limit.py but:
- Fixed site count from Exp 0
- Vary `max_blocks_per_sm` parameter: `model._max_blocks_per_sm = args.max_blocks_per_sm`
- Run 10 ticks each for stable timing
- Track: time per tick, GPU memory usage
- **CRITICAL: Re-verify memory usage doesn't exceed limits at each setting**

**Expected results:**
- Low (2-4): Underutilizes GPU, many waves, slower
- Optimal (4-16?): Best balance of occupancy vs register pressure
- Too high (32+): Register spilling or occupancy limits, slower or error

**Verification:** After finding optimal value, run one more test with full S_max sites to confirm memory usage is still within 64GB limit.

**Output:** Recommended `max_blocks_per_sm` value for all subsequent experiments.

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

**Script:** `gap/weak_scaling_conus.py` - Following superneuroabm pattern with CSV output

```bash
# Automated submission for N nodes
./scaling_jobs/submit_weak_scaling.sh <N_NODES>

# Or manual run:
NODES=<N>
srun -N$NODES -n$((NODES*8)) -c7 --ntasks-per-gpu=1 --gpu-bind=closest \
    python gap/weak_scaling_conus.py \
        --prefix CONUS --sites-per-gpu $S \
        --num_gaps 500 --maxtrees 1000 --years 100 \
        --partition-strategy spatial_metis \
        --csv results/weak_scaling_N${NODES}nodes_${SLURM_JOB_ID}.csv
```

**Key implementation details:**
- Use spatial partitioning (METIS) to minimize cross-rank site connections
- Each rank gets spatially contiguous sites
- CSV output: `job_id, nodes, workers, sites, sites_per_gpu, trees, edge_cuts, network_load_time, model_creation_time, gpu_setup_time, simulation_time, mpi_time, total_time`
- `edge_cuts` from METIS partition indicates MPI communication volume

Note: 1424 CONUS sites caps GPUs at 1424/S. Beyond that, replicate sites with offset IDs or use 50K high-res data.

Measure:
- Time per tick: `T_tick(N)`
- Weak scaling efficiency: `E = T_tick(1) / T_tick(N)`
- Breakdown: GPU kernel time vs MPI ghost exchange time vs data-prep
- Edge cuts: Number of cross-rank site connections

Plots:
1. Time-per-tick vs GPUs (log-log) with ideal horizontal line
2. Stacked bar: compute vs communication at each GPU count
3. Efficiency curve
4. Edge cuts vs GPU count (communication overhead)

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


## Implementation: Scripts to Build

### Phase 1: Spatial Partitioning (4-6 hours)

**File:** `gap/gap_model.py` - Modify `partition_sites()` method

Add spatial partitioning strategies (replacing round-robin):

```python
def partition_sites(self, site_ids, strategy='spatial_metis'):
    """
    Partition sites across MPI ranks.

    Args:
        site_ids: List of site IDs to partition
        strategy: 'spatial_metis' | 'spatial_kd' | 'spatial_hilbert' | 'round_robin'

    spatial_metis: Graph partitioning (METIS) - minimizes edge cuts (RECOMMENDED)
    spatial_kd: KD-tree decomposition on lat/lon (fallback if no METIS)
    spatial_hilbert: Hilbert space-filling curve (fallback)
    round_robin: Original (BAD - maximizes cross-rank communication)
    """
```

**Implementation:**
1. Check if `pymetis` available: `import pymetis`
2. Build site connectivity graph from lat/lon + species dispersal distances
3. METIS partition: `n_cuts, partition = pymetis.part_graph(num_workers, adjacency)`
4. Report edge cuts: print(f"METIS: {n_cuts} cross-rank edges across {num_workers} ranks")

**Fallback (if METIS unavailable):**
- KD-tree: Recursive median split on lat/lon
- Hilbert curve: Map 2D to 1D preserving locality

**New script:** `gap/analyze_site_connectivity.py` - Visualize partition quality

### Phase 2: SAGESim Timing API (1-2 hours)

**File:** `/lustre/orion/proj-shared/lrn088/objective3/xxz/SAGESim/sagesim/model.py`

Add 3 methods to Model class:

```python
def get_timing_data(self) -> list:
    """Return collected per-tick timing data."""
    return self._tick_timings

def get_timing_summary(self) -> dict:
    """Compute timing statistics."""
    if not self._tick_timings:
        return {}
    times = [t['total'] for t in self._tick_timings]
    return {
        'num_ticks': len(times),
        'total_time': sum(times),
        'mean_time_per_tick': sum(times) / len(times),
        'min_time_per_tick': min(times),
        'max_time_per_tick': max(times),
    }

def export_timing_json(self, filepath: str) -> None:
    """Export timing data to JSON."""
    import json
    with open(filepath, 'w') as f:
        json.dump({
            'tick_timings': self._tick_timings,
            'summary': self.get_timing_summary()
        }, f, indent=2)
```

### Phase 3: GGap Scaling Scripts (6-8 hours)

### 3a. `gap/find_memory_limit.py`

Single GPU, incrementally adds sites until OOM. Following superneuroabm pattern.

**Key features:**
- Test S = [1, 5, 10, 25, 50, 100, 200, 300]
- Track: `cupy.get_default_memory_pool().used_bytes()`
- Catch OOM gracefully
- Report: max feasible S_max

### 3b. `gap/weak_scaling_conus.py`

Main weak scaling runner. Following superneuroabm CSV output pattern.

**CLI:**
```python
parser.add_argument('--sites-per-gpu', type=int, default=10)
parser.add_argument('--num-gaps', type=int, default=500)
parser.add_argument('--maxtrees', type=int, default=1000)
parser.add_argument('--years', type=int, default=100)
parser.add_argument('--partition-strategy', default='spatial_metis')
parser.add_argument('--csv', type=str, help='CSV file for timing results')
parser.add_argument('--seed', type=int, default=42)
```

**Key logic:**
```python
# Use spatial partitioning
model = GAPModel()
model.load_globals(prefix='CONUS')

# Auto-select sites
total_sites = args.sites_per_gpu * size
site_ids = list(range(min(total_sites, 1424)))

# CRITICAL: Spatial partitioning (not round-robin)
model.partition_sites(site_ids, strategy=args.partition_strategy)

# Each rank initializes only its local sites
for site_id in site_ids:
    if model._site_partition[site_id] == rank:
        model.initialize_site_with_gaps(site_id, args.num_gaps, args.maxtrees)

# Connect sites (respects partitioning)
model.connect_sites()

# Enable timing
model._verbose_timing = True

# Run simulation
model.setup()
model.simulate(ticks=args.years, sync_workers_every_n_ticks=1)

# Export timing (rank 0 only)
if rank == 0 and args.csv:
    # Append to CSV: job_id, nodes, workers, sites, sites_per_gpu, trees,
    #                 edge_cuts, network_load_time, model_creation_time,
    #                 gpu_setup_time, simulation_time, mpi_time, total_time
```

### 3c. `gap/analyze_site_connectivity.py`

Analyze CONUS site network and partition quality.

**Output:**
- Site adjacency statistics (avg neighbors, max distance)
- Partition comparison: round-robin vs spatial (edge cuts)
- Cross-rank communication estimate

### 3d. `gap/combine_weak_scaling_results.py` & `gap/analyze_weak_scaling.py`

Following superneuroabm pattern - CSV aggregation and text-based analysis.

### Phase 4: Frontier SLURM Scripts (3-4 hours)

**Directory:** `scaling_jobs/`

Common header (following superneuroabm):
```bash
#!/bin/bash
#SBATCH -A lrn088
#SBATCH -J ggap_scaling
#SBATCH -p batch
#SBATCH -t 02:00:00

unset SLURM_EXPORT_ENV
module load PrgEnv-gnu/8.6.0 rocm/6.4.1 craype-accel-amd-gfx90a
source activate /lustre/orion/proj-shared/lrn088/objective3/envs/sagesim_env
export MPICH_GPU_SUPPORT_ENABLED=1

cd /lustre/orion/lrn088/proj-shared/objective3/xxz/GGap
mkdir -p results logs
```

**Scripts:**
1. `test_memory.sh` - Run find_memory_limit.py
2. `submit_weak_scaling.sh <N_NODES>` - Dynamic batch generation (like superneuroabm)
3. `run_conus_demo.sbatch` - Full 1,424 sites
4. `profile_single_site.sh` - rocprofv3 wrapper

**srun pattern:**
```bash
srun -N$N -n$((N*8)) -c7 --ntasks-per-gpu=1 --gpu-bind=closest \
    python gap/weak_scaling_conus.py ...
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

1. **Exp 0** — memory test → determines S_max
2. **Exp 0b** — tune max_blocks_per_sm → optimal GPU occupancy (verify memory!)
3. **Exp 5** — profiling → find if optimization needed
4. **Exp 1** — single-GPU baseline → needed for efficiency calc (use optimal blocks/SM)
5. **Exp 2** — weak scaling → core paper result (use optimal blocks/SM)
6. **Exp 3** — communication analysis → supplementary
7. **Exp 4** — CONUS demo → headline figure

## Summary of Implementation Tasks

### Core Infrastructure Changes

1. **SAGESim timing API** (1-2 hours)
   - Add 3 methods to `model.py`: `get_timing_data()`, `get_timing_summary()`, `export_timing_json()`

2. **Spatial partitioning** (4-6 hours)
   - Modify `gap/gap_model.py:partition_sites()` - add METIS/KD-tree/Hilbert strategies
   - Create `gap/analyze_site_connectivity.py` - partition quality analysis

3. **Scaling scripts** (6-8 hours)
   - `gap/find_memory_limit.py` - Memory capacity test
   - `gap/weak_scaling_conus.py` - Main benchmark runner with CSV output
   - `gap/combine_weak_scaling_results.py` - Aggregate CSVs
   - `gap/analyze_weak_scaling.py` - Statistics and analysis

4. **SLURM infrastructure** (3-4 hours)
   - `scaling_jobs/test_memory.sh`
   - `scaling_jobs/submit_weak_scaling.sh` - Dynamic batch generation
   - `scaling_jobs/run_conus_demo.sbatch`

**Total Effort:** 20-28 hours

### Key Dependencies to Check

- [ ] `pymetis` or `metis` package in sagesim_env (for spatial partitioning)
- [ ] `rocm/6.4.1` module on Frontier
- [ ] LRN088 allocation active

### Execution Workflow

1. **Exp 0:** `sbatch scaling_jobs/test_memory.sh` → determines S_max
2. **Exp 0b:** `sbatch scaling_analysis/scripts/tune_occupancy.sh` → finds optimal max_blocks_per_sm=8 ✓ COMPLETED
3. **Exp 2 (Test):** `sbatch scaling_analysis/scripts/submit_test_weak_scaling.sh` → quick 1-8 GPU validation with synthetic torus
4. **Exp 2 (Full):** `./scaling_jobs/submit_weak_scaling.sh <N>` for N ∈ {1, 2, 5, 10, 20} (use max_blocks_per_sm=8)
5. **Exp 4:** `sbatch scaling_jobs/run_conus_demo.sbatch` → full CONUS 1,424 sites
6. **Analysis:** `python gap/combine_weak_scaling_results.py && python gap/analyze_weak_scaling.py`

**Note:** Experiment 5 (profiling) was removed. SAGESim's `_verbose_timing` flag provides comprehensive per-tick timing breakdown sufficient for weak scaling analysis.

## Verification Checklist

- [ ] Memory test completes, S_max determined
- [x] Occupancy tuning complete: max_blocks_per_sm=8 optimal
- [ ] Torus test completes: weak scaling efficiency >95% at 1-8 GPUs
- [ ] Spatial partitioning reduces edge cuts >10x vs round-robin
- [ ] Determinism: 1-GPU vs 8-GPU same 8 sites with same seed → identical output
- [ ] Timing sanity: ~0.5-1.0s/tick for 1 site, 500 gaps x 1000 trees
- [ ] Weak scaling efficiency >95% at 8 GPUs (intra-node), >85% at 64 GPUs (8 nodes)
- [ ] CONUS demo completes 500 years without crash
