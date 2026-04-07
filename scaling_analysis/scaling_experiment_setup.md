# Scaling Experiment Setup

## 1. Hardware Platform

All experiments run on **OLCF Frontier** (Oak Ridge National Laboratory):
- **Node architecture**: AMD EPYC 7A53 (64 cores) + 4x AMD MI250X GPUs (8 GCDs per node)
- **GPU per node**: 8 GCDs (each MI250X has 2 GCDs, each GCD = 1 logical GPU with 110 CUs)
- **Interconnect**: HPE Slingshot-11 with 4x 25 GB/s injection bandwidth per node
- **GPU-aware MPI**: Cray MPICH with `MPICH_GPU_SUPPORT_ENABLED=1`

**Software stack**:
- PrgEnv-gnu 8.6.0, ROCm 6.4.1
- Python 3.13+, CuPy (ROCm backend), mpi4py
- SAGESim framework with `SAGESIM_NUM_SMS=110` (matching MI250X GCD CU count)
- 7 CPU cores per task, `--gpu-bind=closest` for NUMA affinity

## 2. Synthetic Data

All scaling experiments use synthetic UVAFME-compatible CSV input files to ensure reproducibility and uniform workload across ranks.

**Synthetic data files** (`scaling_analysis/synthetic_data/`):
- `SYNTHETIC_specieslist.csv` — 100 synthetic tree species across 10 genera, with varied traits (age 100-500 years, DBH 20-120 cm, height 15-45 m, shade tolerance 1-5)
- `SYNTHETIC_site.csv` — Site definitions (latitude, longitude, elevation, soil properties)
- `SYNTHETIC_climate.csv` — Monthly climate data (temperature, precipitation) per site
- `SYNTHETIC_climate_stddev.csv` — Climate variability per site
- `SYNTHETIC_rangelist.csv` — Species-site range mapping (all species present at all sites)
- `SYNTHETIC_altitudes.csv` — Altitude data per site

All species are present at all sites, ensuring identical per-site workload regardless of site ID.

## 3. Network Topology

### Torus Grid with Moore Neighborhood

Sites are arranged on a **2D toroidal grid** with 8-neighbor (Moore) connectivity. Each site connects bidirectionally to its 8 neighbors (N, S, E, W, NE, NW, SE, SW) with wraparound at boundaries:

```
neighbor_i = (i + di) % height     for di in {-1, 0, 1}
neighbor_j = (j + dj) % width      for dj in {-1, 0, 1}
```

**Total edges** = 8 x total_sites (each site has exactly 8 neighbors)

### 1D Column-Slab Partitioning

The grid is decomposed into vertical column slabs for MPI distribution:
- Each rank owns `height x block_width` contiguous columns
- `block_width = width / num_gpus`
- Site `(i, j)` is assigned to rank `j // block_width`

### Cross-Rank Communication

Cross-rank edges occur at the left and right boundaries of each slab. Each boundary column has `height` sites, each with 3 neighbors in the adjacent slab:

- **Cross-rank edges per rank** = `height x 3 x 2` (left boundary + right boundary)
- **Total edges per rank** = `sites_per_gpu x 8`
- **Cross-rank fraction** = `(height x 6) / (sites_per_gpu x 8)`

Only **site-level** properties are exchanged across ranks (`neighbor_visible=True`). All gap and tree agents belonging to a site are guaranteed to reside on the same rank — no cross-rank communication for within-site operations.

**Communication protocol per tick**:
1. GPU pack: gather ghost site data into send buffers on GPU
2. MPI exchange: non-blocking Isend/Irecv + Waitall (GPU-aware MPI, direct GPU buffer access)
3. GPU unpack: scatter received data into property tensors on GPU

## 4. Timing Instrumentation

### Phase-Level Timing (wall-clock)

Measured externally around each phase of the simulation setup:

| Phase | Metric | Description |
|-------|--------|-------------|
| 1 | `model_creation_time` | GAPModel constructor |
| 2 | `load_globals_time` | Load species traits + climate into global tensors |
| 3 | `partitioning_time` | Compute 1D slab partition mapping |
| 4 | `site_init_time` | Create all local site/gap/tree agents |
| 5 | `connectivity_time` | Establish torus edges (allgather + connect) |
| 6a | `register_arrays_time` | Register breed-local arrays |
| 6b | `gpu_setup_time` | GPU kernel compilation, buffer allocation, JIT |
| 7 | `simulation_time` | Total wall-clock for all ticks |

### Per-Tick Timing (SAGESim verbose_timing)

SAGESim's built-in instrumentation collects per-tick timing via `verbose_timing=True`. After simulation, the script separates tick 1 (buffer build) from ticks 2-N (steady state) and averages:

| Metric | Description |
|--------|-------------|
| `first_tick_time` | Tick 1 total (includes GPU buffer construction, ghost topology discovery, communication map build) |
| `steady_state_time` | Sum of ticks 2-N |
| `mean_tick_time` | Average of ticks 2-N |
| `mean_gpu_compute` | Kernel launch overhead (async, returns immediately) |
| `mean_gpu_sync` | CPU waiting for GPU kernel to complete |
| `mean_gpu_execution` | `gpu_compute + gpu_sync` = actual GPU work time |
| `mean_data_prep` | Pre-kernel data preparation (includes MPI ghost exchange) |
| `mean_mpi_gpu_pack` | GPU gather into MPI send buffers |
| `mean_mpi_exchange` | MPI Isend/Irecv + Waitall |
| `mean_mpi_gpu_unpack` | GPU scatter from MPI receive buffers |
| `mean_mpi_total` | `mpi_gpu_pack + mpi_exchange + mpi_gpu_unpack` |
| `mean_write_back` | GPU→GPU write buffer copy after kernel |
| `mean_kernel_args_build` | Kernel argument preparation |
| `gpu_execution_fraction` | `gpu_execution / total` per tick |
| `mpi_fraction` | `mpi_total / total` per tick |

**Note on GPU timing**: The GPU kernel launches asynchronously. `gpu_compute` captures only the CPU-side launch overhead (~0.5ms). The actual kernel execution completes during `gpu_sync` when the CPU calls `stream.synchronize()`. Therefore, `gpu_execution = gpu_compute + gpu_sync` is the correct measure of GPU work time.

---

## 5. Weak Scaling Experiment A: 10 Sites/GPU (Communication-Intensive)

### Configuration

| Parameter | Value |
|-----------|-------|
| Sites per GPU | 10 |
| Grid height | 5 |
| Block per GPU | 5 rows x 2 columns |
| Gaps per site | 500 |
| Trees per gap | 1,000 |
| Agents per GPU | ~5,005,010 (10 x 500,501) |
| Ticks | 1,000 |
| sync_workers_every_n_ticks | 1 |

### Scaling Range

| Nodes | GPUs | Total Sites | Grid (HxW) | Total Agents |
|-------|------|-------------|------------|-------------|
| 1     | 8    | 80          | 5 x 16    | ~40M |
| 2     | 16   | 160         | 5 x 32    | ~80M |
| 4     | 32   | 320         | 5 x 64    | ~160M |
| 8     | 64   | 640         | 5 x 128   | ~320M |
| 16    | 128  | 1,280       | 5 x 256   | ~640M |
| 32    | 256  | 2,560       | 5 x 512   | ~1.28B |
| 64    | 512  | 5,120       | 5 x 1024  | ~2.56B |

### Cross-Rank Analysis

- **Cross-rank edges per rank**: 5 x 3 x 2 = **30**
- **Total edges per rank**: 10 x 8 = **80**
- **Cross-rank fraction**: 30/80 = **37.5%**

This configuration provides high communication intensity relative to the per-GPU workload. With 37.5% of edges crossing rank boundaries, any MPI communication overhead is directly visible in the scaling results. This serves as a stress test for the communication subsystem.

### Rationale for 10 Sites/GPU

The choice of 10 sites per GPU with height=5 produces a 5x2 slab per rank, where each site interacts with up to 8 neighbors via the Moore neighborhood. This yields:
- A **physically motivated** neighborhood size: each site exchanges seed dispersal and microclimate data with its spatial neighbors, matching the connectivity pattern of real-world forest sites
- **High cross-rank density** (37.5%): proves the model handles inter-GPU communication efficiently
- **Sufficient per-GPU workload**: 10 sites x 500 gaps x 1,000 trees = ~5M agents per GPU, providing adequate GPU occupancy on MI250X GCDs

---

## 6. Weak Scaling Experiment B: 100 Sites/GPU (Compute-Intensive)

### Configuration

| Parameter | Value |
|-----------|-------|
| Sites per GPU | 100 |
| Grid height | 10 |
| Block per GPU | 10 rows x 10 columns |
| Gaps per site | 200 |
| Trees per gap | 500 |
| Agents per GPU | ~10,020,100 (100 x 100,201) |
| Ticks | 1,000 |
| sync_workers_every_n_ticks | 1 |

### Scaling Range

| Nodes | GPUs | Total Sites | Grid (HxW) | Total Agents |
|-------|------|-------------|------------|-------------|
| 1     | 8    | 800         | 10 x 80   | ~80M |
| 2     | 16   | 1,600       | 10 x 160  | ~160M |
| 4     | 32   | 3,200       | 10 x 320  | ~320M |
| 8     | 64   | 6,400       | 10 x 640  | ~640M |
| 16    | 128  | 12,800      | 10 x 1280 | ~1.28B |
| 32    | 256  | 25,600      | 10 x 2560 | ~2.56B |
| 64    | 512  | 51,200      | 10 x 5120 | ~5.12B |

### Cross-Rank Analysis

- **Cross-rank edges per rank**: 10 x 3 x 2 = **60**
- **Total edges per rank**: 100 x 8 = **800**
- **Cross-rank fraction**: 60/800 = **7.5%**

### Rationale

This experiment complements Experiment A by demonstrating:
- **10x more sites per GPU**: 100 sites vs 10, testing scaling with larger local workloads
- **Larger problem sizes**: up to 51,200 sites (5.12 billion agents) at 512 GPUs
- **Moderate cross-rank density** (7.5%): with heavier per-GPU computation, MPI costs are amortized more effectively

Together, Experiments A and B cover both axes of weak scaling:
- **Experiment A** (full-fidelity) proves the model handles heavy communication (37.5% cross-rank) with high efficiency
- **Experiment B** (high site density) proves the model scales with large spatial domains (100 sites/GPU), and extends to billions of total agents

The strong scaling experiment uses the same per-site configuration as Experiment B (200 gaps, 500 trees), providing a consistent workload across the B experiments.

---

## 7. Strong Scaling Experiment

### Configuration

| Parameter | Value |
|-----------|-------|
| Total sites (fixed) | 2,048 |
| Grid height | 4 |
| Grid width | 512 |
| Gaps per site | 200 |
| Trees per gap | 500 |
| Total agents (fixed) | ~205 million |
| Ticks | 1,000 |
| sync_workers_every_n_ticks | 1 |

### Scaling Range

| Nodes | GPUs | Sites/GPU | Block (HxW) | Cross-rank Edges | Cross-rank % |
|-------|------|-----------|-------------|-----------------|-------------|
| 1     | 8    | 256       | 4 x 64     | 24              | 1.2%        |
| 2     | 16   | 128       | 4 x 32     | 24              | 2.3%        |
| 4     | 32   | 64        | 4 x 16     | 24              | 4.7%        |
| 8     | 64   | 32        | 4 x 8      | 24              | 9.4%        |
| 16    | 128  | 16        | 4 x 4      | 24              | 18.8%       |
| 32    | 256  | 8         | 4 x 2      | 24              | 37.5%       |
| 64    | 512  | 4         | 4 x 1      | 24              | 75.0%       |

### Cross-Rank Analysis

With `grid_height=4`, cross-rank edges per rank is constant at **4 x 3 x 2 = 24** regardless of GPU count. As the problem is split across more GPUs:
- Sites per GPU decreases (256 → 4)
- Cross-rank fraction increases (1.2% → 75.0%)
- This naturally tests the transition from compute-bound to communication-bound regimes
- At 512 GPUs, each rank owns a single column (4x1 block) — the minimum partition for height=4

### Rationale for 2,048 Total Sites

Starting at 256 sites/GPU on a single node (8 GPUs) ensures:
- The baseline measurement has sufficient per-GPU work for meaningful GPU utilization
- At the smallest partition (4 sites/GPU at 512 GPUs), each GPU still manages 4 sites x 200 gaps x 500 trees = ~400K agents
- The fixed total of ~205 million agents represents a scientifically meaningful workload
- The reduced per-site configuration (200 gaps, 500 trees) matches Weak Scaling B, enabling 256 sites/GPU on a single node as the baseline

### Strong Scaling Metrics

For strong scaling, the primary metric is **speedup** relative to the single-node (8 GPU) baseline:

- **Speedup(N)** = T(8 GPUs) / T(N GPUs)
- **Parallel efficiency(N)** = Speedup(N) / (N / 8) x 100%

Where T is the `simulation_time` (or `steady_state_time` to exclude first-tick buffer build overhead).

---

## 8. Summary of Experiment Design

| Experiment | Type | Sites/GPU | Gaps | Trees | Height | Cross-rank % | GPU Range | What it Proves |
|-----------|------|-----------|------|-------|--------|-------------|-----------|----------------|
| Weak A | Weak | 10 | 500 | 1,000 | 5 | 37.5% | 8-2048 | Communication-heavy, full-fidelity |
| Weak B | Weak | 100 | 200 | 500 | 10 | 7.5% | 8-2048 | Compute-heavy, high site density |
| Strong | Strong | 256→4 | 200 | 500 | 4 | 1.2%→75.0% | 8-512 | Time-to-solution improvement |
