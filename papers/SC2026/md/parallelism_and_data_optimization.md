# GGap: Architecture for SC Application-Track Paper

## Agent Hierarchy

GGap uses a three-level **Site-Gap-Tree** hierarchy:

- **Site** agents hold soil pools, climate data, and species availability.
- **Gap** agents aggregate litter/demand from trees and relay climate/nutrients.
- **Tree** agents represent individual trees with growth state and biomass.

All gaps and trees belonging to a site are assigned to the **same MPI rank**
(enforced by `initialize_site_with_gaps()` in `gap/gap_model.py:916`, which
creates the site agent and all of its child gap/tree agents with one shared
`rank` argument). Each rank maps to one GPU via the job scheduler (e.g. SLURM
`--gpus-per-task=1 --gpu-bind=closest`).

The CONUS production run uses **METIS** graph partitioning over the directed
seed-dispersal graph to balance sites across ranks while keeping tightly
connected sites colocated (`conus_simulations/scripts/run_conus.py:214`).
Round-robin partitioning is the framework default for smaller runs.

## Two-Level Parallelism

### Level 1: MPI -- Concurrent Ranks Across GPUs

One MPI rank per GPU. The CONUS production configuration runs on
**Frontier (OLCF)** with 20 nodes × 8 GCDs/node = **160 ranks**, processing
1,424 CONUS sites at ~9 sites per GPU
(`conus_simulations/scripts/submit_conus.sh:4`,
`conus_simulations/scripts/run_conus.py:355`).

Multiple ranks execute **simultaneously**, each running the same priority
pipeline on its own subset of sites. Between ticks, ranks exchange
**ghost data** -- the neighbor-visible properties that remote sites need
to read (see Data Optimization below).

```
Time ──▶

Rank 0 (GPU 0):  [kernel tick N] → [MPI sync] → [kernel tick N+1] → ...
Rank 1 (GPU 1):  [kernel tick N] → [MPI sync] → [kernel tick N+1] → ...
Rank N (GPU N):  [kernel tick N] → [MPI sync] → [kernel tick N+1] → ...
```

GPU compute and MPI sync alternate -- they do not overlap. The
`sync_workers_every_n_ticks` parameter controls how many ticks run on GPU
before an MPI exchange. The CONUS run uses `sync_workers_every_n_ticks=1`
(`run_conus.py:544`), so every annual tick is followed by ghost-data
exchange. Setting it higher batches multiple ticks into one kernel launch,
reducing MPI frequency at the cost of stale neighbor data.

### Level 2: GPU -- Thread-Parallel Agents Within a Rank

Each tick is **one fused GPU kernel launch** (`SAGESim/sagesim/model.py:1466`)
that processes every local agent across all priorities, with on-device grid
barriers between priorities. Agents at the same priority execute in parallel;
priorities run sequentially.

**Launch geometry on AMD MI250X (per GCD):**

| Parameter | Value | Source |
|---|---|---|
| Compute units (CUs) per GCD | **110** | hardware |
| Threads per block | **128** | hardcoded `SAGESim/sagesim/model.py:1330` |
| Max blocks per CU (occupancy-tuned) | **8** | `gap/docs/scaling_experiments.md:503` (Exp 0b) |
| Max co-resident blocks per GPU | **880** | `8 × 110` |
| Max concurrent threads per GPU | **112,640** | `880 × 128` |
| Agents per GPU (CONUS production) | **~4.5 M trees** | ~9 sites × 500 gaps × 1000 trees |

Co-residency is required because the kernel uses an on-device grid barrier
(`SAGESim/sagesim/model.py:1424-1441`): launching more blocks than can be
co-resident causes deadlock, since active blocks would wait at a barrier
for non-resident blocks that cannot start until the active blocks finish.
The 8-blocks-per-CU limit was confirmed safe by occupancy tuning
(Exp 0b: 12, 16, and 110 blocks/SM all deadlocked).

Because there are far more trees than concurrent threads, the kernel uses
a **grid-stride loop** so each thread processes multiple agents.

### How the Grid Barrier Is Implemented (Atomic Counter, Not Cooperative Groups)

Fusing the per-priority steps into one kernel collapses what would otherwise
be **ten kernel launches per tick** (one per priority) into a **single
launch**. The savings come from eliminating per-launch host overhead:
CPU→GPU command-queue submission, driver scheduling, grid-launch dispatch,
and the cold-cache penalty that comes with starting a fresh kernel on each
SM. For GGap this matters because several priorities operate on tiny agent
populations -- P1 `site_soil` runs one thread per site, P9 `site_final`
runs one thread per site -- and would not have enough work on their own
to amortize a per-launch host round-trip. Fusion makes those small
priorities essentially free. **The grid barriers themselves are not
eliminated** -- the data dependencies between priorities are real. They
have just moved from "kernel-launch boundary on the host" to "in-kernel
grid barrier on the device".

**The barrier is a hand-rolled atomic-counter spinlock**, not
`cooperative_groups::grid_group::sync()`. SAGESim generates the barrier
code at kernel-build time
(`SAGESim/sagesim/model.py:1908-1926`, `_gen_barrier_code()`):

```python
jit.syncthreads()                                       # 1. intra-block sync
if jit.threadIdx.x == 0:
    jit.threadfence()                                   # 2. publish my writes globally
    jit.atomic_add(barrier_counter, 0, 1)               # 3. signal "this block done"
    _barrier_target = (barrier_id + 1) * num_blocks_param
    while jit.atomic_add(barrier_counter, 0, 0) < _barrier_target:
        pass                                            # 4. spin until all blocks arrive
    jit.threadfence()                                   # 5. ensure others' writes visible
jit.syncthreads()                                       # 6. release the warps
barrier_id = barrier_id + 1
```

Each block elects its first thread as the leader. The leader issues a
`__threadfence()` to publish its block's writes to global memory, then
atomically increments a global counter, then spin-waits until the counter
equals `(barrier_id + 1) × num_blocks` -- meaning every block has arrived
at this particular barrier. A second `__threadfence()` ensures the leader
sees other blocks' writes before releasing the warps in its own block.
CuPy's JIT does not expose `__threadfence()` natively, so SAGESim
monkey-patches it in
(`SAGESim/sagesim/jit_extensions.py`).

**Why not cooperative groups?** `cooperative_groups::grid_group::sync()`
is the modern CUDA / HIP idiom for in-kernel grid barriers and has
hardware support on Pascal+ (NVIDIA) and GFX9+ (AMD). We chose not to use
it for three concrete reasons:

1. **CuPy JIT (`cupyx.jit.rawkernel`) does not expose the
   `cooperative_groups` header.** A JIT'd kernel cannot call
   `grid_group::sync()` or any other cooperative-groups primitive.
2. **`cudaLaunchCooperativeKernel` is not reachable from CuPy's launch
   path.** Cooperative kernels require a special launch flag that the
   GPU driver only honors when launched via this dedicated API; CuPy's
   `RawKernel.__call__` and JIT-compiled launches both go through the
   standard `cudaLaunchKernel` path, which silently ignores cooperative
   semantics.
3. **Bypassing CuPy JIT** to use the lower-level cuda-python launch API
   would mean giving up SAGESim's automatic AST transformations
   (double-buffering, CSR neighbor rewriting, seed injection, scalar
   global auto-extract) -- the very abstractions that let users write
   idiomatic Python step functions and have them compile to GPU code.
   That is a much higher cost than rolling a software grid barrier.

The atomic-counter pattern works correctly today on both CUDA (NVIDIA)
and HIP (AMD MI250X / MI300A) backends with no header dependencies and
no special launch API. Cooperative-groups support is a planned future
upgrade for SAGESim, at which point the barrier mechanism becomes
hardware-accelerated and the manual co-residency cap can be replaced by
the driver's cooperative-launch validation -- but the *pipeline
structure* (10 priorities, 1 launch per tick, in-kernel barriers between
priorities) stays the same.

**The co-residency invariant is identical to cooperative groups.**
Whether the barrier is hardware (`grid_group::sync`) or software (atomic
counter + spinlock), every block launched as part of the fused kernel
must be co-resident on the GPU when a barrier fires. Otherwise an
"active" block spinning at the barrier would wait forever for an
"inactive" block that cannot start until the active block releases its
SM resources -- a guaranteed deadlock. Cooperative groups protect against
this by having `cudaLaunchCooperativeKernel` *reject* an oversized launch;
SAGESim's atomic-counter version has no such guard, so it computes the
cap explicitly
(`SAGESim/sagesim/model.py:1424-1441`):

```python
num_sms = attrs['MultiProcessorCount']
max_blocks_per_sm = getattr(self, '_max_blocks_per_sm', 8)
max_grid_blocks = max_blocks_per_sm * num_sms
effective_blocks = min(blockspergrid, max_grid_blocks)
```

On the AMD MI250X (110 CUs per GCD) the production setting is
`max_blocks_per_sm = 8`, which clamps the launch at **880 blocks per
GPU** -- the same number already shown in the launch-geometry table
above. Total agent counts always exceed `880 × 128 = 112,640` at
production scale (one site has half a million trees), so the kernel
relies on the grid-stride loop documented above to walk through the
remaining agents.

### The Fused-Kernel Pipeline (10 priorities)

Within one tick, priorities run in this order, separated by grid barriers
(verified against `gap/gap_model.py:147-222`):

```
Tick N on one GPU (1 launch, 10 priorities):

  P0  Gap   gap_litter_aggregate   ──▶ barrier
  P1  Site  site_soil              ──▶ barrier   (365-day daily loop)
  P2  Gap   gap_climate_relay      ──▶ barrier
  P3  Tree  tree_potential_growth  ──▶ barrier   (env stress, N demand)
  P4  Gap   gap_demand_aggregate   ──▶ barrier   (sum N demand → N supply ratio)
  P5  Tree  tree_template_renewal  ──▶ barrier   (seedbank, seedling weights)
  P6  Gap   gap_recruit_aggregate  ──▶ barrier   (compute recruit_prob)
  P7  Tree  tree_actual_growth     ──▶ barrier   (N-limited growth, mortality, free-slot recruit)
  P8  Gap   gap_nconsumed_aggregate──▶ barrier
  P9  Site  site_final             ──▶ barrier   (N balance + seed dispersal)
```

The pipeline has **10 priorities** (P0–P9). The former P9/P10 split
(N balance vs. inter-site seed dispersal) was merged into a single
`site_final_step` (`gap/step_functions/site/site_final_step.py:1-24`),
eliminating one grid barrier per tick.

## Decomposing a Sequential Ecological Model into a GPU Pipeline

### The Problem: UVAFME's Non-Parallelizable Structure

UVAFME's growth algorithm has a **mandatory 3-loop structure** because
nitrogen is a shared pool -- all trees must declare demand before any tree
gets allocated supply. The original sequential code (GAPpy,
`GAPpy/src/model.py:386` `growth()`) does:

```
Loop 1: all trees compute env_stress + N_demand
         → sum N_demand into plot-level pool
         → compute N_supply_demand = avail_N / total_N_demand
         → update each species' nutrient[i] = poor_soil_rsp(N_supply_demand)
Loop 2: all trees read nutrient[k], compute actual N-limited growth,
         apply mortality
Loop 3: canopy pruning, litter output
```

These loops cannot be fused: Loop 2 depends on a plot-level aggregate
computed after Loop 1 finishes, and Loop 3 depends on final biomass from
Loop 2. The nitrogen pool is shared state that creates a **producer-consumer
dependency** between individual trees and the plot level.

### The Solution: Inter-Breed Data Flow Across Priorities

GGap expresses this dependency as **explicit data flow through Gap agents**,
which act as intermediaries between the tree level and site level:

```
P3  Tree  →  potential_growth: each tree independently computes
             env_stress, diam_max, n_demand
                 │
                 ▼  (trees write N_DEMAND into states[TreeS.N_DEMAND])
P4  Gap   →  demand_aggregate: sum all child trees' N_DEMAND
             → N_SUPPLY_RATIO = avail_N / total_demand
                 │
                 ▼  (gap writes N_SUPPLY_RATIO into states[GapS.N_SUPPLY_RATIO])
P7  Tree  →  actual_growth: each tree reads N_SUPPLY_RATIO from
             parent gap, applies nutrient limitation, computes
             final diameter increment, mortality, litter
```

The same producer-consumer pattern repeats throughout the pipeline:

| Pattern                       | Producer            | Consumer                   |
|-------------------------------|---------------------|----------------------------|
| Litter → soil                 | P0 Gap aggregate    | P1 Site soil               |
| Climate → trees               | P1 Site soil        | P2 Gap relay → P3 Tree     |
| N demand → supply ratio       | P3 Tree             | P4 Gap aggregate → P7 Tree |
| Seedling weights → recruit    | P5 Tree template    | P6 Gap aggregate → P7 Tree |
| N consumed → balance          | P7 Tree             | P8 Gap aggregate → P9 Site |
| Species avail → dispersal     | P0 Gap aggregate    | P9 Site final              |

GAPpy's implicit shared-state mutations become **explicit writes to
agent properties read at later priorities**. This restructuring is what
makes GPU parallelism possible: within each priority, all agents of that
breed execute independently.

### Probability-Based Parallel Recruitment

GAPpy's recruitment (`GAPpy/src/model.py:792` `renewal()`) iterates sequentially
over free tree slots in a Python `for` loop; each slot draws a uniform random
number, performs a binary search over the cumulative species probability, and
then instantiates a seedling. The slot-by-slot loop is the bottleneck, not the
species selection.

GGap parallelizes this by splitting the work across two priorities:

- **P6 Gap** computes `recruit_prob = num_to_recruit / free_slot_tree_count`
  (`gap_recruit_aggregate_step.py:108`).
- **P7 Tree** lets each free slot independently perform a Bernoulli
  trial against `recruit_prob`, and on success draws species via the same
  cumulative-probability scheme over `gap_seedling_weights`.

Hundreds of free slots per gap can therefore activate in the same kernel
phase while preserving the expected number of recruits.

> Note: GAPpy has **no inter-site seed dispersal** -- recruitment is
> strictly within-plot via the local `plot.seedling[]` array. Inter-site
> dispersal (haversine distance + negative-exponential kernel) is a new
> capability added by GGap at P9.

## 365-Day Daily Biogeochemistry on GPU

### Why This Is Non-Trivial

`site/soil_step.py` runs a **Markovian daily loop**
(`gap/step_functions/site/soil_step.py:572` `while day < 365:`) -- each day's
soil water state feeds into the next day's decomposition. This is inherently
sequential *within* a site:

```
for day in range(365):
    aet = soil_water_balance(previous_day_state)
    avail_n += decomposition(aet, temperature[day])
    # water state updated for next day
```

The daily loop cannot be parallelized across days. However, it is
**embarrassingly parallel across sites** -- each site's soil state is
independent. Inside the kernel, each *site agent* is one GPU thread that
executes all 365 daily iterations in registers.

### What Runs Inside the Kernel

Per day, each site computes:

- **Stochastic climate generation**: Monthly means + stddev expanded to
  daily values via Box-Muller normal samples with rational-polynomial
  inverse-CDF approximation (all on-GPU, no host RNG).
- **Hamon PET**: Solar declination, day length, and temperature-based
  potential evapotranspiration.
- **3-layer soil water routing**: Canopy interception → A0 → A → Base,
  with field capacity / permanent wilting point constraints and
  slope-dependent runoff.
- **Temperature-dependent decomposition**: Q10 kinetics (rates double per
  10 °C) applied to 3 soil layers (A0 litter → A humus → Base stable)
  with species-specific C:N ratios.
- **Nitrogen mineralization**: Released from A-layer organic matter
  decomposition, accumulated into annual available N.

After 365 days, the kernel writes annual summaries (degree-days, dry-days,
available N, fire/wind intensity) to site states for downstream priorities
to read.

### Mixed Temporal Scales

The model operates at two timescales simultaneously:

- **Annual**: tree demographics (growth, mortality, recruitment) at P3, P5, P7
  (and the gap/site aggregations at P4, P6, P8, P9).
- **Daily**: soil biogeochemistry (365 iterations) at P1.

The priority pipeline bridges these: P1 produces annual climate summaries
and available nitrogen from 365 daily steps, which P3+ consumes for
annual tree growth decisions.

## Data Optimization

### Colocation Eliminates Intra-Site Communication

Because `initialize_site_with_gaps()` places each site's entire ensemble
(site + gaps + trees) on one rank, **all intra-site neighbor access is
local GPU memory reads**. Tree-to-Gap, Gap-to-Site, and Site-to-Gap lookups
never cross rank boundaries.

This is why almost all properties are registered with `neighbor_visible=False`
(`gap/gap_model.py:147-222`):

| Breed | Property            | `neighbor_visible` | Reason                                       |
|-------|---------------------|--------------------|----------------------------------------------|
| Tree  | params (14 floats)  | False              | Self-only physiological intermediates        |
| Tree  | states (11 floats)  | False              | Read by parent gap on same rank              |
| Gap   | params (2 floats)   | False              | Self-only (`SITE_ID`, `TOTAL_N_DEMAND`)      |
| Gap   | states (16 floats)  | False              | Read by parent site / child trees on same rank |
| Site  | params (20 floats)  | False              | Self-only soil pools, climate aggregates     |
| Site  | states (8 floats)   | **True**           | Read by neighbor sites for climate + dispersal |
| Site  | site_avail_spec     | **True**           | Per-species availability for cross-rank dispersal |

Only **Site states** and **site_avail_spec** are neighbor-visible, because only
site-to-site interactions cross rank boundaries. This means MPI ghost-exchange
volume scales with the number of **sites**, not the number of trees or gaps.

### Flattening Objects into Tensors

UVAFME/GAPpy represents trees and species as Python objects with methods:

```python
tree.update_tree(species)    # load species data
tree.leaf_biomass_c()        # update leaf from diameter
tree.max_growth()            # compute potential growth
species.light_rsp(al)        # light response
species.temp_rsp(x)          # temperature response
species.poor_soil_rsp(n)     # N limitation
```

GPU kernels cannot serialize objects. GGap flattens everything into
fixed-size numeric arrays (sizes from `gap/constants.py`):

- **Tree params[14]**: age, biomC, biomN, leaf_bm, x/y, light_avail,
  fc_degday, fc_drought, fc_flood, diam_max_calc, forska_shade, seedbank,
  seedling (private to owning tree, `TreeP` enum).
- **Tree states[11]**: is_alive, diam, height, canopy_ht, seedling_weight,
  litter_c, litter_n, n_demand, n_consumed, species_id, env_stress
  (`TreeS` enum; readable by parent gap on the same rank).
- **Gap states[16]**: deg_days, dry_days, avail_n, n_supply_ratio,
  litter_accum_c/n, num_to_recruit, recruit_rand_seed, flood_days,
  total_seedling_weight, fire_intensity, total_lai, n_consumed,
  dry_days_base, wind_intensity, recovery_years (`GapS` enum).
- **Site params[20]** + **states[8]**: soil C/N pools (A0/A/Base),
  water content, annual runoff, climate summaries.
- **Species traits[num_species, 26]**: 2D global tensor, read-only broadcast
  to all kernels (`Trait` enum, `NUM_SPECIES_TRAITS = 26`). Each kernel does
  `species_traits[species_id, TRAIT_IDX]` instead of method calls.
- **Site configs[num_sites, 107]**: monthly tmin / tmax / prcp (12 × 3 = 36),
  their stddev (12 × 3 = 36), monthly temp + prcp lapse rates (12 × 2 = 24),
  field capacity, permanent wilting point, slope, sigma, LAI, latitude,
  longitude, rain N, fire/wind probability, base height -- read-only global
  (`Cfg` enum, `NUM_SITE_CONFIGS = 107`).

All methods are inlined into kernel code (e.g. Forska height-diameter
allometry becomes an explicit formula). This trades code compactness for
GPU compatibility.

### Persistent GPU-Resident Buffers

All agent property tensors, write buffers, neighbor structures (CSR format),
and breed-local arrays are allocated on GPU once (first tick) and **reused
across all subsequent ticks** (`SAGESim/sagesim/model.py:1340-1422`,
`is_initialized` fast path). There are no per-tick CPU-to-GPU uploads of
agent state and no per-tick rebuild of neighbor structures.

### Breed-Local Arrays

Per-breed GPU tensors avoid padding across all breeds. For example,
`gap_lai` allocates LAI bins on Gap agents only -- Tree and Site agents
don't waste memory on it. GGap registers six breed-local arrays
(`gap/gap_model.py:849-897`):

| Array | Breed | Shape per agent | `neighbor_visible` | `double_buffer` | Purpose |
|---|---|---|---|---|---|
| `gap_lai`              | Gap  | `(50, 2)`           | False | False | Per-gap LAI by height bin × (deciduous, conifer) for Beer-Lambert light attenuation; built P0, read P3/P5 |
| `gap_avail_spec`       | Gap  | `(num_species,)`    | False | False | Per-gap species availability flags; built P0, read P5/P9 |
| `gap_imported_seeds`   | Gap  | `(num_species,)`    | False | False | Per-gap imported seed relay; copied from site at P2, read at P5 |
| `gap_seedling_weights` | Gap  | `(num_species,)`    | False | False | Per-gap per-species seedling weight; written by template trees at P5, read by free slots at P7 |
| `site_avail_spec`      | Site | `(num_species,)`    | **True** | **True** | Site-averaged species availability; only array that crosses rank boundaries; double-buffered because P9 reads neighbors' values while writing its own |
| `site_imported_seeds`  | Site | `(num_species,)`    | False | False | Imported seeds from P9 dispersal, read locally by P2 in the next tick |

### GPU-Aware MPI

When running on systems with GPU-aware MPI (e.g. Cray MPICH on Frontier),
ghost-data exchange flows **GPU → NIC → GPU** via RDMA without CPU staging.
The CONUS submission script enables this with `MPICH_GPU_SUPPORT_ENABLED=1`
(`conus_simulations/scripts/submit_conus.sh:42`). Without GPU-aware MPI,
each exchange falls back to a `.get()` per peer (GPU → CPU copy) before
MPI send/recv.

The ghost exchange uses a `CommunicationManager`
(`SAGESim/sagesim/model.py:1374`) that packs all neighbor-visible properties
into per-peer buffers in one pass, rather than issuing separate MPI calls
per property.

## SAGESim Framework Contributions

### Automatic Double-Buffer Generation

SAGESim parses step function source code with Python's `ast` module to
detect which properties are written. For properties not explicitly listed
in `no_double_buffer`, it auto-generates write buffers and rewrites
assignments:

```python
# User writes:
params[agent_index][DIAM] += increment

# Framework transforms to:
write_params[agent_index][DIAM] = params[agent_index][DIAM] + increment
```

This eliminates manual race-condition handling. For GGap, only
`site_avail_spec` requires double-buffering (P9 reads neighbors' values
while writing its own at the same priority); every other breed property and
breed-local array is in `no_double_buffer` because the priority pipeline
already separates writers and readers.

### CSR Neighbor Auto-Transformation

User code uses intuitive array-of-neighbors syntax for neighbor iteration.
The framework's `_CSRBodyTransformer` (an AST `NodeTransformer`) automatically
rewrites it to compressed sparse row (CSR) format:

- Replaces `len(neighbors)` with `offsets[i+1] - offsets[i]`
- Replaces `neighbors[j]` with `values[offsets[i] + j]`
- Removes sentinel checks (e.g. `!= -1`)

This gives efficient coalesced GPU memory access without requiring users
to reason about sparse data structures.

### Deterministic Reproducibility via Logical IDs

Stochastic processes (climate perturbation, fire/wind, mortality, recruitment
selection) use Philox-based counter RNG keyed on `(tick, logical_id, salt)`.
Logical IDs are stable identifiers assigned at creation (e.g. site slot
index, gap index within site) -- not runtime agent indices that change with
partitioning. This ensures identical results regardless of MPI rank count
or agent creation order, which is critical for scientific reproducibility
and validation against the sequential reference implementation.

## Ecological Fidelity

GGap preserves the full UVAFME physics without approximation:

- **Forska height-diameter allometry** with species-specific parameters
- **5 shade-tolerance classes** with distinct sigmoidal light response curves
- **Parabolic temperature response** based on species degree-day ranges
  (min/optimum/max)
- **Species-specific C:N ratios** for litter decomposition across deciduous,
  conifer, and broadleaf types
- **Fire and wind disturbance** with stochastic probability, sampled
  intensity, and multi-year recovery suppression of recruitment
- **Distance-weighted seed dispersal** using haversine distances and
  negative exponential kernel:
  `weight = exp(-distance / max_dispersal_dist)` -- a new capability beyond
  the GAPpy reference, which has no inter-site seed dispersal
- **3-layer soil model** (A0 litter / A humus / Base stable) with
  C and N pools, temperature-dependent decomposition, and water-balance-
  driven mineralization

Tree species (32 in the regional run, 1,424 sites in the CONUS run) across
20 genera are parameterized from UVAFME input files, with tolerance-dependent
adjustments to leaf area, growth rate, and mortality thresholds.
