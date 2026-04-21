# GGap: Architecture for SC Application-Track Paper

## Agent Hierarchy

GGap uses a three-level **Site-Gap-Tree** hierarchy:

- **Site** agents hold soil pools, climate data, and species availability.
- **Gap** agents aggregate litter/demand from trees and relay climate/nutrients.
- **Tree** agents represent individual trees with growth state and biomass.

All gaps and trees belonging to a site are assigned to the **same MPI rank**
(enforced by `partition_sites()` in `gap_model.py`). Each rank maps to one GPU
via the job scheduler (e.g. SLURM `--gpu-bind`).

## Two-Level Parallelism

### Level 1: GPU -- Thread-Parallel Agents Within a Rank

Each tick, a single GPU kernel processes all local agents. Agents at the same
priority level execute **in parallel** across GPU threads (grid-stride loop,
128 threads/block). Priorities are separated by grid barriers, so P0 fully
completes before P1 starts:

```
Tick N on one GPU:

  P0  gap_litter_aggregate     ── all gap agents in parallel ──▶ barrier
  P1  site_soil                ── all site agents in parallel ──▶ barrier
  P2  gap_climate_relay        ── all gap agents in parallel ──▶ barrier
  P3  tree_potential_growth    ── all tree agents in parallel ──▶ barrier
  P4  gap_demand_aggregate     ── all gap agents in parallel ──▶ barrier
  P5  tree_template_renewal    ── all tree agents in parallel ──▶ barrier
  P6  gap_recruit_aggregate    ── all gap agents in parallel ──▶ barrier
  P7  tree_actual_growth       ── all tree agents in parallel ──▶ barrier
  P8  gap_nconsumed_aggregate  ── all gap agents in parallel ──▶ barrier
  P9  site_nbalance            ── all site agents in parallel ──▶ barrier
  P10 site_seed_dispersal      ── all site agents in parallel ──▶ barrier
```

The multi-priority pipeline ensures step functions execute in the correct
ecological order (e.g. soil processes before tree growth, demand aggregation
before nutrient allocation) while keeping all computation on the GPU.

### Level 2: MPI -- Concurrent Ranks Across GPUs

Multiple ranks execute **simultaneously**, each running the same priority
pipeline on its own subset of sites. Between ticks, ranks exchange
**ghost data** -- the neighbor-visible properties that remote sites need
to read (see Data Optimization below).

```
Time ──▶

Rank 0 (GPU 0):  [kernel tick N] → [MPI sync] → [kernel tick N+1] → ...
Rank 1 (GPU 1):  [kernel tick N] → [MPI sync] → [kernel tick N+1] → ...
Rank 2 (GPU 2):  [kernel tick N] → [MPI sync] → [kernel tick N+1] → ...
```

GPU compute and MPI sync alternate -- they do not overlap. The
`sync_workers_every_n_ticks` parameter controls how many ticks run on GPU
before an MPI exchange. With `sync_workers_every_n_ticks=1` (the default for
multi-GPU), every tick is followed by a sync. Setting it higher batches
multiple ticks into one kernel launch, reducing MPI frequency at the cost
of stale neighbor data.

## Decomposing a Sequential Ecological Model into a GPU Pipeline

### The Problem: UVAFME's Non-Parallelizable Structure

UVAFME's growth algorithm has a **mandatory 3-loop structure** because
nitrogen is a shared pool -- all trees must declare demand before any tree
gets allocated supply. The original sequential code (GAPpy) does:

```
Loop 1: all trees compute env_stress + N_demand
         → sum N_demand into plot-level pool
Loop 2: all trees read N_supply_ratio, compute actual growth
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
                 ▼  (trees write N_DEMAND to their states)
P4  Gap   →  demand_aggregate: sum all trees' N_DEMAND
             → compute N_SUPPLY_RATIO = avail_N / total_demand
                 │
                 ▼  (gap writes N_SUPPLY_RATIO to its states)
P7  Tree  →  actual_growth: each tree reads N_SUPPLY_RATIO from
             parent gap, applies nutrient limitation, computes
             final diameter increment, mortality, litter
```

The same producer-consumer pattern repeats throughout the pipeline:

| Pattern                   | Producer         | Consumer         |
|---------------------------|------------------|------------------|
| Litter → soil             | P0 Gap aggregate | P1 Site soil     |
| Climate → trees           | P1 Site soil     | P2 Gap relay → P3 Tree |
| N demand → supply ratio   | P3 Tree          | P4 Gap aggregate → P7 Tree |
| Seedling weights → recruit| P5 Tree template | P6 Gap aggregate → P7 Tree |
| N consumed → balance      | P7 Tree          | P8 Gap aggregate → P9 Site |
| Species avail → dispersal | P0 Gap aggregate | P10 Site dispersal |

GAPpy's implicit shared-state mutations become **explicit writes to
agent properties read at later priorities**. This restructuring is what
makes GPU parallelism possible: within each priority, all agents of that
breed execute independently.

### Probability-Based Parallel Recruitment

GAPpy recruits new trees by iterating sequentially over free slots.
GGap instead computes `recruit_prob = nrenew / free_slots` at the Gap
level (P6), then lets each free slot independently decide via a Bernoulli
trial at P7. This enables hundreds of free slots to activate simultaneously
while preserving expected recruitment counts.

## 365-Day Daily Biogeochemistry on GPU

### Why This Is Non-Trivial

`soil_step.py` (~600 lines of GPU kernel code) runs a **Markovian daily
loop** -- each day's soil water state feeds into the next day's decomposition.
This is inherently sequential *within* a site:

```
for day in range(365):
    aet = soil_water_balance(previous_day_state)
    avail_n += decomposition(aet, temperature[day])
    # water state updated for next day
```

The daily loop cannot be parallelized across days. However, it is
**embarrassingly parallel across sites and gaps** -- each site's soil
state is independent.

### What Runs Inside the Kernel

Each site agent executes all 365 daily iterations in GPU registers within
a single kernel invocation. Per day, it computes:

- **Stochastic climate generation**: Monthly means + stddev expanded to
  daily values via Box-Muller normal samples with rational-polynomial
  inverse-CDF approximation (all on-GPU, no host RNG)
- **Hargreaves PET**: Solar declination, day length, extraterrestrial
  radiation, and temperature-based evapotranspiration (~8 trig terms)
- **3-layer soil water routing**: Canopy interception → A0 → A → Base,
  with field capacity / permanent wilting point constraints and
  slope-dependent runoff
- **Temperature-dependent decomposition**: Q10 kinetics (rates double
  per 10C) applied to 3 soil layers (A0 litter → A humus → Base stable)
  with species-specific C:N ratios
- **Nitrogen mineralization**: Released from A-layer organic matter
  decomposition, accumulated into annual available N

After 365 days, the kernel writes annual summaries (degree-days, dry-days,
available N, fire/wind intensity) to site state for downstream priorities
to read.

### Mixed Temporal Scales

The model operates at two timescales simultaneously:
- **Annual**: tree demographics (growth, mortality, recruitment) at P3-P9
- **Daily**: soil biogeochemistry (365 iterations) at P1

The priority pipeline bridges these: P1 produces annual climate summaries
and available nitrogen from 365 daily steps, which P3+ consumes for
annual tree growth decisions.

## Data Optimization

### Colocation Eliminates Intra-Site Communication

Because `partition_sites()` places each site's entire ensemble (site + gaps +
trees) on one rank, **all intra-site neighbor access is local GPU memory
reads**. Tree-to-Gap, Gap-to-Site, and Site-to-Gap lookups never cross rank
boundaries.

This is why Tree and Gap properties are registered with `neighbor_visible=False`:

| Breed | Property     | `neighbor_visible` | Reason                                  |
|-------|-------------|--------------------|-----------------------------------------|
| Tree  | params      | False              | Only read by owning tree                |
| Tree  | states      | False              | Only read by parent gap (same rank)     |
| Gap   | params      | False              | Only read by parent site (same rank)    |
| Gap   | states      | False              | Only read by child trees (same rank)    |
| Site  | params      | False              | Immutable site config                   |
| Site  | states      | **True**           | Read by neighbor sites for dispersal    |
| Site  | avail_spec  | **True**           | Species availability for seed dispersal |

Only **Site states** and **site_avail_spec** are neighbor-visible, because only
site-to-site seed dispersal crosses rank boundaries. This means MPI traffic
scales with the number of **sites**, not the number of trees or gaps.

### Flattening Objects into Tensors

UVAFME/GAPpy represents trees as Python objects with methods:
```python
tree.update_tree(species)    # load species data
tree.leaf_biomass_c()        # update leaf from diameter
tree.max_growth()            # compute max increment
```

GPU kernels cannot serialize objects. GGap flattens everything into
fixed-size numeric arrays:

- **Tree params[14]**: age, biomass_c, biomass_n, leaf_bm, and 10
  physiological intermediates (private to owning tree)
- **Tree states[11]**: is_alive, diam, height, canopy_ht, litter_c/n,
  n_demand, species_id, env_stress (readable by parent gap)
- **Species traits[32][26]**: 2D global tensor, read-only broadcast to
  all kernels. Each kernel does `species_traits[species_id][TRAIT_IDX]`
  instead of method calls.
- **Site configs[N][107]**: Monthly climate (12 months x 3 variables),
  soil properties, fire/wind probabilities, latitude -- read-only global.

All methods are inlined into kernel code (e.g. Forska height-diameter
allometry becomes an explicit formula). This trades code compactness for
GPU compatibility.

### GPU-Resident Buffers

All agent property tensors, write buffers, neighbor structures (CSR format),
and breed-local arrays are allocated on GPU once (first tick) and **reused
across all subsequent ticks**. There are no per-tick CPU-to-GPU uploads of
agent state.

The only per-tick CPU involvement is:
- Kernel launch overhead (microseconds)
- Write-back of double-buffer results (GPU-to-GPU copy, no CPU)

### Breed-Local Arrays

Per-breed GPU tensors avoid padding across all breeds. For example,
`gap_lai[50]` allocates 50 height-layer LAI values per Gap agent only --
Tree and Site agents don't waste memory on it. GGap uses breed-local
arrays for:

- `gap_lai` (50 height layers): cumulative LAI profile for Beer-Lambert
  light attenuation, built at P0, read at P3/P5
- `gap_avail_spec` (num_species): per-gap species availability flags,
  built at P0, read at P5/P10
- `site_avail_spec` (num_species): per-site species availability
  (neighbor_visible=True for cross-rank dispersal)
- `gap_imported_seeds` (num_species): seeds received from neighboring
  sites, relayed from site at P2
- `site_imported_seeds` (num_species): accumulated seed imports from
  dispersal kernel at P10

### GPU-Aware MPI

When running on systems with GPU-aware MPI (e.g. Cray MPICH on Frontier),
ghost data exchange flows **GPU -> NIC -> GPU** via RDMA without CPU staging.
The framework detects this via environment variables (`MPICH_GPU_SUPPORT_ENABLED`,
`OMPI_MCA_opal_cuda_support`).

Without GPU-aware MPI, each exchange falls back to one `.get()` per peer
(GPU -> CPU copy) before MPI send/recv.

The ghost exchange uses batched GPU gather/scatter -- all neighbor-visible
properties are packed into a single per-peer buffer in one pass, rather
than issuing separate MPI calls per property.

## SAGESim Framework Contributions

### Automatic Double-Buffer Generation

SAGESim parses step function source code with Python's `ast` module to
detect which properties are written. It then auto-generates write buffers
and rewrites assignments:

```python
# User writes:
params[agent_index][DIAM] += increment

# Framework transforms to:
write_params[agent_index][DIAM] = params[agent_index][DIAM] + increment
```

This eliminates manual race-condition handling. Domain scientists write
natural array-indexing code; the framework guarantees that parallel threads
reading and writing the same property see consistent values within a
priority level.

### CSR Neighbor Auto-Transformation

User code uses intuitive array-of-neighbors syntax for neighbor iteration.
The framework's `_CSRBodyTransformer` (an AST NodeTransformer) automatically
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
index, gap index within site) -- not runtime agent indices that change
with partitioning.

This ensures identical results regardless of MPI rank count or agent
creation order, which is critical for scientific reproducibility and
validation against the sequential reference implementation.

## Ecological Fidelity

GGap preserves the full UVAFME physics without approximation:

- **Forska height-diameter allometry** with species-specific parameters
- **5 shade-tolerance classes** with distinct sigmoidal light response curves
- **Parabolic temperature response** based on species degree-day ranges
  (min/optimum/max)
- **Species-specific C:N ratios** for litter decomposition across
  deciduous, conifer, and broadleaf types
- **Fire and wind disturbance** with stochastic probability (base 1%,
  increasing with dry conditions to 15% max), sampled intensity, and
  multi-year recovery suppression of recruitment
- **Distance-weighted seed dispersal** using haversine distances and
  negative exponential kernel: `weight = exp(-distance / max_dispersal_dist)`
- **3-layer soil model** (A0 litter / A humus / Base stable) with
  C and N pools, temperature-dependent decomposition, and water-balance-
  driven mineralization

32 tree species across 20 genera are parameterized from UVAFME input files,
with tolerance-dependent adjustments to leaf area, growth rate, and
mortality thresholds.
