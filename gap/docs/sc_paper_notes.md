# GGap: SC Application-Track Paper Notes

## 1. Project Identity

**GGap** -- GPU-accelerated Gap model. A pure-Python, agent-based forest
dynamics model built on the SAGESim framework. Implements the full UVAFME
(University of Virginia Forest Model Enhanced) ecological processes using
a Site-Gap-Tree agent hierarchy running on GPU via CuPy JIT kernels and
MPI for multi-GPU distribution.

- **Authors**: Xi Zhang, Chathika Gunaratne (ORNL)
- **License**: MIT
- **Language**: Python 3.13+, CuPy JIT (compiles to CUDA/ROCm at runtime)
- **Hardware targets**: NVIDIA (CUDA), AMD MI250X (ROCm) on Frontier
- **ORNL allocation**: LRN088

---

## 2. The Scientific Problem

Forest gap models simulate individual-tree competition, growth, mortality,
and recruitment within forest "gaps" (canopy openings). They are essential
tools for projecting forest composition under climate change, but are
computationally expensive due to:

- **Individual-tree resolution**: each tree is a distinct entity with
  species-specific physiology
- **Long timescales**: 500-1000 year simulations needed for succession
- **Ensemble requirements**: 200+ replicate gaps per site for statistical
  robustness
- **Daily soil processes**: 365-day inner loop per annual tree timestep
- **Multi-site coupling**: seed dispersal between geographically separated
  sites

UVAFME is a well-established gap model (Fortran origin, Python translation
as GAPpy). A typical run: 200 gaps x 1000 tree slots x 1000 years =
200M tree-years, taking minutes to hours on CPU per site. Regional-scale
simulations (thousands of sites) become prohibitive.

---

## 3. The Computational Challenge

### 3.1 Why This Is Not Embarrassingly Parallel

UVAFME's annual cycle contains **shared-pool dependencies** that prevent
naive parallelization:

**The 3-loop growth algorithm** (GAPpy `model.py:364-607`):

```
Loop 1: all trees compute env_stress + N_demand
         → sum N_demand into plot-level pool    ← REDUCTION
Loop 2: all trees read N_supply_ratio           ← BROADCAST
         → compute actual growth
Loop 3: canopy pruning, litter output           ← DEPENDS ON Loop 2
```

Nitrogen is a **shared resource**: all trees must declare demand before
any tree gets allocated supply. This creates a producer-consumer dependency
that must be resolved within each annual timestep.

Similarly:
- **Light competition**: each tree's growth depends on cumulative LAI above
  it, which depends on all taller trees' leaf areas
- **Soil biogeochemistry**: 365-day sequential water balance where each
  day's state feeds the next
- **Recruitment**: density-dependent, requiring knowledge of all living
  trees before new trees can be established

### 3.2 The GAPpy Sequential Execution

GAPpy processes the annual cycle as nested loops:

```python
for year in range(num_years):           # 1000 iterations
    for site in sites:                  # 1-6 sites
        bio_geo_climate(site)           # 365-day soil loop
        canopy(site)                    # LAI + Beer-Lambert
        growth(site)                    # 3 sequential loops
        mortality(site)                 # disturbance + individual
        renewal(site)                   # seedbank → recruitment
```

Within `growth()`, three passes over the tree array are mandatory because
Loop 2 requires a plot-level aggregate computed by Loop 1.

---

## 4. GGap Architecture

### 4.1 Agent Hierarchy

Three agent breeds form a containment hierarchy:

| Breed | Count (typical) | Role |
|-------|-----------------|------|
| Site  | 1-N (one per geographic location) | Soil pools, climate, dispersal |
| Gap   | 200 per site | Aggregate trees, relay climate/nutrients |
| Tree  | 1000 slots per gap | Individual trees (living, dormant, or template) |

**Topology**: Site <-> Gap (bidirectional), Gap <-> Tree (bidirectional),
Free-slot -> Template (directed, for species selection).

All agents of a site are assigned to the **same MPI rank** via
`partition_sites()` (round-robin). Each rank maps to one GPU.

### 4.2 Property Layout

Every agent stores state in flat float arrays (GPU-compatible):

**Tree** (25 floats total):
- `params[14]` (neighbor_visible=False): AGE, BIOMC, BIOMN, LEAF_BM,
  X, Y, LIGHT_AVAIL, FC_DEGDAY, FC_DROUGHT, FC_FLOOD, DIAM_MAX_CALC,
  FORSKA_SHADE, SEEDBANK, SEEDLING
- `states[11]` (neighbor_visible=False): IS_ALIVE, DIAM, HEIGHT,
  CANOPY_HT, SEEDLING_WEIGHT, LITTER_C, LITTER_N, N_DEMAND,
  N_CONSUMED, SPECIES_ID, ENV_STRESS

**Gap** (18 floats total):
- `params[2]` (neighbor_visible=False): SITE_ID, TOTAL_N_DEMAND
- `states[16]` (neighbor_visible=False): DEG_DAYS, DRY_DAYS, AVAIL_N,
  N_SUPPLY_RATIO, LITTER_ACCUM_C, LITTER_ACCUM_N, NUM_TO_RECRUIT,
  RECRUIT_RAND_SEED, FLOOD_DAYS, TOTAL_SEEDLING_WEIGHT, FIRE_INTENSITY,
  TOTAL_LAI, N_CONSUMED, DRY_DAYS_BASE, WIND_INTENSITY, RECOVERY_YEARS

**Site** (29 floats total):
- `params[21]` (neighbor_visible=False): A0_C, A0_N, A_C, A_N, BL_C,
  BL_N, A0_W, A_W, BL_W, LAI_W0, ANNUAL_RUNOFF, SITE_ID, plus
  9 output-only fields (N_SUPPLY_RATIO, ANNUAL_RAIN, GROW_DAYS,
  POT_EVAP, ACT_EVAP, SOIL_RESP, C_INTO_A0, N_INTO_A0, NET_N_INTO_A0)
- `states[8]` (**neighbor_visible=True**): DEG_DAYS, DRY_DAYS, AVAIL_N,
  FLOOD_DAYS, FIRE_INTENSITY, DRY_DAYS_BASE, WIND_INTENSITY, SITE_ID

Only Site states cross rank boundaries (for inter-site seed dispersal).

### 4.3 Global Read-Only Tensors

Broadcast to all kernels, never modified during simulation:

| Name | Shape | Content |
|------|-------|---------|
| species_traits | [num_species x 26] | 26 traits per species (MAX_AGE, MAX_DIAM, MAX_HT, G, shade/drought/flood tolerance, C:N ratios, etc.) |
| site_configs | [num_sites x 107] | Monthly climate (12x tmin/tmax/prcp), soil props, fire/wind probs, std devs, lapse rates |
| rangelists | [num_sites x num_species] | Binary species presence mask per site |
| site_distances | [num_sites x num_sites] | Pre-computed haversine distances (km) |

### 4.4 Breed-Local Arrays

Per-breed GPU tensors (no padding across breeds):

| Name | Breed | Shape per agent | neighbor_visible | Purpose |
|------|-------|-----------------|------------------|---------|
| gap_lai | Gap | (50, 2) | False | LAI profile: 50 height layers x {deciduous, coniferous} |
| gap_avail_spec | Gap | (num_species,) | False | Per-gap species maturity flags |
| gap_imported_seeds | Gap | (num_species,) | False | Seeds from dispersal, relayed from site |
| site_avail_spec | Site | (num_species,) | **True** | Site-averaged species availability (cross-rank) |
| site_imported_seeds | Site | (num_species,) | False | Accumulated seed imports |

---

## 5. The 11-Priority Kernel Pipeline

### 5.1 Priority Map

The GAPpy annual cycle is decomposed into 11 priorities that execute
sequentially (enforced by grid barriers), with all agents at each
priority executing in parallel:

| P | Breed | Step Function | GAPpy Equivalent | Key Algorithm |
|---|-------|---------------|------------------|---------------|
| 0 | Gap | gap_litter_aggregate | canopy() partial | Aggregate litter from trees; bin LAI into 50 height layers; top-down cumulative prefix sum for O(1) Beer-Lambert lookup; mark species maturity flags |
| 1 | Site | site_soil | bio_geo_climate() | **365-day daily loop**: stochastic climate generation (Box-Muller), Hargreaves PET, 3-layer soil water routing, temperature-dependent decomposition (Q10), N mineralization, fire/wind stochastic events |
| 2 | Gap | gap_climate_relay | (no equivalent) | Copy Site climate -> Gap states (eliminates 1-tick lag); relay imported seeds for template access |
| 3 | Tree | tree_potential_growth | growth() Loop 1 + canopy() | Environmental response functions: parabolic temperature, sqrt drought, sigmoidal light (5 shade classes), Beer-Lambert light from gap_lai; Forska diameter increment; N demand projection |
| 4 | Gap | gap_demand_aggregate | growth() N aggregation | Sum tree N_DEMAND; compute per-gap N_SUPPLY_RATIO = avail_N / total_demand; clear accumulators for next tick |
| 5 | Tree | tree_template_renewal | renewal() | Templates only: seedbank dynamics, regrowth = fc_degday x fc_drought x fc_nutrient x fc_light; fire/wind disturbance branching; seedling weight for recruitment |
| 6 | Gap | gap_recruit_aggregate | renewal() density calc | Count living/free/templates; density-based nrenew = min(PLOTSIZE x growmax - numtrees, PLOTSIZE x 0.5); convert to per-slot recruit_prob; suppress during recovery |
| 7 | Tree | tree_actual_growth | growth() Loops 2-3 + mortality() + renewal() activation | **Living trees**: nutrient response (quadratic), final growth, allometric updates, canopy self-pruning, mortality (age + growth stress + fire/wind), litter output. **Free slots**: probabilistic activation, species selection via CDF scan over template weights, seedling initialization |
| 8 | Gap | gap_nconsumed_aggregate | (implicit in GAPpy) | Sum N_CONSUMED from all trees for soil balance |
| 9 | Site | site_nbalance | growth() N closure | Surplus/deficit: avail_N -= total_consumed; leaching proportional to runoff; stoichiometric C:N transfer to base layer |
| 10 | Site | site_seed_dispersal | inter-site dispersal | Average gap avail_spec -> site_avail_spec (ghosted to neighbors); distance-weighted seed import: weight = exp(-dist / max_dispersal_dist); filter by rangelist |

### 5.2 Data Flow Across Priorities

The pipeline resolves GAPpy's implicit shared-state dependencies as
**explicit inter-breed data flow**:

```
P0  Gap reads Trees (litter, LAI)
     │
P1  Site reads Gaps (aggregated litter, LAI) → writes climate, avail_N
     │
P2  Gap reads Site → writes climate relay to Gap states
     │
P3  Tree reads Gap (climate, gap_lai) → writes ENV_STRESS, N_DEMAND
     │
P4  Gap reads Trees (N_DEMAND) → writes N_SUPPLY_RATIO
     │
P5  Tree reads Gap (N_SUPPLY_RATIO, climate, avail_spec) → writes SEEDLING_WEIGHT
     │
P6  Gap reads Trees (SEEDLING_WEIGHT) → writes NUM_TO_RECRUIT
     │
P7  Tree reads Gap (N_SUPPLY_RATIO, disturbance, NUM_TO_RECRUIT)
     → writes DIAM, HEIGHT, IS_ALIVE, LITTER, N_CONSUMED
     │
P8  Gap reads Trees (N_CONSUMED) → writes aggregated N_CONSUMED
     │
P9  Site reads Gaps (N_CONSUMED) → writes soil N balance
     │
P10 Site reads neighbor Sites (avail_spec via ghost) → writes imported_seeds
```

Each arrow represents a **grid barrier** -- all agents must complete the
current priority before any agent starts the next. Within each priority,
all agents of the active breed execute in parallel across GPU threads.

### 5.3 Same-Tick Closure

The priority pipeline achieves **same-tick nutrient closure**:
- P3: trees declare N demand
- P4: gap aggregates demand, computes supply ratio
- P7: trees consume N based on supply ratio
- P8: gap aggregates consumption
- P9: site updates soil pools

All within a single simulation tick. GAPpy achieves this through sequential
loops; GGap achieves it through priority ordering with barriers.

---

## 6. Key Ecological Algorithms on GPU

### 6.1 365-Day Soil Biogeochemistry (P1, ~600 lines)

The most computationally intensive step function. Each site agent executes
365 sequential daily iterations in GPU registers:

**Per-day computation:**
1. **Stochastic climate**: Monthly means + stddev expanded to daily via
   Box-Muller normal samples with rational-polynomial inverse-CDF
   (all on-GPU, no host RNG)
2. **Hargreaves PET**: Solar declination from day-of-year, day length
   from latitude, extraterrestrial radiation (~8 trig terms),
   temperature-based evapotranspiration
3. **3-layer soil water routing**: Precipitation -> canopy interception
   (1 - exp(-0.001 x LAI)) -> A0 -> A -> Base, with field capacity /
   permanent wilting point constraints, slope-dependent runoff
4. **Temperature-dependent decomposition**: Q10 kinetics applied to
   3 soil layers (A0 litter -> A humus -> Base stable) with
   species-specific C:N ratios (A0_CN=30, SA_CN=4, SB_CN=20)
5. **Nitrogen mineralization**: Released from A-layer decomposition,
   accumulated into annual available N

**Annual outputs**: degree-days, dry-days, available N, fire/wind
intensity (stochastic: base 1%, scales with dry conditions to 15% max,
intensity sampled uniformly in [0.3, 1.0]).

**Why notable**: This is a Markovian inner loop (each day depends on
previous day's water state) -- it cannot be parallelized across days.
But it is embarrassingly parallel across sites/gaps.

### 6.2 Environmental Response Functions (P3)

Five multiplicative stress factors, all species-specific:

- **Temperature** (parabolic): fc = (x/opt)^a x ((max-x)/(max-opt))^b
  where a = (opt-min)/(max-min), b = (max-opt)/(max-min)
- **Drought** (sqrt): fc = sqrt((gamma - dry_days) / gamma) where
  gamma = [0.50, 0.45, 0.35, 0.25, 0.15, 0.05] by tolerance class.
  Dual metric for drought_tol=1 (uses base-layer dry days with
  conifer x 0.33 or deciduous x 0.2 multiplier)
- **Light** (sigmoidal): fc = c1 x (1 - exp(-c2 x (light - c3))) with
  5 shade-tolerance classes having different (c1, c2, c3) curves
- **Flood** (constant 1.0, preserved for fidelity)
- **Nutrient** (quadratic, at P7): fc = (c1 + c2*sf + c3*sf^2) x sf
  where sf = clamped N_supply_ratio, coefficients by 3 tolerance classes

Light availability at each tree's height from Beer-Lambert:
`light = exp(-0.40 x cumLAI / PLOTSIZE)`, read in O(1) from
pre-aggregated gap_lai array (built at P0).

### 6.3 UVAFME Growth Equation (P3)

Maximum diameter increment (Forska model):
```
diam_max = g x d x (1 - d*h / (Dmax*Hmax))
           / (2*h + a0 * exp(-a0*d / (Hmax-1.3)) * d)
```
where g = growth parameter (species-specific, adjusted by shade tolerance),
d = current diameter, h = current height, a0 = Forska shape parameter.

Height from Forska allometry:
```
h = 1.3 + (Hmax - 1.3) x (1 - exp(-a0 x d / (Hmax - 1.3)))
```

### 6.4 Parallel Recruitment (P6 + P7)

GAPpy recruits sequentially: iterate free slots, pick species one by one.

GGap converts to **probability-based parallel activation**:
- P6 computes `recruit_prob = nrenew / free_slots`
- P7: each free slot independently draws a Bernoulli trial
- Selected species via 2-pass CDF scan over template weights:
  1. Sum all template SEEDLING_WEIGHT values
  2. Draw uniform random, scan CDF until cumulative >= threshold
  3. Copy species traits from selected template to new tree

### 6.5 Canopy Self-Pruning (P7)

When environmental conditions at canopy base are poor:
```
forska_check = fc_degday x fc_drought x fc_flood x forska_shade x fc_nutrient
if forska_check <= 0.05:
    canopy_ht += 1  (advance one integer meter)
```
Twig biomass recomputed with narrower crown; pruned biomass added to litter.

### 6.6 Mortality (P7)

Two independent survival checks (both must pass to survive):
- **Age**: prob = age_check / max_age, where age_check from
  {4.605, 6.908, 11.51} by stress tolerance class
- **Growth stress**: if growth below threshold, prob = stress_check
  from {0.31, 0.34, 0.37, 0.40, 0.43} by tolerance class

Fire/wind: instant death if disturbance intensity > 0.01.

### 6.7 Distance-Weighted Seed Dispersal (P10)

Inter-site spatial coupling:
- Haversine distance matrix pre-computed at setup
  (EARTH_RADIUS_KM = 6371.0)
- Connectivity threshold: DISPERSAL_CUTOFF_FACTOR (5.0) x max dispersal distance
- Weight = exp(-distance / max_dispersal_dist) per species
- imported_seeds += seed_num x neighbor_avail x weight
- Divided by gap_count for even distribution across gaps

---

## 7. Two-Level Parallelism

### Level 1: GPU -- Thread-Parallel Agents Within a Rank

Each tick, a GPU kernel processes all local agents. Grid-stride loop
(128 threads/block) distributes agents across threads. All agents at the
same priority execute in parallel; grid barriers enforce sequential
ordering between priorities.

```
Tick N on one GPU:

  P0  ── all gap agents in parallel ──▶ barrier
  P1  ── all site agents in parallel ──▶ barrier
  ...
  P10 ── all site agents in parallel ──▶ barrier
```

### Level 2: MPI -- Concurrent Ranks Across GPUs

Multiple ranks execute simultaneously, each running the same priority
pipeline on its own subset of sites. Between ticks, ranks exchange
ghost data (neighbor-visible properties).

```
Time ──▶

Rank 0 (GPU 0):  [kernel] → [MPI sync] → [kernel] → ...
Rank 1 (GPU 1):  [kernel] → [MPI sync] → [kernel] → ...
```

GPU compute and MPI sync alternate (do not overlap).

---

## 8. Data Movement Optimization

### 8.1 Colocation Eliminates Intra-Site Communication

`partition_sites()` places each site's entire ensemble (site + gaps +
trees) on one rank. All intra-site neighbor access is **local GPU memory
reads**. Tree-to-Gap, Gap-to-Site, Site-to-Gap lookups never cross rank
boundaries.

Consequence: Tree and Gap properties are `neighbor_visible=False`. Only
Site states (8 floats) and site_avail_spec (num_species floats) cross
ranks. **MPI traffic scales with the number of sites, not the number of
trees or gaps.**

### 8.2 GPU-Resident Persistent Buffers

All property tensors, write buffers, CSR neighbor structures, and
breed-local arrays are allocated on GPU **once** (first tick) and reused
across all subsequent ticks. No per-tick CPU-to-GPU uploads of agent state.

Slack factors (1.5x agents, 2.0x CSR edges) amortize reallocation cost.
Kernel arguments are cached (`_cached_all_args`) to avoid rebuild overhead.

### 8.3 GPU-Aware MPI

When available (Cray MPICH on Frontier, Open MPI with ROCm/CUDA), ghost
data flows **GPU -> NIC -> GPU** via RDMA without CPU staging. Detected
at runtime via `MPICH_GPU_SUPPORT_ENABLED` / `OMPI_MCA_opal_cuda_support`.

Fallback: one `.get()` per peer (GPU -> CPU copy) before MPI send/recv.

### 8.4 Batched Ghost Exchange

Ghost exchange uses vectorized GPU gather/scatter:
1. One GPU gather per property across all peers (not per-peer)
2. Split into per-peer send buffers
3. Non-blocking Isend/Irecv with Waitall
4. One GPU scatter per property to ghost slots

This batching reduces GPU memory bandwidth overhead vs. per-peer gathers.

### 8.5 CSR Neighbor Format

Sparse ragged neighbor lists automatically converted to Compressed Sparse
Row format. Eliminates padding overhead; only actual edges stored. The
`_CSRBodyTransformer` (AST NodeTransformer) transparently rewrites user
code: `neighbors[i]` -> `values[offsets[agent_index] + i]`.

### 8.6 Flattening Objects into Tensors

GAPpy trees are Python objects with methods. GPU kernels can't serialize
objects. GGap flattens:
- Tree objects -> params[14] + states[11] (flat float arrays)
- Species objects -> species_traits[32][26] (read-only global tensor)
- Site climate -> site_configs[N][107] (read-only global tensor)
- All methods inlined into kernel code (Forska allometry, Beer-Lambert,
  response functions become explicit formulas)

---

## 9. SAGESim Framework Contributions

### 9.1 Automatic Double-Buffer Generation

SAGESim parses step function source with Python `ast` to detect writes.
Auto-generates write buffers and rewrites assignments:

```python
# User writes:
params[agent_index][DIAM] += increment
# Framework transforms to:
write_params[agent_index][DIAM] = params[agent_index][DIAM] + increment
```

Detects: direct assignments, subscript assignments, augmented assignments,
helper function calls (`set_this_agent_data_from_tensor`). Eliminates
manual race-condition handling.

### 9.2 Three-Phase Code Generation Pipeline

**Phase 1: CSR auto-transformation** (`_auto_transform_csr`)
- Replace `locations` parameter with `neighbor_offsets, neighbor_values`
- Rewrite loop bounds: `len(var)` -> `offsets[i+1] - offsets[i]`
- Remove sentinel checks (`!= -1`)
- Transform subscript access to CSR indexing

**Phase 2: Double-buffer rewriting**
- Identify written properties via AST walk
- Insert write parameters into function signature
- Replace write targets: `param[idx] = expr` -> `write_param[idx] = expr`
- Convert augmented assignments to explicit read-modify-write

**Phase 3: Fused kernel generation**
- Wrap all priorities in a persistent-thread grid-stride loop
- Insert grid barriers between priorities
- Generate breed dispatch (`if breed_id == 0: ... elif breed_id == 1: ...`)
- Handle breed-local array index maps and write buffers

Output: single CuPy `@jit.rawkernel` function containing all priorities.

### 9.3 Grid Barrier Mechanism

Software grid barriers using atomic operations:

```
syncthreads()                          // block-level sync
if threadIdx.x == 0:
    threadfence()                      // flush to global memory
    atomic_add(barrier_counter, 0, 1)  // signal block arrival
    while atomic_add(barrier_counter, 0, 0) < target:
        pass                           // spin until all blocks arrive
    threadfence()                      // fence before release
syncthreads()                          // release all threads
```

Co-residency safety: kernel launch limited to `max_blocks_per_sm x num_sms`
blocks (conservative default: 2 blocks/SM) to guarantee all blocks are
resident simultaneously (prevents deadlock).

`threadfence` exposed via CuPy JIT monkeypatch (`jit_extensions.py`).

### 9.4 Deterministic RNG with Logical IDs

Counter-based Philox-2x32-10 PRNG keyed on (seed, tick, logical_id, salt).
Logical IDs are stable identifiers (e.g., site_slot x 10_000_000 +
gap_index x 10_000 + tree_index) -- not runtime agent indices.

Guarantees identical results regardless of MPI rank count or agent
creation order. Also provides Xorshift-Multiply (3x faster, lower quality)
and Box-Muller normal distribution.

### 9.5 Selective Neighbor Visibility

Fine-grained `neighbor_visible` flag per property. Properties marked
False are never included in MPI ghost exchange. GGap exploits this:
only 2 of 11 property groups are neighbor-visible (Site states and
site_avail_spec), drastically reducing MPI data volume.

### 9.6 Breed-Local Arrays with Ghost Exchange

Per-breed GPU storage with independent allocation, double-buffering,
and optional ghost exchange. Each breed-local array has an index map
(`idx_map[global_agent_index] = breed_local_row`) for O(1) kernel access.

---

## 10. Species and Input Data

### 10.1 Species Inventory

32 species across 20 genera loaded from `UVAFME2012_specieslist.csv`:

- Acer (rubrum, saccharum), Aesculus, Betula, Carya (3 spp.),
  Cercis, Cornus, Diospyros, Fagus, Fraxinus (2 spp.), Juglans,
  Juniperus, Liquidambar, Liriodendron, Nyssa, Oxydendron,
  Pinus (3 spp.), Prunus, Quercus (7 spp.), Robinia, Sassafras, Tilia

Per-species: 26 traits including max age/diam/height, growth rate,
shade/drought/flood/fire/nutrient tolerance, C:N ratios, wood density,
leaf parameters, dispersal distance.

Shade-tolerance adjustments at load time:
- g_adj = g x ss[shade_tol-1], where ss = [1.1, 1.15, 1.2, 1.23, 1.25]
- leafdiam_a_adj = D_L x [1.5, 1.55, 1.6, 1.65, 1.7][shade_tol-1]
- Max height capped: min(Hmax, rootdepth x 80 / (1 + rootdepth))

### 10.2 Site Data

6 sites from `UVAFME2012_site.csv` (Oak Ridge, TN area):
- Grid pattern: lat 35.25-36.25, lon 83.75-84.25
- Elevation: 275.8 m
- Soil: field_cap=25, perm_wp=12.5
- Initial soil pools: A0_c=5, A0_n=0.1, A_c=33.7, A_n=2.6

Monthly climate from `UVAFME2012_climate.csv`:
- tmin: -2.39 to 19.55 C, tmax: 8.39 to 31.37 C
- Standard deviations from `UVAFME2012_climate_stddev.csv`

### 10.3 Typical Simulation Scale

```
Sites:        1-6
Gaps/site:    200
Tree slots:   1000 per gap
Total agents: 1 + 200 + 200,000 = 200,201 per site
Simulation:   500-1000 years
```

---

## 11. Output System

### 11.1 GAPpy-Compatible CSV Files

5 output files matching GAPpy format for direct comparison:

| File | Content | Key Columns |
|------|---------|-------------|
| site_data.csv | Annual climate | rain, pet, aet, grow_days, deg_days, dry_days, flood_days |
| soil_data.csv | Soil pools | a0_c/n, a_c/n, bl_c/n, avail_n, soilresp, biomass_c/n |
| genus_data.csv | Per-genus aggregates | 7 diameter classes (<=8, <=28, <=48, <=68, <=88, >88 cm), max_diam, max_hgt, LAI, basal_area, biomass |
| species_data.csv | Per-species aggregates | Same as genus |
| tree_data.csv | Individual trees (optional) | plot, tree, genus, species, diam, height, leaf_bm, biomC, biomN |

**Scaling factors**: plotscale = HEC_TO_M2 / plotsize = 10000 / 500 = 20.
Cross-gap averaging for genus/species data.

### 11.2 Visualization

`plot_outputs.py` generates 4 plot types:
- forest_dynamics: species/genus biomass time series
- soil_biogeochemistry: C/N pool evolution
- environmental_conditions: temperature, drought, flood
- summary_dashboard: multi-panel overview

---

## 12. Validation Infrastructure

### 12.1 GAPpy Reference Comparison

GGap uses identical CSV input files and produces GAPpy-compatible output
format. Validation approach:
1. Run same input on GAPpy (CPU) and GGap (GPU)
2. Compare CSV outputs at report intervals (every 10 years)
3. Hash tree litter values for deterministic comparison

### 12.2 Debug Tools

- `debug_mpi_diff.py`: Bisect runs to isolate GPU vs CPU differences.
  Prints site soil pools, climate, hashes of tree litter, template
  seedling weights, gap species distribution.

### 12.3 SAGESim Tests

4 test files in `SAGESim/tests/`:
- `test_double_buffer.py`: Double-buffering semantics
- `test_no_double_buffer.py`: Cross-priority data visibility
- `test_space.py`: Network topology and neighbor storage
- `test_worker_sync.py`: MPI synchronization barriers

---

## 13. Existing Performance Data

### 13.1 SAGESim Weak Scaling (Frontier, January 2026)

SIR model, MI250X GPUs, 5000 agents/GPU, 10 ticks:

| Nodes | GPUs | Agents | Time (s) | Efficiency |
|-------|------|--------|----------|------------|
| 1     | 8    | 40K    | 19.9     | 100%       |
| 2     | 16   | 80K    | 31.8     | 62.5%      |
| 5     | 40   | 200K   | 63.6     | 31.3%      |
| 10    | 80   | 400K   | 119.4    | 16.7%      |
| 17    | 136  | 680K   | 201.6    | 9.9%       |

**Bottleneck**: GPU kernel only ~10% of per-tick wall time. ~90% is
communication and synchronization. Ghost exchange is dominant cost
(CPU staging path pre-GPU-aware-MPI optimization).

### 13.2 GGap Estimated Performance

From run_one_site.py (single GPU, 200 gaps x 1000 slots x 1000 years):
- GPU setup: 2-10s (kernel compilation + species upload)
- Simulation: ~300-500s depending on GPU
- Approximate breakdown: soil P1 ~40%, tree growth P3/P7 ~35%,
  aggregations ~15%, renewal/recruitment ~10%

---

## 14. Code Volume

| Component | Lines of Code |
|-----------|---------------|
| GGap gap/ directory | ~9,500 |
| GGap total (all .py) | ~27,000 |
| SAGESim sagesim/ | ~9,900 |
| GAPpy src/ | ~3,500 |

Step function sizes:
- soil_step.py (P1): ~600 lines (largest -- 365-day loop)
- tree_actual_growth_step.py (P7): ~400 lines
- tree_potential_growth_step.py (P3): ~370 lines
- tree_template_renewal_step.py (P5): ~200 lines
- gap_litter_aggregate_step.py (P0): ~165 lines
- gap_recruit_aggregate_step.py (P6): ~124 lines
- gap_demand_aggregate_step.py (P4): ~89 lines
- Others: <50 lines each

---

## 15. Physics Constants Reference

### Tree Growth
| Constant | Value | Description |
|----------|-------|-------------|
| STD_HT | 1.3 m | Standard height for DBH measurement |
| TC_KG | pi/80 = 0.039270 | Stem volume constant |
| XT | -0.40 | Beer-Lambert extinction coefficient |
| PLOTSIZE | 500.0 m^2 | Gap area |
| STEM_C_N | 450.0 | Stem carbon-to-nitrogen ratio |
| CON_LEAF_C_N | 60.0 | Conifer leaf C:N |
| DEC_LEAF_C_N | 40.0 | Deciduous leaf C:N |
| CON_LEAF_B | 1.3 | Conifer leaf biomass multiplier |
| UNIT_CONV | 0.02 | kg (tree) -> tn/ha (soil) |
| MAX_HEIGHT_BINS | 50 | LAI profile resolution |

### Soil Biogeochemistry
| Constant | Value | Description |
|----------|-------|-------------|
| AO_CN_0 | 30.0 | A0 layer initial C:N |
| SA_CN_0 | 4.0 | A layer initial C:N |
| SB_CN_0 | 20.0 | Base layer initial C:N |
| AO_RESP | 5.24e-4 | A0 daily respiration rate |
| SA_RESP | 1.24e-5 | A daily respiration rate |
| SB_RESP | 2.74e-7 | Base daily respiration rate |
| PRCP_N | 0.00002 | Atmospheric N deposition (tn/cm precip) |

### Hargreaves PET
| Constant | Value |
|----------|-------|
| H_B | 0.017214 |
| H_AS | 0.409 |
| H_AC | 0.033 |
| H_PHASE | -1.39 |
| H_AMP | 37.58603 |
| H_COEFF | 0.000093876 |
| H_ADDON | 17.8 |

### Spatial
| Constant | Value | Description |
|----------|-------|-------------|
| DISPERSAL_CUTOFF_FACTOR | 5.0 | Connectivity radius multiplier |
| EARTH_RADIUS_KM | 6371.0 | For haversine distance |

---

## 16. Paper Outline Suggestion

### Title
GGap: A GPU-Accelerated Agent-Based Forest Gap Model on the
Site-Gap-Tree Hierarchy

### Structure

**1. Introduction**
- Forest gap models for climate projection
- Computational bottleneck: individual-tree resolution x long timescales
  x ensemble requirements
- Contribution: GPU-accelerated UVAFME implementation preserving full
  ecological fidelity

**2. Background**
- UVAFME model overview and its 3-loop growth constraint
- Gap model computational characteristics (shared nitrogen pool,
  365-day soil loop, light competition)
- Why naive GPU parallelization fails

**3. GGap Design**
- Site-Gap-Tree agent hierarchy and colocation invariant
- 11-priority kernel pipeline resolving sequential dependencies
- Gap agents as intermediaries for reduction/broadcast patterns
- Probability-based parallel recruitment

**4. Implementation**
- SAGESim framework: automatic double-buffering, CSR transformation,
  code generation pipeline
- 365-day soil biogeochemistry as a GPU kernel
- Stochastic processes with deterministic reproducibility (Philox RNG)
- GPU-aware MPI for inter-site dispersal

**5. Data Movement Optimization**
- Colocation: MPI traffic scales with sites not trees
- Selective neighbor visibility
- GPU-resident persistent buffers
- Batched ghost exchange

**6. Results**
- Validation against GAPpy reference (identical output)
- Single-GPU speedup over GAPpy
- Multi-GPU weak/strong scaling on Frontier
- Timing breakdown by priority (identify bottleneck in soil P1)

**7. Discussion**
- Soil step as case study: Markovian inner loop on GPU
- Generalizability to other biogeochemical models
- Path to further optimization (thread-parallel daily loop,
  compute-communication overlap)
