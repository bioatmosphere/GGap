# GGap Implementation Logic

This document explains the detailed logic of the GGap model implementation, including the step function execution flow, data dependencies, and how it maps to GAPpy/UVAFME processes.

## Overview

GGap is a GPU-accelerated forest gap dynamics model that implements GAPpy/UVAFME processes using the SAGESim agent-based framework. It uses a two-phase tree growth pattern with soil-first ordering and renewal-last ordering, matching GAPpy's annual cycle.

### Agent Hierarchy

```
Site (1 per simulation)
  └── Gap (N per site, default 200)
        └── Tree (M per gap, default 1000 slots)
              ├── Living trees (is_alive = 1)
              ├── Dormant slots (is_alive = 0)
              └── Template trees (is_alive = -1, one per species)
```

### Why Gap Agents?

Gap agents serve as intermediaries between Site and Trees to:

1. **Memory Efficiency**: Without Gaps, Site would need 200,000+ tree neighbors. Gap keeps neighbor lists short (~200 gaps per site, ~1000 tree slots per gap).

2. **Data Aggregation**: Gap collects litter from trees before passing to Site, reducing Site's workload.

3. **Future Extensibility**: Gap enables gap-to-gap interactions (seed dispersal, edge effects).

### Initialization

GGap matches GAPpy's initialization: forests start **empty** (all dormant slots, no living trees). The forest establishes entirely through the renewal process. This differs from some UVAFME implementations that pre-populate plots with trees.

`initialize_trees()` creates:
- One **template** per species (is_alive = -1) for permanent species pool preservation
- N **dormant slots** (is_alive = 0) that get activated through recruitment
- Zero initial living trees

---

## GAPpy Annual Cycle Mapping

GAPpy's annual cycle (`model.py:1005-1023`):
```
bio_geo_climate → canopy → growth → mortality → renewal
```

Our priority mapping:
```
P0: Gap litter aggregate   ─┐
P1: Site soil               ─┘ bio_geo_climate (soil first)
P2: Tree potential growth      canopy + growth phase 1 (env stress, light, n_demand)
P3: Gap N demand aggregate  ─┐
P4: Site nutrient            │ growth phase 2 (nutrient feedback)
P5: Gap sync                ─┘
P6: Tree actual growth         growth finalization + mortality + renewal (last)
```

---

## Step Function Execution Flow

Each simulation tick executes seven step functions in priority order:

```
┌─────────────────────────────────────────────────────────────────┐
│ TICK N                                                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Priority 0: gap_litter_aggregate_step (Gap)                    │
│  ─────────────────────────────────────────────────────────────  │
│                                                                 │
│  1. AGGREGATE LITTER FROM TREES                                 │
│     - Loop through Tree neighbors                               │
│     - Sum litter_c, litter_n (above-ground) from prev tick P6   │
│     - Sum litter_c_bg, litter_n_bg (below-ground roots)         │
│     - Count living trees and dormant slots                      │
│                                                                 │
│  2. READ GROWMAX FROM TEMPLATES                                 │
│     - For template neighbors: read env_stress (= regrowth)     │
│     - growmax = max regrowth across all templates               │
│     - env_stress written at P6 of previous tick                 │
│                                                                 │
│  3. DENSITY-BASED RECRUITMENT (GAPpy model.py:833-837)          │
│     - total_capacity = living_count + dormant_count             │
│     - max_renew = total_capacity * growmax - living_count       │
│     - Cap at half capacity: min(max_renew, total_capacity*0.5)  │
│     - Floor at 3: max(nrenew, 3)                                │
│     - Cap by available: min(nrenew, total_capacity - living)    │
│     - Cap by dormant slots: min(nrenew, dormant_count)          │
│                                                                 │
│  WRITES:                                                        │
│     - states: litter_accum_c/n, litter_accum_c/n_bg (for Site) │
│     - states: num_to_recruit, recruit_rand_seed (for Trees)     │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Priority 1: site_soil_step (Site)                              │
│  ─────────────────────────────────────────────────────────────  │
│                                                                 │
│  1. READ LITTER FROM GAPS                                       │
│     - Sum litter_accum_c/n + litter_accum_c/n_bg from Gaps     │
│     - Above-ground litter → A0 layer (pulse at year start)     │
│     - Below-ground litter → A layer (pulse at year start)      │
│                                                                 │
│  2. MONTHLY CLIMATE PERTURBATION                                │
│     - For each month (0-11), generate pseudo-random perturbation│
│     - Temperature perturbation clamped to [-1, 1]              │
│     - Precipitation perturbation clamped to [-0.5, 0.5]        │
│     - Apply: tmin/tmax += pert * std_dev, prcp += pert * std   │
│     - Compute annual precip and dry_days from perturbed climate │
│       dry_days = max(0, 100 - annual_precip_cm)                │
│                                                                 │
│  3. DAILY LOOP (365 days)                                       │
│     For each day:                                               │
│                                                                 │
│     a. CLIMATE INTERPOLATION                                    │
│        - Determine month from day of year                       │
│        - Get daily tmin, tmax, precip from perturbed monthly    │
│        - Track freeze days                                      │
│        - Accumulate atmospheric N from precipitation (mm units) │
│        - Convert precip mm → cm for water balance               │
│                                                                 │
│     b. POTENTIAL EVAPOTRANSPIRATION (Hamon method)              │
│        - Solar declination, day length, PET from temperature    │
│                                                                 │
│     c. SOIL WATER BALANCE                                       │
│        - Route precipitation through canopy → A0 → A → Base    │
│        - Apply slope runoff                                     │
│        - Evapotranspiration draws from layers                   │
│        - Track flood days (A layer saturated)                   │
│                                                                 │
│     d. SOIL DECOMPOSITION (three-layer)                         │
│        - A0 respiration → transfer to A layer                   │
│        - A layer respiration → N mineralization (avail_n)       │
│        - A layer transfer to Base layer                         │
│        - Base layer respiration                                 │
│        - Temperature and moisture adjustments                   │
│        - Division guards: skip decomposition when C/N ≤ 0.001  │
│          (prevents NaN when pools deplete with empty start)     │
│                                                                 │
│  3. FIRE PROBABILITY                                            │
│     - Base probability: 1%, increases with dry conditions       │
│     - Cap at 15%, stochastic fire occurrence                    │
│     - Fire intensity: 0.3 - 1.0                                │
│                                                                 │
│  WRITES:                                                        │
│     - params: soil pools (A0/A/BL C, N, W)                     │
│     - states: avail_n, deg_days, dry_days, flood_days,          │
│              fire_intensity                                      │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Priority 2: tree_potential_growth_step (Tree)                  │
│  ─────────────────────────────────────────────────────────────  │
│  Executes for living trees only (is_alive > 0.5)               │
│                                                                 │
│  1. ENVIRONMENTAL RESPONSE                                      │
│     - Read climate from Gap states (deg_days, dry_days, etc.)   │
│     - Read neighbor heights from Tree states_db (light comp.)   │
│     - Calculate growth factors:                                 │
│       fc_degday:  parabolic temperature response                │
│       fc_drought: exponential drought response (tol 1-5)        │
│       fc_flood:   linear flood response (tol 1-6)              │
│       fc_light:   Beer-Lambert canopy light attenuation         │
│                                                                 │
│  2. POTENTIAL GROWTH                                            │
│     - env_stress = fc_degday * fc_drought * fc_light * fc_flood │
│       (NO fc_nutrient — applied later at P6)                    │
│     - diam_max = optimal diameter increment for current size    │
│     - Compute potential biomass change                          │
│     - Compute n_demand from potential growth                    │
│                                                                 │
│  WRITES:                                                        │
│     - params: env_stress, diam_max, light_avail, fc_* factors   │
│     - states: n_demand (Gap aggregates at P3)                   │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Priority 3: gap_demand_aggregate_step (Gap)                    │
│  ─────────────────────────────────────────────────────────────  │
│                                                                 │
│  - Sum n_demand from all living Tree neighbors (from P2)        │
│  - Write total_n_demand to Gap states (Site reads at P4)        │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Priority 4: site_nutrient_step (Site)                          │
│  ─────────────────────────────────────────────────────────────  │
│                                                                 │
│  - Read avail_n from own states (from P1, same tick)            │
│  - Sum total_n_demand from all Gap neighbors (from P3)          │
│  - n_supply_ratio = avail_n / total_n_demand                    │
│  - Cap at 2.0, guard against small demand (< 0.0001)           │
│  - Write n_supply_ratio to own states (Gap reads at P5)         │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Priority 5: gap_sync_step (Gap)                                │
│  ─────────────────────────────────────────────────────────────  │
│                                                                 │
│  - Read climate + n_supply_ratio from Site neighbor              │
│  - Copy all values to own states (Trees read at P2 and P6)     │
│  - Clear litter accumulators (consumed by Site at P1)           │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Priority 6: tree_actual_growth_step (Tree)                     │
│  ─────────────────────────────────────────────────────────────  │
│  Three branches based on is_alive:                              │
│                                                                 │
│  BRANCH 1: LIVING TREES (is_alive > 0.5)                       │
│  ─────────────────────────────────────────                      │
│  a. NUTRIENT RESPONSE                                           │
│     - Read n_supply_ratio from Gap (written at P5, same tick)   │
│     - fc_nutrient based on lownutr_tol (1-3)                   │
│                                                                 │
│  b. FINAL GROWTH                                                │
│     - growth_factor = env_stress (from P2) * fc_nutrient        │
│     - diam_increment = diam_max * growth_factor                 │
│     - Update diameter, height (Forska equation), biomass        │
│                                                                 │
│  c. CANOPY SELF-PRUNING                                         │
│     - When growth_factor <= 0.05, crown base rises              │
│     - Biomass difference becomes litter                         │
│                                                                 │
│  d. MORTALITY (GAPpy two-check model)                           │
│     - Age survival: age_check / max_age (tol 1-3)              │
│     - Growth survival: stress_check if growth below threshold   │
│     - Tree dies if EITHER check fails                           │
│     - Dead tree: all biomass → litter (70% above, 30% below)   │
│     - Surviving tree: annual leaf litter                        │
│                                                                 │
│  e. LITTER OUTPUT                                               │
│     - Living: leaf litter (deciduous 100%, conifer ~32%)        │
│     - Dying: all biomass as litter (70/30 above/below split)    │
│                                                                 │
│  BRANCH 2: TEMPLATE RENEWAL (is_alive < -0.5)                  │
│  ─────────────────────────────────────────────                  │
│  Matches GAPpy renewal() (model.py:792-982)                    │
│                                                                 │
│  a. ENVIRONMENTAL RESPONSE                                      │
│     - Read climate from Gap (same-tick from P5)                 │
│     - Compute fc_degday, fc_drought, fc_flood, fc_nutrient      │
│     - Compute fc_light at ground level (all neighbors shade)    │
│                                                                 │
│  b. REGROWTH                                                    │
│     - regrowth = fc_degday * fc_drought * fc_flood              │
│                  * fc_nutrient * fc_light                        │
│     - If regrowth <= 0.05: set to 0                             │
│                                                                 │
│  c. SAME-SPECIES COUNT                                          │
│     - Count living trees with matching species_id (avail_spec)  │
│                                                                 │
│  d. SEEDBANK UPDATE (GAPpy model.py:843-856)                   │
│     - seedbank += invader + seed * avail_spec                   │
│                  + sprout * avail_spec                           │
│     - If regrowth >= 0.05: seedling += seedbank; seedbank = 0   │
│     - Else: seedbank *= seed_surv (decay)                       │
│                                                                 │
│  e. RECRUITMENT WEIGHT                                          │
│     - weight = seedling * regrowth                              │
│                                                                 │
│  f. SEEDLING SURVIVAL (GAPpy model.py:969-972)                  │
│     - seedling *= seedling_lg (annual survival rate)            │
│                                                                 │
│  g. OUTPUTS                                                     │
│     - params: seedbank, seedling, env_stress (= regrowth)       │
│     - states_db: seedling_weight (dormant reads same priority)  │
│                                                                 │
│  BRANCH 3: DORMANT ACTIVATION (is_alive == 0)                  │
│  ─────────────────────────────────────────────                  │
│  Recruitment happens last, matching GAPpy ordering.             │
│  Seedlings don't grow until next tick.                          │
│                                                                 │
│  a. READ RECRUITMENT INFO                                       │
│     - num_to_recruit from Gap (written at P0)                   │
│     - recruit_rand_seed for deterministic selection             │
│                                                                 │
│  b. SLOT PRIORITY                                               │
│     - Hash agent_index with rand seed for selection priority    │
│     - If slot_priority < recruit_threshold: recruit             │
│                                                                 │
│  c. SPECIES SELECTION                                           │
│     - Iterate template neighbors                                │
│     - Read seedling_weight from states_db (written same P6)     │
│     - Select species weighted by seedling_weight                │
│     - Minimum weight 0.01 per template (ensures all species     │
│       have a chance even with zero seedlings)                   │
│                                                                 │
│  d. INITIALIZE SEEDLING                                         │
│     - Copy species traits [0-21] + seed_surv + seedling_lg      │
│     - Seedling diameter: uniform [0.5, 2.5] cm                  │
│       (approximates GAPpy's 1.5 + N(0,1))                      │
│     - Height from Forska equation                               │
│     - Set is_alive = 1.0 (visible next tick via double buffer)  │
│                                                                 │
│  ALL BRANCHES WRITE:                                            │
│     - states: litter_c/n, litter_c/n_bg (0 for dormant/template)│
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Diagram

```
                    TICK N                           TICK N+1
                    ══════                           ════════

    ┌────────────────────────────────────────────────────────────┐
    │                                                            │
    │   P0: Gap reads Tree litter/counts from prev tick          │
    │        → writes litter_accum, num_to_recruit               │
    │                     │                                      │
    │                     ↓                                      │
    │   P1: Site reads Gap litter_accum                          │
    │        → soil decomposition (365 daily steps)              │
    │        → writes avail_n, flood_days, fire_intensity        │
    │                     │                                      │
    │                     ↓                                      │
    │   P2: Trees read Gap climate (from P5 prev tick)           │
    │        + read Tree heights (states_db, double buffered)    │
    │        → potential growth, light competition               │
    │        → writes env_stress, diam_max, n_demand             │
    │                     │                                      │
    │                     ↓                                      │
    │   P3: Gap reads Tree n_demand                              │
    │        → writes total_n_demand                             │
    │                     │                                      │
    │                     ↓                                      │
    │   P4: Site reads own avail_n + Gap total_n_demand          │
    │        → n_supply_ratio = avail_n / total_n_demand         │
    │                     │                                      │
    │                     ↓                                      │
    │   P5: Gap reads Site climate + n_supply_ratio              │
    │        → relays to own states, clears litter accumulators  │
    │                     │                                      │
    │                     ↓                                      │
    │   P6: Trees read Gap n_supply_ratio (same tick!)           │
    │        Living: final growth + mortality + litter            │
    │        Templates: renewal (seedbank/seedling/weight)       │
    │        Dormant: species selection + activation              │
    │                     │                                      │
    │                     ↓                                      │
    │   ┌─────────────────────────────────────────────┐          │
    │   │ SWAP states_db (write → read)               │          │
    │   │ New is_alive, diam, height, seedling_weight │          │
    │   │ visible to all readers next tick             │          │
    │   └─────────────────────────────────────────────┘          │
    │                                                            │
    └────────────────────────────────────────────────────────────┘
```

---

## Mapping to GAPpy Processes

| GAPpy Process | GGap Implementation |
|---------------|---------------------|
| `bio_geo_climate()` | P1: site_soil_step (daily climate + soil decomposition) |
| `canopy()` (light competition) | P2: tree_potential_growth_step (Beer-Lambert) |
| `growth()` first loop (env stress) | P2: tree_potential_growth_step |
| `growth()` second loop (nutrient + final growth) | P6: tree_actual_growth_step (living branch) |
| `growth()` third loop (canopy pruning) | P6: tree_actual_growth_step (self-pruning section) |
| `mortality()` | P6: tree_actual_growth_step (mortality section) |
| `renewal()` | P6: tree_actual_growth_step (template + dormant branches) |
| `sum_litter()` (in bio_geo_climate) | P0: gap_litter_aggregate_step |
| N ratio computation | P3→P4→P5 chain: demand aggregate → nutrient allocation → sync |

---

## Environmental Response Functions

### Temperature Response (fc_degday)
Parabolic response curve (GAPpy species.py):
```
fc_degday = ((dd - dd_min)/(dd_opt - dd_min))^a * ((dd_max - dd)/(dd_max - dd_opt))^b
where a = (dd_opt - dd_min)/(dd_max - dd_min)
      b = (dd_max - dd_opt)/(dd_max - dd_min)
```

### Drought Response (fc_drought)
Inverse square root based on drought tolerance (1-5):
```
gamma = [0.50, 0.45, 0.35, 0.25, 0.15] by tolerance class
if dry_days < gamma * 365:
    fc_drought = sqrt((gamma*365 - dry_days) / (gamma*365))
```

### Light Response (fc_light)
Exponential saturation based on shade tolerance (1-5):
```
fc_light = c1 * (1 - exp(-c2 * (light - c3)))
where c1, c2, c3 vary by shade tolerance class
```

### Nutrient Response (fc_nutrient)
Linear response below threshold based on lownutr_tol (1-3):
```
gamma_n = [0.5, 0.25, 0.05] by tolerance class
if n_supply_ratio < gamma_n:
    fc_nutrient = n_supply_ratio / gamma_n
```

### Flood Response (fc_flood)
Linear reduction based on flood_tol (1-6):
```
threshold = (1.0 - (flood_tol - 1) * 0.1) * 365
if flood_days >= threshold:
    fc_flood = 0
else:
    fc_flood = 1 - flood_days / threshold
```

---

## Light Competition

Beer-Lambert light attenuation with species-specific extinction coefficients:

```
for each taller neighbor:
    neighbor_lai = (diam^2 * 0.01) / canopy_depth
    overlap = min(neighbor_height - my_height, canopy_depth)

    if neighbor_evergreen:
        xt = -0.70  (Conifer)
    else:
        xt = -0.80  (Deciduous)

    total_lai += neighbor_lai * overlap * (xt / -0.80)

light_avail = exp(-0.80 * total_lai)
```

Templates compute light at ground level (height = 0), so all living neighbors shade them.

---

## Renewal Process (GAPpy renewal())

Renewal is the last step in each tick (P6), matching GAPpy's ordering where `renewal()` runs after `mortality()`. This ensures seedlings don't grow until the next year.

### 1. Template Renewal (P6, templates)

Each template represents one species and maintains persistent seedbank/seedling state:

**Seedbank accumulation:**
```
seedbank += invader + seed * avail_spec + sprout * avail_spec
```
- `invader`: base colonization rate (species trait from CSV)
- `seed`: seed production rate per tree
- `avail_spec`: count of living trees of same species

**Seedbank → Seedling transfer:**
```
if regrowth >= 0.05:
    seedling += seedbank    (conditions favorable, seeds germinate)
    seedbank = 0
else:
    seedbank *= seed_surv   (conditions poor, seedbank decays)
```

**Recruitment weight:**
```
weight = seedling * regrowth
```
Higher weight means this species has both more seedlings AND better environmental conditions.

**Annual seedling survival:**
```
seedling *= seedling_lg
```

### 2. Density-Based nrenew (P0, Gap)

The number of recruits per tick is density-dependent (GAPpy model.py:833-837):
```
total_capacity = living_count + dormant_count
max_renew = total_capacity * growmax - living_count
nrenew = clamp(max_renew, min=3, max=total_capacity*0.5)
nrenew = min(nrenew, total_capacity - living_count, dormant_count)
```
- `growmax`: max regrowth across all templates (from prev tick)
- When forest is near capacity, nrenew is small
- When forest is sparse, nrenew can be up to half capacity

### 3. Dormant Slot Activation (P6, dormant slots)

Dormant slots compete for recruitment based on a priority hash:
```
slot_priority = hash(agent_index, recruit_rand_seed) / 10000
if slot_priority < num_to_recruit / 100:
    select species from templates weighted by seedling_weight
    copy traits, initialize as seedling
```

Species selection uses templates' `seedling_weight` from `states_db` (double-buffered, written same priority). A minimum weight of 0.01 ensures all species have a baseline recruitment chance.

### One-Tick Lag (Double Buffering)

Template renewal writes to `states_db` at P6 tick T. Dormant slots read from `states_db` at P6 tick T (same priority, same buffer). However, the newly activated seedling's `is_alive = 1.0` is written to the states_db **write buffer** and becomes visible at P2 of tick T+1 after the buffer swap. This means seedlings don't participate in light competition or growth until the next year.

---

## Mortality Process (GAPpy Two-Check Model)

Tree survival requires passing BOTH independent checks:

**Age survival (GAPpy tree.py:178-202):**
```
age_check = [4.605, 6.908, 11.51] by age_tol (1-3)
age_mort_prob = age_check / max_age
if random < age_mort_prob: age_dies
```

**Growth survival (GAPpy tree.py:204-222):**
```
stress_check = [0.31, 0.34, 0.37, 0.40, 0.43] by stress_tol (1-5)
if growth_below_threshold AND random < stress_check: growth_dies
```

**Combined:** tree dies if age_dies OR growth_dies.

---

## Soil Biogeochemistry

Three-layer model (A0 → A → Base):

### A0 Layer (Litter)
- Receives above-ground tree litter as pulse at year start
- Respiration rate: `AO_RESP = 5.24e-4`
- Transfer to A layer based on C/N ratio
- Division guard: skip decomposition when C/N ratio ≤ 0.001

### A Layer (Humus)
- Receives from A0 decomposition + below-ground root litter (pulse at year start)
- Respiration rate: `SA_RESP = 1.24e-5`
- **N mineralization**: main source of available N
- N efficiency: `max(0.5, (sa_cn - SA_CN_0) / sa_cn)`
- Transfer to Base layer
- Division guard: skip mineralization when C/N ratio ≤ 0.001

### Base Layer
- Receives from A layer
- Respiration rate: `SB_RESP = 2.74e-7`
- Slow turnover, long-term storage

### Division Guards (Empty-Start Protection)

With empty-start initialization, soil pools can deplete before the first trees produce litter. When C/N ratio reaches 0 (carbon = 0 but nitrogen > 0), direct division would produce NaN. Guards skip decomposition when C/N ≤ 0.001, which is physically correct: zero carbon means nothing to decompose.

Note: GAPpy's `soil.py` does not include these guards because its renewal process produces trees (and litter) quickly enough that pools never fully deplete. The GPU double-buffering lag in GGap can delay litter production, making this guard necessary.

### Moisture Effects
- Decomposition scaled by moisture function
- Optimal moisture around 30% saturation for A0, 80% for A layer
- Reduced at both extremes

### Temperature Effects
- `temp_adjustment = 3.0^(0.1 * (temp - 1))` for A0
- `temp_adjustment = 2.5^(0.1 * (temp - 1))` for A layer
- No decomposition below -5C

---

## Fire Dynamics

1. **Fire probability** (Site P1):
   - Base: 1% annual
   - Increases with dry soil moisture
   - Cap: 15% annual

2. **Fire intensity** (if fire occurs):
   - Range: 0.3 - 1.0
   - Based on dry conditions

3. **Fire mortality** (Tree P6):
   - Currently simplified; full fire mortality not yet implemented

---

## Step Function Files

| File | Priority | Breed | Purpose |
|------|----------|-------|---------|
| `step_functions/gap/gap_litter_aggregate_step.py` | P0 | Gap | Aggregate litter, density-based nrenew |
| `step_functions/site/soil_step.py` | P1 | Site | Soil biogeochemistry (365-day loop) |
| `step_functions/tree/tree_potential_growth_step.py` | P2 | Tree | Env stress, light competition, n_demand |
| `step_functions/gap/gap_demand_aggregate_step.py` | P3 | Gap | Aggregate N demand from trees |
| `step_functions/site/site_nutrient_step.py` | P4 | Site | Compute n_supply_ratio |
| `step_functions/gap/gap_sync_step.py` | P5 | Gap | Relay climate + clear accumulators |
| `step_functions/tree/tree_actual_growth_step.py` | P6 | Tree | Final growth + mortality + renewal + recruitment |

---

## Output System

GGap produces 5 GAPpy-compatible CSV files via `output_utils.py`:

### Scaling

All output values are scaled to per-hectare units using GAPpy conventions:
- `plotsize = 500 m²` (area of one gap)
- `plotscale = HEC_TO_M2 / plotsize = 20` (scale factor from gap to hectare)
- `plotadj = plotscale / num_gaps` (per-gap contribution to hectare average)
- `plotrenorm = 1 / (plotsize * num_gaps)` (basal area normalization)

### CSV Files

| File | Key Columns | Notes |
|------|------------|-------|
| `site_data.csv` | year, lat, lon, elevation, slope, deg_days, flood_days, dry_days, annual_rain, grow_days | Annual site conditions |
| `soil_data.csv` | year, A0_C/N, A_C/N, BL_C/N, avail_n, biomC, biomN | Soil pools in tn/ha |
| `genus_data.csv` | year, genus, biomC/N (mean/std), basal_area, max_ht/diam, n_trees, diam_cats (6 classes) | Per-genus averages across gaps |
| `species_data.csv` | year, genus, species, biomC/N (mean/std), basal_area, max_ht/diam, n_trees, diam_cats | Per-species averages |
| `tree_data.csv` | year, gap_id, species_id, diam, height, biomC, biomN, leaf_bm, age, canopy_ht | Individual tree data (optional) |

### Biomass Calculation

- **Species/genus output**: `biomC = tree.biomC + leaf_bm` (always includes leaf biomass)
- **Soil output**: `biomC += leaf_bm` only for conifers (evergreen trees)
- Diameter categories: <=8, <=28, <=48, <=68, <=88, >88 cm
