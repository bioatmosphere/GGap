# GGap Implementation Logic

This document explains the detailed logic of the GGap model implementation, including the step function execution flow, data dependencies, and how it maps to GAPpy/UVAFME processes.

## Overview

GGap is a GPU-accelerated forest gap dynamics model that implements GAPpy/UVAFME processes using the SAGESim agent-based framework. It uses an 11-priority execution pipeline (P0-P10) with soil-first ordering, two-phase tree growth, and renewal-last ordering, matching GAPpy's annual cycle.

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
P0:  Gap litter aggregate    ─┐
P1:  Site soil                ─┘ bio_geo_climate (soil first)
P2:  Gap climate relay           climate Site→Gap (eliminates 1-tick lag)
P3:  Tree potential growth       canopy + growth phase 1 (env stress, light, n_demand)
P4:  Gap N demand aggregate  ─┐
P5:  Gap sync (N ratio)      ─┘ growth phase 2 (per-gap nutrient feedback)
P6:  Tree template renewal       renewal phase 1 (seedbank/seedling/weight)
P7:  Gap recruit aggregate       renewal phase 2 (density-based nrenew)
P8:  Tree actual growth          growth finalization + mortality + recruitment
P9:  Gap N consumed aggregate ─┐
P10: Site N balance           ─┘ same-tick N balance
```

---

## Step Function Execution Flow

Each simulation tick executes eleven step functions in priority order:

```
┌─────────────────────────────────────────────────────────────────┐
│ TICK N                                                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Priority 0: gap_litter_aggregate_step (Gap)                    │
│  ─────────────────────────────────────────────────────────────  │
│                                                                 │
│  1. AGGREGATE LITTER + LAI FROM TREES                           │
│     - Loop through Tree neighbors                               │
│     - Sum litter_c, litter_n (above-ground) from prev tick P8   │
│     - Compute per-tree LAI = dc² × leafdiam_a                  │
│     - Sum total LAI, normalize by PLOTSIZE=500                  │
│     - Sum total_seedling_weight across templates                │
│     - Count living trees and dormant slots                      │
│                                                                 │
│  WRITES:                                                        │
│     - states: litter_accum_c/n (for Site at P1)                │
│     - states: total_lai (for Site at P1, canopy water balance)  │
│     - states: total_seedling_weight (for P6 proportional decr.) │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Priority 1: site_soil_step (Site)                              │
│  ─────────────────────────────────────────────────────────────  │
│                                                                 │
│  1. READ LITTER + LAI FROM GAPS                                 │
│     - Sum litter_accum_c/n from Gaps (unit scaled: UNIT_CONV)   │
│     - Average total_lai across Gaps for canopy water balance    │
│     - Above-ground litter → A0 layer (pulse at year start)     │
│                                                                 │
│  2. MONTHLY CLIMATE PERTURBATION                                │
│     - For each month (0-11), generate pseudo-random perturbation│
│     - BSM inverse normal CDF (rational polynomial approx)       │
│     - Temperature perturbation clamped to [-1, 1]              │
│     - Precipitation perturbation clamped to [-0.5, 0.5]        │
│     - Apply: tmin/tmax += pert * std_dev, prcp += pert * std   │
│     - Compute annual precip and dry_days from perturbed climate │
│                                                                 │
│  3. DAILY LOOP (365 days)                                       │
│     For each day:                                               │
│                                                                 │
│     a. CLIMATE INTERPOLATION (linear between monthly midpoints) │
│        - Get daily tmin, tmax from perturbed monthly values     │
│        - Track freeze days, accumulate degree days (base 5°C)   │
│        - Accumulate atmospheric N from precipitation            │
│                                                                 │
│     b. PRECIPITATION (Bernoulli rain-day allocation, cov365a)   │
│        - Per month: raindays = min(25, prcp/4+1)               │
│        - Each day: hash-based probability of rain event         │
│        - Rain day gets prcp/ik; dry day gets 0                  │
│        - Remainder dumped on last day of month                  │
│                                                                 │
│     c. POTENTIAL EVAPOTRANSPIRATION (Hargreaves method)         │
│        - Solar declination, extraterrestrial radiation           │
│        - PET from tmin, tmax, temperature range                 │
│                                                                 │
│     d. SOIL WATER BALANCE                                       │
│        - Route precipitation through canopy → A0 → A → Base    │
│        - Apply slope runoff, accumulate annual_runoff           │
│        - Evapotranspiration draws from layers                   │
│        - Track flood days (A layer saturated)                   │
│                                                                 │
│     e. SOIL DECOMPOSITION (three-layer)                         │
│        - A0 respiration → transfer to A layer                   │
│        - A layer respiration → N mineralization (avail_n)       │
│        - A layer transfer to Base layer                         │
│        - Base layer respiration                                 │
│        - Temperature and moisture adjustments                   │
│        - Division guards: skip decomposition when C/N ≤ 0.001  │
│                                                                 │
│  4. FIRE/WIND PROBABILITY                                       │
│     - Fire: base 1%, increases with dry conditions, cap 15%    │
│     - Wind: from CSV wind_prob, fire takes precedence           │
│     - Stochastic intensity: fire 0.3-1.0, wind 0.5-1.0        │
│                                                                 │
│  WRITES:                                                        │
│     - params: soil pools (A0/A/BL C, N, W), annual_runoff      │
│     - states: deg_days, dry_days, avail_n, flood_days,          │
│              fire_intensity, wind_intensity, dry_days_base       │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Priority 2: gap_climate_relay_step (Gap)                       │
│  ─────────────────────────────────────────────────────────────  │
│                                                                 │
│  - Read all climate + disturbance from Site (P1, same tick)     │
│  - Copy to own states: deg_days, dry_days, avail_n, flood_days, │
│    fire_intensity, wind_intensity, dry_days_base                │
│  - Eliminates 1-tick climate lag for trees at P3                │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Priority 3: tree_potential_growth_step (Tree)                  │
│  ─────────────────────────────────────────────────────────────  │
│  Executes for living trees only (is_alive > 0.5)               │
│                                                                 │
│  1. ENVIRONMENTAL RESPONSE                                      │
│     - Read climate from Gap states (from P2, same tick)         │
│     - Read neighbor heights from Tree states_db (light comp.)   │
│     - Calculate growth factors:                                 │
│       fc_degday:  parabolic temperature response                │
│       fc_drought: exponential drought response (tol 1-6)        │
│       fc_flood:   linear flood response (tol 1-6)              │
│       fc_light:   Beer-Lambert canopy light attenuation         │
│       forska_shade: light at canopy base (self-pruning check)   │
│                                                                 │
│  2. POTENTIAL GROWTH                                            │
│     - env_stress = fc_degday * fc_drought * fc_light * fc_flood │
│       (NO fc_nutrient — applied later at P8)                    │
│     - diam_max = optimal diameter increment for current size    │
│     - Compute potential biomass change                          │
│     - Compute n_demand from potential growth                    │
│                                                                 │
│  WRITES:                                                        │
│     - params: env_stress, diam_max, light_avail, fc_*,          │
│              forska_shade                                        │
│     - states: n_demand (Gap aggregates at P4)                   │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Priority 4: gap_demand_aggregate_step (Gap)                    │
│  ─────────────────────────────────────────────────────────────  │
│                                                                 │
│  - Sum n_demand from all living Tree neighbors (from P3)        │
│  - Write total_n_demand to Gap states (read at P5)              │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Priority 5: gap_sync_step (Gap)                                │
│  ─────────────────────────────────────────────────────────────  │
│                                                                 │
│  - Compute PER-GAP n_supply_ratio:                              │
│    avail_n / (total_n_demand * UNIT_CONV), capped at 1.0       │
│  - Write n_supply_ratio to own states                           │
│  - Clear litter_accum, total_lai, n_consumed accumulators       │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Priority 6: tree_template_renewal_step (Tree)                  │
│  ─────────────────────────────────────────────────────────────  │
│  Templates only (is_alive < -0.5)                               │
│  Matches GAPpy renewal() (model.py:792-982)                    │
│                                                                 │
│  a. ENVIRONMENTAL RESPONSE                                      │
│     - Read climate from Gap (same-tick from P2)                 │
│     - Compute fc_degday, fc_drought, fc_flood, fc_nutrient      │
│     - Compute fc_light at ground level (all neighbors shade)    │
│     - Forska shade self-pruning check                           │
│                                                                 │
│  b. REGROWTH                                                    │
│     - regrowth = fc_degday * fc_drought * fc_flood              │
│                  * fc_nutrient * fc_light                        │
│     - Two-threshold: >= 0.05 germinate, >= 0.01 keep seedbank  │
│                                                                 │
│  c. AVAIL_SPEC (binary maturity flag)                           │
│     - 1.0 if any neighbor tree of same species has              │
│       diam > max_diam * 0.05; else 0.0                          │
│                                                                 │
│  d. FIRE/WIND SEEDLING RESET (outside avail_N gate)            │
│     - Fire: seedling = (invader*10 + sprout*avail) * fc_fire    │
│     - Wind: seedling += invader + sprout * avail                │
│     - weight = 0 (no recruitment in disturbance year)           │
│                                                                 │
│  e. SEEDBANK UPDATE (gated on avail_n > 0)                     │
│     - seedbank += invader + seed * avail_spec                   │
│                  + sprout * avail_spec                           │
│     - If regrowth >= 0.05: seedling += seedbank; seedbank = 0   │
│     - Else if regrowth >= 0.01: seedbank *= seed_surv (decay)  │
│     - Else: seedbank = 0 (conditions too poor)                  │
│                                                                 │
│  f. RECRUITMENT WEIGHT                                          │
│     - weight = seedling * regrowth                              │
│     - Floor: if weight > 0 and < 0.01 → weight = 0.01          │
│                                                                 │
│  g. SEEDLING DECREMENT (proportional, skipped during fire/wind) │
│     - my_share = num_to_recruit * old_weight / total_weight     │
│     - seedling -= my_share / PLOTSIZE                           │
│                                                                 │
│  h. SEEDLING SURVIVAL (gated on avail_n > 0)                   │
│     - seedling *= seedling_lg (annual survival rate)            │
│                                                                 │
│  i. OUTPUTS                                                     │
│     - params: seedbank, seedling, env_stress (= regrowth),      │
│              seedling_weight (P8 dormant slots read same tick)   │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Priority 7: gap_recruit_aggregate_step (Gap)                   │
│  ─────────────────────────────────────────────────────────────  │
│                                                                 │
│  - Read template regrowth from Tree.params (from P6, same tick) │
│  - Count living trees and free slots from Tree.states_db        │
│  - growmax = max regrowth across all templates                  │
│  - DENSITY-BASED RECRUITMENT (GAPpy model.py:833-837):          │
│    max_renew = int(PLOTSIZE * growmax) - living_count            │
│    cap at int(PLOTSIZE * 0.5), floor at 3                       │
│    cap by int(PLOTSIZE) - living_count                           │
│  - recruit_prob = nrenew / free_slot_count                      │
│  - Write recruit_prob + recruit_rand_seed to Gap states          │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Priority 8: tree_actual_growth_step (Tree)                     │
│  ─────────────────────────────────────────────────────────────  │
│  Two branches based on is_alive:                                │
│                                                                 │
│  BRANCH 1: LIVING TREES (is_alive > 0.5)                       │
│  ─────────────────────────────────────────                      │
│  a. GROWTH (runs first, before fire/wind check)                 │
│     - Read env_stress, diam_max from params (P3, same tick)     │
│     - Read n_supply_ratio from Gap (P5, same tick)              │
│     - fc_nutrient based on lownutr_tol (1-3)                   │
│     - growth_factor = env_stress * fc_nutrient                  │
│     - diam_increment = diam_max * growth_factor                 │
│     - Update diameter, height (Forska equation), biomass        │
│                                                                 │
│  b. CANOPY SELF-PRUNING (Forska)                                │
│     - check = fc_degday * fc_drought * fc_flood * forska_shade  │
│              * fc_nutrient                                       │
│     - When check <= 0.5, crown base rises (integer layers)      │
│     - Biomass difference becomes litter                         │
│                                                                 │
│  c. FIRE/WIND MORTALITY (checked after growth)                  │
│     - If fire_intensity > 0.01 OR wind_intensity > 0.01:       │
│       immediate death, all post-growth biomass → A0 litter      │
│                                                                 │
│  d. NATURAL MORTALITY (GAPpy two-check model)                   │
│     - Age survival: age_check / max_age (tol 1-3)              │
│     - Growth survival: stress_check if growth below threshold   │
│     - Tree dies if EITHER check fails                           │
│     - Dead tree: all biomass → litter                           │
│     - Surviving tree: annual leaf litter                        │
│                                                                 │
│  e. N CONSUMED (pre-pruning values)                             │
│     - n_consumed = ΔbiomC/STEM_C_N + Δleaf_N                   │
│     - Conifer leaf N: CON_LEAF_B * Δleaf / CON_LEAF_C_N        │
│     - Deciduous leaf N: Δleaf / DEC_LEAF_C_N                   │
│                                                                 │
│  BRANCH 2: DORMANT ACTIVATION (is_alive == 0)                  │
│  ─────────────────────────────────────────────                  │
│  Recruitment happens last, matching GAPpy ordering.             │
│  Seedlings don't grow until next tick.                          │
│                                                                 │
│  a. READ RECRUITMENT INFO                                       │
│     - recruit_prob from Gap (written at P7, same tick)          │
│     - recruit_rand_seed for deterministic selection             │
│                                                                 │
│  b. SLOT SELECTION                                              │
│     - Hash agent_index with rand seed for selection priority    │
│     - If slot_priority < recruit_prob: recruit                  │
│                                                                 │
│  c. SPECIES SELECTION                                           │
│     - Iterate template neighbors                                │
│     - Read seedling_weight from params (written at P6, same     │
│       tick via params — no double buffer needed)                 │
│     - Select species weighted by seedling_weight                │
│     - Minimum weight 0.01 per template (baseline chance)        │
│                                                                 │
│  d. INITIALIZE SEEDLING                                         │
│     - Copy species traits [0-21] + seed_surv + seedling_lg      │
│     - Seedling diameter from species-specific calculation        │
│     - Height from Forska equation                               │
│     - Conifer: leaf_bm = 0.0 (first-tick, no leaf mass yet)     │
│     - Set is_alive = 1.0 (visible next tick via double buffer)  │
│                                                                 │
│  e. SEEDLING N CONSUMED                                         │
│     - n_consumed = leaf/C_N + biomC/STEM_C_N                   │
│     - Seedling litter: conifer leaf_bm*0.3 + C/N; dec leaf_bm   │
│                                                                 │
│  ALL BRANCHES WRITE:                                            │
│     - states: litter_c/n, n_consumed                            │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Priority 9: gap_nconsumed_aggregate_step (Gap)                 │
│  ─────────────────────────────────────────────────────────────  │
│                                                                 │
│  - Sum n_consumed from all trees (from P8, same tick)           │
│  - Skip templates (is_alive < -0.5)                             │
│  - Write total n_consumed to Gap states (Site reads at P10)     │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Priority 10: site_nbalance_step (Site)                         │
│  ─────────────────────────────────────────────────────────────  │
│                                                                 │
│  - Read avail_n (from P1) + annual_runoff (from P1)             │
│  - Sum n_consumed from all Gap neighbors (from P9, same tick)   │
│  - Scale: total_n_consumed * UNIT_CONV / gap_count              │
│  - surplus = avail_n - scaled_n_consumed                        │
│  - If surplus > 0: return to A layer minus leach fraction       │
│    leach_frac = min(annual_runoff/1000, 0.1)                    │
│  - If surplus ≤ 0: debit A layer (surplus is negative)          │
│  - Runoff leaching (always): A_n -= 0.00002 * annual_runoff    │
│  - Transfer leached N×20 C + leached N to base layer            │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Swap double-buffered data (states_db write → read)             │
│  New is_alive, diam, height, canopy_ht, seedling_weight         │
│  visible to all readers next tick                               │
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
    │   P0: Gap reads Tree litter/LAI/counts from prev tick      │
    │        → writes litter_accum, total_lai, seedling_weights  │
    │                     │                                      │
    │                     ↓                                      │
    │   P1: Site reads Gap litter + LAI                          │
    │        → soil decomposition (365 daily steps)              │
    │        → writes avail_n, fire/wind, climate                │
    │                     │                                      │
    │                     ↓                                      │
    │   P2: Gap relays Site climate (same tick!)                 │
    │        → copies deg_days, avail_n, fire/wind to Gap states │
    │                     │                                      │
    │                     ↓                                      │
    │   P3: Trees read Gap climate (from P2, same tick)          │
    │        + read Tree heights (states_db, double buffered)    │
    │        → potential growth, light competition, forska shade │
    │        → writes env_stress, diam_max, n_demand             │
    │                     │                                      │
    │                     ↓                                      │
    │   P4: Gap reads Tree n_demand → total_n_demand             │
    │                     │                                      │
    │                     ↓                                      │
    │   P5: Gap computes per-gap N ratio + clears accumulators   │
    │        → n_supply_ratio = avail_n / (demand * UNIT_CONV)   │
    │                     │                                      │
    │                     ↓                                      │
    │   P6: Templates compute renewal (seedbank/seedling/weight) │
    │        → fire/wind seedling reset (outside N gate)         │
    │        → writes seedling_weight to params (P8 reads)       │
    │                     │                                      │
    │                     ↓                                      │
    │   P7: Gap counts recruits from template regrowth           │
    │        → recruit_prob = nrenew / free_slots                │
    │                     │                                      │
    │                     ↓                                      │
    │   P8: Trees read n_supply_ratio (same tick!)               │
    │        Living: growth → fire/wind → mortality → litter     │
    │        Dormant: species selection + activation              │
    │        → writes litter, n_consumed                         │
    │                     │                                      │
    │                     ↓                                      │
    │   P9: Gap aggregates n_consumed from trees                 │
    │                     │                                      │
    │                     ↓                                      │
    │   P10: Site applies N balance (surplus/deficit + leaching) │
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
| `canopy()` (light competition) | P3: tree_potential_growth_step (Beer-Lambert, XT=-0.40) |
| `growth()` first loop (env stress) | P3: tree_potential_growth_step |
| `growth()` second loop (nutrient + final growth) | P8: tree_actual_growth_step (living branch) |
| `growth()` third loop (canopy pruning) | P8: tree_actual_growth_step (Forska self-pruning) |
| `mortality()` fire/wind | P8: tree_actual_growth_step (after growth, before natural mort.) |
| `mortality()` natural | P8: tree_actual_growth_step (age + stress two-check) |
| `renewal()` | P6: tree_template_renewal_step (templates) + P8 (dormant activation) |
| `sum_litter()` (in bio_geo_climate) | P0: gap_litter_aggregate_step (litter + LAI) |
| N ratio computation | P4→P5: demand aggregate → per-gap N ratio |
| Climate relay | P2: gap_climate_relay_step (Site→Gap, eliminates 1-tick lag) |
| N balance | P9→P10: n_consumed aggregate → site N balance (same-tick) |

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
Inverse square root based on drought tolerance (1-6):
```
gamma = [0.50, 0.45, 0.35, 0.25, 0.15, 0.05] by tolerance class
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

Beer-Lambert light attenuation matching GAPpy's canopy model (model.py:280-362):

```
XT = -0.40  (extinction coefficient, GAPpy model.py:280)

for each neighbor (all living trees in gap):
    neighbor_lai = dc² × leafdiam_a  (canopy diameter + species leaf area)
    canopy_layers = int(height) - int(canopy_ht)

    Distribute LAI across integer height layers:
    for each layer from int(canopy_ht) to int(height):
        if neighbor_evergreen:
            con_lai[layer] += layer_lai    (100%)
            dec_lai[layer] += layer_lai    (100%)
        else:
            con_lai[layer] += layer_lai * 0.8  (80%)
            dec_lai[layer] += layer_lai        (100%)

    cumLAI = sum of LAI in layers above int(my_height)
    light_avail = exp(XT * cumLAI / PLOTSIZE)
```

- Conifers read con_light, deciduous read dec_light (separate accumulations)
- PLOTSIZE = 500.0 constant, matching GAPpy's /plotsize
- Templates compute light at ground level (height = 0), so all living neighbors shade them
- Forska shade: light at canopy base height (int(canopy_ht)), stored in params[40]
- Used at both P3 (living trees) and P6 (template renewal)

---

## Renewal Process (GAPpy renewal())

Renewal spans three priorities (P6→P7→P8), matching GAPpy's ordering where `renewal()` runs after `mortality()`. Seedlings don't grow until the next year.

### 1. Template Renewal (P6, templates)

Each template represents one species and maintains persistent seedbank/seedling state:

**avail_spec (binary maturity flag):**
```
avail_spec = 1.0 if any neighbor tree has diam > max_diam * 0.05
             0.0 otherwise
```
Matches GAPpy model.py:411-414 (binary flag, not count).

**Fire/wind seedling reset (outside avail_N gate):**
```
fire: seedling = (invader*10 + sprout*avail_spec) * fc_fire
wind: seedling += invader + sprout * avail_spec
both: weight = 0 (no recruitment in disturbance year)
```

**Seedbank accumulation (gated on avail_n > 0):**
```
seedbank += invader + seed * avail_spec + sprout * avail_spec
```
- `invader`: base colonization rate (species trait from CSV)
- `seed`: seed production rate per tree
- `avail_spec`: binary flag (0 or 1) for species presence with maturity threshold

**Seedbank → Seedling transfer (two-threshold):**
```
if regrowth >= 0.05:
    seedling += seedbank    (conditions favorable, seeds germinate)
    seedbank = 0
elif regrowth >= 0.01:
    seedbank *= seed_surv   (marginal conditions, seedbank decays)
else:
    seedbank = 0            (conditions too poor)
```

**Recruitment weight:**
```
weight = seedling * regrowth
if weight > 0.0 and weight < 0.01: weight = 0.01  (floor)
```

**Seedling decrement (proportional, skipped during fire/wind):**
```
my_share = num_to_recruit * old_weight / total_seedling_weight
seedling -= my_share / PLOTSIZE
```

**Annual seedling survival (gated on avail_n > 0):**
```
seedling *= seedling_lg
```

### 2. Density-Based nrenew (P7, Gap)

The number of recruits per tick is density-dependent (GAPpy model.py:833-837):
```
growmax = max regrowth across all templates (from P6, same tick)
max_renew = int(PLOTSIZE * growmax) - living_count
nrenew = min(max_renew, int(PLOTSIZE * 0.5))
nrenew = max(nrenew, 3)
nrenew = min(nrenew, int(PLOTSIZE) - living_count)
recruit_prob = nrenew / free_slot_count
```
- PLOTSIZE = 500.0 (GAPpy plotsize), NOT maxtrees (1000)
- `growmax`: max regrowth across all templates (from P6, same tick)
- When forest is near capacity, nrenew is small
- When forest is sparse, nrenew can be up to half capacity
- recruit_prob is per-slot probability (not raw count)

### 3. Dormant Slot Activation (P8, dormant slots)

Dormant slots compete for recruitment based on a priority hash:
```
slot_priority = hash(agent_index, recruit_rand_seed) / 1000000
if slot_priority < recruit_prob:
    select species from templates weighted by seedling_weight
    copy traits, initialize as seedling
```

Species selection reads `seedling_weight` from template `params[41]` (written at P6, same tick — no double buffer needed since P8 > P6). A minimum weight of 0.01 ensures all species have a baseline recruitment chance.

Conifer seedlings are initialized with `leaf_bm = 0.0` (matching GAPpy's zeroed local arrays in growth()).

### One-Tick Lag (Double Buffering)

The newly activated seedling's `is_alive = 1.0` is written to the states_db **write buffer** and becomes visible at P3 of tick T+1 after the buffer swap. This means seedlings don't participate in light competition or growth until the next year.

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

### Daily Climate
- Temperature: linear interpolation between monthly midpoints (GAPpy cov365)
- Precipitation: Bernoulli rain-day allocation (GAPpy cov365a)
  - raindays = min(25, prcp/4+1), each day has prob raindays/days_in_month
  - Rain days get prcp/ik; dry days get 0; remainder on last day of month
- PET: Hargreaves method (GAPpy climate.py:85-112)
- Climate perturbation: BSM inverse normal CDF (hash-based RNG)

### A0 Layer (Litter)
- Receives above-ground tree litter as pulse at year start
- Respiration rate: `AO_RESP = 5.24e-4`
- Transfer to A layer based on C/N ratio
- Division guard: skip decomposition when C/N ratio ≤ 0.001

### A Layer (Humus)
- Receives from A0 decomposition
- Respiration rate: `SA_RESP = 1.24e-5`
- **N mineralization**: main source of available N
- N efficiency: `max(0.5, (sa_cn - SA_CN_0) / sa_cn)`
- Transfer to Base layer
- Division guard: skip mineralization when C/N ratio ≤ 0.001

### Base Layer
- Receives from A layer + leached N/C from P10
- Respiration rate: `SB_RESP = 2.74e-7`
- Slow turnover, long-term storage

### N Balance (P10, same-tick)
- surplus = avail_N - total_N_consumed (scaled by UNIT_CONV / gap_count)
- Surplus > 0: return to A layer minus leach fraction (min(runoff/1000, 0.1))
- Surplus ≤ 0: debit A layer
- Runoff leaching (always): A_n -= 0.00002 × annual_runoff
- Leached N×20 C + leached N transferred to base layer

### Division Guards (Empty-Start Protection)

With empty-start initialization, soil pools can deplete before the first trees produce litter. When C/N ratio reaches 0 (carbon = 0 but nitrogen > 0), direct division would produce NaN. Guards skip decomposition when C/N ≤ 0.001, which is physically correct: zero carbon means nothing to decompose.

Note: GAPpy's `soil.py` does not include these guards because its renewal process produces trees (and litter) quickly enough that pools never fully deplete. The GPU double-buffering lag in GGap can delay litter production, making this guard necessary.

### Unit Scaling
- Tree/Gap data in raw kg; Site/soil data in tn/ha
- UNIT_CONV = 0.02 (= HEC_TO_M2 / PLOTSIZE / 1000 = 10000/500/1000)
- Applied at P1 (litter input) and P10 (N consumed input)

### Moisture Effects
- Decomposition scaled by moisture function
- Optimal moisture around 30% saturation for A0, 80% for A layer
- Reduced at both extremes

### Temperature Effects
- `temp_adjustment = 3.0^(0.1 * (temp - 1))` for A0
- `temp_adjustment = 2.5^(0.1 * (temp - 1))` for A layer
- No decomposition below -5C

---

## Fire/Wind Disturbance

1. **Probability** (Site P1):
   - Fire: base 1%, increases with dry soil conditions, cap 15%
   - Wind: from CSV wind_prob, fire takes precedence
   - Both stochastic (hash-based random)

2. **Intensity** (if event occurs):
   - Fire: 0.3 - 1.0 (based on dry conditions)
   - Wind: 0.5 - 1.0 (based on hash)

3. **Propagation**: Site states → Gap relay (P2) → Trees read at P6/P8

4. **Living tree mortality** (Tree P8):
   - If fire_intensity > 0.01 OR wind_intensity > 0.01: immediate death
   - Growth runs BEFORE fire check (post-growth biomass goes to litter)
   - All biomass → A0 litter, fire-killed trees contribute to n_consumed

5. **Template seedling reset** (Tree P6):
   - Fire: seedling = (invader×10 + sprout×avail_spec) × fc_fire
     - fc_fire uses gama table indexed by fire_tol (1-6)
   - Wind: seedling += invader + sprout × avail_spec
   - Both: weight = 0 (no recruitment in disturbance year)
   - Both: outside avail_N gate (GAPpy mortality runs before renewal)

---

## Step Function Files

| File | Priority | Breed | Purpose |
|------|----------|-------|---------|
| `step_functions/gap/gap_litter_aggregate_step.py` | P0 | Gap | Aggregate litter + LAI + seedling weights |
| `step_functions/site/soil_step.py` | P1 | Site | Soil biogeochemistry (365-day loop, Bernoulli rain) |
| `step_functions/gap/gap_climate_relay_step.py` | P2 | Gap | Relay climate Site→Gap (eliminates 1-tick lag) |
| `step_functions/tree/tree_potential_growth_step.py` | P3 | Tree | Env stress, light competition, n_demand |
| `step_functions/gap/gap_demand_aggregate_step.py` | P4 | Gap | Aggregate N demand from trees |
| `step_functions/gap/gap_sync_step.py` | P5 | Gap | Per-gap N ratio + clear accumulators |
| `step_functions/tree/tree_template_renewal_step.py` | P6 | Tree | Seedbank/seedling dynamics + regrowth (templates) |
| `step_functions/gap/gap_recruit_aggregate_step.py` | P7 | Gap | Density-based nrenew from template regrowth |
| `step_functions/tree/tree_actual_growth_step.py` | P8 | Tree | Final growth + mortality + recruitment |
| `step_functions/gap/gap_nconsumed_aggregate_step.py` | P9 | Gap | Aggregate N consumed from trees |
| `step_functions/site/site_nbalance_step.py` | P10 | Site | N surplus/deficit + leaching |

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
