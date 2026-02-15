# Agent Properties Guide for GGap

This document explains the property system used in GGap for defining agent data. Understanding this system is essential when adding new variables or modifying the model.

## Overview

Each agent type (Tree, Gap, Site) stores its data in **three property arrays**:

| Property | Visibility | Buffering | Purpose |
|----------|------------|-----------|---------|
| `params` | Private | None | Internal data only this agent uses |
| `states` | Public | None | Data that neighbors read at **different** priorities |
| `states_db` | Public | Double-buffered | Data that neighbors read at the **same** priority |

## Key Concepts

### Priority Execution

Each step function is assigned a **priority** (lower numbers run first). SAGESim executes priorities sequentially with synchronization barriers between them.

**How it works:**
1. All agents with step functions at priority N run **in parallel** on the GPU
2. A **synchronization barrier** waits for ALL agents to complete priority N
3. Then all agents with step functions at priority N+1 run in parallel
4. This continues through all priorities
5. After ALL priorities complete, double-buffered data is swapped

```
Tick N:
  ┌─────────────────────────────────────────────────────────────┐
  │ Priority 0: All Gaps run (litter aggregate)                  │
  └─────────────────────────────────────────────────────────────┘
                              ↓ BARRIER
  ┌─────────────────────────────────────────────────────────────┐
  │ Priority 1: All Sites run (soil biogeochemistry)             │
  └─────────────────────────────────────────────────────────────┘
                              ↓ BARRIER
  ┌─────────────────────────────────────────────────────────────┐
  │ Priority 2: All Trees run (potential growth, light comp.)    │
  └─────────────────────────────────────────────────────────────┘
                              ↓ BARRIER
  ┌─────────────────────────────────────────────────────────────┐
  │ Priority 3: All Gaps run (N demand aggregate)                │
  └─────────────────────────────────────────────────────────────┘
                              ↓ BARRIER
  ┌─────────────────────────────────────────────────────────────┐
  │ Priority 4: All Sites run (nutrient allocation)              │
  └─────────────────────────────────────────────────────────────┘
                              ↓ BARRIER
  ┌─────────────────────────────────────────────────────────────┐
  │ Priority 5: All Gaps run (sync climate + clear accumulators) │
  └─────────────────────────────────────────────────────────────┘
                              ↓ BARRIER
  ┌─────────────────────────────────────────────────────────────┐
  │ Priority 6: All Trees run (actual growth + renewal + recruit)│
  └─────────────────────────────────────────────────────────────┘
                              ↓ BARRIER
  ┌─────────────────────────────────────────────────────────────┐
  │ Swap double-buffered data (states_db write → read)           │
  └─────────────────────────────────────────────────────────────┘
Tick N+1: ...
```

**Key implications:**
- Within a priority: agents run in parallel, may have race conditions → use `states_db` for same-priority reads
- Across priorities: execution is sequential, earlier priorities complete before later ones start → use `states` for cross-priority reads
- Multiple breeds can share the same priority number (they run together in parallel)
- One breed can have multiple step functions at different priorities (Gap has steps at P0, P3, P5)

### Neighbor Visibility

When an agent needs to read data from a connected neighbor, that data must be in a **public** property (`states` or `states_db`).

- **Private (`params`)**: Only the agent itself can read/write. Neighbors cannot see this data.
- **Public (`states`, `states_db`)**: Connected neighbors can read this data.

**Example**: Trees need to read climate conditions from their Gap neighbor. Therefore, climate variables (deg_days, dry_days) are stored in Gap's `states` (public), not `params` (private).

**Cross-Breed neighbor_visible Behavior (OR Logic)**

Since all breeds must have the same property names, what happens if different breeds set different `neighbor_visible` values for the same property?

SAGESim uses **OR logic**: If ANY breed marks a property as `neighbor_visible=True`, it becomes visible for ALL breeds.

```python
# From SAGESim's agent.py:
# Update neighbor_visible (use OR to be conservative - if any breed marks it visible, it's visible)
self._property_name_2_neighbor_visible[property_name] = (
    self._property_name_2_neighbor_visible.get(property_name, False) or neighbor_visible
)
```

**Practical implication**: When designing properties, assume that if ANY breed needs a property to be visible, ALL breeds will have that property visible. This affects MPI synchronization overhead.

### Double Buffering

Double buffering prevents **race conditions** when multiple agents read and write the same type of data simultaneously.

**When is it needed?**
- When agents of the **same type** read each other's data at the **same priority**

**Example**: All Trees run at P2. Tree A needs to read Tree B's height to calculate light competition, while Tree B simultaneously reads Tree A's height. Without double buffering, one tree might read partially-updated data.

**How it works**:
- SAGESim maintains two copies: a "read buffer" and a "write buffer"
- All agents read from the read buffer
- All agents write to the write buffer
- **IMPORTANT**: Buffers are swapped only **after ALL priorities complete**, not after each priority

**Critical Implication for Cross-Priority Reads**:

Because buffers swap only after ALL priorities finish, if a property is double-buffered:
- Changes written at priority 2 are **NOT visible** to readers at priority 6 within the same tick
- Priority 6 will read the value from the **start of the tick**, not the updated value

**When double buffering must be DISABLED (`no_double_buffer`)**:
- When a later priority needs to read changes made by an earlier priority **within the same tick**
- Example: Gap writes `litter_accum` at P0, Site reads it at P1. If double-buffered, Site would see the old value. Therefore `states` is in `no_double_buffer`.

**Summary Table**:

| Scenario | Reader Priority | Writer Priority | Need Double Buffer? | Why |
|----------|----------------|-----------------|---------------------|-----|
| Same-priority parallel reads | P2 | P2 | **YES** | Prevent race conditions |
| Later priority reads earlier write (same tick) | P4 | P1 | **NO** | Must see updated value |
| Earlier priority reads later write | P0 | P6 | No (doesn't matter) | Reader runs first anyway |

**Special Case: Property Needs BOTH Behaviors**

What if a property is read by same-type agents at the same priority (needs buffer) AND by different agents at a later priority (needs no buffer)?

Since `no_double_buffer` applies at the property level (not per-priority), you cannot have both behaviors for one property. **Solution: Duplicate the data into two properties.**

**In GGap currently:** Tree `states_db` is double-buffered (Trees read each other's heights at P2 for light competition). Tree structure written at P6 is visible to other Trees at P2 of the **next tick** via the buffer swap.

## Property Assignments by Agent Type

### Tree Agent

| Property | Array | Contents |
|----------|-------|----------|
| `params[40]` | Private | Species traits [0-21] + Physiology [22-31] + Intermediates [32-33] + Renewal [34-37] + Leaf area [38-39] |
| `states[5]` | Public | Litter output: litter_c, litter_n, n_demand, litter_c_bg, litter_n_bg |
| `states_db[5]` | Public, buffered | Structure: is_alive, diam, height, canopy_ht, seedling_weight |

**params breakdown:**
- `[0-21]` Species traits: species_id, max_age, max_diam, max_ht, arfa_0, g, shade_tol, deg_day_min/opt/max, invader, seed, sprout, wood_bulk_dens, lownutr_tol, flood_tol, drought_tol, evergreen, fire_tol, rootdepth, stress_tol, age_tol
- `[22-31]` Internal physiology: age, biomC, biomN, leaf_bm, x, y, light_avail, fc_degday, fc_drought, fc_flood
- `[32-33]` Intermediates: env_stress (P2→P6), diam_max_calc (P2→P6)
- `[34-37]` Renewal (template-only): seed_surv, seedling_lg, seedbank, seedling
- `[38-39]` Leaf area: leafdiam_a (adjusted by shade_tol), leafarea_c (normalized by HEC_TO_M2)

**states breakdown:**
- `[0-1]` Above-ground litter: litter_c, litter_n
- `[2]` Nitrogen demand: n_demand
- `[3-4]` Below-ground litter: litter_c_bg, litter_n_bg

**states_db breakdown:**
- `[0-3]` Structure: is_alive, diam, height, canopy_ht
- `[4]` Renewal: seedling_weight (templates write at P6, dormant reads at P6)

**Three tree states (encoded in is_alive):**
- `is_alive = 1.0`: Living tree (grows, produces litter, can die)
- `is_alive = 0.0`: Dormant slot (available for recruitment)
- `is_alive = -1.0`: Template (permanent per-species reference, computes renewal state)

**Template trees:**
- One per species per gap, preserving species pool for recruitment
- Hold persistent seedbank/seedling state in params[36-37]
- Compute environmental response (regrowth) and write seedling_weight to states_db[4]
- Not counted in statistics, connected to Gap and all Trees in the gap

**Why this assignment?**
- Species traits and physiology are never read by neighbors → `params`
- Litter output is read by Gap at P0 (after Trees finish at P6 of prev tick) → `states`
- N demand is read by Gap at P3 (after Trees write at P2) → `states`
- Structure is read by other Trees at P2 (same priority) for light competition → `states_db`
- Seedling weight is written by templates at P6, read by dormant slots at P6 (same priority) → `states_db`

### Gap Agent

| Property | Array | Contents |
|----------|-------|----------|
| `params[2]` | Private | gap_id, total_n_demand |
| `states[14]` | Public | Climate + nutrients + litter + recruitment + disturbance + n_demand + bg_litter |
| `states_db[1]` | Public, buffered | Placeholder (not used) |

**states breakdown:**
- `[0-3]` Climate/nutrients: deg_days, dry_days, avail_n, n_supply_ratio
- `[4-5]` Above-ground litter accumulators: litter_accum_c, litter_accum_n
- `[6-7]` Recruitment: num_to_recruit, recruit_rand_seed
- `[8]` Flood days
- `[9]` Seed bank (legacy, unused in current renewal)
- `[10]` Fire intensity
- `[11]` Total N demand (public, Site reads at P4)
- `[12-13]` Below-ground litter accumulators: litter_accum_c_bg, litter_accum_n_bg

**Why this assignment?**
- total_n_demand internal copy → `params`
- Trees read climate/n_supply_ratio from Gap at P2 and P6 → `states`
- Site reads litter_accum from Gap at P1 → `states`
- Site reads total_n_demand from Gap at P4 → `states`
- No same-priority reads → `states_db` unused

### Site Agent

| Property | Array | Contents |
|----------|-------|----------|
| `params[116]` | Private | Soil pools [0-8] + Monthly climate [9-44] + Site properties [45-55] + Climate std dev [56-91] + Lapse rates [92-115] |
| `states[6]` | Public | Climate + nutrients + disturbance + n_supply_ratio |
| `states_db[1]` | Public, buffered | Placeholder (not used) |

**params breakdown:**
- `[0-8]` Soil pools: A0_c, A0_n, A_c, A_n, BL_c, BL_n, A0_w, A_w, BL_w
- `[9-20]` Monthly tmin (12 months)
- `[21-32]` Monthly tmax (12 months)
- `[33-44]` Monthly precipitation (12 months)
- `[45-55]` Site properties: field_cap, perm_wp, slope, sigma, lai, lai_w0, latitude, rain_n, fire_prob, wind_prob, base_h
- `[56-67]` Monthly tmin std dev (12 months)
- `[68-79]` Monthly tmax std dev (12 months)
- `[80-91]` Monthly precipitation std dev (12 months)
- `[92-103]` Monthly temperature lapse rate (12 months)
- `[104-115]` Monthly precipitation lapse rate (12 months)

**states breakdown:**
- `[0-1]` Climate: deg_days, dry_days
- `[2]` Nutrient: avail_n
- `[3-4]` Disturbance: flood_days, fire_intensity
- `[5]` Nutrient: n_supply_ratio (computed at P4, Gap reads at P5)

**Why this assignment?**
- Soil pool dynamics and monthly climate are internal to Site → `params`
- Gap reads climate, avail_n, flood_days, fire_intensity, n_supply_ratio from Site at P5 → `states`
- No same-priority reads → `states_db` unused

## Data Flow Diagram

```
Priority 0: Gap Litter Aggregate
    Reads:  Tree.states (litter_c/n, litter_c/n_bg) - from P6 prev tick
            Tree.states_db (is_alive: count living/dormant)
            Tree.params (env_stress from templates: regrowth/growmax)
    Writes: Gap.states (litter_accum_c/n, litter_accum_c/n_bg)
            Gap.states (num_to_recruit, recruit_rand_seed)

Priority 1: Site Soil Step
    Reads:  Gap.states (litter_accum_c/n, litter_accum_c/n_bg) - from P0 same tick
    Writes: Site.params (soil pools: A0/A/BL carbon, nitrogen, water)
            Site.states (avail_n, flood_days, fire_intensity)

Priority 2: Tree Potential Growth
    Reads:  Gap.states (deg_days, dry_days, flood_days) - relayed at P5 prev tick
            Tree.states_db (neighbor heights for light) ← SAME PRIORITY, needs buffer
    Writes: Tree.params (env_stress, diam_max, light_avail, fc_degday/drought/flood)
            Tree.states (n_demand)

Priority 3: Gap N Demand Aggregate
    Reads:  Tree.states (n_demand) - from P2 same tick
    Writes: Gap.states (total_n_demand)

Priority 4: Site Nutrient Allocation
    Reads:  Site.states (avail_n) - from P1 same tick
            Gap.states (total_n_demand) - from P3 same tick
    Writes: Site.states (n_supply_ratio)

Priority 5: Gap Sync
    Reads:  Site.states (climate, n_supply_ratio) - from P1/P4 same tick
    Writes: Gap.states (climate copy, n_supply_ratio relay)
    Clears: Gap.states (litter_accum_c/n, litter_accum_c/n_bg)

Priority 6: Tree Actual Growth + Renewal + Recruitment
    Living trees:
      Reads:  Tree.params (env_stress, diam_max from P2)
              Gap.states (n_supply_ratio, fire_intensity from P5)
      Writes: Tree.params (age, biomC, biomN, leaf_bm)
              Tree.states (litter_c/n, litter_c/n_bg)
              Tree.states_db (is_alive, diam, height, canopy_ht)
    Templates:
      Reads:  Gap.states (climate, n_supply_ratio from P5)
              Tree.states_db (neighbor structure for light, species_id for avail_spec)
      Writes: Tree.params (seedbank, seedling, env_stress=regrowth)
              Tree.states_db (seedling_weight)
    Dormant slots:
      Reads:  Gap.states (num_to_recruit, recruit_rand_seed from P0)
              Tree.states_db (seedling_weight from templates, same priority)
      Writes: Tree.params (species traits, physiology init)
              Tree.states_db (is_alive=1, diam, height, canopy_ht)
```

## Adding New Variables

### Decision Tree

When adding a new variable, ask these questions:

1. **Does any neighbor need to read this variable?**
   - No → Put in `params`
   - Yes → Continue to question 2

2. **Do neighbors of the SAME TYPE read this at the SAME PRIORITY?**
   - Yes → Put in `states_db` (needs double buffering)
   - No → Continue to question 3

3. **Does a LATER priority need to read this value written by an EARLIER priority (same tick)?**
   - Yes → Put in `states` (must NOT be double-buffered)
   - No → Put in `states` (can be either, but `states` is simpler)

### Examples

**Adding a new tree variable: "root_depth"**
- Q1: Does any neighbor need to read root_depth?
  - If only used internally for water uptake calculation → `params`
  - If Gap needs it to calculate soil water competition → `states`

**Adding a new tree variable: "crown_radius"**
- Q1: Do neighbors need it? Yes, other trees for crown overlap calculation.
- Q2: Same type at same priority? Yes, all trees at P2.
- Answer: `states_db`

**Adding a new site variable: "soil_temperature"**
- Q1: Do neighbors need it? Yes, Gap relays it to trees.
- Q2: Same type at same priority? No, Gap reads at P5, Site writes at P1.
- Answer: `states`

## Index Constants

Each property array uses named constants for indices. Follow this naming convention:

```python
# params indices use _P_ prefix
TREE_P_MAX_AGE = 1
SITE_P_A0_C = 0

# states indices use _S_ prefix
TREE_S_LITTER_C = 0
GAP_S_DEG_DAYS = 0

# states_db indices use _DB_ prefix
TREE_DB_IS_ALIVE = 0
TREE_DB_HEIGHT = 2
TREE_DB_SEEDLING_WEIGHT = 4
```

## Summary

| Question | Answer | Property |
|----------|--------|----------|
| Only I use this data? | Yes | `params` |
| Neighbors read at different priority? | Yes | `states` |
| Same-type neighbors read at same priority? | Yes | `states_db` |

**Key Rule for Cross-Priority Communication**:
If priority X writes a value that priority Y (where Y > X) needs to read **in the same tick**, that value must be in `states` (NOT `states_db`). Double-buffered properties only become visible to other agents in the **next tick**.

When in doubt:
- Start with `params` (safest, most private)
- Move to `states` if neighbors need to read it
- Move to `states_db` only if same-priority race conditions occur

**GGap Examples**:
- Gap writes `litter_accum` at P0, Site reads at P1 → must be in `states`
- Site writes `avail_n` at P1, Site nutrient reads at P4 → must be in `states`
- Tree writes `height` at P6, other Trees read at P2 → must be in `states_db`
- Template writes `seedling_weight` at P6, dormant slots read at P6 → must be in `states_db`
