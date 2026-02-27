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
  │ Priority 0: All Gaps run (litter + LAI aggregate)            │
  └─────────────────────────────────────────────────────────────┘
                              ↓ BARRIER
  ┌─────────────────────────────────────────────────────────────┐
  │ Priority 1: All Sites run (soil biogeochemistry, 365 days)   │
  └─────────────────────────────────────────────────────────────┘
                              ↓ BARRIER
  ┌─────────────────────────────────────────────────────────────┐
  │ Priority 2: All Gaps run (climate relay Site→Gap)            │
  └─────────────────────────────────────────────────────────────┘
                              ↓ BARRIER
  ┌─────────────────────────────────────────────────────────────┐
  │ Priority 3: All Trees run (potential growth, light comp.)    │
  └─────────────────────────────────────────────────────────────┘
                              ↓ BARRIER
  ┌─────────────────────────────────────────────────────────────┐
  │ Priority 4: All Gaps run (N demand aggregate)                │
  └─────────────────────────────────────────────────────────────┘
                              ↓ BARRIER
  ┌─────────────────────────────────────────────────────────────┐
  │ Priority 5: All Gaps run (per-gap N ratio + clear accum.)    │
  └─────────────────────────────────────────────────────────────┘
                              ↓ BARRIER
  ┌─────────────────────────────────────────────────────────────┐
  │ Priority 6: All Trees run (template renewal)                 │
  └─────────────────────────────────────────────────────────────┘
                              ↓ BARRIER
  ┌─────────────────────────────────────────────────────────────┐
  │ Priority 7: All Gaps run (recruit aggregate)                 │
  └─────────────────────────────────────────────────────────────┘
                              ↓ BARRIER
  ┌─────────────────────────────────────────────────────────────┐
  │ Priority 8: All Trees run (actual growth + mortality + recruit)│
  └─────────────────────────────────────────────────────────────┘
                              ↓ BARRIER
  ┌─────────────────────────────────────────────────────────────┐
  │ Priority 9: All Gaps run (N consumed aggregate)              │
  └─────────────────────────────────────────────────────────────┘
                              ↓ BARRIER
  ┌─────────────────────────────────────────────────────────────┐
  │ Priority 10: All Sites run (N balance)                       │
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
- One breed can have multiple step functions at different priorities (Gap has steps at P0, P2, P4, P5, P7, P9)

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

**In GGap currently:** Tree `states_db` is double-buffered (Trees read each other's heights at P3 for light competition). Tree structure written at P8 is visible to other Trees at P3 of the **next tick** via the buffer swap.

## Property Assignments by Agent Type

### Tree Agent

| Property | Array | Contents |
|----------|-------|----------|
| `params[42]` | Private | Species traits [0-21] + Physiology [22-31] + Intermediates [32-33] + Renewal [34-37] + Leaf area [38-39] + Forska [40] + Seedling weight [41] |
| `states[5]` | Public | Litter output: litter_c, litter_n, n_demand, n_consumed, (unused) |
| `states_db[5]` | Public, buffered | Structure: is_alive, diam, height, canopy_ht, seedling_weight |

**params breakdown:**
- `[0-21]` Species traits: species_id, max_age, max_diam, max_ht, arfa_0, g, shade_tol, deg_day_min/opt/max, invader, seed, sprout, wood_bulk_dens, lownutr_tol, flood_tol, drought_tol, evergreen, fire_tol, rootdepth, stress_tol, age_tol
- `[22-31]` Internal physiology: age, biomC, biomN, leaf_bm, x, y, light_avail, fc_degday, fc_drought, fc_flood
- `[32-33]` Intermediates: env_stress (P3→P8), diam_max_calc (P3→P8)
- `[34-37]` Renewal (template-only): seed_surv, seedling_lg, seedbank, seedling
- `[38-39]` Leaf area: leafdiam_a (adjusted by shade_tol), leafarea_c (normalized by HEC_TO_M2)
- `[40]` Forska shade: forska_shade (P3→P8, self-pruning threshold)
- `[41]` Seedling weight: seedling_weight (P6 templates→P8 dormant slots, same tick via params)

**states breakdown:**
- `[0-1]` Above-ground litter: litter_c, litter_n
- `[2]` Nitrogen demand: n_demand (P3→P4 same tick)
- `[3]` Nitrogen consumed: n_consumed (P8→P9 same tick)
- `[4]` (unused)

**states_db breakdown:**
- `[0-3]` Structure: is_alive, diam, height, canopy_ht
- `[4]` Renewal: seedling_weight (templates write at P6, dormant reads at P8 via states_db)

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
- Seedling weight is in both `params[41]` (P6 templates→P8 dormant, same tick) and `states_db[4]` (for P0 aggregation next tick)
- Litter output is read by Gap at P0 (after Trees finish at P8 of prev tick) → `states`
- N demand is read by Gap at P4 (after Trees write at P3) → `states`
- N consumed is read by Gap at P9 (after Trees write at P8) → `states`
- Structure is read by other Trees at P3 (same priority) for light competition → `states_db`

### Gap Agent

| Property | Array | Contents |
|----------|-------|----------|
| `params[2]` | Private | gap_id, total_n_demand |
| `states[16]` | Public | Climate + nutrients + litter + recruitment + disturbance + LAI + N consumed |
| `states_db[1]` | Public, buffered | Placeholder (not used) |

**states breakdown:**
- `[0-3]` Climate/nutrients: deg_days, dry_days, avail_n, n_supply_ratio
- `[4-5]` Above-ground litter accumulators: litter_accum_c, litter_accum_n
- `[6-7]` Recruitment: num_to_recruit, recruit_rand_seed
- `[8]` Flood days
- `[9]` Total seedling weight (sum across templates, for proportional decrement)
- `[10]` Fire intensity
- `[11]` Total N demand (public, read at P5 for per-gap ratio)
- `[12]` Total LAI (normalized by PLOTSIZE, for canopy water balance)
- `[13]` N consumed (aggregated at P9 for same-tick N balance)
- `[14]` Dry days base (base-layer dry days)
- `[15]` Wind intensity

**Why this assignment?**
- total_n_demand internal copy → `params`
- Trees read climate/n_supply_ratio from Gap at P3, P6, P8 → `states`
- Site reads litter_accum + LAI from Gap at P1 → `states`
- Gap reads own total_n_demand at P5 for per-gap N ratio → `states`
- Site reads n_consumed from Gap at P10 → `states`
- No same-priority reads → `states_db` unused

### Site Agent

| Property | Array | Contents |
|----------|-------|----------|
| `params[116]` | Private | Soil pools [0-8] + Monthly climate [9-44] + Site properties [45-55] + Climate std dev [56-91] + Annual runoff [92] + Lapse rates [93-115] |
| `states[8]` | Public | Climate + nutrients + disturbance |
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
- `[92]` Annual runoff (accumulated in P1, used at P10 for leaching)
- `[93-104]` Monthly temperature lapse rate (12 months)
- `[105-115]` Monthly precipitation lapse rate (12 months)

**states breakdown:**
- `[0-1]` Climate: deg_days, dry_days
- `[2]` Nutrient: avail_n
- `[3]` Flood days
- `[4]` Fire intensity
- `[5]` Dry days base (base-layer dry days)
- `[6]` Wind intensity
- `[7]` (unused)

**Why this assignment?**
- Soil pool dynamics and monthly climate are internal to Site → `params`
- Annual runoff used only by P10 (same breed, internal) → `params`
- Gap reads climate, avail_n, flood_days, fire/wind_intensity from Site at P2 → `states`
- No same-priority reads → `states_db` unused

## Data Flow Diagram

```
Priority 0: Gap Litter Aggregate
    Reads:  Tree.states (litter_c/n, n_consumed) - from P8 prev tick
            Tree.states_db (is_alive, diam, height, canopy_ht) - structure for LAI
            Tree.params (leafdiam_a) - leaf area coefficient
    Writes: Gap.states (litter_accum_c/n, total_lai, total_seedling_weight)

Priority 1: Site Soil Step
    Reads:  Gap.states (litter_accum_c/n, total_lai) - from P0 same tick
    Writes: Site.params (soil pools: A0/A/BL carbon, nitrogen, water, annual_runoff)
            Site.states (deg_days, dry_days, avail_n, flood_days, fire/wind_intensity)

Priority 2: Gap Climate Relay
    Reads:  Site.states (all climate + disturbance) - from P1 same tick
    Writes: Gap.states (climate copy: deg_days, dry_days, avail_n, flood_days,
                        fire/wind_intensity, dry_days_base)

Priority 3: Tree Potential Growth
    Reads:  Gap.states (deg_days, dry_days, flood_days) - from P2 same tick
            Tree.states_db (neighbor heights for light) ← SAME PRIORITY, needs buffer
    Writes: Tree.params (env_stress, diam_max, light_avail, fc_*, forska_shade)
            Tree.states (n_demand)

Priority 4: Gap N Demand Aggregate
    Reads:  Tree.states (n_demand) - from P3 same tick
    Writes: Gap.params + Gap.states (total_n_demand)

Priority 5: Gap Sync (Per-Gap N Ratio + Clear)
    Reads:  Gap.states (avail_n from P2, total_n_demand from P4)
    Writes: Gap.states (n_supply_ratio)
    Clears: Gap.states (litter_accum_c/n, total_lai, n_consumed)

Priority 6: Tree Template Renewal
    Templates only:
      Reads:  Gap.states (climate, n_supply_ratio from P2/P5)
              Gap.states (num_to_recruit, total_seedling_weight from P7 prev tick)
              Tree.states_db (neighbor structure for light, species_id for avail_spec)
      Writes: Tree.params (seedbank, seedling, env_stress=regrowth, seedling_weight)

Priority 7: Gap Recruit Aggregate
    Reads:  Tree.params (env_stress=regrowth from templates) - from P6 same tick
            Tree.states_db (is_alive: count living/free)
    Writes: Gap.states (num_to_recruit as recruit_prob, recruit_rand_seed)

Priority 8: Tree Actual Growth + Recruitment
    Living trees:
      Reads:  Tree.params (env_stress, diam_max from P3, forska_shade)
              Gap.states (n_supply_ratio from P5, fire/wind from P2)
      Writes: Tree.params (age, biomC, biomN, leaf_bm)
              Tree.states (litter_c/n, n_consumed)
              Tree.states_db (is_alive, diam, height, canopy_ht)
    Dormant slots:
      Reads:  Gap.states (recruit_prob, recruit_rand_seed from P7)
              Tree.params (seedling_weight from templates at P6, same tick)
      Writes: Tree.params (species traits, physiology init)
              Tree.states (litter_c/n, n_consumed for seedlings)
              Tree.states_db (is_alive=1, diam, height, canopy_ht)

Priority 9: Gap N Consumed Aggregate
    Reads:  Tree.states (n_consumed) - from P8 same tick
    Writes: Gap.states (n_consumed)

Priority 10: Site N Balance
    Reads:  Site.states (avail_n) - from P1 same tick
            Site.params (annual_runoff) - from P1 same tick
            Gap.states (n_consumed) - from P9 same tick
    Writes: Site.params (A_n, A_c, BL_c, BL_n - surplus/deficit + leaching)
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
- Site writes `avail_n` at P1, Gap relays at P2 → must be in `states`
- Tree writes `height` at P8, other Trees read at P3 → must be in `states_db`
- Template writes `seedling_weight` to `params[41]` at P6, dormant slots read at P8 → can use `params` (cross-priority, no buffer needed)
- Tree writes `n_consumed` at P8, Gap aggregates at P9 → must be in `states`
