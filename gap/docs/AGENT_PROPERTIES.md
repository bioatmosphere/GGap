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
1. All agents with step functions at priority 0 run **in parallel** on the GPU
2. A **synchronization barrier** waits for ALL agents to complete priority 0
3. Then all agents with step functions at priority 1 run in parallel
4. Another barrier waits for completion
5. This continues through all priorities
6. After ALL priorities complete, double-buffered data is swapped

```
Tick N:
  ┌─────────────────────────────────────────────────────────────┐
  │ Priority 0: All Trees run in parallel                       │
  │   (Tree A, Tree B, Tree C, ... all execute simultaneously)  │
  └─────────────────────────────────────────────────────────────┘
                              ↓ BARRIER (wait for all)
  ┌─────────────────────────────────────────────────────────────┐
  │ Priority 1: All Gaps run in parallel                        │
  └─────────────────────────────────────────────────────────────┘
                              ↓ BARRIER (wait for all)
  ┌─────────────────────────────────────────────────────────────┐
  │ Priority 2: All Sites run in parallel                       │
  └─────────────────────────────────────────────────────────────┘
                              ↓ BARRIER (wait for all)
  ┌─────────────────────────────────────────────────────────────┐
  │ Priority 3: All Gaps run in parallel (second step function) │
  └─────────────────────────────────────────────────────────────┘
                              ↓ BARRIER (wait for all)
  ┌─────────────────────────────────────────────────────────────┐
  │ Swap double-buffered data (states_db write → read)          │
  └─────────────────────────────────────────────────────────────┘
Tick N+1: ...
```

**Key implications:**
- Within a priority: agents run in parallel, may have race conditions → use `states_db` for same-priority reads
- Across priorities: execution is sequential, earlier priorities complete before later ones start → use `states` for cross-priority reads
- Multiple breeds can share the same priority number (they run together in parallel)
- One breed can have multiple step functions at different priorities (like Gap at P1 and P3)

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

**Example:**
```python
# Tree breed registers states as visible
tree_breed.register_property("states", [0.0] * 3, neighbor_visible=True)

# Site breed registers states as NOT visible (perhaps it doesn't need neighbors to read its states)
site_breed.register_property("states", [0.0] * 4, neighbor_visible=False)

# Result: states is neighbor_visible=True for BOTH breeds
# Because Tree set it to True, Site's False is ignored
```

**Why OR logic?**
- **Conservative approach**: If any breed needs a property to be visible, it must be synchronized across MPI ranks
- **Simplifies implementation**: One visibility setting per property name across all breeds
- **Prevents subtle bugs**: Ensures data is available when any agent might need to read it

**Practical implication**: When designing properties, assume that if ANY breed needs a property to be visible, ALL breeds will have that property visible. This affects MPI synchronization overhead.

### Double Buffering

Double buffering prevents **race conditions** when multiple agents read and write the same type of data simultaneously.

**When is it needed?**
- When agents of the **same type** read each other's data at the **same priority**

**Example**: All Trees run at priority 0. Tree A needs to read Tree B's height to calculate light competition, while Tree B simultaneously reads Tree A's height. Without double buffering, one tree might read partially-updated data.

**How it works**:
- SAGESim maintains two copies: a "read buffer" and a "write buffer"
- All agents read from the read buffer
- All agents write to the write buffer
- **IMPORTANT**: Buffers are swapped only **after ALL priorities complete**, not after each priority

**Critical Implication for Cross-Priority Reads**:

Because buffers swap only after ALL priorities finish, if a property is double-buffered:
- Changes written at priority 1 are **NOT visible** to readers at priority 3 within the same tick
- Priority 3 will read the value from the **start of the tick**, not the updated value

**When double buffering must be DISABLED (`no_double_buffer`)**:
- When a later priority needs to read changes made by an earlier priority **within the same tick**
- Example: Gap writes `litter_accum` at P1, Site reads it at P2. If double-buffered, Site would see the old value. Therefore `states` (which contains `litter_accum`) is in `no_double_buffer`.

**When double buffering is NOT needed**:
- When the reader runs **before** the writer (reader at P0, writer at P1) - reader sees start-of-tick value anyway
- When readers and writers are at **different priorities** and cross-tick visibility is acceptable

**Summary Table**:

| Scenario | Reader Priority | Writer Priority | Need Double Buffer? | Why |
|----------|----------------|-----------------|---------------------|-----|
| Same-priority parallel reads | P0 | P0 | **YES** | Prevent race conditions |
| Later priority reads earlier write (same tick) | P3 | P1 | **NO** | Must see updated value |
| Earlier priority reads later write | P0 | P1 | No (doesn't matter) | Reader runs first anyway |

**Special Case: Property Needs BOTH Behaviors**

What if a property is read by same-type agents at the same priority (needs buffer) AND by different agents at a later priority (needs no buffer)?

Since `no_double_buffer` applies at the property level (not per-priority), you cannot have both behaviors for one property. **Solution: Duplicate the data into two properties.**

```python
# Example: Tree height needed by both Trees (P0) and Gap (P1)

# states_db - double buffered (Trees read each other at P0)
TREE_DB_HEIGHT = 2

# states - NOT buffered (Gap reads at P1, needs updated value)
TREE_S_HEIGHT_COPY = 3  # duplicate for cross-priority reads

# In tree_step, write to BOTH:
states_db_tensor[agent_index][TREE_DB_HEIGHT] = new_height
states_tensor[agent_index][TREE_S_HEIGHT_COPY] = new_height
```

**Trade-offs:**
- Extra memory for duplicate data
- Extra write operations
- But ensures correct behavior for both use cases

**In GGap currently:** This situation doesn't arise because:
- Tree structure (`states_db`) is only read by other Trees at P0
- Gap doesn't need to read individual tree heights (it reads aggregated litter from `states`)

## Property Assignments by Agent Type

### Tree Agent

| Property | Array | Contents |
|----------|-------|----------|
| `params[29]` | Private | Species traits [0-14] + Internal physiology [15-28] |
| `states[3]` | Public | Litter output: litter_c, litter_n, n_demand |
| `states_db[4]` | Public, buffered | Structure: is_alive, diameter, height, canopy_height |

**params breakdown:**
- `[0-14]` Species traits: max_age, max_diam, max_ht, g, dd_min, dd_opt, dd_max, shade_tol, drought_tol, lownutr_tol, invader, seed, sprout_prob, evergreen, fire_tol
- `[15-28]` Internal physiology: age, biomC, biomN, leafC, leafN, rootC, rootN, stemC, stemN, fc_degday, fc_drought, fc_light, fc_nutrient, fc_flood

**Template trees:**
- Templates have `is_alive = -1` in states_db
- They preserve species pool for recruitment even if all living trees of a species die
- Not counted in statistics, not connected to other trees

**Why this assignment?**
- Species traits (max_age, shade_tolerance, fire_tol, etc.) are never read by neighbors → `params`
- Internal physiology (age, biomass, growth factors) are never read by neighbors → `params`
- Litter output is read by Gap at priority 1 (after Trees finish at priority 0) → `states`
- Structure is read by other Trees at priority 0 (same priority) → `states_db`

### Gap Agent

| Property | Array | Contents |
|----------|-------|----------|
| `params[2]` | Private | gap_id, total_n_demand |
| `states[12]` | Public | Climate + nutrients + litter + recruitment + disturbance |
| `states_db[1]` | Public, buffered | Placeholder (not used) |

**states breakdown:**
- `[0-4]` Climate/nutrients: deg_days, dry_days, base_mortality, avail_n, n_supply_ratio
- `[5-6]` Litter accumulators: litter_accum_c, litter_accum_n
- `[7-8]` Recruitment: num_to_recruit, recruit_rand_seed
- `[9-11]` Disturbance: flood_days, seed_bank, fire_intensity

**Why this assignment?**
- total_n_demand is an intermediate calculation, not read by neighbors → `params`
- Trees (priority 0) read climate, n_supply_ratio, flood_days, fire_intensity from Gap → `states`
- Site (priority 2) reads litter_accum from Gap → `states`
- No same-priority reads → `states_db` unused

### Site Agent

| Property | Array | Contents |
|----------|-------|----------|
| `params[53]` | Private | Soil pools [0-8] + Monthly climate [9-44] + Site properties [45-52] |
| `states[6]` | Public | Climate + nutrients + disturbance |
| `states_db[1]` | Public, buffered | Placeholder (not used) |

**params breakdown:**
- `[0-8]` Soil pools: A0_c, A0_n, A_c, A_n, BL_c, BL_n, A0_w, A_w, BL_w
- `[9-20]` Monthly tmin (12 months)
- `[21-32]` Monthly tmax (12 months)
- `[33-44]` Monthly precipitation (12 months)
- `[45-52]` Site properties: field_cap, perm_wp, slope, sigma, lai, lai_w0, latitude, rain_n

**states breakdown:**
- `[0-3]` Climate: deg_days, dry_days, base_mortality, avail_n
- `[4-5]` Disturbance: flood_days, fire_intensity

**Why this assignment?**
- Soil pool dynamics are internal to Site → `params`
- Monthly climate is used for daily interpolation in site_soil_step → `params`
- Gap (priority 3) reads climate, avail_n, flood_days, fire_intensity from Site → `states`
- No same-priority reads → `states_db` unused

## Data Flow Diagram

```
Priority 0: Tree Step
    Reads:  Gap.states (climate, n_supply_ratio, flood_days, fire_intensity)
            Gap.states (num_to_recruit, recruit_rand_seed) - for dormant trees
            Tree.states_db (neighbor heights for light)  ← SAME PRIORITY, needs buffer
    Writes: Tree.params (internal physiology: age, biomass, growth factors)
            Tree.states (litter output: litter_c, litter_n, n_demand)
            Tree.states_db (updated structure: is_alive, diam, height)

Priority 1: Gap Aggregate Step
    Reads:  Tree.states (litter_c, litter_n, n_demand)  ← DIFFERENT PRIORITY
            Tree.states_db (is_alive to filter living vs dormant)
            Tree.params (invader, seed for recruitment calc)
    Writes: Gap.params (total_n_demand)
            Gap.states (litter_accum_c/n, num_to_recruit, seed_bank)

Priority 2: Site Soil Step
    Reads:  Gap.states (litter_accum_c/n)  ← DIFFERENT PRIORITY
    Writes: Site.params (soil pools: A0/A/BL carbon, nitrogen, water)
            Site.states (deg_days, dry_days, avail_n, flood_days, fire_intensity)

Priority 3: Gap Sync Step
    Reads:  Site.states (climate, avail_n, flood_days, fire_intensity)  ← DIFFERENT PRIORITY
            Gap.params (total_n_demand for n_supply_ratio calc)
    Writes: Gap.states (climate copy, n_supply_ratio)
            Gap.states (clears litter_accum and num_to_recruit)
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
- Q2: Same type at same priority? Yes, all trees at priority 0.
- Answer: `states_db`

**Adding a new site variable: "soil_temperature"**
- Q1: Do neighbors need it? Yes, Gap relays it to trees.
- Q2: Same type at same priority? No, Gap reads at priority 3, Site writes at priority 2.
- Answer: `states`

## Customizing Property Names

The current property names (`params`, `states`, `states_db`) are grouped by **code requirements** (visibility and buffering behavior), not by domain meaning. This is a practical choice for the framework.

**If you prefer more domain-meaningful names**, you can create them. For example:
- Instead of `params`, use `traits` for Tree and `soil_pools` for Site
- Instead of `states`, use `litter_output` for Tree and `climate` for Site

**Critical Constraint: All breeds MUST have the same property names.**

SAGESim requires a uniform step function signature across all breeds. If you create a property for one breed, all other breeds must have a property with the same name, even if they don't use it.

```python
# Example: If you want domain-specific names

# Tree breed
tree_breed.register_property("traits", [0.0] * 20, neighbor_visible=False)      # used
tree_breed.register_property("litter", [0.0] * 3, neighbor_visible=True)        # used
tree_breed.register_property("structure", [0.0] * 4, neighbor_visible=True)     # used
tree_breed.register_property("soil_pools", [0.0] * 1, neighbor_visible=False)   # NOT used, but must exist

# Site breed
site_breed.register_property("traits", [0.0] * 1, neighbor_visible=False)       # NOT used, but must exist
site_breed.register_property("litter", [0.0] * 1, neighbor_visible=True)        # NOT used, but must exist
site_breed.register_property("structure", [0.0] * 1, neighbor_visible=True)     # NOT used, but must exist
site_breed.register_property("soil_pools", [0.0] * 9, neighbor_visible=False)   # used
```

**Recommendation:** The current `params`/`states`/`states_db` naming is simple and clearly indicates visibility and buffering behavior. Domain meaning is captured in the index constants (e.g., `TREE_P_MAX_AGE`, `SITE_S_AVAIL_N`). This avoids proliferation of unused placeholder properties.

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

**GGap Example**:
- Gap writes `litter_accum` at P1, Site reads at P2 → must be in `states`
- Site writes `avail_n` at P2, Gap reads at P3 → must be in `states`
- Tree writes `height` at P0, other Trees read at P0 → must be in `states_db`
