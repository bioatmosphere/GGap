# GGap Implementation Logic

This document explains the detailed logic of the GGap model implementation, including the step function execution flow, data dependencies, and how it maps to UVAFME processes.

## Overview

GGap is a GPU-accelerated forest gap dynamics model that implements UVAFME (University of Virginia Forest Model Enhanced) processes using the SAGESim agent-based framework.

### Agent Hierarchy

```
Site (1 per simulation)
  └── Gap (N per site, typically 1-10)
        └── Tree (M per gap, typically 100-1000)
              └── Template Trees (one per species, for species pool preservation)
```

### Why Gap Agents?

Gap agents serve as intermediaries between Site and Trees to:

1. **Memory Efficiency**: Without Gaps, Site would need 1000+ tree neighbors. Gap keeps neighbor lists short (~10 gaps per site, ~100 trees per gap).

2. **Data Aggregation**: Gap collects litter from trees before passing to Site, reducing Site's workload.

3. **Future Extensibility**: Gap enables gap-to-gap interactions (seed dispersal, edge effects).

---

## Step Function Execution Flow

Each simulation tick executes four step functions in priority order:

```
┌─────────────────────────────────────────────────────────────────┐
│ TICK N                                                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Priority 0: tree_step                                          │
│  ─────────────────────────────────────────────────────────────  │
│  Executes for ALL Tree agents in parallel on GPU                │
│                                                                 │
│  1. RECRUITMENT (dormant slots only)                            │
│     - Read num_to_recruit from Gap states                       │
│     - If selected for recruitment:                              │
│       - Find species source (living tree OR template)           │
│       - Copy species traits to own params                       │
│       - Initialize as seedling (1cm diameter)                   │
│       - Set is_alive = 1                                        │
│                                                                 │
│  2. GROWTH (living trees only)                                  │
│     - Read climate from Gap states (deg_days, dry_days, etc.)   │
│     - Read neighbor heights from Tree states_db                 │
│     - Calculate light availability (Beer-Lambert)               │
│     - Calculate growth factors (temperature, drought, light,    │
│       nutrient, flood)                                          │
│     - Apply diameter/height growth                              │
│     - Update biomass                                            │
│                                                                 │
│  3. MORTALITY                                                   │
│     - Calculate mortality probability:                          │
│       - Age-based (4.605/max_age)                              │
│       - Stress-based (if growth below threshold)               │
│       - Seedling-specific (age <= 2)                           │
│       - Fire-based (if fire_intensity > 0)                     │
│     - Random check for death                                    │
│     - If dies: check for sprout regeneration                    │
│     - If sprouts: reset to 2cm diameter, age=1                  │
│                                                                 │
│  4. LITTER OUTPUT                                               │
│     - Living trees: leaf litter (50% of leaves)                 │
│     - Dying trees: all biomass as litter                        │
│     - Sprouting trees: 80% of old biomass as litter             │
│                                                                 │
│  WRITES:                                                        │
│     - params: age, biomC, growth factors (internal)             │
│     - states: litter_c, litter_n, n_demand (for Gap)            │
│     - states_db: is_alive, diam, height (double buffered)       │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                     [Double buffer swap]                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Priority 1: gap_aggregate_step                                 │
│  ─────────────────────────────────────────────────────────────  │
│  Executes for ALL Gap agents in parallel on GPU                 │
│                                                                 │
│  1. AGGREGATE FROM TREES                                        │
│     - Loop through Tree neighbors                               │
│     - Sum litter_c, litter_n, n_demand from living trees        │
│     - Count living trees and dormant slots                      │
│     - Accumulate seed production (invader * seed)               │
│                                                                 │
│  2. SEED BANK MANAGEMENT                                        │
│     - Add current production to seed bank                       │
│     - Apply 30% annual decay                                    │
│     - Cap at 100 seeds maximum                                  │
│                                                                 │
│  3. RECRUITMENT CALCULATION                                     │
│     - potential_recruits = available_seeds * 0.3 (germination)  │
│     - Cap by dormant slots available                            │
│     - Cap at 10 per tick (prevent explosive growth)             │
│     - Generate random seed for species selection                │
│                                                                 │
│  WRITES:                                                        │
│     - params: total_n_demand (for n_supply_ratio calc)          │
│     - states: litter_accum_c/n (for Site)                       │
│     - states: num_to_recruit, recruit_rand_seed (for Trees)     │
│     - states: seed_bank (persistent)                            │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Priority 2: site_soil_step                                     │
│  ─────────────────────────────────────────────────────────────  │
│  Executes for ALL Site agents in parallel on GPU                │
│                                                                 │
│  1. READ LITTER FROM GAPS                                       │
│     - Sum litter_accum_c/n from all Gap neighbors               │
│     - Convert annual litter to daily inputs                     │
│                                                                 │
│  2. DAILY LOOP (365 days)                                       │
│     For each day:                                               │
│                                                                 │
│     a. CLIMATE INTERPOLATION                                    │
│        - Determine month from day of year                       │
│        - Get daily tmin, tmax, precip from monthly values       │
│        - Track freeze days                                      │
│        - Accumulate atmospheric N from precipitation            │
│                                                                 │
│     b. POTENTIAL EVAPOTRANSPIRATION (Hamon method)              │
│        - Calculate solar declination                            │
│        - Calculate day length                                   │
│        - Calculate PET from temperature and day length          │
│                                                                 │
│     c. SOIL WATER BALANCE                                       │
│        - Route precipitation through canopy                     │
│        - Infiltration to A0, A, and Base layers                 │
│        - Evapotranspiration draws from layers                   │
│        - Track flood days (A layer saturated)                   │
│                                                                 │
│     d. SOIL DECOMPOSITION                                       │
│        - Add daily litter to A0 layer                           │
│        - A0 respiration → transfer to A layer                   │
│        - A layer respiration → N mineralization                 │
│        - A layer transfer to Base layer                         │
│        - Base layer respiration                                 │
│        - Moisture and temperature adjustments                   │
│                                                                 │
│  3. FIRE PROBABILITY                                            │
│     - Base probability: 1%                                      │
│     - Increases with dry conditions                             │
│     - Cap at 15%                                                │
│     - Stochastic fire occurrence                                │
│     - Fire intensity: 0.3 - 1.0 based on conditions             │
│                                                                 │
│  WRITES:                                                        │
│     - params: soil pools (A0/A/BL C, N, W)                      │
│     - states: avail_n (mineralization + atmospheric)            │
│     - states: flood_days, fire_intensity                        │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Priority 3: gap_sync_step                                      │
│  ─────────────────────────────────────────────────────────────  │
│  Executes for ALL Gap agents in parallel on GPU                 │
│                                                                 │
│  1. READ FROM SITE                                              │
│     - deg_days, dry_days, base_mortality                        │
│     - avail_n, flood_days, fire_intensity                       │
│                                                                 │
│  2. CALCULATE N SUPPLY RATIO                                    │
│     - n_supply_ratio = avail_n / total_n_demand                 │
│     - Cap at 2.0 (can't over-supply)                            │
│                                                                 │
│  3. RELAY TO TREES                                              │
│     - Copy all climate values to own states                     │
│     - Trees will read these at P0 next tick                     │
│                                                                 │
│  4. CLEAR ACCUMULATORS                                          │
│     - litter_accum_c/n = 0 (consumed by Site)                   │
│     - num_to_recruit = 0 (consumed by Trees)                    │
│                                                                 │
│  WRITES:                                                        │
│     - states: climate values (for Trees next tick)              │
│     - states: n_supply_ratio                                    │
│     - states: cleared accumulators                              │
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
    │   ┌─────────┐    P0      ┌─────────┐                       │
    │   │  Tree   │──────────→│  Tree   │                       │
    │   │ states  │  litter   │ states  │                       │
    │   └────┬────┘           └─────────┘                       │
    │        │                                                   │
    │        │ P1                                                │
    │        ↓                                                   │
    │   ┌─────────┐           ┌─────────┐    P0                  │
    │   │   Gap   │           │   Gap   │◄────── Trees read     │
    │   │ states  │           │ states  │        climate        │
    │   └────┬────┘           └────┬────┘                       │
    │        │                     ↑                             │
    │        │ P2                  │ P3                          │
    │        ↓                     │                             │
    │   ┌─────────┐           ┌────┴────┐                       │
    │   │  Site   │──────────→│  Site   │                       │
    │   │ states  │  avail_n  │ states  │                       │
    │   └─────────┘           └─────────┘                       │
    │                                                            │
    └────────────────────────────────────────────────────────────┘

    Legend:
    P0 = Priority 0 (tree_step)
    P1 = Priority 1 (gap_aggregate_step)
    P2 = Priority 2 (site_soil_step)
    P3 = Priority 3 (gap_sync_step)
```

---

## Mapping to UVAFME Processes

| UVAFME Process | GGap Implementation |
|----------------|---------------------|
| `bio_geo_climate()` | P2: site_soil_step (daily climate loop) |
| `grow_plot()` | P0: tree_step growth section |
| `mortality()` | P0: tree_step mortality section |
| `regenerate()` | P0: tree_step recruitment section |
| `soil_water()` | P2: site_soil_step water balance |
| `soil_decomp()` | P2: site_soil_step decomposition |
| `sum_litter()` | P1: gap_aggregate_step |
| Species response functions | P0: fc_degday, fc_drought, fc_light, fc_nutrient, fc_flood |

---

## Environmental Response Functions

### Temperature Response (fc_degday)
Parabolic response curve:
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
    neighbor_lai = (diam² * 0.01) / canopy_depth
    overlap = min(neighbor_height - my_height, canopy_depth)

    if neighbor_evergreen:
        xt = -0.70  # Conifer
    else:
        xt = -0.80  # Deciduous

    total_lai += neighbor_lai * overlap * (xt / -0.80)

light_avail = exp(-0.80 * total_lai)
```

---

## Recruitment Process

1. **Gap calculates recruitment count** (P1):
   - Count dormant slots
   - Sum seed production from living trees
   - Add to seed bank, apply decay
   - Calculate num_to_recruit based on seed availability

2. **Dormant trees check for recruitment** (P0 next tick):
   - Hash agent_index with random seed for selection priority
   - If selected:
     - Find species source (living tree OR template)
     - Weight by invader * seed
     - Copy species traits
     - Initialize as seedling

3. **Template trees ensure species persistence**:
   - Templates (is_alive = -1) are always available
   - Species never lost even if all living trees die

---

## Mortality Process

Combined mortality probability:
```
total_mort = age_mort + stress_mort * (1 - age_mort)
           + seedling_mort * (1 - previous)
           + fire_mort * (1 - previous)
```

Where:
- **age_mort** = 4.605 / max_age (UVAFME age_tol=1)
- **stress_mort** = 0.37 if growth below threshold (UVAFME stress_tol=3)
- **seedling_mort** = 0.15 + light-dependent term for age <= 2
- **fire_mort** = intensity * (1 - survival_factor) + size penalty

### Sprout Regeneration
If tree dies and `sprout_rand < sprout_prob * 0.5` and age > 5:
- Tree respawns at 2cm diameter
- 80% of old biomass becomes litter
- Reset age to 1

---

## Soil Biogeochemistry

Three-layer model (A0 → A → Base):

### A0 Layer (Litter)
- Receives tree litter input
- Respiration rate: `AO_RESP = 5.24e-4`
- Transfer to A layer based on C/N ratio

### A Layer (Humus)
- Receives from A0 decomposition
- Respiration rate: `SA_RESP = 1.24e-5`
- **N mineralization**: main source of available N
- Transfer to Base layer

### Base Layer
- Receives from A layer
- Respiration rate: `SB_RESP = 2.74e-7`
- Slow turnover, long-term storage

### Moisture Effects
- Decomposition scaled by moisture function
- Optimal moisture around 30% saturation
- Reduced at both extremes

### Temperature Effects
- `temp_adjustment = 3.0^(0.1 * (temp - 1))`
- No decomposition below -5°C

---

## Fire Dynamics

1. **Fire probability** (Site P2):
   - Base: 1% annual
   - Increases with dry soil moisture
   - Cap: 15% annual

2. **Fire intensity** (if fire occurs):
   - Range: 0.3 - 1.0
   - Based on dry conditions

3. **Fire mortality** (Tree P0):
   - Base: intensity * (1 - fire_tol_survival)
   - Small tree penalty: +20% for diam < 10cm
   - Cap: 99%
