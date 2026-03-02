# GGap - GPU-Accelerated Forest Gap Dynamics Model

GGap integrates the UVAFME (University of Virginia Forest Model Enhanced) forest gap dynamics model with the SAGESim GPU-accelerated agent-based modeling framework.

## Overview

This implementation translates UVAFME's forest gap model into a scalable, GPU-enabled simulation using:

- **3-level agent hierarchy** (Site → Gap → Tree) with SAGESim breeds
- **32 tree species** loaded from UVAFME CSV input files (20 genera)
- **7-priority step function pipeline** running on GPU via CuPy JIT
- **Full soil biogeochemistry** with daily timestep (365 days/year)
- **GAPpy-compatible CSV output** (5 files matching GAPpy format)
- **MPI parallelization** for multi-rank execution

## Architecture

### Agent Hierarchy

```
Site Agent (1 per simulation)
├── Soil pools: A0, A, Base layers (C and N)
├── Climate: 12 months × (tmin, tmax, prcp, prcp_std, tmin_std, tmax_std)
├── State: avail_n, deg_days, dry_days, flood_days
│
├── Gap Agent (num_gaps per site, default 200)
│   ├── Light profile, seedbank, seedling arrays
│   ├── Aggregated litter (C/N) and N demand from trees
│   │
│   ├── Tree Agent (pool_size slots per gap, default 1000)
│   │   ├── Species params: max_age, max_diam, max_ht, growth rate, tolerances
│   │   ├── State: diam, height, canopy_ht, biomC, biomN, leaf_bm, age
│   │   └── Growth factors: light_rsp, temp_rsp, drought_rsp, nutrient_rsp
│   └── ...
└── ...
```

### Step Function Pipeline (7 Priorities)

Each simulation tick (= 1 year) executes these GPU kernels in order:

| Priority | Step Function | Agent | Description |
|----------|--------------|-------|-------------|
| P0 | `gap_litter_aggregate_step` | Gap | Aggregate litter from trees, density-based recruitment count |
| P1 | `soil_step` | Site | Daily soil biogeochemistry (365 days), climate variability, water balance |
| P2 | `tree_potential_growth_step` | Tree | Environmental responses, potential diameter growth |
| P3 | `gap_demand_aggregate_step` | Gap | Sum nitrogen demand across trees per gap |
| P4 | `site_nutrient_step` | Site | Compute N supply ratio from available N vs demand |
| P5 | `gap_sync_step` | Gap | Relay climate and N ratio to trees, clear accumulators |
| P6 | `tree_actual_growth_step` | Tree | N-limited actual growth, mortality check, biomass update, renewal |

### Core Files

| File | Description |
|------|-------------|
| `gap_model.py` | GAPModel class: breed registration, CSV initialization, agent creation, data collection |
| `run_one_site.py` | Main simulation runner with CLI arguments and console output |
| `output_utils.py` | OutputWriter: 5 GAPpy-compatible CSV files with proper scaling |
| `plot_outputs.py` | Plotting script: 4 plot types from CSV output data |
| `step_func_code.py` | Auto-generated GPU kernel code (do not edit manually) |
| `__init__.py` | Package exports |

### Step Functions (`step_functions/`)

```
step_functions/
├── gap/
│   ├── gap_litter_aggregate_step.py    # P0: litter aggregation, recruitment count
│   ├── gap_demand_aggregate_step.py    # P3: N demand aggregation
│   └── gap_sync_step.py               # P5: relay climate/N, clear accumulators
├── site/
│   ├── soil_step.py                    # P1: daily soil biogeochemistry
│   └── site_nutrient_step.py           # P4: compute N supply ratio
└── tree/
    ├── tree_potential_growth_step.py    # P2: env responses, potential growth
    └── tree_actual_growth_step.py       # P6: actual growth, mortality, renewal
```

## Species

32 species loaded from `input_data/UVAFME2012_specieslist.csv`, filtered by site range. Species are grouped into 20 genera:

Acer, Betula, Carya, Castanea, Cornus, Fagus, Fraxinus, Juniperus, Liquidambar, Liriodendron, Magnolia, Nyssa, Oxydendrum, Picea, Pinus, Prunus, Quercus, Robinia, Tilia, Tsuga

Each species has parameters for: max age/diameter/height, growth rate (g), shade/drought/flood tolerance, degree-day range (dd_min/dd_max), leaf area, wood density, conifer flag, and more.

## Model Processes

### Growth (P2 + P6)

Trees grow each year through a two-phase process:

**Phase 1 - Potential Growth (P2):**
1. Temperature response — parabolic function of degree days
2. Light response — exponential saturation based on shade tolerance
3. Drought response — exponential decay with dry days
4. Flood response — tolerance-based factor
5. Potential diameter increment from species growth parameters

**Phase 2 - Actual Growth (P6):**
1. Nitrogen demand calculated from potential growth
2. Gap-level N limitation applied (demand vs available)
3. Final diameter/height/biomass update
4. Mortality check (age-dependent + stress-induced)
5. Renewal: seedbank → seedling → new tree recruitment

### Soil Biogeochemistry (P1)

Daily timestep (365 iterations per tick):
1. Monthly climate perturbation (temperature ±1°C, precip ±50%)
2. Daily temperature and precipitation from monthly values
3. PET calculation (Hamon method)
4. Soil water balance with freeze/thaw, runoff, percolation
5. Three-layer decomposition (A0 → A → Base)
6. Nitrogen mineralization and availability
7. Annual dry_days recomputation from perturbed precipitation

### Light Competition (P0 + P5)

- Gap-level light profile calculated from tree heights and leaf areas
- Beer-Lambert law attenuation through canopy layers
- Deciduous vs coniferous light arrays
- Distributed back to individual trees at their canopy height

## Usage

### Running a Simulation

```bash
cd gap
python run_one_site.py --num_gaps 200 --pool_size 1000 --years 500
```

### Command-Line Options

```
Options:
  --num_gaps INT          Number of gaps per site (default: 200)
  --pool_size INT         Max tree slots per gap (default: 1000)
  --years INT             Simulation duration in years (default: 1000)
  --report_interval INT   Years between reports and CSV output (default: 10)
  --site_id INT           Site ID from UVAFME CSV files (default: 0)
  --data_dir PATH         Directory with UVAFME CSV files (default: input_data)
  --prefix STRING         File prefix for CSV files (default: UVAFME2012)
  --output_dir PATH       Directory for output CSV files (default: output_data)
  --no_tree_data          Skip writing tree_data.csv
```

### Plotting Results

```bash
python plot_outputs.py --output-dir ../output_data --format png
```

Generates 4 plot types:
- **forest_dynamics** — biomass, basal area, tree counts, species composition over time
- **soil_biogeochemistry** — soil C/N pools, available N, biomass N over time
- **environmental_conditions** — degree days, dry days, flood days, precipitation
- **summary_dashboard** — combined overview of key metrics

### Output Files

Five GAPpy-compatible CSV files in `output_dir/`:

| File | Content |
|------|---------|
| `site_data.csv` | Annual site-level climate and conditions |
| `soil_data.csv` | Soil C/N pools (A0, A, Base layers), available N, biomass |
| `genus_data.csv` | Per-genus biomass, basal area, tree counts, diameter classes |
| `species_data.csv` | Per-species biomass, basal area, tree counts, diameter classes |
| `tree_data.csv` | Individual tree data (optional, can be very large) |

Scaling follows GAPpy conventions: plotscale = HEC_TO_M2/plotsize, plotadj = plotscale/num_gaps.

## Input Data

Required CSV files in `input_data/` (with configurable prefix):

| File | Content |
|------|---------|
| `{prefix}_specieslist.csv` | Species trait parameters (32+ species) |
| `{prefix}_site.csv` | Site characteristics (location, elevation, soil) |
| `{prefix}_climate.csv` | Monthly temperature and precipitation |
| `{prefix}_climate_stddev.csv` | Climate variability (std dev) |
| `{prefix}_rangelist.csv` | Species range by site |
| `{prefix}_altitudes.csv` | Elevation data |

## GPU/CuPy Constraints

Step functions are compiled to GPU kernels with strict constraints:

- No Python dicts, objects, or nested functions
- No `return`, `break`, or `continue` statements
- Must use CuPy operations, not NumPy
- Cannot use `-1` indexing (use `len(array)-1`)
- Variables cannot be reassigned in `if`/`for` blocks — declare at function top level
- For-each loops not supported (use `for i in range(n)`)

See `CLAUDE.md` for complete CuPy JIT limitation details.

## Documentation

- `docs/AGENT_PROPERTIES.md` — Complete listing of all agent properties (params, states, states_db)
- `docs/implementation_logic.md` — Detailed implementation logic for each step function

## References

### UVAFME
- University of Virginia Forest Model Enhanced
- Translated from Fortran to Python
- See `UVAFME/` and `GAPpy/` directories

### SAGESim
- GPU-accelerated agent-based modeling framework
- CuPy for GPU computation, MPI for parallel execution
- See `SAGESim/` directory

### Forest Gap Models
- Botkin, D.B., et al. (1972). "Some Ecological Consequences of a Computer Model of Forest Growth"
- Shugart, H.H. (1984). "A Theory of Forest Dynamics"
- Bugmann, H. (2001). "A Review of Forest Gap Models"
