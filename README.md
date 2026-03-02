# GGap - GPU-Accelerated Forest Gap Dynamics Model

A scalable GPU-enabled agent-based forest gap model that integrates UVAFME (University of Virginia Forest Model Enhanced) with the SAGESim framework to create a modern, high-performance forest simulation system.

## Overview

GGap combines three major components:

1. **UVAFME** - Traditional forest gap model with detailed ecological processes
2. **SAGESim** - GPU-accelerated agent-based modeling framework
3. **GGap Integration** - Implementation bridging UVAFME processes with SAGESim's GPU capabilities

## Quick Start

### Single-Site Simulation (GPU Required)

```bash
cd gap
python run_one_site.py --num_gaps 200 --pool_size 1000 --years 500
```

### Plot Results

```bash
cd gap
python plot_outputs.py --output-dir ../output_data --format png
```

## Features

- **3-level agent hierarchy**: Site → Gap → Tree with 7-priority step function pipeline
- **32 tree species** (20 genera) loaded from UVAFME CSV input files
- **Full soil biogeochemistry**: 3-layer (A0/A/Base) C/N cycling with daily timestep
- **GPU-accelerated computation** using CuPy JIT kernels
- **MPI parallelization** for multi-rank execution
- **GAPpy-compatible CSV output**: 5 output files (site, soil, genus, species, tree data)
- **Visualization**: 4 plot types (forest dynamics, soil, environment, dashboard)
- **Environmental responses**: temperature, light, drought, flood, nutrient limitation
- **Tree lifecycle**: growth, mortality, seedbank, seedling recruitment, renewal

## Project Structure

```
GGap/
├── gap/                          # Main GGap implementation
│   ├── gap_model.py              # GAPModel class (breeds, initialization, data collection)
│   ├── run_one_site.py           # Main simulation runner
│   ├── output_utils.py           # GAPpy-compatible CSV output writer
│   ├── plot_outputs.py           # Visualization (4 plot types)
│   ├── step_func_code.py         # Auto-generated GPU kernels
│   ├── step_functions/           # GPU step functions by agent type
│   │   ├── gap/                  # P0: litter, P3: demand, P5: sync
│   │   ├── site/                 # P1: soil, P4: nutrient
│   │   └── tree/                 # P2: potential growth, P6: actual growth
│   └── docs/                     # AGENT_PROPERTIES.md, implementation_logic.md
├── GAPpy/                        # Python UVAFME translation (submodule, reference)
├── SAGESim/                      # SAGESim framework (submodule)
│   ├── sagesim/                  # Framework core
│   └── examples/                 # Reference implementations
├── input_data/                   # UVAFME CSV input files
│   ├── UVAFME2012_specieslist.csv
│   ├── UVAFME2012_site.csv
│   ├── UVAFME2012_climate.csv
│   ├── UVAFME2012_climate_stddev.csv
│   ├── UVAFME2012_rangelist.csv
│   └── UVAFME2012_altitudes.csv
├── output_data/                  # Simulation CSV outputs
├── main.py                       # Quick demo script (no GPU)
└── CLAUDE.md                     # Developer documentation
```

## Requirements

- **Python 3.13+**
- **GPU**: NVIDIA with CUDA or AMD with ROCm 5.7.1+
- **MPI**: For parallel execution
- **Dependencies**: See `pyproject.toml`

### Installation

```bash
# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

## Usage

### Single-Site Simulation

```bash
cd gap
python run_one_site.py [OPTIONS]

Options:
  --num_gaps INT          Number of gaps per site (default: 200)
  --pool_size INT         Max tree slots per gap (default: 1000)
  --years INT             Simulation years (default: 1000)
  --report_interval INT   Years between reports and CSV output (default: 10)
  --site_id INT           Site ID from UVAFME CSV files (default: 0)
  --data_dir PATH         Directory with UVAFME CSV files (default: input_data)
  --prefix STRING         File prefix for CSV files (default: UVAFME2012)
  --output_dir PATH       Output directory for CSV files (default: output_data)
  --no_tree_data          Skip tree_data.csv (can be very large)
```

**Example runs:**
```bash
# Quick test (10 gaps, 50 years)
python run_one_site.py --num_gaps 10 --years 50

# Full simulation
python run_one_site.py --num_gaps 200 --pool_size 1000 --years 1000

# Different site
python run_one_site.py --site_id 1 --data_dir input_data --prefix UVAFME2012
```

### Plotting

```bash
cd gap
python plot_outputs.py --output-dir ../output_data --format png

Options:
  --output-dir PATH     Directory with CSV output (default: ../output_data)
  --plots-dir PATH      Directory for plot images (default: ../plots)
  --format FORMAT       Image format: png, pdf, svg (default: png)
  --dpi INT             Resolution (default: 150)
  --style STYLE         Matplotlib style (default: seaborn-v0_8-whitegrid)
  --show / --no-show    Show interactive plots (default: --no-show)
```

Generates 4 plot types:
- **forest_dynamics** — biomass, basal area, tree counts, species composition
- **soil_biogeochemistry** — soil C/N pools, available N, biomass
- **environmental_conditions** — degree days, dry days, flood days, precipitation
- **summary_dashboard** — combined overview

### Output Files

Five GAPpy-compatible CSV files:

| File | Content |
|------|---------|
| `site_data.csv` | Annual site climate and conditions |
| `soil_data.csv` | Soil C/N pools (A0, A, Base), available N, biomass |
| `genus_data.csv` | Per-genus biomass, basal area, counts, diameter classes |
| `species_data.csv` | Per-species biomass, basal area, counts, diameter classes |
| `tree_data.csv` | Individual tree data (optional) |

## Model Architecture

### Step Function Pipeline

Each simulation tick (= 1 year) executes 7 GPU kernels in priority order:

| Priority | Kernel | Agent | Purpose |
|----------|--------|-------|---------|
| P0 | `gap_litter_aggregate_step` | Gap | Litter aggregation, recruitment count |
| P1 | `soil_step` | Site | Daily biogeochemistry (365 days) |
| P2 | `tree_potential_growth_step` | Tree | Env responses, potential growth |
| P3 | `gap_demand_aggregate_step` | Gap | Sum N demand from trees |
| P4 | `site_nutrient_step` | Site | Compute N supply ratio |
| P5 | `gap_sync_step` | Gap | Relay climate and N ratio to trees |
| P6 | `tree_actual_growth_step` | Tree | N-limited growth, mortality, renewal |

### Comparison: GAPpy vs GGap

| Feature | GAPpy (Reference) | GGap |
|---------|-------------------|------|
| Execution | CPU, serial | GPU, parallel (MPI) |
| Scale | ~20 plots | 200+ gaps |
| Architecture | Procedural Python | Agent-based GPU kernels |
| Processes | Full biogeochemistry | Full biogeochemistry |
| Output | 7 CSV files | 5 GAPpy-compatible CSV files |
| Species | 32 from CSV | 32 from same CSV |

## Documentation

- **gap/README.md** — Detailed GGap implementation guide
- **gap/docs/AGENT_PROPERTIES.md** — Complete agent property reference
- **gap/docs/implementation_logic.md** — Step function implementation details
- **CLAUDE.md** — Architecture and development guide
- **SAGESim/README.md** — SAGESim framework documentation

## References

### UVAFME
- University of Virginia Forest Model Enhanced
- Fortran original translated to Python
- Detailed ecological and biogeochemical processes

### SAGESim
- GPU-accelerated agent-based modeling
- CuPy for GPU computation, MPI for distributed execution
- Oak Ridge National Laboratory

### Forest Gap Models
- Botkin, D.B., et al. (1972). "Some Ecological Consequences of a Computer Model of Forest Growth"
- Shugart, H.H. (1984). "A Theory of Forest Dynamics"
- Bugmann, H. (2001). "A Review of Forest Gap Models"

## License

Compatible with UVAFME and SAGESim licenses.
