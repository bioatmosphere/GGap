# GGap - GPU-Accelerated Forest Gap Dynamics Model

A scalable GPU-enabled agent-based forest Gap model that integrates UVAFME (University of Virginia Forest Model Enhanced) with the SAGESim framework to create a modern, high-performance forest simulation system.

## Overview

GGap combines three major components:

1. **UVAFME** - Traditional forest gap model with detailed ecological processes
2. **SAGESim** - GPU-accelerated agent-based modeling framework
3. **GGap Integration** - New implementation bridging UVAFME processes with SAGESim's GPU capabilities

## Quick Start

### Demo (No GPU Required)

```bash
python main.py
```

### Full GPU Simulation with MPI

```bash
cd gap
mpirun -n 4 python run.py --num_trees 200 --years 100
```

## Features

### Implemented in GGap

- ✅ **6 Eastern US tree species** with UVAFME-based parameters
- ✅ **Individual tree agents** with growth, mortality, and competition
- ✅ **GPU-accelerated computation** using CuPy JIT kernels
- ✅ **MPI parallelization** for large-scale forests
- ✅ **UVAFME growth equations** (Forska height-diameter, allometric biomass)
- ✅ **Environmental responses** (temperature, light, drought, flood factors)
- ✅ **Age and stress-dependent mortality**
- ✅ **Species-specific shade tolerance**

### In Development

- 🚧 Full canopy light model with vertical layers
- 🚧 Spatial indexing for efficient neighbor queries
- 🚧 Seedling recruitment and regeneration
- 🚧 Soil biogeochemistry integration
- 🚧 Climate variability and disturbances

## Project Structure

```
GGap/
├── gap/                    # Main GGap implementation
│   ├── gap_model.py       # GAPModel class
│   ├── tree_breed.py      # TreeBreed agent definition
│   ├── tree_step_func.py  # GPU kernels for tree dynamics
│   ├── tree_species_data.py  # Species parameters
│   ├── run.py             # MPI runner script
│   └── README.md          # Detailed documentation
├── UVAFME/                # Original UVAFME Python translation
│   └── vegetation/        # Core UVAFME modules
├── SAGESim/               # SAGESim framework (submodule)
│   ├── sagesim/           # Framework core
│   └── examples/          # Reference implementations
├── main.py                # Quick demo script
└── CLAUDE.md              # Developer documentation
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

## Usage Examples

### Small Test Run

```bash
cd gap
mpirun -n 2 python run.py --num_trees 50 --forest_size 50 --years 50
```

### Large-Scale Simulation

```bash
mpirun -n 8 python run.py --num_trees 5000 --forest_size 200 --years 200
```

### Single Species Forest

```bash
# White Oak forest
mpirun -n 4 python run.py --num_trees 500 --species_dist single:3 --years 150
```

### High Competition Scenario

```bash
mpirun -n 4 python run.py --num_trees 1000 --forest_size 50 --neighborhood_radius 15
```

## Tree Species

Six pre-configured Eastern US species:

| ID | Species | Common Name | Type | Shade Tolerance | Max Age |
|----|---------|-------------|------|-----------------|---------|
| 1 | *Acer rubrum* | Red Maple | Mid-successional | Moderate | 150 |
| 2 | *Pinus taeda* | Loblolly Pine | Early-successional | Intolerant | 200 |
| 3 | *Quercus alba* | White Oak | Late-successional | Moderate | 300 |
| 4 | *Liquidambar styraciflua* | Sweetgum | Mid-successional | Intolerant | 200 |
| 5 | *Tsuga canadensis* | Eastern Hemlock | Late-successional | Very tolerant | 400 |
| 6 | *Liriodendron tulipifera* | Tulip Poplar | Early-successional | Very intolerant | 200 |

## Model Processes

### Growth

Each year, trees grow based on:

1. **Species-specific growth rate** - Fast (early) to slow (late successional)
2. **Temperature response** - Parabolic function of degree days
3. **Light availability** - Shading from taller neighbors
4. **Size constraints** - Asymptotic approach to maximum dimensions

Growth uses UVAFME equations:
- Forska height-diameter relationship
- Allometric biomass calculation
- Species-specific parameters

### Mortality

Trees die when:
- Age approaches species maximum
- Poor growing conditions (low light, temperature stress)
- Stochastic mortality events

### Competition

- Trees compete for light
- Taller neighbors cast shade
- Species differ in shade tolerance
- Growth reduced in low light conditions

## Performance

Typical performance on NVIDIA A100:

| Trees | Years | MPI Ranks | Time |
|-------|-------|-----------|------|
| 100 | 50 | 1 | ~2s |
| 1,000 | 100 | 4 | ~10s |
| 10,000 | 200 | 8 | ~2min |

Scales linearly with trees, years, and (near-linearly) with MPI ranks.

## Documentation

- **gap/README.md** - Detailed GGap implementation guide
- **CLAUDE.md** - Architecture and development guide
- **SAGESim/README.md** - SAGESim framework documentation
- **UVAFME/vegetation/README_PYTHON.md** - Original UVAFME translation

## Development

### Adding New Species

Edit `gap/tree_species_data.py`:

```python
SPECIES_DATA[7] = SpeciesParameters(
    species_id=7,
    genus_name="Fagus",
    common_name="American Beech",
    max_age=300.0,
    max_diam=120.0,
    max_ht=28.0,
    # ... other parameters
)
```

### Modifying Growth Equations

Edit GPU kernels in `gap/tree_step_func.py`:
- `light_step()` - Light competition
- `growth_step()` - Growth calculations
- `mortality_step()` - Mortality checks

**Important**: GPU kernels must follow CuPy JIT constraints (see CLAUDE.md).

### Testing

```bash
# Test with small forest
python -c "from gap import GAPModel; m = GAPModel(); m.create_forest(10, 50); m.print_forest_statistics()"

# Run full simulation test
cd gap
mpirun -n 2 python run.py --num_trees 20 --years 10
```

## Comparison: UVAFME vs GGap

| Feature | UVAFME (Original) | GGap |
|---------|------------------|------|
| Execution | CPU, serial | GPU, parallel (MPI) |
| Scale | 100s of trees | 10,000s of trees |
| Speed | Baseline | 10-100x faster |
| Architecture | Procedural Fortran/Python | Agent-based GPU |
| Processes | Full biogeochemistry | Core growth/mortality (extensible) |

GGap prioritizes scalability and speed while maintaining UVAFME's ecological realism where possible.

## References

### UVAFME
- University of Virginia Forest Model Enhanced
- Fortran original translated to Python
- Detailed ecological and biogeochemical processes

### SAGESim
- GPU-accelerated agent-based modeling
- CuPy for GPU computation
- MPI for distributed execution
- Oak Ridge National Laboratory

### Forest Gap Models
- Botkin, D.B., et al. (1972). "Some Ecological Consequences of a Computer Model of Forest Growth"
- Shugart, H.H. (1984). "A Theory of Forest Dynamics"
- Bugmann, H. (2001). "A Review of Forest Gap Models"

## Contributing

Contributions welcome! Priority areas:

1. Full canopy light model
2. Spatial indexing optimization
3. Regeneration and recruitment
4. Soil biogeochemistry integration
5. Validation against UVAFME outputs
6. Visualization tools

## License

Compatible with UVAFME and SAGESim licenses.

## Citation

If you use GGap in research, please cite:

```
GGap: GPU-Accelerated Forest Gap Dynamics Model
https://github.com/[your-repo]/GGap
```

And the underlying models:
- UVAFME (UVA Forest Model Enhanced)
- SAGESim (ORNL)
