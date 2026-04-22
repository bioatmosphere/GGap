# SC2026 Artifact

Artifact for reproducing all experiments, figures, and tables in the SC2026 paper.

## Hardware Requirements

- **OLCF Frontier** (or equivalent AMD MI250X GPU cluster)
- Up to 256 nodes (2,048 GPUs) for full weak scaling
- Up to 64 nodes (512 GPUs) for full strong scaling
- 20 nodes (160 GPUs) for CONUS simulation

## Software Requirements

- Python 3.11+
- SAGESim (installed separately, see `../SAGESim/`)
- Python packages: `cupy`, `mpi4py`, `numpy`, `matplotlib`, `pillow`, `cartopy`, `netcdf4`, `shapely`
- System modules (Frontier): `PrgEnv-gnu/8.6.0`, `rocm/6.4.1`, `craype-accel-amd-gfx90a`, `metis/5.1.0`

Environment setup on Frontier:
```bash
source ../setup_frontier.sh
```

## Reproduction Workflow

### 1. Scaling Experiments (`scaling_analysis/`)

Generate synthetic data and run weak/strong scaling experiments.

```bash
cd scaling_analysis/scripts
python create_synthetic_csv.py
sbatch submit_weak_scaling.sh
sbatch submit_strong_scaling.sh
```

**Produces**: `scaling_analysis/results/weak_scaling.csv` and `strong_scaling.csv`

See [`scaling_analysis/README.md`](scaling_analysis/README.md) for details.

### 2. CONUS Simulation (`conus_simulation/`)

Run the 1,424-site CONUS forest simulation and extract results.

```bash
cd conus_simulation/scripts
sbatch submit_conus.sh              # Production run with snapshots
sbatch submit_conus_nosave.sh       # No-snapshot run (timing comparison)
python extract_10sites_species.py   # 10 representative sites → Table 1 + map figure
python extract_last_species_data.py # All 1,424 sites → size distribution + biomass figures
```

**Produces**: `conus_simulation/results/{10sites/, last_year_species/}`

See [`conus_simulation/README.md`](conus_simulation/README.md) for details.

### 3. Generate Figures (`figures/`)

Generate all paper figures from experiment results.

```bash
cd figures/scripts

# Architecture and hierarchy (no experiment data needed)
python ggap_architecture.py
python site_hierarchy.py

# Scaling figures (needs scaling_analysis results)
python strong_scaling_speedup.py
python weak_scaling_efficiency.py
python phase_breakdown.py
python setup_amortization.py

# CONUS figures (needs conus_simulation results)
python conus_10sites_with_bars.py
python conus_size_distribution.py
python conus_biomass_by_genus.py
```

**Produces**: PDF + PNG figures in `figures/figs/`

See [`figures/README.md`](figures/README.md) for the script-to-figure map.

## Complete Paper Artifact Map

| Paper Element | Source |
|--------------|--------|
| **Table 1**: 10 representative sites | `conus_simulation/results/10sites/10sites_species.csv` |
| **Figure**: GGap architecture | `figures/figs/ggap_architecture.pdf` |
| **Figure**: Site hierarchy | `figures/figs/site_hierarchy.pdf` |
| **Figure**: Strong scaling speedup | `figures/figs/strong_scaling_speedup.pdf` |
| **Figure**: Weak scaling efficiency | `figures/figs/weak_scaling_efficiency.pdf` |
| **Figure**: Phase breakdown | `figures/figs/phase_breakdown.pdf` |
| **Figure**: Setup amortization | `figures/figs/setup_amortization.pdf` |
| **Figure**: CONUS 10-site map | `figures/figs/conus_10sites_with_bars.pdf` |
| **Figure**: Size distribution | `figures/figs/conus_size_distribution.pdf` |
| **Figure**: Biomass by genus | `figures/figs/conus_biomass_by_genus.pdf` |
| **Timing**: Pure simulation (112s) | `conus_simulation/logs/conus_nosave_*.out` |
| **Timing**: With snapshots (808s) | `conus_simulation/logs/conus_*.out` |

## Directory Structure

```
SC2026/
  scaling_analysis/         # Weak and strong scaling experiments
    scripts/                #   Experiment runners and submit scripts
    results/                #   CSV results (provided)
  conus_simulation/         # CONUS-wide forest simulation
    scripts/                #   Simulation, extraction, and submit scripts
    input_data/             #   CONUS forest configuration CSVs (provided)
  figures/                  # All paper figures
    scripts/                #   Plotting scripts
    figs/                   #   Generated figures (PDF + PNG)
    assets/                 #   Static input images and data
    md/                     #   Scaling results markdown
```
