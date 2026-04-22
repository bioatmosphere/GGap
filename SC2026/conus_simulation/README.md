# CONUS Simulation

CONUS-wide forest gap dynamics simulation using GGap. Simulates 1,424 forested sites across the contiguous United States for 1,000 years on OLCF Frontier.

## Reproduction Workflow

All commands run from the `scripts/` directory.

### Step 1: Run the CONUS simulation

**Important**: Submit jobs from the `scripts/` directory.

```bash
cd scripts
sbatch submit_conus.sh
```

- **Nodes**: 20 (160 GPUs)
- **Wall time**: ~30 minutes
- **Logs**: `../logs/conus_<jobid>.out`

**Output** (`results/simulation/`):
- `snapshots/year_NNNN_rank_NNN.npz` — GPU state arrays saved every 10 simulation years (100 snapshots per rank)
- `rank_NNN_sites.pkl` — Per-rank metadata (160 files: site assignments, species data, tree-to-gap mapping)

### Step 2: Run the no-snapshot variant (for I/O overhead comparison)

```bash
sbatch submit_conus_nosave.sh
```

Runs the same simulation but calls `model.simulate(1000)` in one shot with no GPU-to-CPU transfer or disk I/O. Used in the paper to measure pure simulation cost vs. snapshot overhead.

**Output**: Timing only (printed to log). No data files.

### Step 3: Extract data for the 10 representative sites

```bash
python extract_10sites_species.py
```

Reads all 100 snapshots for 10 selected sites to produce full time-series data.

**Output** (`results/10sites/`):

| File | Description | Used by |
|------|-------------|---------|
| `10sites_species.csv` | Species counts at year 1000 for 10 sites | **Table 1** in paper |
| `site_NNNN/species_data.csv` (10 dirs) | Per-species size distributions and biomass over all 100 years | **Figure: `conus_10sites_with_bars`** |

### Step 4: Extract data for all 1,424 sites

```bash
python extract_last_species_data.py
```

Reads the final snapshot (year 1000) for all 160 ranks, extracts species data for every site.

**Output** (`results/last_year_species/`):

| File | Description | Used by |
|------|-------------|---------|
| `site_NNNN/species_data.csv` (1,424 dirs) | Per-species counts and biomass at year 1000 | **Figure: `conus_size_distribution`** and **Figure: `conus_biomass_by_genus`** |

### Step 5: Generate figures

After Steps 3 and 4, generate all CONUS figures from `../figures/scripts/`:

```bash
cd ../../figures/scripts

# Needs Step 3 output (results/10sites/)
python conus_10sites_with_bars.py     # CONUS map with 10 sites + bar/composition charts

# Needs Step 4 output (results/last_year_species/)
python conus_size_distribution.py     # Diameter-class distribution across all 1,424 sites
python conus_biomass_by_genus.py      # Biomass by genus pie chart across all 1,424 sites
```

**Output**: PDF + PNG figures in `../figures/figs/`.

## Complete Output-to-Figure/Table Map

| Script | Output Location | Paper Figure/Table |
|--------|----------------|-------------------|
| `submit_conus.sh` | `results/simulation/` | Raw data for all downstream |
| `submit_conus_nosave.sh` | Log file only | I/O overhead timing comparison |
| `extract_10sites_species.py` | `results/10sites/10sites_species.csv` | **Table 1**: Species counts at 10 sites |
| `extract_10sites_species.py` | `results/10sites/site_NNNN/species_data.csv` | **Figure**: `conus_10sites_with_bars.{pdf,png}` |
| `extract_last_species_data.py` | `results/last_year_species/site_NNNN/species_data.csv` | **Figure**: `conus_size_distribution.{pdf,png}` |
| `extract_last_species_data.py` | `results/last_year_species/site_NNNN/species_data.csv` | **Figure**: `conus_biomass_by_genus.{pdf,png}` |

## Input Data

`input_data/` contains the CONUS forest configuration (~2.6 MB total):

| File | Description |
|------|-------------|
| `CONUS_site.csv` | 1,424 site locations, soil, and climate parameters |
| `CONUS_specieslist.csv` | 235 tree species with growth/tolerance traits |
| `CONUS_rangelist.csv` | Species presence/absence per site |
| `CONUS_climate.csv` | Monthly temperature and precipitation per site |
| `CONUS_climate_stddev.csv` | Climate variability per site |
| `CONUS_altitudes.csv` | Site elevations |

## Simulation Parameters

| Parameter | With Snapshots | No Snapshots |
|-----------|---------------|--------------|
| Sites | 1,424 | 1,424 |
| Nodes | 20 (160 GPUs) | 20 (160 GPUs) |
| Gaps/site | 500 | 500 |
| Trees/gap | 1,000 | 1,000 |
| Years | 1,000 | 1,000 |
| Snapshot interval | Every 10 years | None |
| Dispersal factor | 2.0 | 2.0 |

Note: Results will vary across runs. Ecological outputs (species counts, biomass, size distributions) differ due to the stochastic nature of forest gap dynamics. Performance timings will also show small variations due to GPU scheduling, MPI synchronization jitter, and Lustre I/O contention.

## Scripts

| Script | Purpose |
|--------|---------|
| `run_conus.py` | Main simulation: loads CONUS data, builds dispersal network, METIS partitioning, runs GGap simulation with periodic snapshots |
| `submit_conus.sh` | SLURM submission for production run with snapshots |
| `submit_conus_nosave.sh` | SLURM submission for no-snapshot timing run (`--no_snapshots` flag) |
| `extract_10sites_species.py` | Extracts full time-series species data for 10 representative sites from all snapshots |
| `extract_last_species_data.py` | Extracts final-year species data for all 1,424 sites from the last snapshot |

## Example Logs

The `logs/` directory includes example output from our runs on Frontier for reference:

- `conus_4447217.out` — Production run with snapshots (20 nodes, 160 GPUs, 808s total: ~112s compute + ~400s Lustre I/O + ~296s MPI barrier wait)
- `conus_nosave_4447196.out` — No-snapshot run (20 nodes, 160 GPUs, 112s pure simulation)

These are provided as examples only. Your job IDs and timings will differ. Compute time is consistent across runs; I/O time varies due to Lustre contention.

## Directory Structure

```
scripts/                    # All executable scripts
input_data/                 # CONUS forest configuration CSVs (provided)
results/                    # Created by scripts
  simulation/               # Raw simulation output (Step 1)
    snapshots/              #   year_NNNN_rank_NNN.npz (100 × 160 files)
    rank_NNN_sites.pkl      #   Metadata (160 files)
  simulation_nosave/        # No-snapshot run output (Step 2, timing only)
  10sites/                  # 10 representative sites (Step 3)
    10sites_species.csv     #   Summary table → Paper Table 1
    site_NNNN/              #   Per-site time-series → conus_10sites_with_bars figure
  last_year_species/        # All 1,424 sites, final year (Step 4)
    site_NNNN/              #   Per-site data → size_distribution + biomass_by_genus figures
logs/                       # SLURM job logs
```
