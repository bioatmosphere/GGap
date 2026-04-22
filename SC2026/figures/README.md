# Figures

All figure generation scripts and outputs for the SC2026 paper.

## Quick Start

```bash
# Generate all scaling and architecture figures (no simulation data needed)
cd scripts
python ggap_architecture.py
python site_hierarchy.py
python strong_scaling_speedup.py
python weak_scaling_efficiency.py
python phase_breakdown.py
python setup_amortization.py

# Generate CONUS figures (requires simulation output, see ../conus_simulation/)
python conus_size_distribution.py
python conus_biomass_by_genus.py
python conus_10sites_with_bars.py    # Also requires cartopy, netCDF4, shapely
```

Figures are written to `figs/` as paired PDF (for LaTeX) and PNG (for preview) files.

## Script-to-Figure Map

| Script | Figure | Data Source |
|--------|--------|-------------|
| `ggap_architecture.py` | `ggap_architecture.{pdf,png}` | None (procedural) |
| `site_hierarchy.py` | `site_hierarchy.{pdf,png}` | `assets/*.png` |
| `strong_scaling_speedup.py` | `strong_scaling_speedup.{pdf,png}` | `../scaling_analysis/results/strong_scaling.csv` |
| `weak_scaling_efficiency.py` | `weak_scaling_efficiency.{pdf,png}` | `../scaling_analysis/results/weak_scaling.csv` |
| `phase_breakdown.py` | `phase_breakdown.{pdf,png}` | `../scaling_analysis/results/strong_scaling.csv` |
| `setup_amortization.py` | `setup_amortization.{pdf,png}` | `../scaling_analysis/results/weak_scaling.csv` |
| `conus_size_distribution.py` | `conus_size_distribution.{pdf,png}` | `../conus_simulation/results/last_year_species/` |
| `conus_biomass_by_genus.py` | `conus_biomass_by_genus.{pdf,png}` | `../conus_simulation/results/last_year_species/` |
| `conus_10sites_with_bars.py` | `conus_10sites_with_bars.{pdf,png,svg}` | `../conus_simulation/results/10sites/` + `assets/surfdata_*.nc` |

## Data Dependencies

**Available immediately** (included in this artifact):
- `../scaling_analysis/results/weak_scaling.csv` and `strong_scaling.csv`
- `assets/` — static images (globe.png, site_inset.png, conus_nldas2_forest_fraction_notitle.png)

**Requires separate download** (too large for Git, ~3.5 GB):
- `assets/surfdata_0.125nldas2_hist_2000_78pfts_c240908.nc` — CESM/CLM surface dataset used by `conus_10sites_with_bars.py` for the forest fraction map. Download from the [CESM input data repository](https://svn-ccsm-inputdata.cgd.ucar.edu/trunk/inputdata/) and place in `assets/`.

**Requires running experiments first** (see `../conus_simulation/`):
- `../conus_simulation/results/last_year_species/` — produced by `extract_last_species_data.py`
- `../conus_simulation/results/10sites/` — 10 selected site outputs for the map figure

## Shared Modules

- `_scaling_common.py` — CSV reading, metric derivation, path resolution, CLI argument helpers
- `_style.py` — Matplotlib style configuration (font sizes, colors, DPI)

## Other Scripts

- `generate_results_md.py` — Regenerates `md/scaling_results.md` from the scaling CSVs
- `extract_last_species_data.py` — Copy of the extraction script (also in `../conus_simulation/scripts/`)

## Directory Structure

```
scripts/         # Figure generation scripts
figs/            # Generated figures (PDF + PNG)
assets/          # Static input images and data
md/              # Scaling results markdown
```
