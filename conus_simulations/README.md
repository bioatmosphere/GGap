# CONUS-Wide GGap Forest Simulation

This directory contains scripts for running continent-scale forest simulations across all 1,424 CONUS sites with MPI+GPU parallelization.

---

## NEW: Integrated Simulation (run_conus.py)

The primary workflow now combines partitioning and simulation in a single run. See [Quick Start](#quick-start) below.

## LEGACY: Standalone Network Partitioning

For visualization and analysis only, `build_network_partition.py` can generate partition files independently.

## Overview

The CONUS simulation uses:
- **1,424 sites** across the continental US
- **235 tree species** with varying dispersal distances (0.2-50 km)
- **Per-site dispersal** based on species actually present at each site (rangelist)
- **Directed graph** where Site A→B if A can disperse seeds to B
- **METIS partitioning** to minimize cross-rank communication for MPI parallelization

## Directory Structure

```
conus_simulations/
├── scripts/
│   └── build_network_partition.py    # Network + METIS partitioning
├── results/                           # Partition outputs
│   ├── partition.csv                  # Site → partition mapping
│   ├── edges_directed.csv             # Directed edge list
│   └── site_dispersal.csv             # Per-site dispersal stats
└── logs/                              # Job logs
```

## Quick Start

### Test Run (2 nodes, 50 years)
```bash
cd conus_simulations/scripts
sbatch submit_conus_test.sh
```

### Production Run (20 nodes, 1,000 years)
```bash
cd conus_simulations/scripts
sbatch submit_conus.sh
```

### Configuration
- **Partition size**: 160 (matches 20 nodes × 8 GPUs)
- **Dispersal factor**: 2.0 (gives ~8 neighbors per site)
- **Sites per rank**: ~9 sites (1,424 sites / 160 ranks)

---

## Simulation Parameters

| Parameter | Test | Production | Description |
|-----------|------|------------|-------------|
| Nodes | 2 | 20 | HPC compute nodes |
| MPI ranks | 16 | 160 | 8 GPUs per node |
| Sites | 1,424 | 1,424 | CONUS forest sites |
| Gaps per site | 10 | 500 | Forest gap agents |
| Max trees per gap | 100 | 1,000 | Tree agent slots |
| Years | 50 | 1,000 | Simulation duration |

### Estimated Resources

**Test run** (2 nodes, 16 ranks):
- Agents per rank: ~90K
- Total agents: ~1.4M
- GPU memory: <1 GB/rank
- Runtime: ~10-30 min

**Production run** (20 nodes, 160 ranks):
- Agents per rank: ~1.9M
- Total agents: ~303M
- GPU memory: 5-10 GB/rank (needs profiling)
- Runtime: TBD (hours to days)

---

## Legacy: Standalone Network Partitioning (for visualization)

### Build Network Graph and Partition

```bash
cd conus_simulations/scripts

# Basic: all 1424 sites, 160 partitions, dispersal_factor=2.0
python build_network_partition.py --num_ranks 160 --dispersal_factor 2.0

# Custom dispersal factor (affects network density)
python build_network_partition.py --num_ranks 16 --dispersal_factor 3.0
```

## Command-Line Arguments

- `--num_ranks`: Number of partitions (default: 4)
- `--dispersal_factor`: Cutoff multiplier (default: 5.0)
  - Cutoff = `dispersal_factor × max_dispersal[site]`
  - Higher factor → denser network, more connections
- `--max_dispersal_dist`: Global override in km (default: None = per-site from species CSV)
- `--data_dir`: Input directory (default: "../../input_data")
- `--prefix`: File prefix (default: "CONUS")
- `--results_dir`: Output directory (default: "../results")

## Output Files

### 1. `partition.csv` - Site Partitioning

METIS partition assignments with metadata.

```csv
site_id,partition,latitude,longitude,num_out_edges,num_in_edges,max_dispersal_km
0,5,48.75,-122.25,34,34,50.00
1,5,48.75,-121.75,35,35,50.00
```

- `partition`: Which MPI rank this site belongs to
- `num_out_edges`: How many sites THIS site can disperse to
- `num_in_edges`: How many sites can disperse to THIS site
- `max_dispersal_km`: Maximum dispersal distance of species at this site

### 2. `edges_directed.csv` - Directed Edge List

All directed edges for SAGESim simulation.

```csv
reader_site,source_site,distance_km,cross_partition,source_max_dispersal
1,0,36.66,0,50.00
2,0,73.32,0,50.00
```

- `reader_site`: Site that READS data (receives seeds)
- `source_site`: Site that PROVIDES data (disperses seeds)
- `cross_partition`: 1 if edge crosses partition boundary (MPI communication needed)
- Each row = `model.connect_agents(reader_site, source_site, directed=True)`

### 3. `site_dispersal.csv` - Per-Site Statistics

```csv
site_id,num_species,max_dispersal_km,out_degree,in_degree
0,44,50.00,34,34
```

## Example Output

```
CONUS Directed Dispersal Network and METIS Partitioning
========================================

Configuration:
  Partitions: 8
  Dispersal factor: 5.0

Phase 1: Loading Data
  Sites: 1424 from CONUS_site.csv
  Species: 235 from CONUS_rangelist.csv

Per-site max dispersal:
  Min: 3.0 km
  Max: 50.0 km
  Mean: 43.6 km

Phase 2: Building Directed Graph
  Directed edges: 60,577

Phase 3: METIS Partitioning
  Edge cut: 3,192

Partition Statistics:
  Sites per partition:
    Rank 0-7: 172-183 sites (12.1-12.9%)
  Load balance: imbalance=6.0%
  Directed cross-partition: 6,301 (10.4%)

Output Files:
  partition.csv (1424 rows)
  edges_directed.csv (60,577 directed edges)
  site_dispersal.csv (1424 sites)

Total time: 1.57s
```

## Network Graph Details

### Per-Site Dispersal Calculation

For each site:
1. Load rangelist to find which species are present
2. Read `max_dispersal_dist` from species CSV
3. Compute `max_dispersal[site] = max(dispersal_dist for sp in species_at_site)`

### Directed Graph Construction

```python
for site_a in all_sites:
    cutoff_a = dispersal_factor × max_dispersal[site_a]

    for site_b in all_sites:
        dist = haversine_distance(site_a, site_b)

        if dist <= cutoff_a:
            # Site A can disperse TO Site B
            # Site B needs to READ Site A's data
            # Add directed edge: B → A
            add_edge(reader=B, source=A, distance=dist)
```

### METIS Partitioning

1. **Convert to undirected**: METIS requires symmetric graphs
   - Edge exists if A→B OR B→A (or both)
2. **Run METIS**: Minimizes edge cuts across partitions
3. **Output directed graph**: Original directed edges with partition metadata

## Future Use in Simulation

```python
# In future run_conus_simulation.py:
import csv

# Load partition
partition = {}
with open('results/partition.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        partition[int(row['site_id'])] = int(row['partition'])

# Initialize sites using partition
for site_id in all_sites:
    rank = partition[site_id]
    model.initialize_site_with_gaps(site_id, num_gaps, maxtrees, rank=rank)

# Create DIRECTED connections
with open('results/edges_directed.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        reader_site = int(row['reader_site'])
        source_site = int(row['source_site'])
        model.connect_agents(reader_site, source_site, directed=True)
```

## Technical Notes

### METIS Implementation

The script supports two METIS implementations:
1. **Python `metis` library**: Tried first (currently disabled due to segfaults on login node)
2. **`gpmetis` binary**: Fallback (stable, requires `module load metis/5.1.0`)

The binary is currently used exclusively because the Python library is unstable.

### Graph Format

METIS graph file format (for binary):
```
<num_vertices> <num_edges>
<neighbor_1> <neighbor_2> ...   (for vertex 1)
<neighbor_1> <neighbor_2> ...   (for vertex 2)
...
```
- Vertices are 1-indexed
- Undirected edges (each edge appears in both adjacency lists)

### SAGESim Connection Semantics

SAGESim directed connections:
- `connect_agents(agent_0, agent_1, directed=True)`
- **agent_0 can READ FROM agent_1** (one-way neighbor access)
- For seed dispersal: receiver reads donor's `site_avail_spec`
- Therefore: `connect_agents(receiver, donor, directed=True)`

## References

- METIS: http://glaros.dtc.umn.edu/gkhome/metis/metis/overview
- SAGESim: https://github.com/bioatmosphere/SAGESim
- GGap model: ../gap/README.md
