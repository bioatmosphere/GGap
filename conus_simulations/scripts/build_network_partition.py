"""
Build directed dispersal network graph and METIS partitioning for CONUS simulation.

This script:
1. Loads all 1424 CONUS sites and species data
2. Computes per-site max dispersal based on species present (rangelist)
3. Builds directed dispersal graph (A→B if A can disperse to B)
4. Temporarily converts to undirected for METIS partitioning
5. Outputs partition and directed edge list for future SAGESim simulation

Usage:
    cd conus_simulations/scripts

    # Basic run
    python build_network_partition.py --num_ranks 8

    # Custom dispersal factor
    python build_network_partition.py --num_ranks 16 --dispersal_factor 3.0

    # Override dispersal distance (for testing)
    python build_network_partition.py --num_ranks 32 --max_dispersal_dist 100.0
"""

import argparse
import csv
import math
import os
import sys
import time
from collections import defaultdict

# Import METIS partitioning (Python library is buggy, skip it)
HAS_METIS_PYTHON = False
# try:
#     import metis
#     HAS_METIS_PYTHON = True
# except ImportError:
#     HAS_METIS_PYTHON = False

# Import subprocess for calling METIS binary (fallback)
import subprocess
import tempfile

def check_metis_binary():
    """Check if gpmetis binary is available."""
    try:
        result = subprocess.run(['which', 'gpmetis'], capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False

HAS_METIS_BINARY = check_metis_binary()

if not HAS_METIS_PYTHON and not HAS_METIS_BINARY:
    print("ERROR: Neither metis Python library nor gpmetis binary found")
    print("  Load METIS module: module load metis/5.1.0")
    sys.exit(1)

# Earth radius for haversine distance
EARTH_RADIUS_KM = 6371.0


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate great circle distance between two points on Earth.

    Args:
        lat1, lon1: Latitude and longitude of point 1 (degrees)
        lat2, lon2: Latitude and longitude of point 2 (degrees)

    Returns:
        Distance in kilometers
    """
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (math.sin(dlat / 2.0) ** 2 +
         math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    dist = EARTH_RADIUS_KM * c

    return dist


def load_site_locations(data_dir, prefix):
    """
    Load site locations from site CSV.

    Returns:
        dict: {site_id: {'latitude': lat, 'longitude': lon}}
    """
    site_file = os.path.join(data_dir, f"{prefix}_site.csv")
    sites = {}

    with open(site_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            site_id = int(row['site'])
            sites[site_id] = {
                'latitude': float(row['latitude']),
                'longitude': float(row['longitude']),
            }

    return sites


def load_rangelist(data_dir, prefix):
    """
    Load rangelist data: which species exist at each site.

    Returns:
        tuple: (rangelist_dict, num_species)
            rangelist_dict: {site_id: [species_indices where rangelist==1]}
            num_species: total number of species columns
    """
    rangelist_file = os.path.join(data_dir, f"{prefix}_rangelist.csv")
    rangelist = {}

    with open(rangelist_file, 'r') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames

        # Headers: site, latitude, longitude, <species_names>
        # Skip non-species columns
        skip_cols = {'site', 'latitude', 'longitude'}
        species_columns = [h for h in headers if h not in skip_cols]
        num_species = len(species_columns)

        for row in reader:
            site_id = int(row['site'])
            present_species = []

            for sp_idx, sp_col in enumerate(species_columns):
                if int(row[sp_col]) == 1:
                    present_species.append(sp_idx)

            rangelist[site_id] = present_species

    return rangelist, num_species


def load_species_dispersal(data_dir, prefix, override_dist=None):
    """
    Load dispersal distances for all species.

    Priority:
    1. If override_dist provided: use for all species
    2. If 'max_dispersal_dist' column exists in CSV: read values
    3. Else: use default 10.0 km

    Returns:
        dict: {species_index: dispersal_distance_km}
    """
    species_file = os.path.join(data_dir, f"{prefix}_specieslist.csv")

    # Count species first
    with open(species_file, 'r') as f:
        reader = csv.DictReader(f)
        num_species = sum(1 for _ in reader)

    # If override provided, use it for all species
    if override_dist is not None:
        return {sp_idx: override_dist for sp_idx in range(num_species)}

    # Check if max_dispersal_dist column exists
    with open(species_file, 'r') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames

        if 'max_dispersal_dist' in headers:
            # Read dispersal distances from CSV
            dispersal = {}
            for sp_idx, row in enumerate(reader):
                if row['max_dispersal_dist']:
                    dispersal[sp_idx] = float(row['max_dispersal_dist'])
                else:
                    dispersal[sp_idx] = 10.0  # Default if missing
            return dispersal
        else:
            # Column doesn't exist yet, use default
            return {sp_idx: 10.0 for sp_idx in range(num_species)}


def compute_site_max_dispersal(all_site_ids, rangelist, species_dispersal):
    """
    Compute maximum dispersal distance for each site based on species present.

    Returns:
        dict: {site_id: max_dispersal_km}
    """
    site_max_disp = {}

    for site_id in all_site_ids:
        species_at_site = rangelist[site_id]

        if len(species_at_site) > 0:
            max_d = max(species_dispersal[sp_idx] for sp_idx in species_at_site)
            site_max_disp[site_id] = max_d
        else:
            site_max_disp[site_id] = 0.0  # No species = no dispersal

    return site_max_disp


def build_directed_dispersal_graph(site_locations, all_site_ids, site_max_dispersal, dispersal_factor):
    """
    Build directed dispersal graph.

    For each site A:
        For each site B != A:
            if distance(A,B) <= dispersal_factor × max_dispersal[A]:
                Site A can disperse TO Site B
                Site B needs to READ Site A's data
                Add directed edge: (reader=B, source=A, distance)

    Returns:
        list: directed_edges as [(reader_site, source_site, distance), ...]
    """
    directed_edges = []

    print(f"  Checking {len(all_site_ids) * (len(all_site_ids) - 1)} directed pairs...")

    for site_a in all_site_ids:
        cutoff_a = dispersal_factor * site_max_dispersal[site_a]

        if cutoff_a == 0.0:
            continue  # No species, skip

        loc_a = site_locations[site_a]

        for site_b in all_site_ids:
            if site_a == site_b:
                continue

            loc_b = site_locations[site_b]

            dist = haversine_distance(
                loc_a['latitude'], loc_a['longitude'],
                loc_b['latitude'], loc_b['longitude']
            )

            if dist <= cutoff_a:
                # Site A can disperse TO Site B
                # Site B reads FROM Site A
                # Directed edge: B → A
                directed_edges.append((site_b, site_a, dist))

    return directed_edges


def make_undirected_adjacency(directed_edges, all_site_ids):
    """
    Convert directed edges to undirected adjacency for METIS.

    Edge exists in undirected graph if ANY directed edge connects the pair.

    Returns:
        dict: {site_id: [neighbor_site_ids]}
    """
    adjacency = {site_id: set() for site_id in all_site_ids}

    for reader, source, dist in directed_edges:
        # Create symmetric edge
        adjacency[reader].add(source)
        adjacency[source].add(reader)

    # Convert sets to lists
    return {k: list(v) for k, v in adjacency.items()}


def write_metis_graph(adjacency, site_ids, filename):
    """
    Write graph in METIS format.

    METIS format:
    Line 1: <num_vertices> <num_edges>
    Line 2+: adjacency list for each vertex (1-indexed)
    """
    num_vertices = len(site_ids)
    num_edges = sum(len(neighbors) for neighbors in adjacency.values()) // 2

    # Create site_id → 1-indexed mapping (METIS uses 1-indexed vertices)
    site_to_idx = {site_id: idx + 1 for idx, site_id in enumerate(site_ids)}

    with open(filename, 'w') as f:
        # Header
        f.write(f"{num_vertices} {num_edges}\n")

        # Adjacency lists (1-indexed)
        for site_id in site_ids:
            neighbors = adjacency[site_id]
            neighbor_indices = [site_to_idx[n] for n in neighbors]
            f.write(" ".join(map(str, sorted(neighbor_indices))) + "\n")


def partition_graph_metis_binary(adjacency, site_ids, num_parts):
    """Partition using gpmetis binary."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.graph', delete=False) as f:
        graph_file = f.name

    write_metis_graph(adjacency, site_ids, graph_file)

    try:
        result = subprocess.run(
            ['gpmetis', graph_file, str(num_parts)],
            capture_output=True, text=True, timeout=300
        )

        if result.returncode != 0:
            raise RuntimeError(f"gpmetis failed: {result.stderr}")

        # Read partition file
        partition_file = f"{graph_file}.part.{num_parts}"
        partition_labels = []
        with open(partition_file, 'r') as f:
            for line in f:
                partition_labels.append(int(line.strip()))

        # Extract edge cut
        edge_cut = 0
        for line in result.stdout.split('\n'):
            if 'Edgecut:' in line:
                edge_cut = int(line.split('Edgecut:')[1].split(',')[0].strip())
                break

        # Clean up
        os.unlink(graph_file)
        os.unlink(partition_file)

        # Build partition map
        partition_map = {site_id: partition_labels[idx] for idx, site_id in enumerate(site_ids)}
        return partition_map, edge_cut

    except Exception as e:
        if os.path.exists(graph_file):
            os.unlink(graph_file)
        partition_file = f"{graph_file}.part.{num_parts}"
        if os.path.exists(partition_file):
            os.unlink(partition_file)
        raise


def partition_graph_metis_python(adjacency, site_ids, num_parts):
    """Partition using Python metis library."""
    site_to_idx = {site_id: idx for idx, site_id in enumerate(site_ids)}
    adjacency_indexed = [
        [site_to_idx[neighbor] for neighbor in adjacency[site_id]]
        for site_id in site_ids
    ]

    edge_cut, partition_labels = metis.part_graph(adjacency_indexed, nparts=num_parts)

    partition_map = {
        site_id: int(partition_labels[site_to_idx[site_id]])
        for site_id in site_ids
    }
    return partition_map, edge_cut


def partition_graph_metis(adjacency, site_ids, num_parts):
    """
    Partition undirected graph using METIS (tries Python, falls back to binary).

    Returns:
        tuple: (partition_map, edge_cut)
    """
    # Try Python library first
    if HAS_METIS_PYTHON:
        try:
            print(f"  Using Python metis library...")
            return partition_graph_metis_python(adjacency, site_ids, num_parts)
        except Exception as e:
            print(f"  Python metis failed ({type(e).__name__}), trying binary...")

    # Fall back to binary
    if HAS_METIS_BINARY:
        print(f"  Using gpmetis binary...")
        return partition_graph_metis_binary(adjacency, site_ids, num_parts)

    raise RuntimeError("No working METIS implementation available")


def analyze_partition_quality(directed_edges, undirected_adjacency, partition_map, num_parts):
    """
    Analyze partition quality for both directed and undirected graphs.

    Returns:
        dict: statistics about partition quality
    """
    # Sites per partition
    sites_per_part = defaultdict(int)
    for site_id, part in partition_map.items():
        sites_per_part[part] += 1

    # Directed edge cuts
    directed_cross_partition = 0
    for reader, source, dist in directed_edges:
        if partition_map[reader] != partition_map[source]:
            directed_cross_partition += 1

    # Undirected edge cuts
    undirected_edges = set()
    for site_id, neighbors in undirected_adjacency.items():
        for neighbor in neighbors:
            if site_id < neighbor:  # Count each edge once
                undirected_edges.add((site_id, neighbor))

    undirected_cross_partition = 0
    for site_i, site_j in undirected_edges:
        if partition_map[site_i] != partition_map[site_j]:
            undirected_cross_partition += 1

    # Out-degree and in-degree per site
    out_degree = defaultdict(int)
    in_degree = defaultdict(int)
    for reader, source, dist in directed_edges:
        out_degree[source] += 1  # source disperses TO reader
        in_degree[reader] += 1   # reader receives FROM source

    return {
        'sites_per_part': dict(sites_per_part),
        'directed_edges': len(directed_edges),
        'undirected_edges': len(undirected_edges),
        'directed_cross_partition': directed_cross_partition,
        'undirected_cross_partition': undirected_cross_partition,
        'out_degree': dict(out_degree),
        'in_degree': dict(in_degree),
    }


def write_outputs(partition_map, directed_edges, site_max_dispersal, site_locations, rangelist, stats, results_dir):
    """Write output CSV files."""
    os.makedirs(results_dir, exist_ok=True)

    # 1. partition.csv
    partition_file = os.path.join(results_dir, "partition.csv")
    with open(partition_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['site_id', 'partition', 'latitude', 'longitude',
                        'num_out_edges', 'num_in_edges', 'max_dispersal_km'])

        for site_id in sorted(partition_map.keys()):
            loc = site_locations[site_id]
            part = partition_map[site_id]
            out_deg = stats['out_degree'].get(site_id, 0)
            in_deg = stats['in_degree'].get(site_id, 0)
            max_disp = site_max_dispersal[site_id]

            writer.writerow([
                site_id, part,
                f"{loc['latitude']:.2f}", f"{loc['longitude']:.2f}",
                out_deg, in_deg, f"{max_disp:.2f}"
            ])

    print(f"  {partition_file}")

    # 2. edges_directed.csv
    edges_file = os.path.join(results_dir, "edges_directed.csv")
    with open(edges_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['reader_site', 'source_site', 'distance_km',
                        'cross_partition', 'source_max_dispersal'])

        for reader, source, dist in directed_edges:
            cross = int(partition_map[reader] != partition_map[source])
            source_max_disp = site_max_dispersal[source]

            writer.writerow([
                reader, source, f"{dist:.2f}", cross, f"{source_max_disp:.2f}"
            ])

    print(f"  {edges_file}")

    # 3. site_dispersal.csv
    dispersal_file = os.path.join(results_dir, "site_dispersal.csv")
    with open(dispersal_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['site_id', 'num_species', 'max_dispersal_km',
                        'out_degree', 'in_degree'])

        for site_id in sorted(partition_map.keys()):
            num_species = len(rangelist[site_id])
            max_disp = site_max_dispersal[site_id]
            out_deg = stats['out_degree'].get(site_id, 0)
            in_deg = stats['in_degree'].get(site_id, 0)

            writer.writerow([
                site_id, num_species, f"{max_disp:.2f}", out_deg, in_deg
            ])

    print(f"  {dispersal_file}")


def print_statistics(stats, site_max_dispersal, num_parts):
    """Print partition statistics to console."""
    print("\nPartition Statistics:")
    print("  Sites per partition:")
    for part in range(num_parts):
        count = stats['sites_per_part'].get(part, 0)
        pct = 100.0 * count / sum(stats['sites_per_part'].values())
        print(f"    Rank {part}: {count} ({pct:.1f}%)")

    min_sites = min(stats['sites_per_part'].values())
    max_sites = max(stats['sites_per_part'].values())
    imbalance = 100.0 * (max_sites - min_sites) / max_sites if max_sites > 0 else 0
    print(f"  Load balance: min={min_sites}, max={max_sites}, imbalance={imbalance:.1f}%")

    print(f"\nEdge Statistics:")
    print(f"  Directed edges: {stats['directed_edges']}")
    print(f"  Undirected edges: {stats['undirected_edges']}")
    print(f"  Directed cross-partition: {stats['directed_cross_partition']} "
          f"({100.0 * stats['directed_cross_partition'] / stats['directed_edges']:.1f}%)")
    print(f"  Undirected cross-partition: {stats['undirected_cross_partition']} "
          f"({100.0 * stats['undirected_cross_partition'] / stats['undirected_edges']:.1f}%)")

    if stats['out_degree']:
        avg_out = sum(stats['out_degree'].values()) / len(site_max_dispersal)
        avg_in = sum(stats['in_degree'].values()) / len(site_max_dispersal)
        print(f"  Avg out-degree: {avg_out:.1f} sites (can disperse to)")
        print(f"  Avg in-degree: {avg_in:.1f} sites (can receive from)")


def main():
    parser = argparse.ArgumentParser(
        description="Build CONUS directed dispersal network and METIS partitioning"
    )
    parser.add_argument(
        "--num_ranks",
        type=int,
        default=4,
        help="Number of partitions (default: 4)"
    )
    parser.add_argument(
        "--dispersal_factor",
        type=float,
        default=5.0,
        help="Cutoff multiplier for dispersal distance (default: 5.0)"
    )
    parser.add_argument(
        "--max_dispersal_dist",
        type=float,
        default=None,
        help="Override max dispersal distance in km (default: None = per-site from species)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../../input_data",
        help="Input data directory (default: ../../input_data)"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="CONUS",
        help="File prefix (default: CONUS)"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="../results",
        help="Output directory (default: ../results)"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("CONUS Directed Dispersal Network and METIS Partitioning")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Prefix: {args.prefix}")
    print(f"  Partitions: {args.num_ranks}")
    print(f"  Dispersal factor: {args.dispersal_factor}")
    if args.max_dispersal_dist is not None:
        print(f"  Max dispersal override: {args.max_dispersal_dist} km")

    t_total_start = time.time()

    # Phase 1: Load data
    print("\nPhase 1: Loading Data")
    print("-" * 80)

    t_start = time.time()
    site_locations = load_site_locations(args.data_dir, args.prefix)
    all_site_ids = sorted(site_locations.keys())
    print(f"  Sites: {len(all_site_ids)} from {args.prefix}_site.csv")

    rangelist, num_species = load_rangelist(args.data_dir, args.prefix)
    print(f"  Species: {num_species} from {args.prefix}_rangelist.csv")
    print(f"  Rangelist: {len(rangelist)} sites × {num_species} species")

    species_dispersal = load_species_dispersal(args.data_dir, args.prefix, args.max_dispersal_dist)

    print(f"\nComputing per-site max dispersal (rangelist-based)...")
    site_max_dispersal = compute_site_max_dispersal(all_site_ids, rangelist, species_dispersal)

    max_vals = [d for d in site_max_dispersal.values() if d > 0]
    if max_vals:
        print(f"  Min: {min(max_vals):.1f} km")
        print(f"  Max: {max(max_vals):.1f} km")
        print(f"  Mean: {sum(max_vals) / len(max_vals):.1f} km")

    t_load = time.time() - t_start

    # Phase 2: Build directed graph
    print(f"\nPhase 2: Building Directed Dispersal Graph")
    print("-" * 80)
    print(f"  Dispersal factor: {args.dispersal_factor}")

    t_start = time.time()
    directed_edges = build_directed_dispersal_graph(
        site_locations, all_site_ids, site_max_dispersal, args.dispersal_factor
    )
    t_graph = time.time() - t_start

    print(f"  Directed edges: {len(directed_edges)}")
    print(f"  Graph built in {t_graph:.2f}s")

    # Phase 3: Convert to undirected
    print(f"\nPhase 3: Converting to Undirected (for METIS)")
    print("-" * 80)

    undirected_adjacency = make_undirected_adjacency(directed_edges, all_site_ids)
    num_undirected = sum(len(neighbors) for neighbors in undirected_adjacency.values()) // 2
    print(f"  Undirected edges: {num_undirected}")

    # Phase 4: METIS partitioning
    print(f"\nPhase 4: METIS Graph Partitioning")
    print("-" * 80)
    print(f"  Partitioning into {args.num_ranks} parts...")

    t_start = time.time()
    partition_map, metis_edge_cut = partition_graph_metis(
        undirected_adjacency, all_site_ids, args.num_ranks
    )
    t_partition = time.time() - t_start

    print(f"  METIS edge cut (undirected): {metis_edge_cut}")
    print(f"  Partitioning completed in {t_partition:.2f}s")

    # Phase 5: Analyze partition quality
    print(f"\nPhase 5: Analyzing Partition Quality")
    print("-" * 80)

    stats = analyze_partition_quality(
        directed_edges, undirected_adjacency, partition_map, args.num_ranks
    )

    print_statistics(stats, site_max_dispersal, args.num_ranks)

    # Write outputs
    print(f"\nWriting Output Files:")
    print("-" * 80)

    write_outputs(
        partition_map, directed_edges, site_max_dispersal,
        site_locations, rangelist, stats, args.results_dir
    )

    # Summary
    t_total = time.time() - t_total_start
    print(f"\n" + "=" * 80)
    print(f"Total time: {t_total:.2f}s")
    print(f"  Load data: {t_load:.2f}s")
    print(f"  Build graph: {t_graph:.2f}s")
    print(f"  METIS partition: {t_partition:.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
