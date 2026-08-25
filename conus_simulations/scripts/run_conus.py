"""
Integrated CONUS Forest Simulation with Network Partitioning.

This script combines network partitioning and simulation in a single run:
1. Build directed dispersal network graph
2. METIS partitioning to assign sites to MPI ranks
3. Initialize GGap model with sites, gaps, and trees
4. Run simulation for specified years

Usage:
    # On HPC cluster with 20 nodes × 8 GPUs = 160 ranks
    salloc -N 20 -A PROJECT_ID -t 2:00:00
    srun -n 160 python run_conus.py --num_gaps 500 --maxtrees 1000 --years 1000

    # Test run with fewer ranks
    srun -n 16 python run_conus.py --num_gaps 10 --maxtrees 100 --years 50 --test
"""

import argparse
import csv
import math
import os
import sys
import time
import subprocess
import tempfile
import pickle
from collections import defaultdict

# Add parent directory to path for gap imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from gap.gap_model import GAPModel
from gap.output_utils import OutputWriter
from gap.constants import SiteP, SiteS, TreeP, TreeS
import numpy as np

# Try to import MPI
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_workers = comm.Get_size()
    HAS_MPI = True
except ImportError:
    rank = 0
    num_workers = 1
    HAS_MPI = False
    print("WARNING: mpi4py not available, running in single-process mode")

# Earth radius for haversine distance
EARTH_RADIUS_KM = 6371.0


def log(message, force=False):
    """Print message from rank 0 only (or force from all ranks)."""
    if rank == 0 or force:
        print(f"[Rank {rank}] {message}", flush=True)


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate great circle distance between two points (km)."""
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
    """Load site locations from CONUS_site.csv."""
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
    """Load rangelist: which species exist at each site."""
    rangelist_file = os.path.join(data_dir, f"{prefix}_rangelist.csv")
    rangelist = {}

    with open(rangelist_file, 'r') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames

        skip_cols = {'site', 'latitude', 'longitude'}
        species_columns = [h for h in headers if h not in skip_cols]

        for row in reader:
            site_id = int(row['site'])
            present_species = []

            for sp_idx, sp_col in enumerate(species_columns):
                if int(row[sp_col]) == 1:
                    present_species.append(sp_idx)

            rangelist[site_id] = present_species

    return rangelist, len(species_columns)


def load_species_dispersal(data_dir, prefix):
    """Load max dispersal distance for each species."""
    species_file = os.path.join(data_dir, f"{prefix}_specieslist.csv")

    with open(species_file, 'r') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames

        if 'max_dispersal_dist' in headers:
            dispersal = {}
            for sp_idx, row in enumerate(reader):
                if row['max_dispersal_dist']:
                    dispersal[sp_idx] = float(row['max_dispersal_dist'])
                else:
                    dispersal[sp_idx] = 10.0
            return dispersal
        else:
            # Column doesn't exist, use default
            with open(species_file, 'r') as f2:
                num_species = sum(1 for _ in csv.DictReader(f2))
            return {sp_idx: 10.0 for sp_idx in range(num_species)}


def compute_site_max_dispersal(all_site_ids, rangelist, species_dispersal):
    """Compute max dispersal for each site based on species present."""
    site_max_disp = {}

    for site_id in all_site_ids:
        species_at_site = rangelist[site_id]

        if len(species_at_site) > 0:
            max_d = max(species_dispersal[sp_idx] for sp_idx in species_at_site)
            site_max_disp[site_id] = max_d
        else:
            site_max_disp[site_id] = 0.0

    return site_max_disp


def build_directed_dispersal_graph(site_locations, all_site_ids, site_max_dispersal, dispersal_factor):
    """Build directed dispersal graph."""
    directed_edges = []

    for site_a in all_site_ids:
        cutoff_a = dispersal_factor * site_max_dispersal[site_a]

        if cutoff_a == 0.0:
            continue

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
                directed_edges.append((site_b, site_a, dist))

    return directed_edges


def make_undirected_adjacency(directed_edges, all_site_ids):
    """Convert directed edges to undirected adjacency for METIS."""
    adjacency = {site_id: set() for site_id in all_site_ids}

    for reader, source, dist in directed_edges:
        adjacency[reader].add(source)
        adjacency[source].add(reader)

    return {k: list(v) for k, v in adjacency.items()}


def write_metis_graph(adjacency, site_ids, filename):
    """Write graph in METIS format."""
    num_vertices = len(site_ids)
    num_edges = sum(len(neighbors) for neighbors in adjacency.values()) // 2

    site_to_idx = {site_id: idx + 1 for idx, site_id in enumerate(site_ids)}

    with open(filename, 'w') as f:
        f.write(f"{num_vertices} {num_edges}\n")

        for site_id in site_ids:
            neighbors = adjacency[site_id]
            neighbor_indices = [site_to_idx[n] for n in neighbors]
            f.write(" ".join(map(str, sorted(neighbor_indices))) + "\n")


def partition_graph_metis(adjacency, site_ids, num_parts):
    """Partition graph using gpmetis binary."""
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

        # Clean up
        os.unlink(graph_file)
        os.unlink(partition_file)

        # Build partition map
        partition_map = {site_id: partition_labels[idx] for idx, site_id in enumerate(site_ids)}
        return partition_map

    except Exception as e:
        if os.path.exists(graph_file):
            os.unlink(graph_file)
        partition_file = f"{graph_file}.part.{num_parts}"
        if os.path.exists(partition_file):
            os.unlink(partition_file)
        raise


def build_network_and_partition(data_dir, prefix, dispersal_factor, num_parts):
    """
    Build dispersal network and partition sites.

    Returns:
        tuple: (partition_map, directed_edges, site_locations)
    """
    log("=" * 80)
    log("Phase 1: Loading CONUS Data")
    log("-" * 80)

    t_start = time.time()

    site_locations = load_site_locations(data_dir, prefix)
    all_site_ids = sorted(site_locations.keys())
    log(f"  Loaded {len(all_site_ids)} sites from {prefix}_site.csv")

    rangelist, num_species = load_rangelist(data_dir, prefix)
    log(f"  Loaded {num_species} species from {prefix}_rangelist.csv")

    species_dispersal = load_species_dispersal(data_dir, prefix)
    site_max_dispersal = compute_site_max_dispersal(all_site_ids, rangelist, species_dispersal)

    t_load = time.time() - t_start
    log(f"  Data loading completed in {t_load:.2f}s")

    # Phase 2: Build dispersal graph
    log("\nPhase 2: Building Directed Dispersal Graph")
    log("-" * 80)
    log(f"  Dispersal factor: {dispersal_factor}")

    t_start = time.time()
    directed_edges = build_directed_dispersal_graph(
        site_locations, all_site_ids, site_max_dispersal, dispersal_factor
    )
    t_graph = time.time() - t_start

    log(f"  Directed edges: {len(directed_edges)}")
    log(f"  Graph built in {t_graph:.2f}s")

    # Phase 3: METIS partitioning
    log("\nPhase 3: METIS Graph Partitioning")
    log("-" * 80)

    undirected_adjacency = make_undirected_adjacency(directed_edges, all_site_ids)
    num_undirected = sum(len(neighbors) for neighbors in undirected_adjacency.values()) // 2
    log(f"  Undirected edges: {num_undirected}")

    t_start = time.time()
    partition_map = partition_graph_metis(undirected_adjacency, all_site_ids, num_parts)
    t_partition = time.time() - t_start

    log(f"  Partitioned {len(all_site_ids)} sites into {num_parts} partitions")
    log(f"  Partitioning completed in {t_partition:.2f}s")

    # Verify partition balance
    partition_counts = defaultdict(int)
    for site_id, part in partition_map.items():
        partition_counts[part] += 1

    min_sites = min(partition_counts.values())
    max_sites = max(partition_counts.values())
    avg_sites = sum(partition_counts.values()) / len(partition_counts)
    log(f"  Sites per partition: min={min_sites}, max={max_sites}, avg={avg_sites:.1f}")

    return partition_map, directed_edges, site_locations


def initialize_simulation(model, partition_map, directed_edges, site_locations,
                         num_gaps, maxtrees, data_dir, prefix):
    """
    Initialize GGap simulation with partitioned sites.

    Args:
        model: GAPModel instance
        partition_map: dict {site_id: partition_id}
        directed_edges: list of (reader, source, distance) tuples
        site_locations: dict {site_id: {'latitude': ..., 'longitude': ...}}
        num_gaps: gaps per site
        maxtrees: max trees per gap
        data_dir: input data directory
        prefix: data file prefix

    Returns:
        local_sites: list of site dicts for sites on this rank
    """
    log("\n" + "=" * 80)
    log("Phase 4: Initializing Simulation")
    log("-" * 80)

    t_start = time.time()

    # Load globals (species traits, site configs) - must be identical on all ranks
    log("  Loading global species and site data...")
    model.load_globals(data_dir=data_dir, prefix=prefix)

    # Apply METIS partition map
    all_site_ids = sorted(partition_map.keys())
    model._site_partition = partition_map

    # Determine local sites for this rank
    local_site_ids = [site_id for site_id in all_site_ids if partition_map[site_id] == rank]
    log(f"  Rank {rank}: assigned {len(local_site_ids)} sites", force=True)

    # Initialize local sites and collect site metadata
    log(f"  Rank {rank}: initializing sites with {num_gaps} gaps, {maxtrees} maxtrees each...", force=True)

    local_sites = []
    site_to_agent = {}
    for site_id in local_site_ids:
        site = model.initialize_site_with_gaps(
            site_id=site_id,
            num_gaps=num_gaps,
            maxtrees=maxtrees,
            data_dir=data_dir,
            prefix=prefix
        )
        local_sites.append(site)
        site_to_agent[site_id] = site['site_agent_id']

    t_init = time.time() - t_start
    log(f"  Rank {rank}: site initialization completed in {t_init:.2f}s", force=True)

    # Connect sites for dispersal
    log("  Connecting sites for seed dispersal...")
    t_start = time.time()

    # directed_edges holds SITE ids, but connect_agents takes AGENT ids, so map
    # through site_to_agent. Agent ids here are per-rank creation order, so a
    # source living on another rank has no id nameable from here; count those
    # rather than passing a site id through and silently wiring a wrong agent.
    n_connected = 0
    n_remote_skipped = 0
    for reader_site, source_site, distance in directed_edges:
        if reader_site not in site_to_agent:
            continue  # this rank only owns edges whose reader lives here
        if source_site not in site_to_agent:
            n_remote_skipped += 1
            continue
        model.connect_agents(site_to_agent[reader_site],
                             site_to_agent[source_site], directed=True)
        n_connected += 1

    t_connect = time.time() - t_start
    log(f"  Rank {rank}: site connections completed in {t_connect:.2f}s "
        f"({n_connected} wired, {n_remote_skipped} cross-rank skipped)", force=True)

    # Register local arrays and setup GPU
    log("  Setting up GPU kernels...")
    t_start = time.time()

    model.register_breed_local_arrays()
    model.setup()

    t_setup = time.time() - t_start
    log(f"  GPU setup completed in {t_setup:.2f}s")

    # Summary
    total_sites = len(all_site_ids)
    total_gaps = total_sites * num_gaps

    log("\n" + "-" * 80)
    log("Initialization Summary:")
    log(f"  Total sites: {total_sites}")
    log(f"  Total gaps: {total_gaps:,}")
    log(f"  Sites per rank (avg): {total_sites / num_workers:.1f}")
    log(f"  MPI ranks: {num_workers}")

    return local_sites


def collect_local_site_data(model, local_sites):
    """
    Collect data for sites on THIS rank only (no MPI gather).

    Returns:
        (results, timings) where results is a list of (site_params, site_states, tree_data, gap_ids)
        and timings is a dict with per-breed GPU→CPU transfer times.
    """
    timings = {}

    # GPU→CPU transfer: Site params
    t0 = time.time()
    all_site_params = model.get_breed_data("Site", "params")
    timings['site_p'] = time.time() - t0

    # GPU→CPU transfer: Site states
    t0 = time.time()
    all_site_states = model.get_breed_data("Site", "states")
    timings['site_s'] = time.time() - t0

    # GPU→CPU transfer: Tree params
    t0 = time.time()
    all_tree_params = model.get_breed_data("Tree", "params")
    timings['tree_p'] = time.time() - t0

    # GPU→CPU transfer: Tree states
    t0 = time.time()
    all_tree_states = model.get_breed_data("Tree", "states")
    timings['tree_s'] = time.time() - t0

    # GPU→CPU transfer: Tree IDs
    t0 = time.time()
    all_tree_ids = model.get_breed_agent_ids("Tree")
    timings['tree_id'] = time.time() - t0

    if all_site_params is None:
        return None, timings

    results = []
    for site_idx, site in enumerate(local_sites):
        # Site data (one row per site agent in breed order)
        site_params = all_site_params[site_idx]
        site_states = all_site_states[site_idx]

        # Filter trees belonging to this site's gaps
        site_gap_ids = set(site['gaps'])
        tree_mask = np.array([
            model.tree_to_gap.get(int(tid), -1) in site_gap_ids
            for tid in all_tree_ids
        ], dtype=bool)

        site_tree_params = all_tree_params[tree_mask]
        site_tree_states = all_tree_states[tree_mask]
        site_tree_ids = all_tree_ids[tree_mask]

        # Filter to living trees
        alive_mask = site_tree_states[:, TreeS.IS_ALIVE] > 0.5
        alive_params = site_tree_params[alive_mask]
        alive_states = site_tree_states[alive_mask]
        alive_ids = site_tree_ids[alive_mask].astype(np.int32)

        gap_ids = np.array([model.tree_to_gap[int(a)] for a in alive_ids], dtype=np.int32)
        species_ids = alive_states[:, TreeS.SPECIES_ID].astype(np.int32)
        evergreen = np.array([
            model.species_by_id.get(int(sid), {}).get('evergreen', 0) > 0.5
            for sid in species_ids
        ], dtype=bool)

        tree_data = {
            'count': int(alive_mask.sum()),
            'gap_agent_id': gap_ids,
            'species_id': species_ids,
            'diam': alive_states[:, TreeS.DIAM],
            'height': alive_states[:, TreeS.HEIGHT],
            'biomC': alive_params[:, TreeP.BIOMC],
            'biomN': alive_params[:, TreeP.BIOMN],
            'leaf_bm': alive_params[:, TreeP.LEAF_BM],
            'age': alive_params[:, TreeP.AGE],
            'canopy_ht': alive_states[:, TreeS.CANOPY_HT],
            'evergreen': evergreen,
        }

        results.append((site_params, site_states, tree_data, list(site_gap_ids)))

    return results, timings


def run_simulation(model, local_sites, years, report_interval, output_dir, no_tree_data):
    """
    Run the simulation with periodic raw data snapshots.

    Saves raw GPU data as .npz files to: {output_dir}/snapshots/year_XXXX_rank_XXX.npz
    Post-processing to generate CSVs can be done offline after simulation completes.
    """
    log("\n" + "=" * 80)
    log(f"Phase 5: Running Simulation for {years} Years")
    log("-" * 80)

    # Create snapshot directory
    snapshot_dir = os.path.join(output_dir, "snapshots")
    os.makedirs(snapshot_dir, exist_ok=True)

    # Save local_sites metadata for post-processing
    metadata_file = os.path.join(output_dir, f"rank_{rank:03d}_sites.pkl")
    with open(metadata_file, 'wb') as f:
        pickle.dump({
            'local_sites': local_sites,
            'species_by_id': model.species_by_id,
            'tree_to_gap': model.tree_to_gap,
            'tree_ids': model.tree_ids,  # Agent ID order (matches array indices)
        }, f)

    if rank == 0:
        log(f"\nSnapshot configuration:")
        log(f"  {len(local_sites)} local sites on rank 0")
        log(f"  Saving raw GPU data to: {snapshot_dir}/")
        log(f"  Metadata saved to: rank_XXX_sites.pkl")
        log(f"  Post-processing: python process_snapshots.py --snapshot_dir {snapshot_dir} --output_dir {output_dir}")
        log("")

    t_total_start = time.time()

    # Run simulation in batches with periodic snapshots
    for year_batch in range(0, years, report_interval):
        years_to_run = min(report_interval, years - year_batch)

        t_batch_start = time.time()

        # Phase 1: Simulation
        model.simulate(ticks=years_to_run, sync_workers_every_n_ticks=1)
        t_sim = time.time()

        current_year = year_batch + years_to_run

        # Phase 2: GPU→CPU transfer (timed per breed) - LOCAL only, no MPI gather
        t_download_start = time.time()

        t0 = time.time()
        site_params = model.get_breed_data("Site", "params", local=True)
        t_site_p = time.time() - t0

        t0 = time.time()
        site_states = model.get_breed_data("Site", "states", local=True)
        t_site_s = time.time() - t0

        t0 = time.time()
        tree_params = model.get_breed_data("Tree", "params", local=True)
        t_tree_p = time.time() - t0

        t0 = time.time()
        tree_states = model.get_breed_data("Tree", "states", local=True)
        t_tree_s = time.time() - t0

        t_download = time.time() - t_download_start

        # Phase 3: Save raw arrays to disk (all ranks save their local data)
        t_save_start = time.time()
        snapshot_file = os.path.join(snapshot_dir, f"year_{current_year:04d}_rank_{rank:03d}.npz")
        np.savez_compressed(
            snapshot_file,
            year=current_year,
            rank=rank,
            site_params=site_params,
            site_states=site_states,
            tree_params=tree_params,
            tree_states=tree_states,
        )
        t_save = time.time() - t_save_start

        # Timing breakdown
        sim_time = t_sim - t_batch_start
        batch_total = time.time() - t_batch_start
        elapsed_total = time.time() - t_total_start

        # Console output (rank 0 only)
        if rank == 0:
            gpu_breakdown = f"[site_p:{t_site_p:.3f}s site_s:{t_site_s:.3f}s tree_p:{t_tree_p:.3f}s tree_s:{t_tree_s:.3f}s]"
            log(f"Year {current_year:>4}/{years}  "
                f"sim:{sim_time:.2f}s  "
                f"gpu→cpu:{gpu_breakdown}  "
                f"save:{t_save:.2f}s  "
                f"({batch_total:.2f}s batch, {elapsed_total:.1f}s total)")

    t_total = time.time() - t_total_start

    log("\n" + "=" * 80)
    log("Simulation Complete")
    log(f"  Total time: {t_total:.2f}s")
    log(f"  Time per year: {t_total / years:.2f}s")
    log(f"  Snapshots saved: {snapshot_dir}/")
    log(f"  To generate CSVs: python process_snapshots.py --snapshot_dir {snapshot_dir}")
    log("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Integrated CONUS forest simulation with network partitioning"
    )

    # Network partitioning parameters
    parser.add_argument("--dispersal_factor", type=float, default=2.0,
                       help="Dispersal cutoff multiplier (default: 2.0 for ~8 neighbors)")

    # Simulation parameters
    parser.add_argument("--num_gaps", type=int, default=500,
                       help="Number of gaps per site (default: 500)")
    parser.add_argument("--maxtrees", type=int, default=1000,
                       help="Max trees per gap (default: 1000)")
    parser.add_argument("--years", type=int, default=1000,
                       help="Simulation duration in years (default: 1000)")
    parser.add_argument("--report_interval", type=int, default=50,
                       help="Years between progress reports (default: 50)")

    # Data parameters
    parser.add_argument("--data_dir", type=str,
                       default="/lustre/orion/proj-shared/lrn088/objective3/xxz/GGap/input_data",
                       help="Input data directory (default: /lustre/.../GGap/input_data)")
    parser.add_argument("--prefix", type=str, default="CONUS",
                       help="Data file prefix (default: CONUS)")
    parser.add_argument("--output_dir", type=str, default="../results/simulation",
                       help="Output directory (default: ../results/simulation)")

    # Output control
    parser.add_argument("--no_tree_data", action="store_true",
                       help="Skip writing tree_data.csv (can be very large)")

    # Test mode
    parser.add_argument("--test", action="store_true",
                       help="Test mode: use smaller subset of sites")

    parser.add_argument("--seed", type=int, default=42,
                       help="RNG seed (same on all ranks -> reproducible & partition-invariant). "
                            "SAGESim defaults to a random per-process seed if unset.")

    args = parser.parse_args()

    # Verify MPI configuration matches expected partitions
    if HAS_MPI and num_workers != 160 and not args.test:
        log(f"WARNING: Expected 160 MPI ranks, got {num_workers}")
        log("  For production run: use -n 160")
        log("  For testing: add --test flag")

    # Build network and partition
    partition_map, directed_edges, site_locations = build_network_and_partition(
        data_dir=args.data_dir,
        prefix=args.prefix,
        dispersal_factor=args.dispersal_factor,
        num_parts=num_workers  # Use actual MPI rank count
    )

    # Initialize GGap model
    model = GAPModel()
    # Fix the RNG seed IDENTICALLY on every rank. SAGESim otherwise picks a random
    # per-process seed, which makes runs irreproducible AND gives each rank a
    # different seed in a multi-rank run. The RNG is keyed on logical_ids
    # (partition-invariant), so a shared fixed seed makes 1-GPU and N-GPU runs
    # produce identical results.
    model.set_seed(args.seed)
    log(f"  RNG seed set to {args.seed} (identical on all ranks)")

    local_sites = initialize_simulation(
        model=model,
        partition_map=partition_map,
        directed_edges=directed_edges,
        site_locations=site_locations,
        num_gaps=args.num_gaps,
        maxtrees=args.maxtrees,
        data_dir=args.data_dir,
        prefix=args.prefix
    )

    # Run simulation
    run_simulation(
        model=model,
        local_sites=local_sites,
        years=args.years,
        report_interval=args.report_interval,
        output_dir=args.output_dir,
        no_tree_data=args.no_tree_data
    )


if __name__ == "__main__":
    main()
