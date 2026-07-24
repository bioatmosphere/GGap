#!/usr/bin/env python3
"""
TN-region GGap example — the one command an artifact reviewer runs.

Reproduces the Tennessee downscaled forest-gap simulation with all parameters
already set to their correct defaults, so a bare

    python run.py                 # 1 GPU
    mpirun -n 2 python run.py     # 2 GPUs  (rank i binds to GPU i automatically)

reproduces the run. Every parameter is still overridable on the command line.

Number of GPUs == number of MPI ranks == the `-n` you pass to mpirun. This script
self-binds each rank to its own GPU (CUDA_VISIBLE_DEVICES = MPI local rank), so no
launcher wrapper is needed. After it finishes, run `python extract.py` to turn the
snapshots into per-site species_data.csv files.
"""
import os
import sys

# ---------------------------------------------------------------------------
# 1. Bind THIS rank to its own GPU *before* importing anything that loads CUDA.
#    mpirun sets the local-rank env var per process before Python starts, so we
#    can read it here without importing mpi4py yet. rank i -> GPU i.
# ---------------------------------------------------------------------------
_local_rank = (
    os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK")
    or os.environ.get("MPI_LOCALRANKID")
    or os.environ.get("PMI_LOCAL_RANK")
    or os.environ.get("SLURM_LOCALID")
    or "0"
)
# Only set it if the user (or a launcher wrapper) hasn't already pinned devices.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", _local_rank)

# ---------------------------------------------------------------------------
# 2. Make the GGap `gap` package importable from a clean clone, with no
#    `pip install -e` required:  tn_example -> SC2026 -> <GGap repo root>.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))  # GGap root (has `gap/`)
sys.path.insert(0, _HERE)                                             # so `import conus_engine` works

# ---------------------------------------------------------------------------
# 3. Reuse the proven CONUS engine verbatim (loading, partitioning, sim, I/O).
#    Importing it also initializes MPI, so E.num_workers == mpirun -n count.
# ---------------------------------------------------------------------------
import argparse
import conus_engine as E


# Tennessee region bounding box (lat 34.9-36.7, lon -90.5 to -81.5) -> 54 CONUS sites.
TN_DEFAULTS = dict(
    min_lat=34.9, max_lat=36.7, min_lon=-90.5, max_lon=-81.5,
    num_gaps=500, maxtrees=1000, years=1000, report_interval=50,
    dispersal_factor=2.0, seed=42,
)


def main():
    p = argparse.ArgumentParser(
        description="TN-region GGap reproduction (defaults reproduce the reference run)."
    )
    p.add_argument("--min_lat", type=float, default=TN_DEFAULTS["min_lat"])
    p.add_argument("--max_lat", type=float, default=TN_DEFAULTS["max_lat"])
    p.add_argument("--min_lon", type=float, default=TN_DEFAULTS["min_lon"])
    p.add_argument("--max_lon", type=float, default=TN_DEFAULTS["max_lon"])
    p.add_argument("--num_gaps", type=int, default=TN_DEFAULTS["num_gaps"],
                   help="Gaps per site (default: 500)")
    p.add_argument("--maxtrees", type=int, default=TN_DEFAULTS["maxtrees"],
                   help="Max trees per gap (default: 1000)")
    p.add_argument("--years", type=int, default=TN_DEFAULTS["years"],
                   help="Simulation duration in years (default: 1000)")
    p.add_argument("--report_interval", type=int, default=TN_DEFAULTS["report_interval"],
                   help="Years between snapshots (default: 50)")
    p.add_argument("--dispersal_factor", type=float, default=TN_DEFAULTS["dispersal_factor"],
                   help="Dispersal cutoff multiplier (default: 2.0)")
    p.add_argument("--seed", type=int, default=TN_DEFAULTS["seed"],
                   help="RNG seed, identical on all ranks -> reproducible (default: 42)")
    p.add_argument("--data_dir", type=str,
                   default=os.path.abspath(os.path.join(_HERE, "..", "conus_simulation", "input_data")),
                   help="Directory with the CONUS_*.csv inputs "
                        "(default: the repo's SC2026/conus_simulation/input_data)")
    p.add_argument("--prefix", type=str, default="CONUS")
    p.add_argument("--output_dir", type=str, default=os.path.join(_HERE, "results"),
                   help="Where snapshots + metadata are written (default: ./results)")
    p.add_argument("--no_tree_data", action="store_true")
    p.add_argument("--no_snapshots", action="store_true",
                   help="Pure-timing mode: run without writing snapshots (no results to extract)")
    args = p.parse_args()

    n = E.num_workers  # == mpirun -n == number of GPUs in use
    if E.rank == 0:
        E.log("=" * 80)
        E.log(f"TN-region GGap example — using {n} rank(s) / {n} GPU(s)")
        E.log(f"  region lat[{args.min_lat},{args.max_lat}] lon[{args.min_lon},{args.max_lon}]")
        E.log(f"  {args.num_gaps} gaps x {args.maxtrees} trees   years {args.years}   seed {args.seed}")
        E.log(f"  output -> {args.output_dir}")
        if n == 1:
            E.log("  (1 GPU: partitioning is skipped; all sites on one device. For 2 GPUs: "
                  "`mpirun -n 2 python run.py`.)")
        E.log("=" * 80)

    # --- The pipeline (same order as the CONUS driver's main()) ---
    partition_map, directed_edges, site_locations = E.build_network_and_partition(
        data_dir=args.data_dir,
        prefix=args.prefix,
        dispersal_factor=args.dispersal_factor,
        num_parts=n,                 # partitions == MPI ranks == GPUs
        min_lat=args.min_lat,
        max_lat=args.max_lat,
        min_lon=args.min_lon,
        max_lon=args.max_lon,
        site_ids=None,
    )

    model = E.GAPModel()
    # Fix the RNG seed identically on every rank. SAGESim otherwise picks a random
    # per-process seed; the RNG is partition-invariant, so a shared fixed seed makes
    # 1-GPU and N-GPU runs produce identical results.
    model.set_seed(args.seed)
    if E.rank == 0:
        E.log(f"  RNG seed set to {args.seed} (identical on all ranks)")

    local_sites = E.initialize_simulation(
        model=model,
        partition_map=partition_map,
        directed_edges=directed_edges,
        site_locations=site_locations,
        num_gaps=args.num_gaps,
        maxtrees=args.maxtrees,
        data_dir=args.data_dir,
        prefix=args.prefix,
    )

    E.run_simulation(
        model=model,
        local_sites=local_sites,
        years=args.years,
        report_interval=args.report_interval,
        output_dir=args.output_dir,
        no_tree_data=args.no_tree_data,
        no_snapshots=args.no_snapshots,
    )

    if E.rank == 0 and not args.no_snapshots:
        E.log("")
        E.log("Simulation done. Next: `python extract.py` -> results/species/site_NNNN/species_data.csv")


if __name__ == "__main__":
    main()
