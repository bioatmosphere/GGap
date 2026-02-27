"""
Run script for a single site simulation with GAPpy-compatible CSV output.
Uses Site -> Gap(s) -> Trees agent hierarchy with soil biogeochemistry.

Produces 5 CSV files in output_dir matching GAPpy format:
  - site_data.csv, soil_data.csv, genus_data.csv, species_data.csv, tree_data.csv
"""

import argparse
import sys
import os
import time

# Add parent directory to path for imports
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Try installed SAGESim first, fall back to submodule
try:
    import sagesim  # noqa: F401
except ImportError:
    _sagesim_path = os.path.join(_parent_dir, "SAGESim")
    if _sagesim_path not in sys.path:
        sys.path.insert(0, _sagesim_path)

from mpi4py import MPI
from gap.gap_model import (
    GAPModel,
    # Site params indices (soil pools are private)
    SITE_P_A0_C, SITE_P_A0_N,
    SITE_P_A_C, SITE_P_A_N,
    SITE_P_BL_C, SITE_P_BL_N,
    SITE_P_PRCP_BASE, SITE_P_TMIN_BASE, SITE_P_TMAX_BASE,
    # Site states indices (public: climate + avail_n)
    SITE_S_AVAIL_N, SITE_S_DEG_DAYS, SITE_S_DRY_DAYS, SITE_S_DRY_DAYS_BASE, SITE_S_FLOOD_DAYS,
)
from gap.output_utils import OutputWriter, compute_site_extras, compute_soil_biomass

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_workers = comm.Get_size()

if num_workers > 1:
    if rank == 0:
        print("ERROR: Multi-rank MPI not yet supported (cross-priority data flow")
        print("requires all agents on same rank). Run with: python run_one_site.py")
    comm.Abort(1)


def main():
    parser = argparse.ArgumentParser(
        description="Run single site simulation with soil biogeochemistry"
    )
    parser.add_argument(
        "--num_gaps",
        type=int,
        default=200,
        help="Number of gaps per site (default: 200)"
    )
    parser.add_argument(
        "--maxtrees",
        type=int,
        default=1000,
        help="Max tree slots per gap (default: 1000, matches GAPpy parameters.py:19)"
    )
    parser.add_argument(
        "--years",
        type=int,
        default=1000,
        help="Number of years to simulate (default: 1000)"
    )
    parser.add_argument(
        "--report_interval",
        type=int,
        default=10,
        help="Years between progress reports and CSV output (default: 10)"
    )
    parser.add_argument(
        "--site_id",
        type=int,
        default=0,
        help="Site ID from UVAFME CSV files (default: 0)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="input_data",
        help="Directory containing UVAFME CSV files"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="UVAFME2012",
        help="File prefix for UVAFME CSV files (default: UVAFME2012)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_data",
        help="Directory for CSV output files (default: output_data)"
    )
    parser.add_argument(
        "--no_tree_data",
        action="store_true",
        help="Skip writing tree_data.csv (can be very large)"
    )

    args = parser.parse_args()

    if rank == 0:
        print("=" * 60)
        print("GGap Single Site Simulation (with Soil)")
        print("=" * 60)
        print(f"\nSimulation Parameters:")
        print(f"  Number of gaps: {args.num_gaps}")
        print(f"  Max trees per gap: {args.maxtrees}")
        print(f"  Total slots: {args.num_gaps * args.maxtrees}")
        print(f"  Simulation duration: {args.years} years")
        print(f"  Data directory: {args.data_dir}")
        print(f"  File prefix: {args.prefix}")
        print(f"  Site ID: {args.site_id}")
        print(f"  Output directory: {args.output_dir}")
        print()

    t_total_start = time.time()

    # Create model
    model = GAPModel()

    if rank == 0:
        print("Initializing site from UVAFME CSV files...")

    t_init_start = time.time()

    # Initialize site (loads from UVAFME CSV files)
    site = model.initialize_site(
        site_id=args.site_id,
        data_dir=args.data_dir,
        prefix=args.prefix,
    )

    site_agent_id = site['site_agent_id']
    num_species = len(site['species'])

    if rank == 0:
        print(f"Site: {site['site_name']} ({site['latitude']:.2f}°N, {site['longitude']:.2f}°W)")
        print(f"  Region: {site.get('region', 'N/A')}, Elevation: {site['elevation']:.1f} m")
        print(f"Loaded {num_species} species for site")
        print(f"  deg_days: {site['deg_days']:.0f}, dry_days: {site['dry_days']:.1f}")
        print(f"  soil_base_h: {site.get('soil_base_h', 'N/A')}, fire_prob: {site.get('fire_prob', 'N/A')}")
        print(f"Site agent ID: {site['site_agent_id']}")
        print(f"\nInitializing {args.num_gaps} gap(s) with trees...")

    # Initialize gaps and trees - creates gap agents and trees, connects them
    total_trees = 0
    total_alive = 0
    for gap_num in range(args.num_gaps):
        tree_ids, initial_alive = model.initialize_trees(
            site=site,
            maxtrees=args.maxtrees,
        )
        total_trees += len(tree_ids)
        total_alive += initial_alive
        if rank == 0:
            print(f"  Gap {gap_num + 1}: {len(tree_ids)} free slots (empty start, renewal fills forest)")

    num_templates = num_species * args.num_gaps

    # All ranks must participate in get_agent_property_value (MPI bcast)
    site_params = model.get_agent_property_value(site_agent_id, "params")
    site_states = model.get_agent_property_value(site_agent_id, "states")

    t_init_end = time.time()

    if rank == 0:
        print(f"\nTotal agents: {len(model.site_agents)} site, {len(model.gap_agents)} gaps, {len(model.tree_ids)} tree slots")
        print(f"  Initial alive: {total_alive}, Free slots: {total_trees - total_alive}")
        print(f"\nInitial soil state:")
        print(f"  A0 layer C: {site_params[SITE_P_A0_C]:.2f} tn/ha")
        print(f"  A layer C: {site_params[SITE_P_A_C]:.2f} tn/ha")
        print(f"  Base layer C: {site_params[SITE_P_BL_C]:.2f} tn/ha")
        print(f"  Available N: {site_states[SITE_S_AVAIL_N]:.4f} tn/ha")

    # Initialize output writer
    writer = OutputWriter(args.output_dir, site_id=args.site_id)
    writer.open(model.species_by_id, args.num_gaps)

    if rank == 0:
        print(f"\nOutput files opened in {args.output_dir}/")
        print(f"  genus_data.csv, species_data.csv, site_data.csv, soil_data.csv"
              + ("" if args.no_tree_data else ", tree_data.csv"))
        print("\nSetting up GPU kernels...")

    # Setup model (generates GPU kernels)
    model.setup(use_gpu=True)

    t_setup_end = time.time()

    if rank == 0:
        print(f"  Init time: {t_init_end - t_init_start:.2f}s, GPU setup time: {t_setup_end - t_init_end:.2f}s")
        print("Starting simulation...")
        print()
        print(f"{'Year':<6} {'Alive':<7} {'Seedlings':<10} {'Free':<8} {'NetChg':<8} {'Biomass':<10} {'Avail_N':<10} {'BatchTime':<10}")
        print("-" * 80)

    # Track living trees for net change calculation
    prev_alive = total_alive
    t_sim_start = time.time()

    # Run simulation
    for year_batch in range(0, args.years, args.report_interval):
        years_to_run = min(args.report_interval, args.years - year_batch)

        t_batch_start = time.time()

        # Simulate
        model.simulate(ticks=years_to_run, sync_workers_every_n_ticks=1)

        current_year = year_batch + years_to_run

        # All ranks must participate in MPI collective calls
        site_params = model.get_agent_property_value(site_agent_id, "params")
        site_states = model.get_agent_property_value(site_agent_id, "states")

        # Collect tree data for output
        tree_data = model.collect_tree_data()

        # Compute derived values
        annual_rain, grow_days = compute_site_extras(
            site_params, SITE_P_PRCP_BASE, SITE_P_TMIN_BASE, SITE_P_TMAX_BASE
        )
        soil_biomC, soil_biomN = compute_soil_biomass(tree_data, writer.plotadj)

        if rank == 0:
            # Write CSV outputs
            writer.write_site_data(
                current_year,
                site['latitude'], site['longitude'],
                site['elevation'], site.get('slope', 0.0),
                site_states[SITE_S_DEG_DAYS],
                site_states[SITE_S_FLOOD_DAYS],
                site_states[SITE_S_DRY_DAYS],
                annual_rain, grow_days,
                dry_days_base=site_states[SITE_S_DRY_DAYS_BASE],
            )
            writer.write_soil_data(
                current_year,
                site_params[SITE_P_A0_C], site_params[SITE_P_A_C],
                site_params[SITE_P_A0_N], site_params[SITE_P_A_N],
                site_params[SITE_P_BL_C], site_params[SITE_P_BL_N],
                site_states[SITE_S_AVAIL_N],
                soil_biomC, soil_biomN,
            )
            writer.write_species_data(current_year, tree_data, model.gap_agents)
            writer.write_genus_data(current_year, tree_data, model.gap_agents)
            if not args.no_tree_data:
                writer.write_tree_data(current_year, tree_data, model.gap_agents)

            # Console output
            num_living = len(tree_data)
            num_seedlings = sum(1 for t in tree_data if t['age'] <= 2.0)
            total_biomass = sum(t['biomC'] for t in tree_data)
            num_free = len(model.tree_ids) - num_living - num_templates
            avail_n = site_states[SITE_S_AVAIL_N] if isinstance(site_states, list) else site_states

            net_change = num_living - prev_alive
            net_str = f"+{net_change}" if net_change >= 0 else str(net_change)
            prev_alive = num_living

            t_batch_elapsed = time.time() - t_batch_start
            print(f"{current_year:<6} {num_living:<7} {num_seedlings:<10} {num_free:<8} {net_str:<8} {total_biomass:<10.1f} {avail_n:<10.4f} {t_batch_elapsed:<10.2f}s")

    # Final state
    site_params = model.get_agent_property_value(site_agent_id, "params")
    site_states = model.get_agent_property_value(site_agent_id, "states")

    t_total_end = time.time()

    if rank == 0:
        avail_n = site_states[SITE_S_AVAIL_N] if isinstance(site_states, list) else site_states
        t_sim_elapsed = t_total_end - t_sim_start
        t_total_elapsed = t_total_end - t_total_start
        print("\n" + "=" * 70)
        print("Simulation Complete")
        print("=" * 70)
        print(f"\nFinal Tree Statistics:")
        print(f"  Living trees: {prev_alive}")
        print(f"  Free slots: {len(model.tree_ids) - prev_alive - num_templates}")
        print(f"  Net change from start: {prev_alive - total_alive:+d}")

        print(f"\nFinal soil state:")
        print(f"  A0 layer C: {site_params[SITE_P_A0_C]:.2f} tn/ha, N: {site_params[SITE_P_A0_N]:.4f} tn/ha")
        print(f"  A layer C: {site_params[SITE_P_A_C]:.2f} tn/ha, N: {site_params[SITE_P_A_N]:.4f} tn/ha")
        print(f"  Base layer C: {site_params[SITE_P_BL_C]:.2f} tn/ha, N: {site_params[SITE_P_BL_N]:.4f} tn/ha")
        print(f"  Available N: {avail_n:.4f} tn/ha/yr")

        print(f"\nTiming Summary:")
        print(f"  Initialization: {t_init_end - t_init_start:.2f}s")
        print(f"  GPU setup:      {t_setup_end - t_init_end:.2f}s")
        print(f"  Simulation:     {t_sim_elapsed:.2f}s ({t_sim_elapsed / args.years:.3f}s/year)")
        print(f"  Total:          {t_total_elapsed:.2f}s")

        print(f"\nOutput files written to {args.output_dir}/")

    writer.close()


if __name__ == "__main__":
    main()
