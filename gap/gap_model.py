"""
GAPModel - GPU-accelerated gap dynamics model.
Integrates UVAFME processes with SAGESim agent framework.

Agent Hierarchy:
- Site agents: hold soil C/N pools (mutable state only, climate in globals)
- Gap agents: relay between Site and Trees, aggregate tree outputs
- Tree agents: individual trees with species_id + mutable state (traits in globals)

Constants (species traits + site config) are stored in SAGESim globals,
shared read-only across all agents and ranks.

Network Connections:
- Site <-> Gap: each gap connects to its parent site (bidirectional)
- Gap <-> Tree: each tree connects to its parent gap (bidirectional)
- Free -> Template: each free slot connects to all templates (directed, for P7 species selection)

Property Scheme (2 properties per breed):
- params: neighbor_visible=False  (self-read only, private mutable state)
- states: neighbor_visible=True   (neighbor-readable, cross-breed communication)

Property Arrays by Breed:
  Tree: params[14], states[11]
        (params: mutable physiology + intermediates + renewal)
        (states: IS_ALIVE/DIAM/HEIGHT/CANOPY_HT + litter/demand + SPECIES_ID/ENV_STRESS)
  Gap:  params[2],  states[16]
  Site: params[21], states[7]
        (params: soil pools[9] + lai_w0 + annual_runoff + site_id + output fields[9])
        (states: climate fields read by Gap P2)
"""

import sys
import os
import csv
import numpy as np

try:
    import sagesim
except ImportError:
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    _sagesim_path = os.path.join(os.path.dirname(_current_dir), "SAGESim")
    if _sagesim_path not in sys.path:
        sys.path.insert(0, _sagesim_path)

from mpi4py import MPI
from sagesim.model import Model
from sagesim.space import NetworkSpace
from sagesim.breed import Breed
from pathlib import Path

comm = MPI.COMM_WORLD
num_workers = comm.Get_size()

# Import step functions from organized folder
from gap.step_functions.gap.gap_litter_aggregate_step import gap_litter_aggregate_step
from gap.step_functions.site.soil_step import site_soil_step
from gap.step_functions.tree.tree_potential_growth_step import tree_potential_growth_step
from gap.step_functions.gap.gap_demand_aggregate_step import gap_demand_aggregate_step
from gap.step_functions.gap.gap_climate_relay_step import gap_climate_relay_step
from gap.step_functions.tree.tree_template_renewal_step import tree_template_renewal_step
from gap.step_functions.gap.gap_recruit_aggregate_step import gap_recruit_aggregate_step
from gap.step_functions.tree.tree_actual_growth_step import tree_actual_growth_step
from gap.step_functions.gap.gap_nconsumed_aggregate_step import gap_nconsumed_aggregate_step
from gap.step_functions.site.site_final_step import site_final_step

# Import constants — all index enums and sizes from single source of truth
from gap.constants import (
    Cfg, NUM_SPECIES_TRAITS, NUM_SITE_CONFIGS,
    DISPERSAL_CUTOFF_FACTOR, EARTH_RADIUS_KM,
    TreeP, TreeS, TREE_PARAMS_SIZE,
    GapP, GapS, GAP_STATES_SIZE,
    SiteP, SiteS, SITE_PARAMS_SIZE,
)

CURRENT_DIR = Path(__file__).resolve().parent

# ============================================================
# Plain integer aliases for output collection (run_one_site.py imports these)
# These MUST match the IntEnum values in gap/constants.py
# ============================================================

# --- Tree params (self-read only) ---
TREE_P_AGE = int(TreeP.AGE)
TREE_P_BIOMC = int(TreeP.BIOMC)
TREE_P_BIOMN = int(TreeP.BIOMN)
TREE_P_LEAF_BM = int(TreeP.LEAF_BM)

# --- Tree states (neighbor-visible) ---
TREE_S_IS_ALIVE = int(TreeS.IS_ALIVE)
TREE_S_DIAM = int(TreeS.DIAM)
TREE_S_HEIGHT = int(TreeS.HEIGHT)
TREE_S_CANOPY_HT = int(TreeS.CANOPY_HT)
TREE_S_SPECIES_ID = int(TreeS.SPECIES_ID)

# --- Site params (self-read only, includes output fields) ---
SITE_P_A0_C = int(SiteP.A0_C)
SITE_P_A0_N = int(SiteP.A0_N)
SITE_P_A_C = int(SiteP.A_C)
SITE_P_A_N = int(SiteP.A_N)
SITE_P_BL_C = int(SiteP.BL_C)
SITE_P_BL_N = int(SiteP.BL_N)
SITE_P_ANNUAL_RAIN = int(SiteP.ANNUAL_RAIN)
SITE_P_GROW_DAYS = int(SiteP.GROW_DAYS)
SITE_P_POT_EVAP = int(SiteP.POT_EVAP)
SITE_P_ACT_EVAP = int(SiteP.ACT_EVAP)
SITE_P_SOIL_RESP = int(SiteP.SOIL_RESP)
SITE_P_C_INTO_A0 = int(SiteP.C_INTO_A0)
SITE_P_N_INTO_A0 = int(SiteP.N_INTO_A0)
SITE_P_NET_N_INTO_A0 = int(SiteP.NET_N_INTO_A0)

# --- Site states (neighbor-visible) ---
SITE_S_DEG_DAYS = int(SiteS.DEG_DAYS)
SITE_S_DRY_DAYS = int(SiteS.DRY_DAYS)
SITE_S_AVAIL_N = int(SiteS.AVAIL_N)
SITE_S_FLOOD_DAYS = int(SiteS.FLOOD_DAYS)
SITE_S_DRY_DAYS_BASE = int(SiteS.DRY_DAYS_BASE)


class GAPModel(Model):
    """
    GAPModel class for gap dynamics simulation.

    Constants (species traits + site config) stored in SAGESim globals.
    Agent params hold only mutable state.
    """

    def __init__(self, verbose_timing=False) -> None:
        space = NetworkSpace()
        super().__init__(space, agent_slack_factor=1.0, csr_slack_factor=1.0,
                         verbose_timing=verbose_timing)

        # Species registry (deduplicated across sites)
        self.unique_species = {}
        self.species_by_id = {}  # global_id -> species info dict

        # Track agents
        self.sites = []  # site info dicts
        self.site_agents = []  # site agent IDs
        self.gap_agents = []  # gap agent IDs
        self.tree_ids = []  # tree agent IDs
        self.tree_to_gap = {}  # tree_agent_id -> gap_agent_id

        # Track number of sites loaded into globals
        self._num_sites_in_globals = 0
        self._breeds_registered = False

    def _register_breeds(self, num_species):
        """Register breeds with property arrays.

        Property design:
          params  = neighbor_visible=False  (self-read only)
          states  = neighbor_visible=True   (neighbor-readable)
          states_db removed — merged into states (no double-buffering needed)

        Must be called after load_globals() determines num_species.
        """
        # === Create Tree breed (breed_id = 0) ===
        self._tree_breed = Breed("Tree")
        self._tree_breed.register_property("params", [0.0] * TREE_PARAMS_SIZE, neighbor_visible=False)
        # neighbor_visible=False: all agents of a site are on the same worker (partition_sites invariant)
        self._tree_breed.register_property("states", [0.0] * 11, neighbor_visible=False)
        self._tree_breed.register_step_func(
            tree_potential_growth_step,
            CURRENT_DIR / "step_functions" / "tree" / "tree_potential_growth_step.py",
            priority=3, no_double_buffer=["params", "states", "gap_lai", "gap_avail_spec", "gap_imported_seeds", "gap_seedling_weights", "site_imported_seeds"],
        )
        self._tree_breed.register_step_func(
            tree_template_renewal_step,
            CURRENT_DIR / "step_functions" / "tree" / "tree_template_renewal_step.py",
            priority=5, no_double_buffer=["params", "states", "gap_lai", "gap_avail_spec", "gap_imported_seeds", "gap_seedling_weights", "site_imported_seeds"],
        )
        self._tree_breed.register_step_func(
            tree_actual_growth_step,
            CURRENT_DIR / "step_functions" / "tree" / "tree_actual_growth_step.py",
            priority=7, no_double_buffer=["params", "states", "gap_lai", "gap_avail_spec", "gap_imported_seeds", "gap_seedling_weights", "site_imported_seeds"],
        )
        self.register_breed(breed=self._tree_breed)

        # === Create Gap breed (breed_id = 1) ===
        self._gap_breed = Breed("Gap")
        self._gap_breed.register_property("params", [0.0] * 2, neighbor_visible=False)
        # neighbor_visible=False: all agents of a site are on the same worker
        self._gap_breed.register_property("states", [0.0] * GAP_STATES_SIZE, neighbor_visible=False)
        self._gap_breed.register_step_func(
            gap_litter_aggregate_step, CURRENT_DIR / "step_functions" / "gap" / "gap_litter_aggregate_step.py",
            priority=0, no_double_buffer=["params", "states", "gap_lai", "gap_avail_spec", "gap_imported_seeds", "gap_seedling_weights", "site_imported_seeds"],
        )
        self._gap_breed.register_step_func(
            gap_climate_relay_step, CURRENT_DIR / "step_functions" / "gap" / "gap_climate_relay_step.py",
            priority=2, no_double_buffer=["params", "states", "gap_lai", "gap_avail_spec", "gap_imported_seeds", "gap_seedling_weights", "site_imported_seeds"],
        )
        self._gap_breed.register_step_func(
            gap_demand_aggregate_step, CURRENT_DIR / "step_functions" / "gap" / "gap_demand_aggregate_step.py",
            priority=4, no_double_buffer=["params", "states", "gap_lai", "gap_avail_spec", "gap_imported_seeds", "gap_seedling_weights", "site_imported_seeds"],
        )
        self._gap_breed.register_step_func(
            gap_recruit_aggregate_step, CURRENT_DIR / "step_functions" / "gap" / "gap_recruit_aggregate_step.py",
            priority=6, no_double_buffer=["params", "states", "gap_lai", "gap_avail_spec", "gap_imported_seeds", "gap_seedling_weights", "site_imported_seeds"],
        )
        self._gap_breed.register_step_func(
            gap_nconsumed_aggregate_step, CURRENT_DIR / "step_functions" / "gap" / "gap_nconsumed_aggregate_step.py",
            priority=8, no_double_buffer=["params", "states", "gap_lai", "gap_avail_spec", "gap_imported_seeds", "gap_seedling_weights", "site_imported_seeds"],
        )
        self.register_breed(breed=self._gap_breed)

        # === Create Site breed (breed_id = 2) ===
        self._site_breed = Breed("Site")
        self._site_breed.register_property("params", [0.0] * SITE_PARAMS_SIZE, neighbor_visible=False)
        self._site_breed.register_property("states", [0.0] * 8, neighbor_visible=True)
        self._site_breed.register_step_func(
            site_soil_step,
            CURRENT_DIR / "step_functions" / "site" / "soil_step.py",
            priority=1,
            no_double_buffer=["params", "states", "gap_lai", "gap_avail_spec", "gap_imported_seeds", "gap_seedling_weights", "site_imported_seeds"],
        )
        self._site_breed.register_step_func(
            site_final_step,
            CURRENT_DIR / "step_functions" / "site" / "site_final_step.py",
            priority=9,
            no_double_buffer=["params", "states", "gap_lai", "gap_avail_spec", "gap_imported_seeds", "gap_seedling_weights", "site_imported_seeds"],
        )
        self.register_breed(breed=self._site_breed)

        # Register site_distances global (placeholder, updated by connect_sites)
        num_sites = self._num_sites_in_globals
        self.register_global_property("site_distances",
                                      np.zeros((num_sites, num_sites)))
        self._breeds_registered = True

    def load_globals(self, data_dir="input_data", prefix="UVAFME2012"):
        """
        Load all species traits and site configs into SAGESim globals.

        Must be called before initialize_site(). All ranks call this identically
        so globals are consistent across MPI workers.

        Loads:
        - Full specieslist.csv → sequential global_ids → species traits in globals
        - All sites from site.csv + climate.csv + climate_stddev.csv + altitudes.csv
        - rangelist.csv → per-site species-present masks
        """
        base_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            data_dir
        )

        months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                  'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

        # === Load ALL species (not filtered by site) ===
        species_file = os.path.join(base_path, f"{prefix}_specieslist.csv")
        all_species_rows = []
        with open(species_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row.get('Species_code', '').strip():
                    continue
                all_species_rows.append(row)

        # Assign global_ids and build species dicts
        for global_id, row in enumerate(all_species_rows):
            species_code = row['Species_code']

            if species_code in self.unique_species:
                continue

            # GAPpy adjustments (species.py:66-101)
            ss = [1.1, 1.15, 1.2, 1.23, 1.25]
            adjust = [1.5, 1.55, 1.6, 1.65, 1.7]
            HEC_TO_M2 = 10000.0

            shade_tol = int(row['l'])
            rootdepth = float(row.get('rootdepth', 0.8))
            g_adj = float(row['g']) * ss[shade_tol - 1]
            max_ht_adj = min(float(row['Hmax']),
                             rootdepth * 80.0 / (1.0 + rootdepth))
            leafdiam_a = float(row['D_L']) * adjust[shade_tol - 1]
            leafarea_c = float(row['L_C']) / HEC_TO_M2

            sp_info = {
                'global_id': global_id,
                'species_code': species_code,
                'common_name': row['Common name'].replace('_', ' ').strip("'"),
                'max_age': float(row['AGEmax']),
                'max_diam': float(row['DBHmax']),
                'max_ht': max_ht_adj,
                'arfa_0': float(row['s']),
                'g': g_adj,
                'shade_tol': shade_tol,
                'deg_day_min': float(row['DEGDmin']),
                'deg_day_opt': float(row['DEGDoptimum']),
                'deg_day_max': float(row['DEGDmax']),
                'wood_bulk_dens': float(row['bulk']),
                'invader': float(row['invader']),
                'seed': float(row['seed']),
                'sprout': float(row['sprout']),
                'lownutr_tol': int(row['n']),
                'flood_tol': int(row['f']),
                'drought_tol': int(row['d']),
                'evergreen': int(row['evergreen']),
                'fire_tol': int(row['fire']),
                'rootdepth': rootdepth,
                'stress_tol': int(row['stress']),
                'age_tol': int(row['old']),
                'genus': row['Genus'],
                'seed_surv': float(row['NDE']),
                'seedling_lg': float(row['NDS']),
                'leafdiam_a': leafdiam_a,
                'leafarea_c': leafarea_c,
                'max_dispersal_dist': float(row.get('max_dispersal_dist', 10.0)),
            }
            self.unique_species[species_code] = sp_info
            self.species_by_id[global_id] = sp_info

        num_species = len(self.unique_species)

        # === Register species traits as 2D tensor global ===
        species_traits = np.zeros((num_species, NUM_SPECIES_TRAITS))
        for sp in range(num_species):
            sp_info = self.species_by_id[sp]
            for t in range(NUM_SPECIES_TRAITS):
                species_traits[sp, t] = self._get_species_trait_value(sp_info, t)
        self.register_global_property("species_traits", species_traits)

        # === Load ALL site configs ===
        site_file = os.path.join(base_path, f"{prefix}_site.csv")
        climate_file = os.path.join(base_path, f"{prefix}_climate.csv")
        climate_std_file = os.path.join(base_path, f"{prefix}_climate_stddev.csv")
        altitude_file = os.path.join(base_path, f"{prefix}_altitudes.csv")
        range_file = os.path.join(base_path, f"{prefix}_rangelist.csv")

        # Read all site rows
        site_rows = {}
        with open(site_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row.get('site', '').strip():
                    continue
                site_rows[int(row['site'])] = row

        # Read all climate rows
        climate_rows = {}
        with open(climate_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row.get('site', '').strip():
                    continue
                climate_rows[int(row['site'])] = row

        # Read climate std devs (optional)
        climate_std_rows = {}
        if os.path.exists(climate_std_file):
            with open(climate_std_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if not row.get('site', '').strip():
                        continue
                    climate_std_rows[int(row['site'])] = row

        # Read altitudes (optional)
        altitude_rows = {}
        if os.path.exists(altitude_file):
            with open(altitude_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if not row.get('site', '').strip():
                        continue
                    altitude_rows[int(row['site'])] = row

        # Read rangelists
        range_rows = {}
        with open(range_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row.get('site', '').strip():
                    continue
                range_rows[int(row['site'])] = row

        # Register site configs as 2D tensor global
        all_site_ids = sorted(site_rows.keys())
        num_sites = len(all_site_ids)
        self._num_sites_in_globals = num_sites

        site_configs = np.zeros((num_sites, NUM_SITE_CONFIGS))
        for site_slot in range(num_sites):
            sid = all_site_ids[site_slot]
            cfg_values = self._build_site_config(
                sid, site_rows[sid], climate_rows.get(sid),
                climate_std_rows.get(sid), altitude_rows.get(sid),
                months
            )
            site_configs[site_slot] = cfg_values
        self.register_global_property("site_configs", site_configs)

        # Register rangelist masks as 2D tensor global
        rangelists = np.zeros((num_sites, num_species))
        for site_slot in range(num_sites):
            sid = all_site_ids[site_slot]
            rrow = range_rows.get(sid)
            if rrow:
                for col, val in rrow.items():
                    if col not in ('site', 'latitude', 'longitude'):
                        if val == '1':
                            sp_info = self.unique_species.get(col)
                            if sp_info is not None:
                                rangelists[site_slot, sp_info['global_id']] = 1.0
        self.register_global_property("rangelists", rangelists)

        # Register breeds now that num_species is known
        if not self._breeds_registered:
            self._register_breeds(num_species)

        # Store site_id mapping for initialize_site
        self._site_id_to_slot = {sid: slot for slot, sid in enumerate(all_site_ids)}
        self._site_rows = site_rows
        self._climate_rows = climate_rows

    def _get_species_trait_value(self, sp_info, trait_idx):
        """Get species trait value by trait index for globals registration."""
        mapping = [
            'max_age', 'max_diam', 'max_ht', 'arfa_0', 'g',
            'shade_tol', 'deg_day_min', 'deg_day_opt', 'deg_day_max',
            'invader', 'seed', 'sprout', 'wood_bulk_dens',
            'lownutr_tol', 'flood_tol', 'drought_tol', 'evergreen',
            'fire_tol', 'rootdepth', 'stress_tol', 'age_tol',
            'seed_surv', 'seedling_lg', 'leafdiam_a', 'leafarea_c',
            'max_dispersal_dist',
        ]
        if trait_idx < len(mapping):
            return float(sp_info[mapping[trait_idx]])
        return 0.0

    def _build_site_config(self, site_id, site_row, climate_row,
                           climate_std_row, altitude_row, months):
        """Build 107-element site config array for globals."""
        cfg = [0.0] * NUM_SITE_CONFIGS

        if climate_row is None:
            return cfg

        # Apply altitude-based climate adjustment
        elevation = float(site_row.get('elevation', 0))
        tmp_lapse = [0.0] * 12
        prcp_lapse = [0.0] * 12
        for i, month in enumerate(months):
            tmp_lapse[i] = float(site_row.get(f'tmp_lapse_{month}', 0.0))
            prcp_lapse[i] = float(site_row.get(f'prcp_lapse_{month}', 0.0))

        altitude = None
        if altitude_row is not None:
            altitude = float(altitude_row['altitude'])

        # Monthly climate
        for i, month in enumerate(months):
            tmin_val = float(climate_row[f'tmin_{month}'])
            tmax_val = float(climate_row[f'tmax_{month}'])
            prcp_val = float(climate_row[f'prcp_{month}']) / 10.0  # Convert to cm

            if altitude is not None:
                tmin_val -= (altitude - elevation) * tmp_lapse[i] * 0.01
                tmax_val -= (altitude - elevation) * tmp_lapse[i] * 0.01
                prcp_val = max(0.0, prcp_val + (altitude - elevation) * prcp_lapse[i] * 0.001)

            cfg[Cfg.TMIN_BASE + i] = tmin_val
            cfg[Cfg.TMAX_BASE + i] = tmax_val
            cfg[Cfg.PRCP_BASE + i] = prcp_val

        # Soil/site params
        soil_rootdepth = 0.8
        cfg[Cfg.FIELD_CAP] = float(site_row['soilA_field_cap']) * soil_rootdepth
        cfg[Cfg.PERM_WP] = float(site_row['soilA_perm_wp']) * soil_rootdepth
        cfg[Cfg.SLOPE] = float(site_row['slope'])
        cfg[Cfg.SIGMA] = float(site_row['sigma'])
        cfg[Cfg.LAI] = float(site_row['lai'])
        cfg[Cfg.LATITUDE] = float(site_row['latitude'])
        cfg[Cfg.LONGITUDE] = float(site_row['longitude'])
        cfg[Cfg.RAIN_N] = 0.0

        cfg[Cfg.FIRE_PROB] = float(site_row.get('fire_prob', 0)) / 1000.0
        cfg[Cfg.WIND_PROB] = float(site_row.get('wind_prob', 0)) / 1000.0
        cfg[Cfg.BASE_H] = float(site_row.get('soil_base_h', 70.0))

        # Climate std devs
        if climate_std_row is not None:
            for i, month in enumerate(months):
                cfg[Cfg.TMIN_STD_BASE + i] = float(climate_std_row[f'tmn_std_{month}'])
                cfg[Cfg.TMAX_STD_BASE + i] = float(climate_std_row[f'tmx_std_{month}'])
                cfg[Cfg.PRCP_STD_BASE + i] = float(climate_std_row[f'prcp_std_{month}']) * 0.1

        # Lapse rates
        for i in range(12):
            cfg[Cfg.TMP_LAPSE_BASE + i] = tmp_lapse[i]
            cfg[Cfg.PRCP_LAPSE_BASE + i] = prcp_lapse[i]

        return cfg

    def initialize_site(
        self,
        site_id: int = 0,
        data_dir: str = "input_data",
        prefix: str = "UVAFME2012",
        rank: int = None,
    ):
        """
        Initialize a site from UVAFME CSV files.

        Expects load_globals() to have been called first. Site params now only
        contain mutable soil state (12 values). All climate/config is in globals.

        :param site_id: Row index in CSV files (default: 0)
        :param data_dir: Directory containing CSV files
        :param prefix: File prefix (e.g., "UVAFME2012")
        :return: site_info dict with site_agent_id
        """
        base_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            data_dir
        )

        months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                  'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

        # Get site slot from globals mapping
        site_slot = self._site_id_to_slot.get(site_id)
        if site_slot is None:
            raise ValueError(f"Site {site_id} not found in globals. Call load_globals() first.")

        site_row = self._site_rows[site_id]
        climate_row = self._climate_rows.get(site_id)

        if climate_row is None:
            raise ValueError(f"Climate data for site {site_id} not found")

        # Load rangelist for this site to determine species present
        range_file = os.path.join(base_path, f"{prefix}_rangelist.csv")
        species_present = set()
        with open(range_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row.get('site', '').strip():
                    continue
                if int(row['site']) == site_id:
                    for col, val in row.items():
                        if col not in ('site', 'latitude', 'longitude'):
                            if val == '1':
                                species_present.add(col)
                    break

        # Filter species for this site (sorted by global_id for deterministic ordering)
        site_species = sorted(
            [self.unique_species[sp_code] for sp_code in species_present
             if sp_code in self.unique_species],
            key=lambda sp: sp['global_id']
        )

        # Calculate degree days from site_configs tensor for display
        BASE_TEMP = 5.0
        days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        deg_days = 0.0
        site_configs = self.get_global_property_value("site_configs")
        for i in range(12):
            tmin = site_configs[site_slot][Cfg.TMIN_BASE + i]
            tmax = site_configs[site_slot][Cfg.TMAX_BASE + i]
            tavg = (tmin + tmax) / 2.0
            if tavg > BASE_TEMP:
                deg_days += (tavg - BASE_TEMP) * days_per_month[i]

        dry_days = 0.0

        # === Build params[12] for Site (mutable soil state only) ===
        soil_rootdepth = 0.8
        params = [0.0] * SITE_PARAMS_SIZE
        params[SITE_P_A0_C] = float(site_row['soilAO_c0'])
        params[SITE_P_A0_N] = float(site_row['soilAO_n0'])
        params[SITE_P_A_C] = float(site_row['soilA_c0'])
        params[SITE_P_A_N] = float(site_row['soilA_n0'])
        params[SITE_P_BL_C] = float(site_row['sbase_c0'])
        params[SITE_P_BL_N] = float(site_row['sbase_n0'])
        params[int(SiteP.A0_W)] = float(site_row['soilAO_w0'])
        params[int(SiteP.A_W)] = float(site_row['soilA_w0']) * soil_rootdepth
        params[int(SiteP.BL_W)] = float(site_row['sbase_w0'])
        params[int(SiteP.LAI_W0)] = float(site_row['lai_w0'])
        params[int(SiteP.ANNUAL_RUNOFF)] = 0.0
        params[int(SiteP.SITE_ID)] = float(site_slot)  # Globals slot index

        # === Build states (8 neighbor-visible fields: climate + SITE_ID) ===
        states = [0.0] * 8
        states[SITE_S_DEG_DAYS] = deg_days
        states[SITE_S_DRY_DAYS] = dry_days
        states[SITE_S_AVAIL_N] = 0.1
        states[SITE_S_FLOOD_DAYS] = 0.0
        states[int(SiteS.FIRE_INTENSITY)] = 0.0
        states[SITE_S_DRY_DAYS_BASE] = 0.0
        states[int(SiteS.WIND_INTENSITY)] = 0.0
        states[int(SiteS.SITE_ID)] = float(site_slot)

        # Create site agent
        site_agent_id = self.create_agent_of_breed(
            self._site_breed,
            rank=rank,
            params=params,
            states=states,
        )
        self.site_agents.append(site_agent_id)
        self.set_agent_logical_id(site_agent_id, site_slot)

        elevation = float(site_row.get('elevation', 0))
        fire_prob = float(site_row.get('fire_prob', 0)) / 1000.0
        wind_prob = float(site_row.get('wind_prob', 0)) / 1000.0
        soil_base_h = float(site_row.get('soil_base_h', 70.0))
        region = site_row.get('region', '')

        site_info = {
            'site_id': site_id,
            'site_slot': site_slot,
            'site_agent_id': site_agent_id,
            'rank': rank,
            'site_name': site_row.get('name', f'Site_{site_id}'),
            'latitude': float(site_row['latitude']),
            'longitude': float(site_row['longitude']),
            'elevation': elevation,
            'slope': float(site_row['slope']),
            'region': region,
            'fire_prob': fire_prob,
            'wind_prob': wind_prob,
            'soil_base_h': soil_base_h,
            'species': site_species,
            'deg_days': deg_days,
            'dry_days': dry_days,
            'gaps': [],
        }
        self.sites.append(site_info)

        return site_info

    def initialize_gap(self, site, rank: int = None):
        """Initialize a gap agent and connect it to the site."""
        gap_id = len(self.gap_agents)
        site_agent_id = site['site_agent_id']

        deg_days = site.get('deg_days', 2500.0)
        dry_days = site.get('dry_days', 0.0)

        params = [0.0] * 2
        params[int(GapP.SITE_ID)] = float(gap_id)
        params[int(GapP.TOTAL_N_DEMAND)] = 0.0

        states = [0.0] * GAP_STATES_SIZE
        states[int(GapS.DEG_DAYS)] = deg_days
        states[int(GapS.DRY_DAYS)] = dry_days
        states[int(GapS.AVAIL_N)] = 0.1
        states[int(GapS.N_SUPPLY_RATIO)] = 1.0

        gap_agent_id = self.create_agent_of_breed(
            self._gap_breed,
            rank=rank,
            params=params,
            states=states,
        )
        self.gap_agents.append(gap_agent_id)
        site['gaps'].append(gap_agent_id)
        site_slot = site.get('site_slot', 0)
        gap_within_site = len(site['gaps']) - 1
        self.set_agent_logical_id(gap_agent_id, site_slot * 10000 + gap_within_site)

        self.connect_agents(site_agent_id, gap_agent_id)

        return gap_agent_id

    def initialize_trees(
        self,
        site,
        gap_agent_id=None,
        maxtrees=1000,
        rank: int = None,
    ):
        """
        Initialize tree slots in a gap. All start as free (renewal fills forest).
        Template trees preserve species diversity for recruitment.
        """
        if maxtrees == 0:
            raise ValueError("Must have at least one tree slot")

        if gap_agent_id is None:
            gap_agent_id = self.initialize_gap(site, rank=rank)

        site_species = site['species']
        num_species = len(site_species)

        if num_species == 0:
            raise ValueError("Site has no species")

        created_trees = []
        template_trees = []
        initial_alive_count = 0

        # Stable logical ID base: site_slot * 10_000_000 + gap * 10_000 + tree_idx
        site_slot = site.get('site_slot', 0)
        gap_within_site = len(site['gaps']) - 1
        logical_base = site_slot * 10000000 + gap_within_site * 10000
        tree_counter = 0

        # Create template trees (one per species)
        for species_info in site_species:
            agent_id = self._create_tree_agent(
                gap_agent_id, species_info, diam=0.0, age=0.0, is_alive=-1.0, rank=rank
            )
            template_trees.append(agent_id)
            self.set_agent_logical_id(agent_id, logical_base + tree_counter)
            tree_counter += 1

        # Create free tree slots
        free_slot_count = maxtrees
        if free_slot_count > 0:
            placeholder_species = site_species[0]
            for _ in range(free_slot_count):
                agent_id = self._create_tree_agent(
                    gap_agent_id, placeholder_species, diam=0.0, age=0.0, is_alive=0.0, rank=rank
                )
                created_trees.append(agent_id)
                self.set_agent_logical_id(agent_id, logical_base + tree_counter)
                tree_counter += 1

        return created_trees, initial_alive_count

    def _create_tree_agent(self, gap_agent_id, species_info, diam, age, is_alive=1.0, rank=None):
        """
        Create a single tree agent. Only stores species_id + mutable state in params.
        Species traits are looked up from globals via species_id.
        """
        STD_HT = 1.3
        TC_KG = 0.039269908

        if is_alive > 0.5 and diam > 0:
            delta_ht = species_info['max_ht'] - STD_HT
            forska_ht = STD_HT + delta_ht * (
                1.0 - (2.71828 ** (-(species_info['arfa_0'] * diam / delta_ht)))
            )
            wood_bulk_dens = species_info['wood_bulk_dens']
            rootdepth = species_info.get('rootdepth', 0.8)
            canopy_ht = STD_HT

            if forska_ht > STD_HT:
                bd = forska_ht / (forska_ht - STD_HT) * diam
            else:
                bd = diam

            if forska_ht > canopy_ht and forska_ht > STD_HT:
                dc = (forska_ht - canopy_ht) / (forska_ht - STD_HT) * diam
            else:
                dc = diam

            stembc = TC_KG * wood_bulk_dens * 0.3 * bd * bd * forska_ht
            crown_depth = forska_ht - canopy_ht
            if crown_depth < 0.0:
                crown_depth = 0.0
            twigbc = TC_KG * wood_bulk_dens * 0.337 * dc * dc * crown_depth

            root_c = 0.0
            if forska_ht > 0.01:
                root_c = stembc * rootdepth / forska_ht + twigbc * 0.5

            biomC = stembc + twigbc + root_c
            biomN = biomC / 450.0
            leaf_bm = dc * dc * species_info['leafdiam_a'] * species_info['leafarea_c'] * 2.0 * 1000.0
        else:
            forska_ht = 0.0
            biomC = 0.0
            biomN = 0.0
            leaf_bm = 0.0

        # Build params[14] — self-read only mutable state
        params = [0.0] * TREE_PARAMS_SIZE
        params[int(TreeP.AGE)] = age
        params[int(TreeP.BIOMC)] = biomC
        params[int(TreeP.BIOMN)] = biomN
        params[int(TreeP.LEAF_BM)] = leaf_bm
        params[int(TreeP.X)] = 0.0
        params[int(TreeP.Y)] = 0.0
        params[int(TreeP.LIGHT_AVAIL)] = 1.0
        params[int(TreeP.FC_DEGDAY)] = 1.0
        params[int(TreeP.FC_DROUGHT)] = 1.0
        params[int(TreeP.FC_FLOOD)] = 1.0

        # Build states[11] — neighbor-visible (merged states + states_db)
        states = [0.0] * 11
        states[int(TreeS.IS_ALIVE)] = float(is_alive)
        states[int(TreeS.DIAM)] = diam
        states[int(TreeS.HEIGHT)] = forska_ht
        states[int(TreeS.CANOPY_HT)] = STD_HT if is_alive > 0.5 else 0.0
        states[int(TreeS.SPECIES_ID)] = float(species_info['global_id'])

        agent_id = self.create_agent_of_breed(
            self._tree_breed,
            rank=rank,
            params=params,
            states=states,
        )

        self.tree_ids.append(agent_id)
        self.tree_to_gap[agent_id] = gap_agent_id

        self.connect_agents(gap_agent_id, agent_id)

        return agent_id

    def connect_agents(self, agent_0, agent_1, directed=False):
        """Connect two agents. Bidirectional by default, one-way if directed=True."""
        self.get_space().connect_agents(agent_0, agent_1, directed=directed)

    def connect_torus_sites(self, width, height, site_agents_dict):
        """Connect sites in torus topology (8-neighbor Moore with wraparound).

        Args:
            width: Grid width
            height: Grid height
            site_agents_dict: dict mapping site_id -> site_agent_id

        Returns:
            int: Total edge count
        """
        neighbors = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]

        edge_count = 0

        for i in range(height):
            for j in range(width):
                site_id = i * width + j
                site_agent = site_agents_dict[site_id]

                for di, dj in neighbors:
                    ni = (i + di) % height
                    nj = (j + dj) % width
                    neighbor_id = ni * width + nj
                    neighbor_agent = site_agents_dict[neighbor_id]

                    self.connect_agents(site_agent, neighbor_agent)
                    edge_count += 1

        if comm.Get_rank() == 0:
            print(f"Connected sites: {edge_count} edges ({edge_count/(width*height):.1f} per site)")

        return edge_count

    def register_breed_local_arrays(self):
        """Register breed-local arrays for LAI, species, and dispersal.

        Must be called after all agents are created but before setup().
        All neighbor_visible=False except site_avail_spec (cross-rank dispersal).
        All in no_double_buffer except site_avail_spec (same-priority race at P10).
        """
        from gap.constants import MAX_HEIGHT_BINS
        num_species = len(self.unique_species)

        # gap_lai: per-gap LAI profile (dec/con per height bin)
        self.register_breed_local_array(
            "gap_lai", breed=self._gap_breed,
            shape_per_agent=(MAX_HEIGHT_BINS, 2),
            neighbor_visible=False)

        # gap_avail_spec: per-gap species availability flags
        self.register_breed_local_array(
            "gap_avail_spec", breed=self._gap_breed,
            shape_per_agent=(num_species,),
            neighbor_visible=False)

        # gap_imported_seeds: per-gap imported seed relay (P2 copies from site)
        self.register_breed_local_array(
            "gap_imported_seeds", breed=self._gap_breed,
            shape_per_agent=(num_species,),
            neighbor_visible=False)

        # gap_seedling_weights: per-gap per-species seedling weight for recruitment
        # Written by templates at P5, read by free slots at P7 and gap at P0.
        # Replaces free→template connections (eliminates num_gaps*pool_size*num_species edges).
        self.register_breed_local_array(
            "gap_seedling_weights", breed=self._gap_breed,
            shape_per_agent=(num_species,),
            neighbor_visible=False)

        # site_avail_spec: site-averaged species availability (cross-rank, double-buffered)
        # neighbor_visible=True: neighbor sites read for dispersal at P10
        # NOT in no_double_buffer: P10 reads neighbor's + writes own at same priority
        self.register_breed_local_array(
            "site_avail_spec", breed=self._site_breed,
            shape_per_agent=(num_species,),
            neighbor_visible=True)

        # site_imported_seeds: imported seeds from dispersal (local read at P2)
        self.register_breed_local_array(
            "site_imported_seeds", breed=self._site_breed,
            shape_per_agent=(num_species,),
            neighbor_visible=False)

    def partition_sites(self, site_ids, strategy='round_robin'):
        """Compute site → rank mapping. Call before initialize_site_with_gaps().

        Strategies:
        - 'round_robin': site_ids[i] → rank i % num_workers (default)
        - Future: 'metis' using site connectivity graph

        :param site_ids: List of site IDs to partition
        :param strategy: Partition strategy name
        """
        self._site_partition = {}
        if strategy == 'round_robin':
            for i, sid in enumerate(site_ids):
                self._site_partition[sid] = i % num_workers
        else:
            raise ValueError(f"Unknown partition strategy: {strategy}")

    def initialize_site_with_gaps(self, site_id, num_gaps, maxtrees,
                                   data_dir="input_data", prefix="UVAFME2012",
                                   rank=None):
        """Initialize a site with all its gaps and trees on the same rank.

        Uses rank from partition_sites() if rank not specified.

        :param site_id: Site ID from CSV files
        :param num_gaps: Number of gaps to create for this site
        :param maxtrees: Max tree slots per gap
        :param data_dir: Directory containing UVAFME CSV files
        :param prefix: File prefix for CSV files
        :param rank: Target rank (overrides partition). If None, uses partition.
        :return: site_info dict
        """
        if rank is None:
            rank = self._site_partition.get(site_id)

        site = self.initialize_site(
            site_id=site_id, data_dir=data_dir, prefix=prefix, rank=rank)
        for _ in range(num_gaps):
            self.initialize_trees(site=site, maxtrees=maxtrees, rank=rank)
        return site

    # def compute_total_agents(self, site_ids, num_gaps, maxtrees,
    #                          data_dir="input_data", prefix="UVAFME2012"):
    #     """Compute total agent count and build site-to-agent-range mapping.

    #     This method calculates how many agents will be created across all sites
    #     and assigns deterministic agent ID ranges to each site.

    #     Agent hierarchy per site:
    #     - 1 site agent
    #     - num_gaps gap agents
    #     - num_gaps * (num_species + maxtrees) tree agents

    #     :param site_ids: List of site IDs to initialize
    #     :param num_gaps: Number of gaps per site
    #     :param maxtrees: Max tree slots per gap
    #     :param data_dir: Directory containing CSV files
    #     :param prefix: File prefix for CSV files
    #     :return: (total_agents, site_ranges) where site_ranges is dict:
    #              site_id -> {
    #                  'site_agent_id': int,
    #                  'gap_start': int,
    #                  'gap_end': int,
    #                  'tree_start': int,
    #                  'tree_end': int,
    #                  'num_species': int,
    #                  'rank': int
    #              }
    #     """
    #     # Load rangelist to get species per site
    #     base_path = os.path.join(
    #         os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    #         data_dir
    #     )
    #     range_file = os.path.join(base_path, f"{prefix}_rangelist.csv")

    #     # Read rangelist once to count species per site
    #     site_species_count = {}
    #     with open(range_file, 'r') as f:
    #         reader = csv.DictReader(f)
    #         for row in reader:
    #             if not row.get('site', '').strip():
    #                 continue
    #             site_id = int(row['site'])
    #             if site_id not in site_ids:
    #                 continue
    #             species_present = set()
    #             for col, val in row.items():
    #                 if col not in ('site', 'latitude', 'longitude'):
    #                     if val == '1':
    #                         species_present.add(col)
    #             # Filter to species in unique_species dict
    #             species_count = len([sp_code for sp_code in species_present
    #                                 if sp_code in self.unique_species])
    #             site_species_count[site_id] = species_count

    #     # Compute agent counts and ranges
    #     site_ranges = {}
    #     current_agent_id = 0

    #     for site_id in sorted(site_ids):
    #         num_species = site_species_count.get(site_id, 0)
    #         if num_species == 0:
    #             print(f"Warning: Site {site_id} has no species, skipping in agent count")
    #             continue

    #         # Site agent
    #         site_agent_id = current_agent_id
    #         current_agent_id += 1

    #         # Gap agents
    #         gap_start = current_agent_id
    #         gap_end = current_agent_id + num_gaps
    #         current_agent_id = gap_end

    #         # Tree agents (templates + free slots per gap)
    #         trees_per_gap = num_species + maxtrees
    #         total_trees = num_gaps * trees_per_gap
    #         tree_start = current_agent_id
    #         tree_end = current_agent_id + total_trees
    #         current_agent_id = tree_end

    #         # Get rank from partition
    #         rank = self._site_partition.get(site_id, 0)

    #         site_ranges[site_id] = {
    #             'site_agent_id': site_agent_id,
    #             'gap_start': gap_start,
    #             'gap_end': gap_end,
    #             'tree_start': tree_start,
    #             'tree_end': tree_end,
    #             'num_species': num_species,
    #             'rank': rank
    #         }

    #     total_agents = current_agent_id
    #     return total_agents, site_ranges

    # def build_agent_id_to_rank_array(self, total_agents, site_ranges):
    #     """Build global agent_id -> rank mapping array.

    #     All agents of a site (site, gaps, trees) are assigned to the same rank
    #     based on site-based partitioning.

    #     :param total_agents: Total number of agents across all sites
    #     :param site_ranges: Dict from compute_total_agents()
    #     :return: numpy array of shape (total_agents,) with rank assignments
    #     """
    #     agent_id_to_rank = np.zeros(total_agents, dtype=np.int32)

    #     for site_id, ranges in site_ranges.items():
    #         rank = ranges['rank']
    #         site_agent_id = ranges['site_agent_id']
    #         gap_start = ranges['gap_start']
    #         gap_end = ranges['gap_end']
    #         tree_start = ranges['tree_start']
    #         tree_end = ranges['tree_end']

    #         # Assign site agent
    #         agent_id_to_rank[site_agent_id] = rank

    #         # Assign gap agents
    #         agent_id_to_rank[gap_start:gap_end] = rank

    #         # Assign tree agents
    #         agent_id_to_rank[tree_start:tree_end] = rank

    #     return agent_id_to_rank

    # def initialize_sites_preallocate(self, site_ids, num_gaps, maxtrees,
    #                                   data_dir="input_data", prefix="UVAFME2012"):
    #     """Initialize multiple sites using preallocate+populate pattern.

    #     Uses SuperNeuroABM-style approach: pre-allocate all agent metadata globally,
    #     then populate only local agents using direct index writes.

    #     NOTE: This method calls populate_agent_at_index() for each agent individually,
    #     which adds function call overhead. See initialize_sites_bulk() for a faster
    #     approach using create_agents_bulk() with numpy arrays.

    #     :param site_ids: List of site IDs to initialize
    #     :param num_gaps: Number of gaps per site
    #     :param maxtrees: Max tree slots per gap
    #     :param data_dir: Directory containing CSV files
    #     :param prefix: File prefix for CSV files
    #     :return: dict mapping site_id -> site_agent_id
    #     """
    #     from mpi4py import MPI
    #     comm = MPI.COMM_WORLD
    #     rank = comm.Get_rank()

    #     # Phase 1: Compute total agents and build global partition
    #     total_agents, site_ranges = self.compute_total_agents(
    #         site_ids, num_gaps, maxtrees, data_dir, prefix
    #     )

    #     # Phase 2: Build agent_id -> rank mapping
    #     agent_id_to_rank = self.build_agent_id_to_rank_array(total_agents, site_ranges)

    #     # Phase 3: Pre-allocate all agent metadata globally
    #     self._agent_factory.preallocate_agent_metadata(total_agents, agent_id_to_rank)
    #     self.get_space().preallocate_location_containers(total_agents)

    #     # Get local index mapping for this rank
    #     rank_to_local = self._agent_factory._rank2agentid2agentidx[rank]

    #     # Load rangelist to get species data
    #     base_path = os.path.join(
    #         os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    #         data_dir
    #     )
    #     range_file = os.path.join(base_path, f"{prefix}_rangelist.csv")

    #     site_species_dict = {}
    #     with open(range_file, 'r') as f:
    #         reader = csv.DictReader(f)
    #         for row in reader:
    #             if not row.get('site', '').strip():
    #                 continue
    #             site_id = int(row['site'])
    #             if site_id not in site_ids:
    #                 continue
    #             species_present = set()
    #             for col, val in row.items():
    #                 if col not in ('site', 'latitude', 'longitude'):
    #                     if val == '1':
    #                         species_present.add(col)
    #             site_species = sorted(
    #                 [self.unique_species[sp_code] for sp_code in species_present
    #                  if sp_code in self.unique_species],
    #                 key=lambda sp: sp['global_id']
    #             )
    #             site_species_dict[site_id] = site_species

    #     # Phase 4: Populate only local agents using direct index writes
    #     site_agents = {}
    #     connections_to_make = []  # Store connections to apply after all agents created

    #     for site_id in sorted(site_ids):
    #         if site_id not in site_ranges:
    #             continue  # Site has no species, was skipped

    #         ranges = site_ranges[site_id]
    #         if ranges['rank'] != rank:
    #             continue  # Skip sites not on this rank

    #         # Get site data
    #         site_params, site_states, site_info = self._build_site_params_states(
    #             site_id, data_dir, prefix
    #         )
    #         site_species = site_species_dict.get(site_id, [])

    #         deg_days = site_info['deg_days']
    #         dry_days = site_info['dry_days']

    #         # Populate site agent
    #         site_agent_id = ranges['site_agent_id']
    #         local_idx = rank_to_local[site_agent_id]
    #         self._agent_factory.populate_agent_at_index(
    #             site_agent_id, local_idx, self._site_breed,
    #             params=site_params, states=site_states
    #         )
    #         self.site_agents.append(site_agent_id)
    #         site_agents[site_id] = site_agent_id

    #         # Build site_info with agent_id
    #         site_info['site_agent_id'] = site_agent_id
    #         site_info['rank'] = rank
    #         site_info['species'] = site_species
    #         site_info['gaps'] = []
    #         self.sites.append(site_info)

    #         # Populate gap agents
    #         gap_ids = list(range(ranges['gap_start'], ranges['gap_end']))
    #         for gap_idx, gap_agent_id in enumerate(gap_ids):
    #             gap_params, gap_states = self._build_gap_params_states(
    #                 gap_agent_id, deg_days, dry_days
    #             )
    #             local_idx = rank_to_local[gap_agent_id]
    #             self._agent_factory.populate_agent_at_index(
    #                 gap_agent_id, local_idx, self._gap_breed,
    #                 params=gap_params, states=gap_states
    #             )
    #             self.gap_agents.append(gap_agent_id)
    #             site_info['gaps'].append(gap_agent_id)

    #             # Store connection: site -> gap
    #             connections_to_make.append((site_agent_id, gap_agent_id, False))

    #         # Populate tree agents
    #         tree_idx = ranges['tree_start']
    #         for gap_agent_id in gap_ids:
    #             template_ids = []

    #             # Template trees (one per species)
    #             for species_info in site_species:
    #                 params, states = self._build_tree_params_states(
    #                     gap_agent_id, species_info, diam=0.0, age=0.0, is_alive=-1.0
    #                 )
    #                 local_idx = rank_to_local[tree_idx]
    #                 self._agent_factory.populate_agent_at_index(
    #                     tree_idx, local_idx, self._tree_breed,
    #                     params=params, states=states
    #                 )
    #                 self.tree_ids.append(tree_idx)
    #                 self.tree_to_gap[tree_idx] = gap_agent_id
    #                 template_ids.append(tree_idx)

    #                 # Store connection: gap -> tree
    #                 connections_to_make.append((gap_agent_id, tree_idx, False))

    #                 tree_idx += 1

    #             # Free tree slots
    #             placeholder_species = site_species[0]
    #             free_slot_ids = []
    #             for _ in range(maxtrees):
    #                 params, states = self._build_tree_params_states(
    #                     gap_agent_id, placeholder_species, diam=0.0, age=0.0, is_alive=0.0
    #                 )
    #                 local_idx = rank_to_local[tree_idx]
    #                 self._agent_factory.populate_agent_at_index(
    #                     tree_idx, local_idx, self._tree_breed,
    #                     params=params, states=states
    #                 )
    #                 self.tree_ids.append(tree_idx)
    #                 self.tree_to_gap[tree_idx] = gap_agent_id
    #                 free_slot_ids.append(tree_idx)

    #                 # Store connection: gap -> tree
    #                 connections_to_make.append((gap_agent_id, tree_idx, False))

    #                 tree_idx += 1

    #             # Store connections: free_slot -> templates (directed)
    #             for free_slot_id in free_slot_ids:
    #                 for template_id in template_ids:
    #                     connections_to_make.append((free_slot_id, template_id, True))

    #     # Phase 5: Apply all connections after agents are created
    #     for agent_0, agent_1, directed in connections_to_make:
    #         self.connect_agents(agent_0, agent_1, directed=directed)

    #     return site_agents

    # def initialize_sites_bulk(self, site_ids, num_gaps, maxtrees,
    #                            data_dir="input_data", prefix="UVAFME2012"):
    #     """Initialize multiple sites using bulk agent creation with numpy arrays.

    #     Creates agents in batches using create_agents_bulk() which takes numpy arrays
    #     and uses list.extend() for efficient property tensor updates. This avoids
    #     per-agent function call overhead.

    #     :param site_ids: List of site IDs to initialize
    #     :param num_gaps: Number of gaps per site
    #     :param maxtrees: Max tree slots per gap
    #     :param data_dir: Directory containing CSV files
    #     :param prefix: File prefix for CSV files
    #     :return: dict mapping site_id -> site_agent_id
    #     """
    #     from mpi4py import MPI
    #     comm = MPI.COMM_WORLD
    #     rank = comm.Get_rank()

    #     site_agents = {}

    #     # Load rangelist to get species per site
    #     base_path = os.path.join(
    #         os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    #         data_dir
    #     )
    #     range_file = os.path.join(base_path, f"{prefix}_rangelist.csv")

    #     # Read rangelist once
    #     site_species_dict = {}
    #     with open(range_file, 'r') as f:
    #         reader = csv.DictReader(f)
    #         for row in reader:
    #             if not row.get('site', '').strip():
    #                 continue
    #             site_id = int(row['site'])
    #             if site_id not in site_ids:
    #                 continue
    #             species_present = set()
    #             for col, val in row.items():
    #                 if col not in ('site', 'latitude', 'longitude'):
    #                     if val == '1':
    #                         species_present.add(col)
    #             site_species = sorted(
    #                 [self.unique_species[sp_code] for sp_code in species_present
    #                  if sp_code in self.unique_species],
    #                 key=lambda sp: sp['global_id']
    #             )
    #             site_species_dict[site_id] = site_species

    #     # Process each site
    #     for site_id in site_ids:
    #         my_rank = self._site_partition.get(site_id, 0)
    #         if my_rank != rank:
    #             continue

    #         # Get site data
    #         site_params, site_states, site_info = self._build_site_params_states(
    #             site_id, data_dir, prefix
    #         )
    #         site_species = site_species_dict.get(site_id, [])
    #         if len(site_species) == 0:
    #             print(f"Warning: Site {site_id} has no species, skipping")
    #             continue

    #         deg_days = site_info['deg_days']
    #         dry_days = site_info['dry_days']

    #         # Create site agent using bulk method
    #         site_agent_ids = self._agent_factory.create_agents_bulk(
    #             self._site_breed, rank, 1,
    #             params=np.array([site_params]),
    #             states=np.array([site_states])
    #         )
    #         site_agent_id = int(site_agent_ids[0])
    #         self.get_space().add_agents_bulk(1)
    #         self.site_agents.append(site_agent_id)
    #         site_agents[site_id] = site_agent_id

    #         # Build site_info with agent_id
    #         site_info['site_agent_id'] = site_agent_id
    #         site_info['rank'] = rank
    #         site_info['species'] = site_species
    #         site_info['gaps'] = []
    #         self.sites.append(site_info)

    #         # Build arrays for all gaps
    #         gap_params_list = []
    #         gap_states_list = []
    #         for gap_idx in range(num_gaps):
    #             gap_id = len(self.gap_agents) + gap_idx
    #             gap_params, gap_states = self._build_gap_params_states(gap_id, deg_days, dry_days)
    #             gap_params_list.append(gap_params)
    #             gap_states_list.append(gap_states)

    #         # Create all gap agents at once using bulk method
    #         gap_agent_ids = self._agent_factory.create_agents_bulk(
    #             self._gap_breed, rank, num_gaps,
    #             params=np.array(gap_params_list),
    #             states=np.array(gap_states_list)
    #         )
    #         self.get_space().add_agents_bulk(num_gaps)
    #         self.gap_agents.extend(gap_agent_ids.tolist())
    #         site_info['gaps'].extend(gap_agent_ids.tolist())

    #         # Connect site to gaps
    #         for gap_agent_id in gap_agent_ids:
    #             self.connect_agents(site_agent_id, int(gap_agent_id))

    #         # Build arrays for all trees in all gaps
    #         trees_per_gap = len(site_species) + maxtrees
    #         total_trees = num_gaps * trees_per_gap

    #         tree_params_list = []
    #         tree_states_list = []
    #         gap_to_tree_mapping = []  # Track which gap each tree belongs to

    #         for gap_idx, gap_agent_id in enumerate(gap_agent_ids):
    #             # Template trees (one per species)
    #             for species_info in site_species:
    #                 params, states = self._build_tree_params_states(
    #                     int(gap_agent_id), species_info, diam=0.0, age=0.0, is_alive=-1.0
    #                 )
    #                 tree_params_list.append(params)
    #                 tree_states_list.append(states)
    #                 gap_to_tree_mapping.append(int(gap_agent_id))

    #             # Free tree slots
    #             placeholder_species = site_species[0]
    #             for _ in range(maxtrees):
    #                 params, states = self._build_tree_params_states(
    #                     int(gap_agent_id), placeholder_species, diam=0.0, age=0.0, is_alive=0.0
    #                 )
    #                 tree_params_list.append(params)
    #                 tree_states_list.append(states)
    #                 gap_to_tree_mapping.append(int(gap_agent_id))

    #         # Create all tree agents at once using bulk method
    #         tree_agent_ids = self._agent_factory.create_agents_bulk(
    #             self._tree_breed, rank, total_trees,
    #             params=np.array(tree_params_list),
    #             states=np.array(tree_states_list)
    #         )
    #         self.get_space().add_agents_bulk(total_trees)
    #         self.tree_ids.extend(tree_agent_ids.tolist())

    #         # Update tree_to_gap mapping and connect trees
    #         for tree_agent_id, gap_agent_id in zip(tree_agent_ids, gap_to_tree_mapping):
    #             self.tree_to_gap[int(tree_agent_id)] = gap_agent_id
    #             self.connect_agents(gap_agent_id, int(tree_agent_id))

    #         # Connect free slots to templates (within each gap)
    #         tree_idx = 0
    #         for gap_idx in range(num_gaps):
    #             num_species = len(site_species)
    #             template_ids = tree_agent_ids[tree_idx:tree_idx + num_species]
    #             tree_idx += num_species
    #             free_slot_ids = tree_agent_ids[tree_idx:tree_idx + maxtrees]
    #             tree_idx += maxtrees

    #             for free_slot_id in free_slot_ids:
    #                 for template_id in template_ids:
    #                     self.connect_agents(int(free_slot_id), int(template_id), directed=True)

    #     return site_agents

    # def _build_site_params_states(self, site_id, data_dir="input_data", prefix="UVAFME2012"):
    #     """Extract params and states building logic from initialize_site().

    #     Returns (params, states, site_info_dict) tuple.
    #     """
    #     # Get site slot from globals mapping
    #     site_slot = self._site_id_to_slot.get(site_id)
    #     if site_slot is None:
    #         raise ValueError(f"Site {site_id} not found in globals. Call load_globals() first.")

    #     site_row = self._site_rows[site_id]
    #     climate_row = self._climate_rows.get(site_id)

    #     if climate_row is None:
    #         raise ValueError(f"Climate data for site {site_id} not found")

    #     # Calculate degree days from site_configs tensor
    #     BASE_TEMP = 5.0
    #     days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    #     deg_days = 0.0
    #     site_configs = self.get_global_property_value("site_configs")
    #     for i in range(12):
    #         tmin = site_configs[site_slot][int(Cfg.TMIN_BASE) + i]
    #         tmax = site_configs[site_slot][int(Cfg.TMAX_BASE) + i]
    #         tavg = (tmin + tmax) / 2.0
    #         if tavg > BASE_TEMP:
    #             deg_days += (tavg - BASE_TEMP) * days_per_month[i]

    #     dry_days = 0.0

    #     # Build params (mutable soil state)
    #     soil_rootdepth = 0.8
    #     params = [0.0] * SITE_PARAMS_SIZE
    #     params[SITE_P_A0_C] = float(site_row['soilAO_c0'])
    #     params[SITE_P_A0_N] = float(site_row['soilAO_n0'])
    #     params[SITE_P_A_C] = float(site_row['soilA_c0'])
    #     params[SITE_P_A_N] = float(site_row['soilA_n0'])
    #     params[SITE_P_BL_C] = float(site_row['sbase_c0'])
    #     params[SITE_P_BL_N] = float(site_row['sbase_n0'])
    #     params[int(SiteP.A0_W)] = float(site_row['soilAO_w0'])
    #     params[int(SiteP.A_W)] = float(site_row['soilA_w0']) * soil_rootdepth
    #     params[int(SiteP.BL_W)] = float(site_row['sbase_w0'])
    #     params[int(SiteP.LAI_W0)] = float(site_row['lai_w0'])
    #     params[int(SiteP.ANNUAL_RUNOFF)] = 0.0
    #     params[int(SiteP.SITE_ID)] = float(site_slot)

    #     # Build states (neighbor-visible fields)
    #     states = [0.0] * 8
    #     states[SITE_S_DEG_DAYS] = deg_days
    #     states[SITE_S_DRY_DAYS] = dry_days
    #     states[SITE_S_AVAIL_N] = 0.1
    #     states[SITE_S_FLOOD_DAYS] = 0.0
    #     states[int(SiteS.FIRE_INTENSITY)] = 0.0
    #     states[SITE_S_DRY_DAYS_BASE] = 0.0
    #     states[int(SiteS.WIND_INTENSITY)] = 0.0
    #     states[int(SiteS.SITE_ID)] = float(site_slot)

    #     # Build site_info dict
    #     elevation = float(site_row.get('elevation', 0))
    #     fire_prob = float(site_row.get('fire_prob', 0)) / 1000.0
    #     wind_prob = float(site_row.get('wind_prob', 0)) / 1000.0
    #     soil_base_h = float(site_row.get('soil_base_h', 70.0))
    #     region = site_row.get('region', '')

    #     site_info = {
    #         'site_id': site_id,
    #         'site_slot': site_slot,
    #         'site_name': site_row.get('name', f'Site_{site_id}'),
    #         'latitude': float(site_row['latitude']),
    #         'longitude': float(site_row['longitude']),
    #         'elevation': elevation,
    #         'slope': float(site_row['slope']),
    #         'region': region,
    #         'fire_prob': fire_prob,
    #         'wind_prob': wind_prob,
    #         'soil_base_h': soil_base_h,
    #         'deg_days': deg_days,
    #         'dry_days': dry_days,
    #     }

    #     return params, states, site_info

    # def _build_gap_params_states(self, gap_id, deg_days, dry_days):
    #     """Extract params and states building logic from initialize_gap().

    #     Returns (params, states) tuple.
    #     """
    #     params = [0.0] * 2
    #     params[int(GapP.SITE_ID)] = float(gap_id)
    #     params[int(GapP.TOTAL_N_DEMAND)] = 0.0

    #     states = [0.0] * GAP_STATES_SIZE
    #     states[int(GapS.DEG_DAYS)] = deg_days
    #     states[int(GapS.DRY_DAYS)] = dry_days
    #     states[int(GapS.AVAIL_N)] = 0.1
    #     states[int(GapS.N_SUPPLY_RATIO)] = 1.0

    #     return params, states

    # def _build_tree_params_states(self, gap_agent_id, species_info, diam, age, is_alive=1.0):
    #     """Extract params and states building logic from _create_tree_agent().

    #     Returns (params, states) tuple.
    #     """
    #     STD_HT = 1.3
    #     TC_KG = 0.039269908

    #     if is_alive > 0.5 and diam > 0:
    #         delta_ht = species_info['max_ht'] - STD_HT
    #         forska_ht = STD_HT + delta_ht * (
    #             1.0 - (2.71828 ** (-(species_info['arfa_0'] * diam / delta_ht)))
    #         )
    #         wood_bulk_dens = species_info['wood_bulk_dens']
    #         rootdepth = species_info.get('rootdepth', 0.8)
    #         canopy_ht = STD_HT

    #         if forska_ht > STD_HT:
    #             bd = forska_ht / (forska_ht - STD_HT) * diam
    #         else:
    #             bd = diam

    #         if forska_ht > canopy_ht and forska_ht > STD_HT:
    #             dc = (forska_ht - canopy_ht) / (forska_ht - STD_HT) * diam
    #         else:
    #             dc = diam

    #         stembc = TC_KG * wood_bulk_dens * 0.3 * bd * bd * forska_ht
    #         crown_depth = forska_ht - canopy_ht
    #         if crown_depth < 0.0:
    #             crown_depth = 0.0
    #         twigbc = TC_KG * wood_bulk_dens * 0.337 * dc * dc * crown_depth

    #         root_c = 0.0
    #         if forska_ht > 0.01:
    #             root_c = stembc * rootdepth / forska_ht + twigbc * 0.5

    #         biomC = stembc + twigbc + root_c
    #         biomN = biomC / 450.0
    #         leaf_bm = dc * dc * species_info['leafdiam_a'] * species_info['leafarea_c'] * 2.0 * 1000.0
    #     else:
    #         forska_ht = 0.0
    #         biomC = 0.0
    #         biomN = 0.0
    #         leaf_bm = 0.0

    #     # Build params
    #     params = [0.0] * TREE_PARAMS_SIZE
    #     params[int(TreeP.AGE)] = age
    #     params[int(TreeP.BIOMC)] = biomC
    #     params[int(TreeP.BIOMN)] = biomN
    #     params[int(TreeP.LEAF_BM)] = leaf_bm
    #     params[int(TreeP.X)] = 0.0
    #     params[int(TreeP.Y)] = 0.0
    #     params[int(TreeP.LIGHT_AVAIL)] = 1.0
    #     params[int(TreeP.FC_DEGDAY)] = 1.0
    #     params[int(TreeP.FC_DROUGHT)] = 1.0
    #     params[int(TreeP.FC_FLOOD)] = 1.0

    #     # Build states
    #     states = [0.0] * 11
    #     states[int(TreeS.IS_ALIVE)] = float(is_alive)
    #     states[int(TreeS.DIAM)] = diam
    #     states[int(TreeS.HEIGHT)] = forska_ht
    #     states[int(TreeS.CANOPY_HT)] = STD_HT if is_alive > 0.5 else 0.0
    #     states[int(TreeS.SPECIES_ID)] = float(species_info['global_id'])

    #     return params, states

    def connect_sites(self):
        """Connect site agents for inter-site seed dispersal.

        Computes haversine distance between each site pair. Only connects
        sites within DISPERSAL_CUTOFF_FACTOR * max(max_dispersal_dist).
        Stores pre-computed distances in site_distances global tensor.
        """
        import math

        num_sites = len(self.site_agents)
        if num_sites < 2:
            return

        # Get max dispersal distance across all species
        max_disp = 0.0
        for sp_info in self.unique_species.values():
            d = sp_info.get('max_dispersal_dist', 0.0)
            if d > max_disp:
                max_disp = d

        cutoff = DISPERSAL_CUTOFF_FACTOR * max_disp

        # Build distance matrix and connect qualifying pairs
        distances = np.zeros((self._num_sites_in_globals, self._num_sites_in_globals))
        for i in range(num_sites):
            for j in range(i + 1, num_sites):
                site_i = self.sites[i]
                site_j = self.sites[j]
                slot_i = site_i['site_slot']
                slot_j = site_j['site_slot']

                lat1 = site_i.get('latitude', 0.0)
                lon1 = site_i.get('longitude', 0.0)
                lat2 = site_j.get('latitude', 0.0)
                lon2 = site_j.get('longitude', 0.0)

                # Haversine distance in km
                lat1_r = math.radians(lat1)
                lat2_r = math.radians(lat2)
                dlat = math.radians(lat2 - lat1)
                dlon = math.radians(lon2 - lon1)
                a = (math.sin(dlat / 2.0) ** 2 +
                     math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2.0) ** 2)
                c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
                dist = EARTH_RADIUS_KM * c

                distances[slot_i, slot_j] = dist
                distances[slot_j, slot_i] = dist

                if cutoff > 0.0 and dist <= cutoff:
                    self.connect_agents(self.site_agents[i], self.site_agents[j])
                    print(f"  Connected site {site_i['site_id']} <-> site {site_j['site_id']} "
                          f"(distance: {dist:.1f} km, cutoff: {cutoff:.1f} km)")

        self.set_global_property_value("site_distances", distances)

    def get_species_count(self):
        return len(self.unique_species)

    def get_statistics(self):
        stats = {
            "total_slots": len(self.tree_ids),
            "living_trees": 0,
            "free_slots": 0,
            "template_trees": 0,
            "seedlings": 0,
            "total_biomass": 0.0,
        }

        for tree_id in self.tree_ids:
            tree_states = self.get_agent_property_value(tree_id, "states")
            alive = tree_states[TREE_S_IS_ALIVE] if isinstance(tree_states, list) else tree_states
            if alive > 0.5:
                stats["living_trees"] += 1
                params = self.get_agent_property_value(tree_id, "params")
                biomC = params[TREE_P_BIOMC] if isinstance(params, list) else 0.0
                age = params[TREE_P_AGE] if isinstance(params, list) else 0.0
                stats["total_biomass"] += biomC
                if age <= 2.0:
                    stats["seedlings"] += 1
            elif alive < -0.5:
                stats["template_trees"] += 1
            else:
                stats["free_slots"] += 1

        return stats

    def collect_tree_data(self):
        """Collect data for all living trees. Returns dict on rank 0, None otherwise.

        Species traits (evergreen) looked up from model's species dict,
        not from tree params.
        """
        params_np = self.get_breed_data("Tree", "params")
        states_np = self.get_breed_data("Tree", "states")
        agent_ids_np = self.get_breed_agent_ids("Tree")

        # Non-root ranks: get_breed_data returns None in multi-worker mode
        if params_np is None:
            return None

        alive_mask = states_np[:, TREE_S_IS_ALIVE] > 0.5
        alive_params = params_np[alive_mask]
        alive_states = states_np[alive_mask]
        alive_ids = agent_ids_np[alive_mask].astype(np.int32)

        gap_ids = np.array([self.tree_to_gap[int(a)] for a in alive_ids], dtype=np.int32)

        # Look up species traits from model's species dict
        species_ids = alive_states[:, TREE_S_SPECIES_ID].astype(np.int32)
        evergreen = np.array([
            self.species_by_id.get(int(sid), {}).get('evergreen', 0) > 0.5
            for sid in species_ids
        ], dtype=bool)

        return {
            'count': int(alive_mask.sum()),
            'gap_agent_id': gap_ids,
            'species_id': species_ids,
            'diam': alive_states[:, TREE_S_DIAM],
            'height': alive_states[:, TREE_S_HEIGHT],
            'biomC': alive_params[:, TREE_P_BIOMC],
            'biomN': alive_params[:, TREE_P_BIOMN],
            'leaf_bm': alive_params[:, TREE_P_LEAF_BM],
            'age': alive_params[:, TREE_P_AGE],
            'canopy_ht': alive_states[:, TREE_S_CANOPY_HT],
            'evergreen': evergreen,
        }

    def collect_site_data(self):
        """Collect site params and states."""
        site_id = self.site_agents[0]
        params = self.get_agent_property_value(site_id, "params")
        states = self.get_agent_property_value(site_id, "states")
        return params, states
