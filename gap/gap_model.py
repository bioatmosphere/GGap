"""
GAPModel - GPU-accelerated gap dynamics model.
Integrates UVAFME processes with SAGESim agent framework.

Agent Hierarchy:
- Site agents: hold climate parameters and soil C/N pools
- Gap agents: relay between Site and Trees, aggregate tree outputs
- Tree agents: individual trees with species params and growth dynamics

Why Gap Agents Exist (Design Rationale):

    Without Gap agents, each Site would need to connect directly to all Trees.
    For a Site with 10 gaps × 100 trees/gap = 1000 tree neighbors. This creates
    two problems:

    1. MEMORY INEFFICIENCY: SAGESim stores neighbor lists per agent. Long neighbor
       lists waste GPU memory and slow down kernel execution (more data to load,
       more iterations in neighbor loops). By introducing Gap as an intermediary:
       - Site only has ~10 Gap neighbors (not 1000 Trees)
       - Each Gap has ~100 Tree neighbors (reasonable)
       - Total neighbor storage is much smaller

    2. FUTURE EXTENSIBILITY: Gap agents provide the foundation for modeling
       gap-level interactions within a site, such as:
       - Seed dispersal between adjacent gaps
       - Edge effects at gap boundaries
       - Gap-specific microclimate variations
       - Disturbance propagation across gaps

    The Gap agent acts as an aggregator (collecting litter from Trees at P1) and
    a relay (copying climate from Site to Trees at P3), keeping the neighbor
    graph sparse while enabling hierarchical data flow.

Network Connections (bidirectional):
- Site <-> Gap: each gap connects to its parent site
- Gap <-> Tree: each tree connects to its parent gap
- Tree <-> Tree: all trees within a gap connect to each other

Property Scheme (3 properties per breed):
- params:    neighbor_visible=False  (private data, not shared)
- states:    neighbor_visible=True, no double buffer (public, cross-priority reads)
- states_db: neighbor_visible=True, double buffered (public, same-priority reads)

Data Flow by Priority:

  Priority 0 - Tree Step (tree_step):
    Reads from Gap neighbor:
      - states: deg_days, dry_days, base_mortality, n_supply_ratio
    Reads from Tree neighbors:
      - states_db: is_alive, diam, height, canopy_ht (for light competition)
    Writes to own:
      - params: age, biomass, growth factors (internal)
      - states: litter_c, litter_n, n_demand
      - states_db: is_alive, diam, height, canopy_ht

  Priority 1 - Gap Aggregate Step (gap_aggregate_step):
    Reads from Tree neighbors:
      - states: litter_c, litter_n, n_demand
    Writes to own:
      - params: total_n_demand (internal)
      - states: litter_accum_c, litter_accum_n

  Priority 2 - Site Soil Step (site_soil_step):
    Reads from Gap neighbors:
      - states: litter_accum_c, litter_accum_n
    Writes to own:
      - params: soil pools (A0/A/BL carbon, nitrogen, moisture)
      - states: avail_n

  Priority 3 - Gap Sync Step (gap_sync_step):
    Reads from Site neighbor:
      - states: deg_days, dry_days, base_mortality, avail_n
    Writes to own:
      - states: climate (copied from Site), n_supply_ratio

Property Arrays by Breed:
  Tree: params[20], states[3], states_db[4]
  Gap:  params[2],  states[7], states_db[1]
  Site: params[53], states[4], states_db[1]
        (params includes soil pools[9] + monthly climate[36] + site properties[8])
"""

import sys
import os
import random
import csv

try:
    import sagesim
except ImportError:
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    _sagesim_path = os.path.join(os.path.dirname(_current_dir), "SAGESim")
    if _sagesim_path not in sys.path:
        sys.path.insert(0, _sagesim_path)

from sagesim.model import Model
from sagesim.space import NetworkSpace
from sagesim.breed import Breed
from pathlib import Path

# Import step functions from organized folder
from gap.step_functions.tree.tree_step import tree_step
from gap.step_functions.gap.gap_step import gap_aggregate_step, gap_sync_step
from gap.step_functions.site.soil_step import site_soil_step

CURRENT_DIR = Path(__file__).resolve().parent

# ============================================================
# Index constants (breed-specific interpretation)
# Each breed interprets its arrays starting from index 0
# ============================================================

# --- Tree breed ---
# params[20]: species traits (static) + physiology (dynamic internal)
#   Species traits [0-9]:
TREE_P_SPECIES_ID = 0
TREE_P_MAX_AGE = 1
TREE_P_MAX_DIAM = 2
TREE_P_MAX_HT = 3
TREE_P_ARFA_0 = 4
TREE_P_G = 5
TREE_P_SHADE_TOL = 6
TREE_P_DEG_DAY_MIN = 7
TREE_P_DEG_DAY_OPT = 8
TREE_P_DEG_DAY_MAX = 9
#   Physiology (internal) [10-19]:
TREE_P_AGE = 10
TREE_P_BIOMC = 11
TREE_P_BIOMN = 12
TREE_P_LEAF_BM = 13
TREE_P_X = 14
TREE_P_Y = 15
TREE_P_LIGHT_AVAIL = 16
TREE_P_FC_DEGDAY = 17
TREE_P_FC_DROUGHT = 18
TREE_P_FC_FLOOD = 19

# states[3]: litter output (public, Gap reads at P1)
TREE_S_LITTER_C = 0
TREE_S_LITTER_N = 1
TREE_S_N_DEMAND = 2

# states_db[4]: structure (public, Trees read at P0 for light competition)
TREE_DB_IS_ALIVE = 0
TREE_DB_DIAM = 1
TREE_DB_HEIGHT = 2
TREE_DB_CANOPY_HT = 3

# --- Gap breed ---
# params[2]: internal only
GAP_P_GAP_ID = 0
GAP_P_TOTAL_N_DEMAND = 1

# states[7]: climate + nutrients + litter_pool (public)
#   Trees read climate at P0, Site reads litter at P2
GAP_S_DEG_DAYS = 0
GAP_S_DRY_DAYS = 1
GAP_S_BASE_MORTALITY = 2
GAP_S_AVAIL_N = 3
GAP_S_N_SUPPLY_RATIO = 4
GAP_S_LITTER_ACCUM_C = 5
GAP_S_LITTER_ACCUM_N = 6

# states_db[1]: placeholder (not used, but keeps uniform signature)
GAP_DB_PLACEHOLDER = 0

# --- Site breed ---
# params[53]: soil pools + monthly climate + soil properties (private internal)
#   Soil pools [0-8]:
SITE_P_A0_C = 0
SITE_P_A0_N = 1
SITE_P_A_C = 2
SITE_P_A_N = 3
SITE_P_BL_C = 4
SITE_P_BL_N = 5
SITE_P_A0_W = 6
SITE_P_A_W = 7
SITE_P_BL_W = 8
#   Monthly climate [9-44]:
SITE_P_TMIN_BASE = 9   # tmin[0..11] at indices 9-20
SITE_P_TMAX_BASE = 21  # tmax[0..11] at indices 21-32
SITE_P_PRCP_BASE = 33  # prcp[0..11] at indices 33-44
#   Additional soil/site params [45-52]:
SITE_P_FIELD_CAP = 45   # A layer field capacity
SITE_P_PERM_WP = 46     # A layer permanent wilting point
SITE_P_SLOPE = 47       # Site slope (degrees)
SITE_P_SIGMA = 48       # Sigma parameter for water table
SITE_P_LAI = 49         # Leaf area index (updated from trees)
SITE_P_LAI_W0 = 50      # Canopy water content
SITE_P_LATITUDE = 51    # Latitude for PET calculation
SITE_P_RAIN_N = 52      # Accumulated atmospheric N from precipitation

SITE_PARAMS_SIZE = 53

# states[4]: climate + available (public, Gap reads at P3)
SITE_S_DEG_DAYS = 0
SITE_S_DRY_DAYS = 1
SITE_S_BASE_MORTALITY = 2
SITE_S_AVAIL_N = 3

# states_db[1]: placeholder (not used)
SITE_DB_PLACEHOLDER = 0


class GAPModel(Model):
    """
    GAPModel class for gap dynamics simulation.

    Agent Structure:
    - Site: climate parameters (public) and soil pools (private)
    - Gap: relay/aggregator between Site and Trees
    - Tree: species traits (private), structure (public), litter output (public)

    Property Scheme:
    - params:    neighbor_visible=False  (private)
    - states:    neighbor_visible=True, no double buffer (public, cross-priority)
    - states_db: neighbor_visible=True, double buffered (public, same-priority)

    Breed IDs (registration order):
    - BREED_TREE = 0
    - BREED_GAP = 1
    - BREED_SITE = 2

    Data Flow (per tick):
    1. Trees (priority 0): read Gap states, compute growth, output litter
    2. Gap (priority 1): aggregate litter from Trees
    3. Site (priority 2): read litter from Gap, decompose soil, output avail_n
    4. Gap (priority 3): read climate+avail_n from Site, compute n_supply_ratio
    """

    def __init__(self) -> None:
        space = NetworkSpace()
        super().__init__(space)

        # === Create Tree breed (breed_id = 0) ===
        self._tree_breed = Breed("Tree")
        # params[20]: species traits + internal physiology (private)
        self._tree_breed.register_property("params", [0.0] * 20, neighbor_visible=False)
        # states[3]: litter output (public, Gap reads at P1)
        self._tree_breed.register_property("states", [0.0] * 3, neighbor_visible=True)
        # states_db[4]: structure (public, other Trees read at P0)
        self._tree_breed.register_property("states_db", [0.0] * 4, neighbor_visible=True)
        self._tree_breed.register_step_func(
            tree_step,
            CURRENT_DIR / "step_functions" / "tree" / "tree_step.py",
            priority=0,
            no_double_buffer=["params", "states"],
        )
        self.register_breed(breed=self._tree_breed)

        # === Create Gap breed (breed_id = 1) ===
        # Gap serves as intermediary between Site and Trees to:
        # (1) Keep neighbor lists short (Site has few Gap neighbors vs many Tree neighbors)
        # (2) Enable future gap-to-gap interactions (seed dispersal, edge effects)
        self._gap_breed = Breed("Gap")
        # params[2]: internal only (private)
        self._gap_breed.register_property("params", [0.0] * 2, neighbor_visible=False)
        # states[7]: climate + nutrients + litter_pool (public)
        self._gap_breed.register_property("states", [0.0] * 7, neighbor_visible=True)
        # states_db[1]: placeholder (public but unused)
        self._gap_breed.register_property("states_db", [0.0] * 1, neighbor_visible=True)
        self._gap_breed.register_step_func(
            gap_aggregate_step,
            CURRENT_DIR / "step_functions" / "gap" / "gap_step.py",
            priority=1,
            no_double_buffer=["params", "states"],
        )
        self._gap_breed.register_step_func(
            gap_sync_step,
            CURRENT_DIR / "step_functions" / "gap" / "gap_step.py",
            priority=3,
            no_double_buffer=["params", "states"],
        )
        self.register_breed(breed=self._gap_breed)

        # === Create Site breed (breed_id = 2) ===
        self._site_breed = Breed("Site")
        # params[53]: soil pools + monthly climate + soil properties (private)
        self._site_breed.register_property("params", [0.0] * SITE_PARAMS_SIZE, neighbor_visible=False)
        # states[4]: climate + available (public, Gap reads at P3)
        self._site_breed.register_property("states", [0.0] * 4, neighbor_visible=True)
        # states_db[1]: placeholder (public but unused)
        self._site_breed.register_property("states_db", [0.0] * 1, neighbor_visible=True)
        self._site_breed.register_step_func(
            site_soil_step,
            CURRENT_DIR / "step_functions" / "site" / "soil_step.py",
            priority=2,
            no_double_buffer=["params", "states"],
        )
        self.register_breed(breed=self._site_breed)

        # Species registry (deduplicated across sites)
        self.unique_species = {}

        # Track agents
        self.sites = []  # site info dicts
        self.site_agents = []  # site agent IDs
        self.gap_agents = []  # gap agent IDs
        self.tree_ids = []  # tree agent IDs

    def initialize_site(
        self,
        site_id: int = 0,
        data_dir: str = "input_data",
        prefix: str = "UVAFME2012",
        base_mortality_rate: float = 0.02,
    ):
        """
        Initialize a site from UVAFME CSV files.

        Reads from {data_dir}/{prefix}_*.csv:
          - site.csv: soil pools, water properties, location
          - climate.csv: monthly tmin/tmax/precip → calculates deg_days
          - specieslist.csv: species parameters
          - rangelist.csv: filters species present at this site

        Matches UVAFME's initialize_site() naming convention.

        :param site_id: Row index in CSV files (default: 0)
        :param data_dir: Directory containing CSV files
        :param prefix: File prefix (e.g., "UVAFME2012")
        :param base_mortality_rate: Base annual mortality rate
        :return: site_info dict with site_agent_id
        """
        base_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            data_dir
        )

        # === Load site data (soil pools, location) ===
        site_file = os.path.join(base_path, f"{prefix}_site.csv")
        site_row = None
        with open(site_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if int(row['site']) == site_id:
                    site_row = row
                    break

        if site_row is None:
            raise ValueError(f"Site {site_id} not found in {site_file}")

        # === Load climate data (monthly temp/precip) ===
        climate_file = os.path.join(base_path, f"{prefix}_climate.csv")
        climate_row = None
        with open(climate_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if int(row['site']) == site_id:
                    climate_row = row
                    break

        if climate_row is None:
            raise ValueError(f"Site {site_id} not found in {climate_file}")

        # Calculate degree days from monthly temperatures (base temp = 5°C)
        BASE_TEMP = 5.0
        deg_days = 0.0
        months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                  'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

        for i, month in enumerate(months):
            tmin = float(climate_row[f'tmin_{month}'])
            tmax = float(climate_row[f'tmax_{month}'])
            tavg = (tmin + tmax) / 2.0
            if tavg > BASE_TEMP:
                deg_days += (tavg - BASE_TEMP) * days_per_month[i]

        # Estimate dry days from precipitation (simplified)
        # UVAFME uses daily soil water balance; this is a rough approximation
        total_precip = sum(float(climate_row[f'prcp_{m}']) for m in months)
        # precip values appear to be in mm/10, so total_precip/10 = annual mm
        annual_precip_mm = total_precip / 10.0
        # Rough estimate: fewer dry days with more precipitation
        dry_days = max(0.0, 120.0 - annual_precip_mm / 20.0)

        # === Load species range list (which species present at this site) ===
        range_file = os.path.join(base_path, f"{prefix}_rangelist.csv")
        species_present = set()
        with open(range_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if int(row['site']) == site_id:
                    # All columns except site, latitude, longitude are species codes
                    for col, val in row.items():
                        if col not in ('site', 'latitude', 'longitude'):
                            if val == '1':
                                species_present.add(col)
                    break

        # === Load species data (filtered by range) ===
        species_file = os.path.join(base_path, f"{prefix}_specieslist.csv")
        site_species = []
        with open(species_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                species_code = row['Species_code']

                # Skip species not present at this site
                if species_code not in species_present:
                    continue

                if species_code not in self.unique_species:
                    global_id = len(self.unique_species)
                    self.unique_species[species_code] = {
                        'global_id': global_id,
                        'species_code': species_code,
                        'common_name': row['Common name'].replace('_', ' ').strip("'"),
                        'max_age': float(row['AGEmax']),
                        'max_diam': float(row['DBHmax']),
                        'max_ht': float(row['Hmax']),
                        'arfa_0': float(row['s']),
                        'g': float(row['g']),
                        'shade_tol': int(row['l']),
                        'deg_day_min': float(row['DEGDmin']),
                        'deg_day_opt': float(row['DEGDoptimum']),
                        'deg_day_max': float(row['DEGDmax']),
                        'wood_bulk_dens': float(row['bulk']),
                    }

                site_species.append(self.unique_species[species_code])

        # === Build params[53] for Site (soil pools + monthly climate + site properties) ===
        params = [0.0] * SITE_PARAMS_SIZE

        # Soil pools [0-8]
        params[SITE_P_A0_C] = float(site_row['soilAO_c0'])    # Litter layer carbon
        params[SITE_P_A0_N] = float(site_row['soilAO_n0'])    # Litter layer nitrogen
        params[SITE_P_A_C] = float(site_row['soilA_c0'])      # Humus layer carbon
        params[SITE_P_A_N] = float(site_row['soilA_n0'])      # Humus layer nitrogen
        params[SITE_P_BL_C] = float(site_row['sbase_c0'])     # Base layer carbon
        params[SITE_P_BL_N] = float(site_row['sbase_n0'])     # Base layer nitrogen
        params[SITE_P_A0_W] = float(site_row['soilAO_w0'])    # A0 moisture
        params[SITE_P_A_W] = float(site_row['soilA_w0'])      # A moisture
        params[SITE_P_BL_W] = float(site_row['sbase_w0'])     # Base moisture

        # Monthly climate [9-44] from climate CSV
        for i, month in enumerate(months):
            params[SITE_P_TMIN_BASE + i] = float(climate_row[f'tmin_{month}'])
            params[SITE_P_TMAX_BASE + i] = float(climate_row[f'tmax_{month}'])
            params[SITE_P_PRCP_BASE + i] = float(climate_row[f'prcp_{month}']) / 10.0  # Convert to cm

        # Additional soil/site parameters [45-52] from site CSV
        params[SITE_P_FIELD_CAP] = float(site_row['soilA_field_cap'])
        params[SITE_P_PERM_WP] = float(site_row['soilA_perm_wp'])
        params[SITE_P_SLOPE] = float(site_row['slope'])
        params[SITE_P_SIGMA] = float(site_row['sigma'])
        params[SITE_P_LAI] = float(site_row['lai'])
        params[SITE_P_LAI_W0] = float(site_row['lai_w0'])
        params[SITE_P_LATITUDE] = float(site_row['latitude'])
        params[SITE_P_RAIN_N] = 0.0  # Will accumulate during simulation

        # === Build states[4] for Site (climate - public) ===
        states = [0.0] * 4
        states[SITE_S_DEG_DAYS] = deg_days
        states[SITE_S_DRY_DAYS] = dry_days
        states[SITE_S_BASE_MORTALITY] = float(base_mortality_rate)
        states[SITE_S_AVAIL_N] = 0.1  # Initial available nitrogen

        # Create site agent
        site_agent_id = self.create_agent_of_breed(
            self._site_breed,
            params=params,
            states=states,
            states_db=[0.0] * 1,
        )
        self.site_agents.append(site_agent_id)

        site_info = {
            'site_id': site_id,
            'site_agent_id': site_agent_id,
            'site_name': site_row.get('name', f'Site_{site_id}'),
            'latitude': float(site_row['latitude']),
            'longitude': float(site_row['longitude']),
            'elevation': float(site_row.get('elevation', 0)),
            'species': site_species,
            'deg_days': deg_days,
            'dry_days': dry_days,
            'gaps': [],  # gap agent IDs for this site
        }
        self.sites.append(site_info)

        return site_info

    def initialize_gap(self, site):
        """
        Initialize a gap agent and connect it to the site (matches UVAFME initialize_plot).

        Gap copies climate from site into its states (trees read from gap).
        Returns gap_agent_id.
        """
        gap_id = len(self.gap_agents)
        site_agent_id = site['site_agent_id']

        # Get climate from site_info (calculated from CSV)
        deg_days = site.get('deg_days', 2500.0)
        dry_days = site.get('dry_days', 30.0)

        # Build params[2] for Gap (internal only)
        params = [0.0] * 2
        params[GAP_P_GAP_ID] = float(gap_id)
        params[GAP_P_TOTAL_N_DEMAND] = 0.0

        # Build states[7] for Gap (climate + nutrients + litter_pool)
        states = [0.0] * 7
        states[GAP_S_DEG_DAYS] = deg_days
        states[GAP_S_DRY_DAYS] = dry_days
        states[GAP_S_BASE_MORTALITY] = 0.02  # Default base mortality
        states[GAP_S_AVAIL_N] = 0.1
        states[GAP_S_N_SUPPLY_RATIO] = 1.0
        states[GAP_S_LITTER_ACCUM_C] = 0.0
        states[GAP_S_LITTER_ACCUM_N] = 0.0

        gap_agent_id = self.create_agent_of_breed(
            self._gap_breed,
            params=params,
            states=states,
            states_db=[0.0] * 1,
        )
        self.gap_agents.append(gap_agent_id)
        site['gaps'].append(gap_agent_id)

        # Bidirectional connection: site <-> gap
        self.connect_agents(site_agent_id, gap_agent_id)

        return gap_agent_id

    def initialize_trees(
        self,
        site,
        gap_agent_id=None,
        maxtrees=100,
        age_range=(5, 50),
        size_range=(3.0, 25.0),
    ):
        """
        Initialize trees in a gap (matches UVAFME initialize_forest).

        If gap_agent_id is None, creates a new gap first via initialize_gap().
        Trees connect bidirectionally to gap and to each other.
        Trees read climate from gap neighbor (not stored on tree).
        """
        if maxtrees == 0:
            raise ValueError("Must have at least one tree")

        # Create gap if not provided
        if gap_agent_id is None:
            gap_agent_id = self.initialize_gap(site)

        site_species = site['species']
        num_species = len(site_species)

        if num_species == 0:
            raise ValueError("Site has no species")

        # Distribute trees across species
        random_weights = [random.random() for _ in range(num_species)]
        total_weight = sum(random_weights)
        proportions = [w / total_weight for w in random_weights]

        trees_per_species = []
        remaining = maxtrees
        for prop in proportions[:-1]:
            count = int(maxtrees * prop)
            trees_per_species.append(count)
            remaining -= count
        trees_per_species.append(remaining)

        created_trees = []

        for species_idx, species_info in enumerate(site_species):
            num_trees = trees_per_species[species_idx]

            for _ in range(num_trees):
                age = random.uniform(age_range[0], age_range[1])
                diam = random.uniform(size_range[0], size_range[1])

                # Calculate height
                STD_HT = 1.3
                delta_ht = species_info['max_ht'] - STD_HT
                forska_ht = STD_HT + delta_ht * (
                    1.0 - (2.71828 ** (-(species_info['arfa_0'] * diam / delta_ht)))
                )

                # Calculate biomass
                wood_bulk_dens = species_info['wood_bulk_dens']
                PI = 3.14159265359
                radius_m = diam / 200.0
                volume_m3 = PI * radius_m * radius_m * forska_ht
                biomC = volume_m3 * wood_bulk_dens * 1000.0 * 0.5
                biomN = biomC / 450.0
                leaf_bm = biomC * 0.1

                # Build params[20] for Tree (species traits + physiology)
                params = [0.0] * 20
                # Species traits [0-9]
                params[TREE_P_SPECIES_ID] = float(species_info['global_id'])
                params[TREE_P_MAX_AGE] = float(species_info['max_age'])
                params[TREE_P_MAX_DIAM] = float(species_info['max_diam'])
                params[TREE_P_MAX_HT] = float(species_info['max_ht'])
                params[TREE_P_ARFA_0] = float(species_info['arfa_0'])
                params[TREE_P_G] = float(species_info['g'])
                params[TREE_P_SHADE_TOL] = float(species_info['shade_tol'])
                params[TREE_P_DEG_DAY_MIN] = float(species_info['deg_day_min'])
                params[TREE_P_DEG_DAY_OPT] = float(species_info['deg_day_opt'])
                params[TREE_P_DEG_DAY_MAX] = float(species_info['deg_day_max'])
                # Physiology [10-19]
                params[TREE_P_AGE] = age
                params[TREE_P_BIOMC] = biomC
                params[TREE_P_BIOMN] = biomN
                params[TREE_P_LEAF_BM] = leaf_bm
                params[TREE_P_X] = 0.0
                params[TREE_P_Y] = 0.0
                params[TREE_P_LIGHT_AVAIL] = 1.0
                params[TREE_P_FC_DEGDAY] = 1.0
                params[TREE_P_FC_DROUGHT] = 1.0
                params[TREE_P_FC_FLOOD] = 1.0

                # Build states[3] for Tree (litter output)
                states = [0.0] * 3

                # Build states_db[4] for Tree (structure)
                states_db = [0.0] * 4
                states_db[TREE_DB_IS_ALIVE] = 1.0
                states_db[TREE_DB_DIAM] = diam
                states_db[TREE_DB_HEIGHT] = forska_ht
                states_db[TREE_DB_CANOPY_HT] = 1.3

                # Create tree agent
                agent_id = self.create_agent_of_breed(
                    self._tree_breed,
                    params=params,
                    states=states,
                    states_db=states_db,
                )

                self.tree_ids.append(agent_id)
                created_trees.append(agent_id)

                # Bidirectional connection: gap <-> tree
                self.connect_agents(gap_agent_id, agent_id)

        # Connect all trees to each other (within gap)
        for i in range(len(created_trees)):
            for j in range(i + 1, len(created_trees)):
                self.connect_agents(created_trees[i], created_trees[j])

        return created_trees

    def connect_agents(self, agent_0, agent_1):
        """Connect two agents bidirectionally."""
        self.get_space().connect_agents(agent_0, agent_1)

    def get_species_count(self):
        return len(self.unique_species)

    def get_statistics(self):
        stats = {
            "total_trees": len(self.tree_ids),
            "living_trees": 0,
            "dead_trees": 0,
            "total_biomass": 0.0,
        }

        for tree_id in self.tree_ids:
            # Access states_db for is_alive
            states_db = self.get_agent_property_value(tree_id, "states_db")
            alive = states_db[TREE_DB_IS_ALIVE] if isinstance(states_db, list) else states_db
            if alive > 0.5:
                stats["living_trees"] += 1
                # Access params for biomC
                params = self.get_agent_property_value(tree_id, "params")
                biomC = params[TREE_P_BIOMC] if isinstance(params, list) else 0.0
                stats["total_biomass"] += biomC
            else:
                stats["dead_trees"] += 1

        return stats
