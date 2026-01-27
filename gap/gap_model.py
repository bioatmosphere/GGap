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
      - states: deg_days, dry_days, base_mortality, n_supply_ratio,
                flood_days, fire_intensity, num_to_recruit
    Reads from Tree neighbors:
      - states_db: is_alive, diam, height, canopy_ht (for light competition)
      - params: species traits (for recruitment, static data)
    Writes to own:
      - params: age, biomass, growth factors (internal)
      - states: litter_c, litter_n, n_demand
      - states_db: is_alive, diam, height, canopy_ht

  Priority 1 - Gap Aggregate Step (gap_aggregate_step):
    Reads from Tree neighbors:
      - states: litter_c, litter_n, n_demand
      - states_db: is_alive (count living/dormant)
      - params: invader, seed (for seed production)
    Writes to own:
      - params: total_n_demand (internal)
      - states: litter_accum_c/n, num_to_recruit, seed_bank

  Priority 2 - Site Soil Step (site_soil_step):
    Reads from Gap neighbors:
      - states: litter_accum_c, litter_accum_n
    Writes to own:
      - params: soil pools (A0/A/BL carbon, nitrogen, moisture)
      - states: avail_n, flood_days, fire_intensity

  Priority 3 - Gap Sync Step (gap_sync_step):
    Reads from Site neighbor:
      - states: deg_days, dry_days, base_mortality, avail_n,
                flood_days, fire_intensity
    Reads from own:
      - params: total_n_demand
    Writes to own:
      - states: climate (copied from Site), n_supply_ratio
    Clears:
      - states: litter_accum_c/n, num_to_recruit (consumed)

See docs/agent_properties.md for detailed property indices.
See docs/implementation_logic.md for step function details.

Property Arrays by Breed:
  Tree: params[29], states[3], states_db[4]
        (params includes species traits[19] + physiology[10])
  Gap:  params[2],  states[12], states_db[1]
        (states includes climate[5] + litter[2] + recruitment[3] + flood[1] + fire[1])
  Site: params[53], states[6], states_db[1]
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
# params[29]: species traits (static) + physiology (dynamic internal)
#   Species traits [0-18]:
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
TREE_P_INVADER = 10      # Colonization ability (for recruitment weighting)
TREE_P_SEED = 11         # Seed production rate
TREE_P_SPROUT = 12       # Sprouting ability
TREE_P_WOOD_BULK_DENS = 13  # Wood density (g/cm³)
TREE_P_LOWNUTR_TOL = 14     # Low nutrient tolerance (1-3)
TREE_P_FLOOD_TOL = 15       # Flood tolerance (1-6)
TREE_P_DROUGHT_TOL = 16     # Drought tolerance (1-5)
TREE_P_EVERGREEN = 17       # 1=evergreen/conifer, 0=deciduous
TREE_P_FIRE_TOL = 18        # Fire tolerance (1-6)
#   Physiology (internal) [19-28]:
TREE_P_AGE = 19
TREE_P_BIOMC = 20
TREE_P_BIOMN = 21
TREE_P_LEAF_BM = 22
TREE_P_X = 23
TREE_P_Y = 24
TREE_P_LIGHT_AVAIL = 25
TREE_P_FC_DEGDAY = 26
TREE_P_FC_DROUGHT = 27
TREE_P_FC_FLOOD = 28

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

# states[12]: climate + nutrients + litter_pool + recruitment + seed_bank + fire (public)
#   Trees read climate at P0, Site reads litter at P2
GAP_S_DEG_DAYS = 0
GAP_S_DRY_DAYS = 1
GAP_S_BASE_MORTALITY = 2
GAP_S_AVAIL_N = 3
GAP_S_N_SUPPLY_RATIO = 4
GAP_S_LITTER_ACCUM_C = 5
GAP_S_LITTER_ACCUM_N = 6
#   Recruitment info (dormant trees read at P0 next tick)
GAP_S_NUM_TO_RECRUIT = 7    # Number of dormant slots to activate this tick
GAP_S_RECRUIT_RAND_SEED = 8 # Random seed for species selection
GAP_S_FLOOD_DAYS = 9        # Annual flood days (from Site)
GAP_S_SEED_BANK = 10        # Accumulated seeds from previous years
GAP_S_FIRE_INTENSITY = 11   # Fire intensity this year (0-1, 0=no fire)

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

# states[6]: climate + available + flood_days + fire (public, Gap reads at P3)
SITE_S_DEG_DAYS = 0
SITE_S_DRY_DAYS = 1
SITE_S_BASE_MORTALITY = 2
SITE_S_AVAIL_N = 3
SITE_S_FLOOD_DAYS = 4  # Days when A layer is saturated
SITE_S_FIRE_INTENSITY = 5  # Fire intensity this year (0-1, 0=no fire)

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
        # params[29]: species traits + internal physiology (private)
        self._tree_breed.register_property("params", [0.0] * 29, neighbor_visible=False)
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
        # states[12]: climate + nutrients + litter_pool + recruitment + flood + seed_bank + fire (public)
        self._gap_breed.register_property("states", [0.0] * 12, neighbor_visible=True)
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
        # states[6]: climate + available + flood_days + fire (public, Gap reads at P3)
        self._site_breed.register_property("states", [0.0] * 6, neighbor_visible=True)
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
                        'invader': float(row['invader']),    # Colonization ability
                        'seed': float(row['seed']),          # Seed production rate
                        'sprout': float(row['sprout']),      # Sprouting ability
                        'lownutr_tol': int(row['n']),        # Low nutrient tolerance (1-3)
                        'flood_tol': int(row['f']),          # Flood tolerance (1-6)
                        'drought_tol': int(row['d']),        # Drought tolerance (1-5)
                        'evergreen': int(row['evergreen']),  # 1=evergreen/conifer, 0=deciduous
                        'fire_tol': int(row['fire']),        # Fire tolerance (1-6)
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

        # === Build states[6] for Site (climate + flood_days + fire - public) ===
        states = [0.0] * 6
        states[SITE_S_DEG_DAYS] = deg_days
        states[SITE_S_DRY_DAYS] = dry_days
        states[SITE_S_BASE_MORTALITY] = float(base_mortality_rate)
        states[SITE_S_AVAIL_N] = 0.1  # Initial available nitrogen
        states[SITE_S_FLOOD_DAYS] = 0.0  # Will be calculated during simulation
        states[SITE_S_FIRE_INTENSITY] = 0.0  # Fire intensity (calculated probabilistically)

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

        # Build states[12] for Gap (climate + nutrients + litter_pool + recruitment + flood + seed_bank + fire)
        states = [0.0] * 12
        states[GAP_S_DEG_DAYS] = deg_days
        states[GAP_S_DRY_DAYS] = dry_days
        states[GAP_S_BASE_MORTALITY] = 0.02  # Default base mortality
        states[GAP_S_AVAIL_N] = 0.1
        states[GAP_S_N_SUPPLY_RATIO] = 1.0
        states[GAP_S_LITTER_ACCUM_C] = 0.0
        states[GAP_S_LITTER_ACCUM_N] = 0.0
        states[GAP_S_NUM_TO_RECRUIT] = 0.0
        states[GAP_S_RECRUIT_RAND_SEED] = 0.0
        states[GAP_S_FLOOD_DAYS] = 0.0
        states[GAP_S_SEED_BANK] = 0.0
        states[GAP_S_FIRE_INTENSITY] = 0.0

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
        maxtrees=1000,
        initial_size_range=(5.0, 30.0),
    ):
        """
        Initialize trees in a gap (matches UVAFME initialize_forest).

        Pre-allocates maxtrees slots for the gap. Initial tree count is calculated
        using UVAFME formula: n_initial = max(1, int(2 * invader + 1)) per species.
        Remaining slots are dormant (is_alive = 0) for future recruitment.

        IMPORTANT - Template Trees (Species Pool Preservation):
            This function creates "template" trees (is_alive = -1) for each species
            in the site's species list. Templates are connected to the gap and other
            trees but never grow, die, or produce litter. They serve as permanent
            species trait references for the recruitment step function.

            Without templates, if all trees of a species die, that species would be
            permanently lost because:
            1. Dormant slots don't preserve species diversity (all use placeholder)
            2. Recruitment only copies traits from neighbor trees
            3. GPU kernels cannot access Python-side species data

            Template trees ensure the full species pool from CSV initialization
            remains available for recruitment throughout the simulation.

        If gap_agent_id is None, creates a new gap first via initialize_gap().
        Trees connect bidirectionally to gap and to each other.

        :param site: Site info dict from initialize_site()
        :param gap_agent_id: Optional existing gap agent ID
        :param maxtrees: Maximum tree slots to pre-allocate (default 1000)
        :param initial_size_range: DBH range for initial trees (default 5-30 cm)
        :return: Tuple of (all_tree_ids, initial_alive_count)
        """
        if maxtrees == 0:
            raise ValueError("Must have at least one tree slot")

        # Create gap if not provided
        if gap_agent_id is None:
            gap_agent_id = self.initialize_gap(site)

        site_species = site['species']
        num_species = len(site_species)

        if num_species == 0:
            raise ValueError("Site has no species")

        # Calculate initial tree count per species using UVAFME formula
        # n_initial = max(1, int(2 * invader + 1))
        initial_per_species = []
        total_initial = 0
        for species_info in site_species:
            invader = species_info.get('invader', 0.01)
            n_initial = max(1, int(2 * invader + 1))
            initial_per_species.append(n_initial)
            total_initial += n_initial

        # Cap total initial to maxtrees
        if total_initial > maxtrees:
            # Scale down proportionally
            scale = maxtrees / total_initial
            initial_per_species = [max(1, int(n * scale)) for n in initial_per_species]
            total_initial = sum(initial_per_species)

        created_trees = []
        template_trees = []
        initial_alive_count = 0

        # === Create template trees (one per species - permanent species references) ===
        # Templates have is_alive = -1, never grow/die, serve as species trait sources
        # for recruitment. This preserves the full species pool from CSV.
        for species_info in site_species:
            agent_id = self._create_tree_agent(
                gap_agent_id, species_info, diam=0.0, age=0.0, is_alive=-1.0
            )
            template_trees.append(agent_id)

        # === Create initial alive trees (distributed across species) ===
        for species_idx, species_info in enumerate(site_species):
            num_initial = initial_per_species[species_idx]

            for _ in range(num_initial):
                # Random initial size (mature trees)
                diam = random.uniform(initial_size_range[0], initial_size_range[1])
                age = random.uniform(10, 50)  # Established trees

                agent_id = self._create_tree_agent(
                    gap_agent_id, species_info, diam, age, is_alive=1.0
                )
                created_trees.append(agent_id)
                initial_alive_count += 1

        # === Create dormant tree slots (for future recruitment) ===
        dormant_count = maxtrees - initial_alive_count
        if dormant_count > 0:
            # Use first species as placeholder for dormant slots
            # (species traits will be copied from template/living tree when recruited)
            placeholder_species = site_species[0]

            for _ in range(dormant_count):
                agent_id = self._create_tree_agent(
                    gap_agent_id, placeholder_species, diam=0.0, age=0.0, is_alive=0.0
                )
                created_trees.append(agent_id)

        # Connect all trees to each other (within gap)
        # Include templates so they appear in neighbor lists for recruitment
        all_trees = template_trees + created_trees
        for i in range(len(all_trees)):
            for j in range(i + 1, len(all_trees)):
                self.connect_agents(all_trees[i], all_trees[j])

        return created_trees, initial_alive_count

    def _create_tree_agent(self, gap_agent_id, species_info, diam, age, is_alive=1.0):
        """
        Create a single tree agent with given species and size.

        :param gap_agent_id: Gap agent to connect to
        :param species_info: Species dict with traits
        :param diam: Diameter at breast height (cm)
        :param age: Tree age (years)
        :param is_alive: Tree state marker:
            -1.0 = template (species reference, never grows/dies)
             0.0 = dormant slot (available for recruitment)
             1.0 = alive (active tree)
        :return: agent_id
        """
        STD_HT = 1.3
        PI = 3.14159265359

        if is_alive > 0.5 and diam > 0:
            # Calculate height from diameter
            delta_ht = species_info['max_ht'] - STD_HT
            forska_ht = STD_HT + delta_ht * (
                1.0 - (2.71828 ** (-(species_info['arfa_0'] * diam / delta_ht)))
            )

            # Calculate biomass
            wood_bulk_dens = species_info['wood_bulk_dens']
            radius_m = diam / 200.0
            volume_m3 = PI * radius_m * radius_m * forska_ht
            biomC = volume_m3 * wood_bulk_dens * 1000.0 * 0.5
            biomN = biomC / 450.0
            leaf_bm = biomC * 0.1
        else:
            # Dormant slot - minimal values
            forska_ht = 0.0
            biomC = 0.0
            biomN = 0.0
            leaf_bm = 0.0

        # Build params[29] for Tree (species traits + physiology)
        params = [0.0] * 29
        # Species traits [0-18]
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
        params[TREE_P_INVADER] = float(species_info['invader'])
        params[TREE_P_SEED] = float(species_info['seed'])
        params[TREE_P_SPROUT] = float(species_info['sprout'])
        params[TREE_P_WOOD_BULK_DENS] = float(species_info['wood_bulk_dens'])
        params[TREE_P_LOWNUTR_TOL] = float(species_info['lownutr_tol'])
        params[TREE_P_FLOOD_TOL] = float(species_info['flood_tol'])
        params[TREE_P_DROUGHT_TOL] = float(species_info['drought_tol'])
        params[TREE_P_EVERGREEN] = float(species_info['evergreen'])
        params[TREE_P_FIRE_TOL] = float(species_info['fire_tol'])
        # Physiology [19-28]
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
        states_db[TREE_DB_IS_ALIVE] = float(is_alive)  # -1=template, 0=dormant, 1=alive
        states_db[TREE_DB_DIAM] = diam
        states_db[TREE_DB_HEIGHT] = forska_ht
        states_db[TREE_DB_CANOPY_HT] = STD_HT if is_alive > 0.5 else 0.0

        # Create tree agent
        agent_id = self.create_agent_of_breed(
            self._tree_breed,
            params=params,
            states=states,
            states_db=states_db,
        )

        self.tree_ids.append(agent_id)

        # Bidirectional connection: gap <-> tree
        self.connect_agents(gap_agent_id, agent_id)

        return agent_id

    def connect_agents(self, agent_0, agent_1):
        """Connect two agents bidirectionally."""
        self.get_space().connect_agents(agent_0, agent_1)

    def get_species_count(self):
        return len(self.unique_species)

    def get_statistics(self):
        stats = {
            "total_slots": len(self.tree_ids),
            "living_trees": 0,
            "dormant_slots": 0,
            "template_trees": 0,  # Species reference trees (is_alive = -1)
            "seedlings": 0,  # Trees with age <= 2 (recently recruited)
            "total_biomass": 0.0,
        }

        for tree_id in self.tree_ids:
            # Access states_db for is_alive
            states_db = self.get_agent_property_value(tree_id, "states_db")
            alive = states_db[TREE_DB_IS_ALIVE] if isinstance(states_db, list) else states_db
            if alive > 0.5:
                # Living tree (is_alive = 1)
                stats["living_trees"] += 1
                # Access params for biomC and age
                params = self.get_agent_property_value(tree_id, "params")
                biomC = params[TREE_P_BIOMC] if isinstance(params, list) else 0.0
                age = params[TREE_P_AGE] if isinstance(params, list) else 0.0
                stats["total_biomass"] += biomC
                # Count seedlings (recently recruited trees)
                if age <= 2.0:
                    stats["seedlings"] += 1
            elif alive < -0.5:
                # Template tree (is_alive = -1) - species reference, not counted as active
                stats["template_trees"] += 1
            else:
                # Dormant slot (is_alive = 0) - available for recruitment
                stats["dormant_slots"] += 1

        return stats
