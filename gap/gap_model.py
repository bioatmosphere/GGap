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

Data Flow by Priority (matching GAPpy: soil first, two-phase tree growth):

  Priority 0 - Gap Litter Aggregate (gap_litter_aggregate_step):
    Reads from Tree neighbors:
      - states: litter_c, litter_n (from P6 of previous tick)
      - states_db: is_alive (count living/free)
      - params: env_stress from templates (regrowth/growmax)
    Writes to own:
      - states: litter_accum_c/n, num_to_recruit

  Priority 1 - Site Soil Step (site_soil_step):
    Reads from Gap neighbors:
      - states: litter_accum_c, litter_accum_n (from P0, same tick)
    Writes to own:
      - params: soil pools (A0/A/BL carbon, nitrogen, moisture)
      - states: avail_n, flood_days, fire_intensity

  Priority 2 - Tree Potential Growth (tree_potential_growth_step):
    Reads from Gap neighbor:
      - states: deg_days, dry_days, flood_days
    Reads from Tree neighbors:
      - states_db: is_alive, diam, height, canopy_ht (for light competition)
    Writes to own:
      - params: env_stress, diam_max, light_avail, fc_degday/drought/flood
      - states: n_demand (for Gap to aggregate at P3)

  Priority 3 - Gap N Demand Aggregate (gap_demand_aggregate_step):
    Reads from Tree neighbors:
      - states: n_demand (from P2, same tick)
    Writes to own:
      - params: total_n_demand (internal)
      - states: total_n_demand (public, Site reads at P4)

  Priority 4 - Site Nutrient Allocation (site_nutrient_step):
    Reads from own:
      - states: avail_n (from P1, same tick)
    Reads from Gap neighbors:
      - states: total_n_demand (from P3, same tick)
    Writes to own:
      - states: n_supply_ratio (Gap reads at P5)

  Priority 5 - Gap Sync Step (gap_sync_step):
    Reads from Site neighbor:
      - states: climate, n_supply_ratio (from P1/P4, same tick)
    Writes to own:
      - states: climate (copied from Site), n_supply_ratio (relayed)
    Clears:
      - states: litter_accum_c/n, num_to_recruit (consumed)

  Priority 6 - Tree Actual Growth + Renewal (tree_actual_growth_step):
    Living trees:
      Reads from own params (written at P2, same tick):
        - env_stress, diam_max, light_avail
      Reads from Gap neighbor:
        - states: n_supply_ratio (from P5, same tick), fire_intensity
      Writes to own:
        - params: age, biomC, biomN, leaf_bm
        - states: litter_c, litter_n
        - states_db: is_alive, diam, height, canopy_ht
    Templates:
      Reads from Gap neighbor: climate, n_supply_ratio (same tick)
      Reads from Tree neighbors: structure (for light), species_id (for avail_spec)
      Writes to own:
        - params: seedbank, seedling, env_stress (regrowth)
        - states_db: seedling_weight (free slots read same priority)
    Free slots:
      Reads from Gap neighbor: num_to_recruit, recruit_rand_seed (from P0)
      Reads from Template neighbors: seedling_weight (from states_db)
      Writes to own:
        - params: species traits, physiology (seedling init)
        - states_db: is_alive=1, diam, height, canopy_ht (visible next tick)

See docs/agent_properties.md for detailed property indices.
See docs/implementation_logic.md for step function details.

Property Arrays by Breed:
  Tree: params[40], states[5], states_db[5]
        (params includes species traits[22] + physiology[10] + intermediates[2] + renewal[4] + leaf_area[2])
  Gap:  params[2],  states[14], states_db[1]
        (states includes climate[5] + litter[4] + recruitment[3] + flood[1] + fire[1] + n_demand[1])
  Site: params[116], states[6], states_db[1]
        (params includes soil pools[9] + monthly climate[36] + site properties[8] + fire/wind/soil[3] + climate_std[36] + lapse_rates[24])
"""

import sys
import os
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
from gap.step_functions.gap.gap_litter_aggregate_step import gap_litter_aggregate_step
from gap.step_functions.site.soil_step import site_soil_step
from gap.step_functions.tree.tree_potential_growth_step import tree_potential_growth_step
from gap.step_functions.gap.gap_demand_aggregate_step import gap_demand_aggregate_step
from gap.step_functions.site.site_nutrient_step import site_nutrient_step
from gap.step_functions.gap.gap_sync_step import gap_sync_step
from gap.step_functions.tree.tree_actual_growth_step import tree_actual_growth_step

CURRENT_DIR = Path(__file__).resolve().parent

# ============================================================
# Index constants (breed-specific interpretation)
# Each breed interprets its arrays starting from index 0
# ============================================================

# --- Tree breed ---
# params[40]: species traits (static) + physiology (dynamic) + intermediates (P2->P6) + renewal (template-only) + leaf_area
#   Species traits [0-21]:
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
TREE_P_ROOTDEPTH = 19       # Root depth (m), default 2.0 (GAPpy)
TREE_P_STRESS_TOL = 20      # Stress tolerance (1-5), from CSV 'stress' column
TREE_P_AGE_TOL = 21         # Age tolerance (1-3), from CSV 'old' column
#   Physiology (internal) [22-31]:
TREE_P_AGE = 22
TREE_P_BIOMC = 23
TREE_P_BIOMN = 24
TREE_P_LEAF_BM = 25
TREE_P_X = 26
TREE_P_Y = 27
TREE_P_LIGHT_AVAIL = 28
TREE_P_FC_DEGDAY = 29
TREE_P_FC_DROUGHT = 30
TREE_P_FC_FLOOD = 31
#   Intermediates [32-33] (P2 writes, P6 reads, same tick via no_double_buffer):
TREE_P_ENV_STRESS = 32      # Composite env stress (no nutrient), P2 -> P6
TREE_P_DIAM_MAX_CALC = 33   # Max diameter increment for current size, P2 -> P6
#   Renewal params [34-37] (template-only: seedbank/seedling dynamics):
TREE_P_SEED_SURV = 34       # Seedbank survival rate (CSV 'NDE')
TREE_P_SEEDLING_LG = 35     # Seedling annual survival rate (CSV 'NDS')
TREE_P_SEEDBANK = 36        # Persistent seedbank (template-only)
TREE_P_SEEDLING = 37        # Persistent seedling pool (template-only)
#   Leaf area params [38-39]:
TREE_P_LEAFDIAM_A = 38     # Leaf diameter (adjusted by shade_tol)
TREE_P_LEAFAREA_C = 39     # Leaf area coefficient (normalized by HEC_TO_M2)

# states[5]: litter output (public, Gap reads at P0)
TREE_S_LITTER_C = 0      # Above-ground litter carbon
TREE_S_LITTER_N = 1      # Above-ground litter nitrogen
TREE_S_N_DEMAND = 2
TREE_S_LITTER_C_BG = 3   # Below-ground litter carbon (roots)
TREE_S_LITTER_N_BG = 4   # Below-ground litter nitrogen (roots)

# states_db[5]: structure (public, Trees read at P0 for light competition) + renewal weight
TREE_DB_IS_ALIVE = 0
TREE_DB_DIAM = 1
TREE_DB_HEIGHT = 2
TREE_DB_CANOPY_HT = 3
TREE_DB_SEEDLING_WEIGHT = 4  # seedling * regrowth (written by templates at P6, read by free slots at P2)

# --- Gap breed ---
# params[2]: internal only
GAP_P_GAP_ID = 0
GAP_P_TOTAL_N_DEMAND = 1

# states[14]: climate + nutrients + litter_pool + recruitment + seed_bank + fire + n_demand + bg_litter (public)
#   Trees read climate at P2, Site reads litter at P1, Site reads n_demand at P4
GAP_S_DEG_DAYS = 0
GAP_S_DRY_DAYS = 1
GAP_S_AVAIL_N = 2
GAP_S_N_SUPPLY_RATIO = 3
GAP_S_LITTER_ACCUM_C = 4
GAP_S_LITTER_ACCUM_N = 5
#   Recruitment info (free slots read at P2)
GAP_S_NUM_TO_RECRUIT = 6    # Number of free slots to activate this tick
GAP_S_RECRUIT_RAND_SEED = 7 # Random seed for species selection
GAP_S_FLOOD_DAYS = 8        # Annual flood days (from Site)
GAP_S_SEED_BANK = 9         # Accumulated seeds from previous years
GAP_S_FIRE_INTENSITY = 10   # Fire intensity this year (0-1, 0=no fire)
GAP_S_TOTAL_N_DEMAND = 11   # Total N demand from trees (public, Site reads at P4)
GAP_S_LITTER_ACCUM_C_BG = 12  # Below-ground litter carbon aggregate
GAP_S_LITTER_ACCUM_N_BG = 13  # Below-ground litter nitrogen aggregate

# states_db[1]: placeholder (not used, but keeps uniform signature)
GAP_DB_PLACEHOLDER = 0

# --- Site breed ---
# params[116]: soil pools + monthly climate + soil properties + fire/wind/soil + climate_std + lapse_rates (private internal)
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
#   Fire/wind/soil [53-55]:
SITE_P_FIRE_PROB = 53   # Annual fire probability (from CSV / 1000)
SITE_P_WIND_PROB = 54   # Annual wind probability (from CSV / 1000)
SITE_P_BASE_H = 55      # Base soil layer depth (from CSV)
#   Climate standard deviations [56-91]:
SITE_P_TMIN_STD_BASE = 56   # tmin_std[0..11] at 56-67
SITE_P_TMAX_STD_BASE = 68   # tmax_std[0..11] at 68-79
SITE_P_PRCP_STD_BASE = 80   # prcp_std[0..11] at 80-91
#   Lapse rates [92-115]:
SITE_P_TMP_LAPSE_BASE = 92   # temp_lapse[0..11] at 92-103
SITE_P_PRCP_LAPSE_BASE = 104 # prcp_lapse[0..11] at 104-115

SITE_PARAMS_SIZE = 116

# states[6]: climate + available + flood_days + fire + n_supply_ratio (public)
SITE_S_DEG_DAYS = 0
SITE_S_DRY_DAYS = 1
SITE_S_AVAIL_N = 2
SITE_S_FLOOD_DAYS = 3       # Days when A layer is saturated
SITE_S_FIRE_INTENSITY = 4   # Fire intensity this year (0-1, 0=no fire)
SITE_S_N_SUPPLY_RATIO = 5   # N supply ratio (computed at P4 by site_nutrient_step)

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

    Data Flow (per tick, matching GAPpy: soil first, two-phase tree growth):
    0. Gap (P0): aggregate litter from trees (prev tick)
    1. Site (P1): decompose litter, compute avail_n
    2. Trees (P2): recruitment + env stress + potential growth + N demand
    3. Gap (P3): aggregate N demand from trees
    4. Site (P4): compute n_supply_ratio = avail_n / total_n_demand
    5. Gap (P5): relay climate + n_supply_ratio from Site to Gap
    6. Trees (P6): apply nutrient factor, final growth, mortality, litter
    """

    def __init__(self) -> None:
        space = NetworkSpace()
        super().__init__(space)

        # === Create Tree breed (breed_id = 0) ===
        self._tree_breed = Breed("Tree")
        # params[40]: species traits + physiology + intermediates + renewal + leaf_area (private)
        self._tree_breed.register_property("params", [0.0] * 40, neighbor_visible=False)
        # states[5]: litter output (public, Gap reads at P0)
        self._tree_breed.register_property("states", [0.0] * 5, neighbor_visible=True)
        # states_db[5]: structure + seedling weight (public, other Trees read at P2)
        self._tree_breed.register_property("states_db", [0.0] * 5, neighbor_visible=True)
        # Phase A (P2): recruitment + env stress + potential growth + N demand
        self._tree_breed.register_step_func(
            tree_potential_growth_step,
            CURRENT_DIR / "step_functions" / "tree" / "tree_potential_growth_step.py",
            priority=2,
            no_double_buffer=["params", "states"],
        )
        # Phase B (P6): nutrient response + final growth + mortality + litter
        self._tree_breed.register_step_func(
            tree_actual_growth_step,
            CURRENT_DIR / "step_functions" / "tree" / "tree_actual_growth_step.py",
            priority=6,
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
        # states[14]: climate + nutrients + litter_pool + recruitment + flood + seed_bank + fire + n_demand + bg_litter (public)
        self._gap_breed.register_property("states", [0.0] * 14, neighbor_visible=True)
        # states_db[1]: placeholder (public but unused)
        self._gap_breed.register_property("states_db", [0.0] * 1, neighbor_visible=True)
        # P0: aggregate litter from trees (prev tick)
        self._gap_breed.register_step_func(
            gap_litter_aggregate_step,
            CURRENT_DIR / "step_functions" / "gap" / "gap_litter_aggregate_step.py",
            priority=0,
            no_double_buffer=["params", "states"],
        )
        # P3: aggregate N demand from trees (same tick, after tree potential growth)
        self._gap_breed.register_step_func(
            gap_demand_aggregate_step,
            CURRENT_DIR / "step_functions" / "gap" / "gap_demand_aggregate_step.py",
            priority=3,
            no_double_buffer=["params", "states"],
        )
        # P5: relay climate + n_supply_ratio from Site to Gap
        self._gap_breed.register_step_func(
            gap_sync_step,
            CURRENT_DIR / "step_functions" / "gap" / "gap_sync_step.py",
            priority=5,
            no_double_buffer=["params", "states"],
        )
        self.register_breed(breed=self._gap_breed)

        # === Create Site breed (breed_id = 2) ===
        self._site_breed = Breed("Site")
        # params[116]: soil pools + monthly climate + site properties + fire/wind/soil + climate_std + lapse_rates (private)
        self._site_breed.register_property("params", [0.0] * SITE_PARAMS_SIZE, neighbor_visible=False)
        # states[6]: climate + available + flood_days + fire + n_supply_ratio (public)
        self._site_breed.register_property("states", [0.0] * 6, neighbor_visible=True)
        # states_db[1]: placeholder (public but unused)
        self._site_breed.register_property("states_db", [0.0] * 1, neighbor_visible=True)
        # P1: soil decomposition (reads litter from Gap P0)
        self._site_breed.register_step_func(
            site_soil_step,
            CURRENT_DIR / "step_functions" / "site" / "soil_step.py",
            priority=1,
            no_double_buffer=["params", "states"],
        )
        # P4: compute n_supply_ratio (reads n_demand from Gap P3, avail_n from P1)
        self._site_breed.register_step_func(
            site_nutrient_step,
            CURRENT_DIR / "step_functions" / "site" / "site_nutrient_step.py",
            priority=4,
            no_double_buffer=["params", "states"],
        )
        self.register_breed(breed=self._site_breed)

        # Species registry (deduplicated across sites)
        self.unique_species = {}
        self.species_by_id = {}  # global_id -> species info dict

        # Track agents
        self.sites = []  # site info dicts
        self.site_agents = []  # site agent IDs
        self.gap_agents = []  # gap agent IDs
        self.tree_ids = []  # tree agent IDs
        self.tree_to_gap = {}  # tree_agent_id -> gap_agent_id

    def initialize_site(
        self,
        site_id: int = 0,
        data_dir: str = "input_data",
        prefix: str = "UVAFME2012",
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
        dry_days = max(0.0, 100.0 - annual_precip_mm / 10.0)

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

                    # GAPpy adjustments (species.py:66-101)
                    ss = [1.1, 1.15, 1.2, 1.23, 1.25]
                    adjust = [1.5, 1.55, 1.6, 1.65, 1.7]
                    HEC_TO_M2 = 10000.0

                    shade_tol = int(row['l'])
                    rootdepth = float(row.get('rootdepth', 2.0))

                    # Adjust g by shade tolerance
                    g_adj = float(row['g']) * ss[shade_tol - 1]

                    # Cap max_ht by rootdepth
                    max_ht_adj = min(float(row['Hmax']),
                                     rootdepth * 80.0 / (1.0 + rootdepth))

                    # Adjust leafdiam_a by shade tolerance
                    leafdiam_a = float(row['D_L']) * adjust[shade_tol - 1]

                    # Normalize leafarea_c
                    leafarea_c = float(row['L_C']) / HEC_TO_M2

                    self.unique_species[species_code] = {
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
                        'invader': float(row['invader']),    # Colonization ability
                        'seed': float(row['seed']),          # Seed production rate
                        'sprout': float(row['sprout']),      # Sprouting ability
                        'lownutr_tol': int(row['n']),        # Low nutrient tolerance (1-3)
                        'flood_tol': int(row['f']),          # Flood tolerance (1-6)
                        'drought_tol': int(row['d']),        # Drought tolerance (1-5)
                        'evergreen': int(row['evergreen']),  # 1=evergreen/conifer, 0=deciduous
                        'fire_tol': int(row['fire']),        # Fire tolerance (1-6)
                        'rootdepth': rootdepth,              # Root depth (m), default 2.0
                        'stress_tol': int(row['stress']),    # Stress tolerance (1-5)
                        'age_tol': int(row['old']),           # Age tolerance (1-3)
                        'genus': row['Genus'],
                        'seed_surv': float(row['NDE']),       # Seedbank survival rate
                        'seedling_lg': float(row['NDS']),     # Seedling annual survival rate
                        'leafdiam_a': leafdiam_a,             # Leaf diameter (adjusted)
                        'leafarea_c': leafarea_c,             # Leaf area coefficient (normalized)
                    }
                    self.species_by_id[global_id] = self.unique_species[species_code]

                site_species.append(self.unique_species[species_code])

        # === Load climate standard deviations (optional) ===
        climate_std_file = os.path.join(base_path, f"{prefix}_climate_stddev.csv")
        tmin_std = [0.0] * 12
        tmax_std = [0.0] * 12
        prcp_std = [0.0] * 12
        if os.path.exists(climate_std_file):
            with open(climate_std_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if int(row['site']) == site_id:
                        for i, month in enumerate(months):
                            tmin_std[i] = float(row[f'tmn_std_{month}'])
                            tmax_std[i] = float(row[f'tmx_std_{month}'])
                            prcp_std[i] = float(row[f'prcp_std_{month}']) * 0.1  # mm to cm
                        break

        # === Load altitudes (optional) ===
        altitude_file = os.path.join(base_path, f"{prefix}_altitudes.csv")
        altitude = None
        if os.path.exists(altitude_file):
            with open(altitude_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if int(row['site']) == site_id:
                        altitude = float(row['altitude'])
                        break

        # === Load additional site CSV columns ===
        fire_prob = float(site_row.get('fire_prob', 0)) / 1000.0
        wind_prob = float(site_row.get('wind_prob', 0)) / 1000.0
        soil_base_h = float(site_row.get('soil_base_h', 70.0))
        region = site_row.get('region', '')

        tmp_lapse = [0.0] * 12
        prcp_lapse = [0.0] * 12
        for i, month in enumerate(months):
            tmp_lapse[i] = float(site_row.get(f'tmp_lapse_{month}', 0.0))
            prcp_lapse[i] = float(site_row.get(f'prcp_lapse_{month}', 0.0))

        # === Apply altitude-based climate adjustment (GAPpy site.py:151-169) ===
        elevation = float(site_row.get('elevation', 0))
        if altitude is not None:
            for i, month in enumerate(months):
                tmin_val = float(climate_row[f'tmin_{month}'])
                tmax_val = float(climate_row[f'tmax_{month}'])
                prcp_val = float(climate_row[f'prcp_{month}']) / 10.0  # Convert to cm

                tmin_val -= (altitude - elevation) * tmp_lapse[i] * 0.01
                tmax_val -= (altitude - elevation) * tmp_lapse[i] * 0.01
                prcp_val = max(0.0, prcp_val + (altitude - elevation) * prcp_lapse[i] * 0.001)

                # Overwrite climate_row-derived values for deg_days/dry_days recalc
                climate_row[f'tmin_{month}'] = str(tmin_val)
                climate_row[f'tmax_{month}'] = str(tmax_val)
                climate_row[f'prcp_{month}'] = str(prcp_val * 10.0)  # Back to original units

            # Recalculate degree days and dry days from adjusted climate
            deg_days = 0.0
            for i, month in enumerate(months):
                tmin = float(climate_row[f'tmin_{month}'])
                tmax = float(climate_row[f'tmax_{month}'])
                tavg = (tmin + tmax) / 2.0
                if tavg > BASE_TEMP:
                    deg_days += (tavg - BASE_TEMP) * days_per_month[i]

            total_precip = sum(float(climate_row[f'prcp_{m}']) for m in months)
            annual_precip_mm = total_precip / 10.0
            dry_days = max(0.0, 100.0 - annual_precip_mm / 10.0)

        # === Build params[116] for Site (soil pools + monthly climate + site properties + new fields) ===
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

        # Fire/wind/soil [53-55]
        params[SITE_P_FIRE_PROB] = fire_prob
        params[SITE_P_WIND_PROB] = wind_prob
        params[SITE_P_BASE_H] = soil_base_h

        # Climate standard deviations [56-91]
        for i in range(12):
            params[SITE_P_TMIN_STD_BASE + i] = tmin_std[i]
            params[SITE_P_TMAX_STD_BASE + i] = tmax_std[i]
            params[SITE_P_PRCP_STD_BASE + i] = prcp_std[i]

        # Lapse rates [92-115]
        for i in range(12):
            params[SITE_P_TMP_LAPSE_BASE + i] = tmp_lapse[i]
            params[SITE_P_PRCP_LAPSE_BASE + i] = prcp_lapse[i]

        # === Build states[6] for Site (climate + flood_days + fire + n_supply_ratio - public) ===
        states = [0.0] * 6
        states[SITE_S_DEG_DAYS] = deg_days
        states[SITE_S_DRY_DAYS] = dry_days
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
            'elevation': elevation,
            'slope': float(site_row['slope']),
            'region': region,
            'fire_prob': fire_prob,
            'wind_prob': wind_prob,
            'soil_base_h': soil_base_h,
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

        # Build states[14] for Gap (climate + nutrients + litter_pool + recruitment + flood + seed_bank + fire + n_demand + bg_litter)
        states = [0.0] * 14
        states[GAP_S_DEG_DAYS] = deg_days
        states[GAP_S_DRY_DAYS] = dry_days
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
    ):
        """
        Initialize tree slots in a gap (matches GAPpy: empty start, renewal fills forest).

        Pre-allocates maxtrees free slots plus one template per species.
        All slots start as free (is_alive = 0). The renewal system at P6 populates
        the forest through seedbank/seedling dynamics, matching GAPpy where
        initialize_forest() is not called.

        IMPORTANT - Template Trees (Species Pool Preservation):
            This function creates "template" trees (is_alive = -1) for each species
            in the site's species list. Templates are connected to the gap and other
            trees but never grow, die, or produce litter. They serve as permanent
            species trait references for the recruitment step function.

            Without templates, if all trees of a species die, that species would be
            permanently lost because:
            1. Free slots don't preserve species diversity (all use placeholder)
            2. Recruitment only copies traits from neighbor trees
            3. GPU kernels cannot access Python-side species data

            Template trees ensure the full species pool from CSV initialization
            remains available for recruitment throughout the simulation.

        If gap_agent_id is None, creates a new gap first via initialize_gap().
        Trees connect bidirectionally to gap and to each other.

        :param site: Site info dict from initialize_site()
        :param gap_agent_id: Optional existing gap agent ID
        :param maxtrees: Maximum tree slots to pre-allocate (default 1000)
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

        # === Create free tree slots (all slots start empty, renewal fills forest) ===
        free_slot_count = maxtrees
        if free_slot_count > 0:
            # Use first species as placeholder for free slots
            # (species traits will be copied from template/living tree when recruited)
            placeholder_species = site_species[0]

            for _ in range(free_slot_count):
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
             0.0 = free slot (available for recruitment)
             1.0 = alive (active tree)
        :return: agent_id
        """
        STD_HT = 1.3
        TC_KG = 0.039269908  # PI / 80, gives biomC in kg (cm, m, g/cm³)

        if is_alive > 0.5 and diam > 0:
            # Calculate height from diameter
            delta_ht = species_info['max_ht'] - STD_HT
            forska_ht = STD_HT + delta_ht * (
                1.0 - (2.71828 ** (-(species_info['arfa_0'] * diam / delta_ht)))
            )

            # Calculate biomass (GAPpy: stem + twig + root)
            wood_bulk_dens = species_info['wood_bulk_dens']
            rootdepth = species_info.get('rootdepth', 2.0)
            canopy_ht = STD_HT

            # Basal diameter (canopy at ground level)
            if forska_ht > STD_HT:
                bd = forska_ht / (forska_ht - STD_HT) * diam
            else:
                bd = diam

            # Canopy diameter
            if forska_ht > canopy_ht and forska_ht > STD_HT:
                dc = (forska_ht - canopy_ht) / (forska_ht - STD_HT) * diam
            else:
                dc = diam

            # Stem biomass (trunk)
            stembc = TC_KG * wood_bulk_dens * 0.3 * bd * bd * forska_ht

            # Twig biomass (crown)
            crown_depth = forska_ht - canopy_ht
            if crown_depth < 0.0:
                crown_depth = 0.0
            twigbc = TC_KG * wood_bulk_dens * 0.337 * dc * dc * crown_depth

            # Root biomass
            root_c = 0.0
            if forska_ht > 0.01:
                root_c = stembc * rootdepth / forska_ht + twigbc * 0.5

            biomC = stembc + twigbc + root_c
            biomN = biomC / 400.0
            leaf_bm = biomC * 0.1
        else:
            # Free slot - minimal values
            forska_ht = 0.0
            biomC = 0.0
            biomN = 0.0
            leaf_bm = 0.0

        # Build params[40] for Tree (species traits + physiology + intermediates + renewal + leaf_area)
        params = [0.0] * 40
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
        params[TREE_P_ROOTDEPTH] = float(species_info.get('rootdepth', 2.0))
        params[TREE_P_STRESS_TOL] = float(species_info.get('stress_tol', 3))
        params[TREE_P_AGE_TOL] = float(species_info.get('age_tol', 3))
        # Renewal params [34-37]
        params[TREE_P_SEED_SURV] = float(species_info.get('seed_surv', 0.5))
        params[TREE_P_SEEDLING_LG] = float(species_info.get('seedling_lg', 0.5))
        params[TREE_P_SEEDBANK] = 0.0
        params[TREE_P_SEEDLING] = 0.0
        # Leaf area params [38-39]
        params[TREE_P_LEAFDIAM_A] = float(species_info.get('leafdiam_a', 0.0))
        params[TREE_P_LEAFAREA_C] = float(species_info.get('leafarea_c', 0.0))
        # Physiology [22-31]
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

        # Build states[5] for Tree (litter output: above + below ground)
        states = [0.0] * 5

        # Build states_db[5] for Tree (structure + seedling weight)
        states_db = [0.0] * 5
        states_db[TREE_DB_IS_ALIVE] = float(is_alive)  # -1=template, 0=free, 1=alive
        states_db[TREE_DB_DIAM] = diam
        states_db[TREE_DB_HEIGHT] = forska_ht
        states_db[TREE_DB_CANOPY_HT] = STD_HT if is_alive > 0.5 else 0.0
        states_db[TREE_DB_SEEDLING_WEIGHT] = 0.0

        # Create tree agent
        agent_id = self.create_agent_of_breed(
            self._tree_breed,
            params=params,
            states=states,
            states_db=states_db,
        )

        self.tree_ids.append(agent_id)
        self.tree_to_gap[agent_id] = gap_agent_id

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
            "free_slots": 0,
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
                # Free slot (is_alive = 0) - available for recruitment
                stats["free_slots"] += 1

        return stats

    def collect_tree_data(self):
        """Collect current data for all living trees.

        Returns list of dicts with per-tree data for output.
        Each dict has: gap_agent_id, species_id, diam, height,
        biomC, biomN, leaf_bm, age, canopy_ht, evergreen.
        """
        trees = []
        for tree_id in self.tree_ids:
            states_db = self.get_agent_property_value(tree_id, "states_db")
            if not isinstance(states_db, list) or states_db[TREE_DB_IS_ALIVE] < 0.5:
                continue
            params = self.get_agent_property_value(tree_id, "params")
            trees.append({
                'gap_agent_id': self.tree_to_gap[tree_id],
                'species_id': int(params[TREE_P_SPECIES_ID]),
                'diam': states_db[TREE_DB_DIAM],
                'height': states_db[TREE_DB_HEIGHT],
                'biomC': params[TREE_P_BIOMC],
                'biomN': params[TREE_P_BIOMN],
                'leaf_bm': params[TREE_P_LEAF_BM],
                'age': params[TREE_P_AGE],
                'canopy_ht': states_db[TREE_DB_CANOPY_HT],
                'evergreen': params[TREE_P_EVERGREEN] > 0.5,
            })
        return trees
