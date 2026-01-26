"""
GAPModel - GPU-accelerated gap dynamics model.
Integrates UVAFME processes with SAGESim agent framework.

Agent hierarchy:
- Site agents: hold environmental params (deg_days, dry_days, base_mortality)
- Gap agents: connect to site (bidirectional) and to all trees (bidirectional)
- Tree agents: connect to gap and to other trees in gap

All breeds register the SAME 5 properties:
- params: static parameters (15 floats)
- state_db: state needing double buffer (5 floats)
- state: state NOT needing double buffer (20 floats)
- output: outputs (8 floats)
- soil: soil state (10 floats)
"""

import sys
import os
import random
import csv

try:
    import sagesim  # noqa: F401
except ImportError:
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    _sagesim_path = os.path.join(os.path.dirname(_current_dir), "SAGESim")
    if _sagesim_path not in sys.path:
        sys.path.insert(0, _sagesim_path)

from sagesim.model import Model
from sagesim.space import NetworkSpace
from gap.tree_breed import TreeBreed
from gap.site_breed import SiteBreed
from gap.gap_breed import GapBreed

# === Index constants (shared across all breeds) ===

# params indices (15 floats)
P_SPECIES_ID = 0
P_SPECIES_PARAMS = 1  # indices 1-9 (9 values)
P_SITE_PARAMS = 10    # indices 10-12 (3 values: deg_days, dry_days, base_mortality)
P_GAP_ID = 13
P_SITE_IDX = 14

# Species params sub-indices (within params[1:10])
SP_MAX_AGE = 1
SP_MAX_DIAM = 2
SP_MAX_HT = 3
SP_ARFA_0 = 4
SP_G = 5
SP_SHADE_TOL = 6
SP_DEG_DAY_MIN = 7
SP_DEG_DAY_OPT = 8
SP_DEG_DAY_MAX = 9

# Site params sub-indices (within params[10:13])
SITE_DEG_DAYS = 10
SITE_DRY_DAYS = 11
SITE_BASE_MORTALITY = 12

# state_db indices (5 floats)
DB_IS_ALIVE = 0
DB_DIAM_BHT = 1
DB_FORSKA_HT = 2
DB_CANOPY_HT = 3

# state indices (20 floats)
S_AGE = 0
S_BIOMC = 1
S_BIOMN = 2
S_LEAF_BM = 3
S_X = 4
S_Y = 5
S_LIGHT_AVAIL = 6
S_FC_DEGDAY = 7
S_FC_DROUGHT = 8
S_FC_FLOOD = 9
S_GROWTH_FACTOR = 10
S_NUTRIENT_FACTOR = 11
S_AVAIL_N = 12
S_TOTAL_N_DEMAND = 13
S_N_SUPPLY_RATIO = 14

# output indices (8 floats)
O_LITTER_C = 0
O_LITTER_N = 1
O_N_DEMAND = 2
O_LITTER_ACCUM_C = 3
O_LITTER_ACCUM_N = 4
O_SOIL_RESP = 5

# soil indices (10 floats)
SOIL_A0_C = 0
SOIL_A0_N = 1
SOIL_A_C = 2
SOIL_A_N = 3
SOIL_BL_C = 4
SOIL_BL_N = 5
SOIL_A0_W = 6
SOIL_A_W = 7
SOIL_BL_W = 8


class GAPModel(Model):
    """
    GAPModel class for gap dynamics simulation.

    Agent structure:
    - Site agents hold site_params [deg_days, dry_days, base_mortality]
    - Gap agents connect to site and trees, hold site_params copy
    - Tree agents have species_params, read site_params from gap neighbor

    Breed registration order matters for breed IDs:
    - BREED_TREE = 0
    - BREED_GAP = 1
    - BREED_SITE = 2
    """

    def __init__(self) -> None:
        space = NetworkSpace()
        super().__init__(space)

        # Register breeds in order (determines breed IDs)
        self._tree_breed = TreeBreed()
        self._gap_breed = GapBreed()
        self._site_breed = SiteBreed()
        self.register_breed(breed=self._tree_breed)   # breed_id = 0
        self.register_breed(breed=self._gap_breed)    # breed_id = 1
        self.register_breed(breed=self._site_breed)   # breed_id = 2

        # Species registry (deduplicated across sites)
        self.unique_species = {}

        # Track agents
        self.sites = []  # site info dicts
        self.site_agents = []  # site agent IDs
        self.gap_agents = []  # gap agent IDs
        self.tree_ids = []  # tree agent IDs

    def load_site(
        self,
        site_csv,
        deg_days=2500.0,
        dry_days=30.0,
        base_mortality_rate=0.02,
    ):
        """
        Load a site from CSV and create a site agent.

        Returns site_info dict with site_agent_id.
        """
        file_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            site_csv
        )

        site_species = []

        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                species_code = row['Species_code']

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

        # Build params list for Site (site_params at indices 10-12)
        params = [0.0] * 15
        params[SITE_DEG_DAYS] = float(deg_days)
        params[SITE_DRY_DAYS] = float(dry_days)
        params[SITE_BASE_MORTALITY] = float(base_mortality_rate)

        # Build state with avail_n initial
        state = [0.0] * 20
        state[S_AVAIL_N] = 0.1

        # Build soil with initial pools
        # [A0_c, A0_n, A_c, A_n, BL_c, BL_n, A0_w, A_w, BL_w, ...]
        soil = [0.0] * 10
        soil[SOIL_A0_C] = 5.0      # Litter layer carbon (tn C/ha)
        soil[SOIL_A0_N] = 0.167    # Litter layer nitrogen (5/30)
        soil[SOIL_A_C] = 50.0      # Humus layer carbon
        soil[SOIL_A_N] = 12.5      # Humus layer nitrogen (50/4)
        soil[SOIL_BL_C] = 100.0    # Base layer carbon
        soil[SOIL_BL_N] = 5.0      # Base layer nitrogen (100/20)
        soil[SOIL_A0_W] = 0.5      # Moisture
        soil[SOIL_A_W] = 0.5
        soil[SOIL_BL_W] = 0.5

        # Create site agent
        site_agent_id = self.create_agent_of_breed(
            self._site_breed,
            params=params,
            state_db=[0.0] * 5,
            state=state,
            output=[0.0] * 8,
            soil=soil,
        )
        self.site_agents.append(site_agent_id)

        site_info = {
            'site_id': len(self.sites),
            'site_agent_id': site_agent_id,
            'species': site_species,
            'site_params': [float(deg_days), float(dry_days), float(base_mortality_rate)],
            'gaps': [],  # gap agent IDs for this site
        }
        self.sites.append(site_info)

        return site_info

    def create_gap(self, site):
        """
        Create a gap agent and connect it bidirectionally to the site.

        Gap gets site_params from site (trees read from gap).
        Returns gap_agent_id.
        """
        gap_id = len(self.gap_agents)
        site_agent_id = site['site_agent_id']
        site_params = site['site_params']

        # Build params list for Gap (site_params at indices 10-12, gap_id at 13, site_idx at 14)
        params = [0.0] * 15
        params[SITE_DEG_DAYS] = site_params[0]
        params[SITE_DRY_DAYS] = site_params[1]
        params[SITE_BASE_MORTALITY] = site_params[2]
        params[P_GAP_ID] = float(gap_id)
        params[P_SITE_IDX] = float(site_agent_id)

        # Build state with initial n_supply_ratio and avail_n
        state = [0.0] * 20
        state[S_AVAIL_N] = 0.1
        state[S_TOTAL_N_DEMAND] = 0.0
        state[S_N_SUPPLY_RATIO] = 1.0

        gap_agent_id = self.create_agent_of_breed(
            self._gap_breed,
            params=params,
            state_db=[0.0] * 5,
            state=state,
            output=[0.0] * 8,
            soil=[0.0] * 10,
        )
        self.gap_agents.append(gap_agent_id)
        site['gaps'].append(gap_agent_id)

        # Bidirectional connection: site ↔ gap
        self.connect_agents(site_agent_id, gap_agent_id)

        return gap_agent_id

    def initialize_gap(
        self,
        site,
        gap_agent_id=None,
        maxtrees=100,
        age_range=(5, 50),
        size_range=(3.0, 25.0),
    ):
        """
        Initialize trees in a gap.

        If gap_agent_id is None, creates a new gap first.
        Trees connect bidirectionally to gap and to each other.
        Trees read site_params from gap neighbor (not stored on tree).
        """
        if maxtrees == 0:
            raise ValueError("Must have at least one tree")

        # Create gap if not provided
        if gap_agent_id is None:
            gap_agent_id = self.create_gap(site)

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

                # Build params list for Tree
                params = [0.0] * 15
                params[P_SPECIES_ID] = float(species_info['global_id'])
                params[SP_MAX_AGE] = float(species_info['max_age'])
                params[SP_MAX_DIAM] = float(species_info['max_diam'])
                params[SP_MAX_HT] = float(species_info['max_ht'])
                params[SP_ARFA_0] = float(species_info['arfa_0'])
                params[SP_G] = float(species_info['g'])
                params[SP_SHADE_TOL] = float(species_info['shade_tol'])
                params[SP_DEG_DAY_MIN] = float(species_info['deg_day_min'])
                params[SP_DEG_DAY_OPT] = float(species_info['deg_day_opt'])
                params[SP_DEG_DAY_MAX] = float(species_info['deg_day_max'])

                # Build state_db
                state_db = [0.0] * 5
                state_db[DB_IS_ALIVE] = 1.0
                state_db[DB_DIAM_BHT] = diam
                state_db[DB_FORSKA_HT] = forska_ht
                state_db[DB_CANOPY_HT] = 1.3

                # Build state
                state = [0.0] * 20
                state[S_AGE] = age
                state[S_BIOMC] = biomC
                state[S_BIOMN] = biomN
                state[S_LEAF_BM] = leaf_bm
                state[S_LIGHT_AVAIL] = 1.0
                state[S_FC_DEGDAY] = 1.0
                state[S_FC_DROUGHT] = 1.0
                state[S_FC_FLOOD] = 1.0
                state[S_GROWTH_FACTOR] = 1.0
                state[S_NUTRIENT_FACTOR] = 1.0

                # Create tree with consolidated properties
                agent_id = self.create_agent_of_breed(
                    self._tree_breed,
                    params=params,
                    state_db=state_db,
                    state=state,
                    output=[0.0] * 8,
                    soil=[0.0] * 10,
                )

                self.tree_ids.append(agent_id)
                created_trees.append(agent_id)

                # Bidirectional connection: gap ↔ tree
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
            # Access state_db for is_alive (index 0)
            state_db = self.get_agent_property_value(tree_id, "state_db")
            alive = state_db[DB_IS_ALIVE] if isinstance(state_db, list) else state_db
            if alive > 0.5:
                stats["living_trees"] += 1
                # Access state for biomC (index 1)
                state = self.get_agent_property_value(tree_id, "state")
                biomC = state[S_BIOMC] if isinstance(state, list) else 0.0
                stats["total_biomass"] += biomC
            else:
                stats["dead_trees"] += 1

        return stats
