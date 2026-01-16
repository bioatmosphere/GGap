"""
GAPModel - GPU-accelerated gap dynamics model.
Integrates UVAFME processes with SAGESim agent framework.

Structure: Model -> Sites -> Gaps -> Trees (agents)
"""

import sys
import os
import random
import csv

# Try installed SAGESim first, fall back to submodule
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


class GAPModel(Model):
    """
    GAPModel class for gap dynamics simulation.
    Inherits from SAGESim Model class.

    Future structure: Model contains Sites, each Site contains Gaps.
    Currently supports single gap for testing.
    """

    def __init__(
        self,
        deg_days=2500.0,
        dry_days=30.0,
        base_mortality_rate=0.02,
    ) -> None:
        """
        Initialize GAPModel with environmental parameters.

        Parameters:
        -----------
        deg_days : float
            Annual degree days for temperature response
        dry_days : float
            Annual drought days
        base_mortality_rate : float
            Base annual mortality probability (0-1)
        """
        # Create network space for agent connections
        space = NetworkSpace()
        super().__init__(space)

        # Create and register tree breed
        self._tree_breed = TreeBreed()
        self.register_breed(breed=self._tree_breed)

        # Register global environmental properties
        self.register_global_property("deg_days", deg_days)
        self.register_global_property("dry_days", dry_days)
        self.register_global_property("base_mortality_rate", base_mortality_rate)

        # Track created trees
        self.tree_ids = []

    def connect_agents(self, agent_0, agent_1):
        """Connect two agents as neighbors."""
        self.get_space().connect_agents(agent_0, agent_1)

    def initialize_gap(
        self,
        specieslist_file="input_data/UVAFME2012_specieslist.csv",
        maxtrees=100,
        maxheight=60,
        age_range=(5, 50),
        size_range=(3.0, 25.0),
    ):
        """
        Initialize a gap with random trees from a species list CSV file.
        All trees in this gap are mutually connected (fully connected network).
        Follows UVAFME initialize_plot pattern.

        Parameters:
        -----------
        specieslist_file : str
            Path to the species list CSV file
        maxtrees : int
            Number of trees to create in this gap
        maxheight : int
            Maximum tree height in meters
        age_range : tuple
            (min_age, max_age) for initial tree ages
        size_range : tuple
            (min_diam, max_diam) for initial tree diameters (cm)

        Returns:
        --------
        tree_ids : list
            List of created tree agent IDs
        """
        # Validate inputs (following UVAFME initialize_plot)
        if maxtrees == 0:
            raise ValueError("Must allow at least a few trees")
        if maxheight == 0:
            raise ValueError("Must have a nonzero maximum height")

        num_trees = maxtrees

        # Read species from CSV file
        species_list = []
        file_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            specieslist_file
        )

        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                species_info = {
                    'species_id': len(species_list) + 1,  # 1-indexed
                    'species_code': row['Species_code'],
                    'common_name': row['Common name'].replace('_', ' ').strip("'"),
                    'genus': row['Genus'].strip("'"),
                    'max_age': float(row['AGEmax']),
                    'max_diam': float(row['DBHmax']),
                    'max_ht': float(row['Hmax']),
                    'arfa_0': float(row['s']),
                    'g': float(row['g']),
                    'shade_tol': int(row['l']),
                    'drought_tol': int(row['d']),
                    'flood_tol': int(row['f']),
                    'deg_day_min': float(row['DEGDmin']),
                    'deg_day_opt': float(row['DEGDoptimum']),
                    'deg_day_max': float(row['DEGDmax']),
                    'wood_bulk_dens': float(row['bulk']),
                }
                species_list.append(species_info)

        num_species = len(species_list)
        if num_species == 0:
            raise ValueError(f"No species found in {specieslist_file}")

        # Distribute trees randomly across species
        random_weights = [random.random() for _ in range(num_species)]
        total_weight = sum(random_weights)
        proportions = [w / total_weight for w in random_weights]

        # Calculate number of trees per species
        trees_per_species = []
        remaining = num_trees
        for i, prop in enumerate(proportions[:-1]):
            count = int(num_trees * prop)
            trees_per_species.append(count)
            remaining -= count
        trees_per_species.append(remaining)  # Last species gets remainder

        # Create trees for each species
        created_trees = []

        for species_idx, species_info in enumerate(species_list):
            num_trees_this_species = trees_per_species[species_idx]

            for _ in range(num_trees_this_species):
                # Random age and size
                age = random.uniform(age_range[0], age_range[1])
                diam = random.uniform(size_range[0], size_range[1])

                # Calculate height using species-specific parameters
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

                # Create tree agent
                agent_id = self.create_agent_of_breed(
                    self._tree_breed,
                    species_id=species_info['species_id'],
                    is_alive=1.0,
                    age=age,
                    diam_bht=diam,
                    forska_ht=forska_ht,
                    canopy_ht=1.3,
                    biomC=biomC,
                    biomN=biomN,
                    leaf_bm=leaf_bm,
                    light_avail=1.0,
                    fc_degday=1.0,
                    fc_drought=1.0,
                    fc_flood=1.0,
                    growth_factor=1.0,
                )

                self.tree_ids.append(agent_id)
                created_trees.append(agent_id)

        # Connect all trees to each other (fully connected network)
        for i in range(len(created_trees)):
            for j in range(i + 1, len(created_trees)):
                self.connect_agents(created_trees[i], created_trees[j])

        return created_trees

    def get_statistics(self):
        """
        Get current gap statistics.

        Returns:
        --------
        stats : dict
            Dictionary containing gap statistics
        """
        stats = {
            "total_trees": len(self.tree_ids),
            "living_trees": 0,
            "dead_trees": 0,
            "species_counts": {},
            "avg_age": 0.0,
            "avg_height": 0.0,
            "avg_diameter": 0.0,
            "total_biomass": 0.0,
        }

        if len(self.tree_ids) == 0:
            return stats

        total_age = 0.0
        total_height = 0.0
        total_diameter = 0.0
        living_count = 0

        for tree_id in self.tree_ids:
            is_alive = self.get_agent_property_value(tree_id, "is_alive")

            if is_alive > 0.5:  # Alive
                stats["living_trees"] += 1
                living_count += 1

                species_id = int(self.get_agent_property_value(tree_id, "species_id"))
                stats["species_counts"][species_id] = (
                    stats["species_counts"].get(species_id, 0) + 1
                )

                age = self.get_agent_property_value(tree_id, "age")
                height = self.get_agent_property_value(tree_id, "forska_ht")
                diameter = self.get_agent_property_value(tree_id, "diam_bht")
                biomC = self.get_agent_property_value(tree_id, "biomC")

                total_age += age
                total_height += height
                total_diameter += diameter
                stats["total_biomass"] += biomC
            else:
                stats["dead_trees"] += 1

        # Calculate averages
        if living_count > 0:
            stats["avg_age"] = total_age / living_count
            stats["avg_height"] = total_height / living_count
            stats["avg_diameter"] = total_diameter / living_count

        return stats

    def print_statistics(self, tick=None):
        """Print formatted gap statistics."""
        stats = self.get_statistics()

        if tick is not None:
            print(f"\n--- Year {tick} Gap Statistics ---")
        else:
            print("\n--- Gap Statistics ---")

        print(f"Total trees: {stats['total_trees']}")
        print(f"Living trees: {stats['living_trees']}")
        print(f"Dead trees: {stats['dead_trees']}")

        if stats["living_trees"] > 0:
            print(f"\nAverage age: {stats['avg_age']:.1f} years")
            print(f"Average height: {stats['avg_height']:.1f} m")
            print(f"Average diameter: {stats['avg_diameter']:.1f} cm")
            print(f"Total biomass: {stats['total_biomass']:.1f} kg C")

            if stats["species_counts"]:
                print(f"\nNumber of species: {len(stats['species_counts'])}")
