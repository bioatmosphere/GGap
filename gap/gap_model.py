"""
GAPModel - GPU-accelerated forest gap dynamics model.
Integrates UVAFME forest processes with SAGESim agent framework.
"""

import sys
import os
import random

# Add SAGESim to path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_sagesim_path = os.path.join(os.path.dirname(_current_dir), "SAGESim")
if _sagesim_path not in sys.path:
    sys.path.insert(0, _sagesim_path)

from sagesim.model import Model
from sagesim.space import Space
from gap.tree_breed import TreeBreed
from gap.tree_species_data import get_species_params, get_all_species_ids


class GAPModel(Model):
    """
    GAPModel class for forest gap dynamics simulation.
    Inherits from SAGESim Model class.
    """

    def __init__(
        self,
        neighborhood_radius=10.0,
        deg_days=2500.0,
        dry_days=30.0,
        base_mortality_rate=0.02,
    ) -> None:
        """
        Initialize GAPModel with environmental parameters.

        Parameters:
        -----------
        neighborhood_radius : float
            Distance threshold for tree interactions (meters)
        deg_days : float
            Annual degree days for temperature response
        dry_days : float
            Annual drought days
        base_mortality_rate : float
            Base annual mortality probability (0-1)
        """
        # Create a simple space (will be enhanced with spatial indexing later)
        space = Space()
        super().__init__(space)

        # Create and register tree breed
        self._tree_breed = TreeBreed()
        self.register_breed(breed=self._tree_breed)

        # Register global environmental properties
        self.register_global_property("neighborhood_radius", neighborhood_radius)
        self.register_global_property("deg_days", deg_days)
        self.register_global_property("dry_days", dry_days)
        self.register_global_property("base_mortality_rate", base_mortality_rate)

        # Track created trees
        self.tree_ids = []

    def create_tree(
        self,
        species_id=1,
        age=10.0,
        diam_bht=5.0,
        x=0.0,
        y=0.0,
        forska_ht=None,
        canopy_ht=1.3,
        is_alive=1.0,
        light_avail=1.0,
        fc_degday=1.0,
        fc_drought=1.0,
        fc_flood=1.0,
    ):
        """
        Create a tree agent with specified properties.

        Parameters:
        -----------
        species_id : int
            Species identifier (1-6)
        age : float
            Tree age in years
        diam_bht : float
            Diameter at breast height (cm)
        x : float
            X coordinate (meters)
        y : float
            Y coordinate (meters)
        forska_ht : float, optional
            Tree height (meters). If None, calculated from diameter
        canopy_ht : float
            Canopy height (meters)
        is_alive : float
            Alive status (1.0=alive, 0.0=dead)
        light_avail : float
            Available light (0-1)
        fc_degday : float
            Temperature response factor (0-1)
        fc_drought : float
            Drought response factor (0-1)
        fc_flood : float
            Flood response factor (0-1)

        Returns:
        --------
        agent_id : int
            ID of created tree agent
        """
        # Get species parameters for height calculation if needed
        if forska_ht is None:
            species_params = get_species_params(species_id)
            STD_HT = 1.3
            delta_ht = species_params.max_ht - STD_HT
            forska_ht = STD_HT + delta_ht * (
                1.0 - (2.71828 ** (-(species_params.arfa_0 * diam_bht / delta_ht)))
            )

        # Calculate initial biomass (simplified)
        wood_bulk_dens = 0.54
        PI = 3.14159265359
        radius_m = diam_bht / 200.0
        volume_m3 = PI * radius_m * radius_m * forska_ht
        biomC = volume_m3 * wood_bulk_dens * 1000.0 * 0.5

        biomN = biomC / 450.0  # C:N ratio of ~450
        leaf_bm = biomC * 0.1  # Rough estimate

        agent_id = self.create_agent_of_breed(
            self._tree_breed,
            species_id=species_id,
            is_alive=is_alive,
            age=age,
            diam_bht=diam_bht,
            forska_ht=forska_ht,
            canopy_ht=canopy_ht,
            biomC=biomC,
            biomN=biomN,
            leaf_bm=leaf_bm,
            x=x,
            y=y,
            light_avail=light_avail,
            fc_degday=fc_degday,
            fc_drought=fc_drought,
            fc_flood=fc_flood,
            growth_factor=1.0,
        )

        self.tree_ids.append(agent_id)
        return agent_id

    def create_forest(
        self,
        num_trees=100,
        forest_size=100.0,
        species_distribution=None,
        age_range=(10, 50),
        size_range=(5.0, 20.0),
    ):
        """
        Create a forest with multiple trees.

        Parameters:
        -----------
        num_trees : int
            Number of trees to create
        forest_size : float
            Size of square forest area (meters)
        species_distribution : dict, optional
            Distribution of species as {species_id: proportion}
            If None, uses equal distribution
        age_range : tuple
            (min_age, max_age) for initial tree ages
        size_range : tuple
            (min_diam, max_diam) for initial tree diameters (cm)

        Returns:
        --------
        tree_ids : list
            List of created tree agent IDs
        """
        if species_distribution is None:
            # Default: equal distribution across all species
            species_ids = get_all_species_ids()
            species_distribution = {sid: 1.0 / len(species_ids) for sid in species_ids}

        # Normalize distribution
        total = sum(species_distribution.values())
        species_distribution = {k: v / total for k, v in species_distribution.items()}

        # Create trees
        created_trees = []
        for i in range(num_trees):
            # Random spatial location
            x = random.uniform(0, forest_size)
            y = random.uniform(0, forest_size)

            # Select species based on distribution
            rand_val = random.random()
            cumulative = 0.0
            selected_species = 1
            for species_id, proportion in species_distribution.items():
                cumulative += proportion
                if rand_val <= cumulative:
                    selected_species = species_id
                    break

            # Random age and size
            age = random.uniform(age_range[0], age_range[1])
            diam = random.uniform(size_range[0], size_range[1])

            # Create tree
            tree_id = self.create_tree(
                species_id=selected_species,
                age=age,
                diam_bht=diam,
                x=x,
                y=y,
            )
            created_trees.append(tree_id)

        return created_trees

    def get_forest_statistics(self):
        """
        Get current forest statistics.

        Returns:
        --------
        stats : dict
            Dictionary containing forest statistics
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

    def print_forest_statistics(self, tick=None):
        """Print formatted forest statistics."""
        stats = self.get_forest_statistics()

        if tick is not None:
            print(f"\n--- Year {tick} Forest Statistics ---")
        else:
            print("\n--- Forest Statistics ---")

        print(f"Total trees: {stats['total_trees']}")
        print(f"Living trees: {stats['living_trees']}")
        print(f"Dead trees: {stats['dead_trees']}")

        if stats["living_trees"] > 0:
            print(f"\nAverage age: {stats['avg_age']:.1f} years")
            print(f"Average height: {stats['avg_height']:.1f} m")
            print(f"Average diameter: {stats['avg_diameter']:.1f} cm")
            print(f"Total biomass: {stats['total_biomass']:.1f} kg C")

            print("\nSpecies distribution:")
            from gap.tree_species_data import get_species_params

            for species_id, count in sorted(stats["species_counts"].items()):
                species = get_species_params(species_id)
                print(f"  {species.common_name}: {count}")
