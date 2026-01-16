"""
TreeBreed for GGap model.
Based on UVAFME tree structure with SAGESim agent framework.
"""

import sys
import os

# Try installed SAGESim first, fall back to submodule
try:
    import sagesim  # noqa: F401
except ImportError:
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    _sagesim_path = os.path.join(os.path.dirname(_current_dir), "SAGESim")
    if _sagesim_path not in sys.path:
        sys.path.insert(0, _sagesim_path)

from sagesim.breed import Breed
from gap.tree_step_func import growth_step, mortality_step, light_step


class TreeBreed(Breed):
    """
    TreeBreed class for the GGap forest model.
    Represents individual trees with UVAFME-based properties.
    """

    def __init__(self) -> None:
        name = "Tree"
        super().__init__(name)

        # Core identification and state
        self.register_property("species_id", 1)  # Which species (1-6)
        self.register_property("is_alive", 1.0)  # 1=alive, 0=dead

        # Age and size
        self.register_property("age", 1.0)  # years
        self.register_property("diam_bht", 2.0)  # diameter at breast height (cm)
        self.register_property("forska_ht", 1.5)  # total height (meters)
        self.register_property("canopy_ht", 1.3)  # canopy height (meters, typically 1.3m)

        # Biomass
        self.register_property("biomC", 0.1)  # carbon biomass (kg C)
        self.register_property("biomN", 0.001)  # nitrogen biomass (kg N)
        self.register_property("leaf_bm", 0.05)  # leaf biomass (kg C)

        # Spatial location
        self.register_property("x", 0.0)  # x coordinate (meters)
        self.register_property("y", 0.0)  # y coordinate (meters)

        # Environmental factors
        self.register_property("light_avail", 1.0)  # Available light (0-1)
        self.register_property("fc_degday", 1.0)  # Temperature response factor (0-1)
        self.register_property("fc_drought", 1.0)  # Drought response factor (0-1)
        self.register_property("fc_flood", 1.0)  # Flood response factor (0-1)

        # Growth tracking
        self.register_property("growth_factor", 1.0)  # Combined growth factor (0-1)

        # Register step functions with priorities
        # Priority 0: Calculate light competition first
        self.register_step_func(light_step, "gap/tree_step_func.py", 0)

        # Priority 1: Growth (depends on light)
        self.register_step_func(growth_step, "gap/tree_step_func.py", 1)

        # Priority 2: Mortality (depends on growth factors)
        self.register_step_func(mortality_step, "gap/tree_step_func.py", 2)
