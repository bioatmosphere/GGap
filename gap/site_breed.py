"""
SiteBreed for GGap model.
Site agent holds soil biogeochemistry state and environmental parameters.

All breeds register the SAME 5 properties:
- params: static parameters (no double buffer)
- state_db: state needing double buffer
- state: state NOT needing double buffer
- output: outputs (no double buffer)
- soil: soil state (no double buffer)
"""

import sys
import os

try:
    import sagesim  # noqa: F401
except ImportError:
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    _sagesim_path = os.path.join(os.path.dirname(_current_dir), "SAGESim")
    if _sagesim_path not in sys.path:
        sys.path.insert(0, _sagesim_path)

from sagesim.breed import Breed
from gap.site_step_func import site_soil_step


class SiteBreed(Breed):
    """
    SiteBreed - registered third (breed_id = 2).

    Site-specific data:
    - params[10:13]: site_params (deg_days, dry_days, base_mortality)
    - state[12]: avail_n (written for Gap to read)
    - output[5]: soil_resp
    - soil[0:9]: soil_pools and soil_water
    """

    def __init__(self) -> None:
        super().__init__("Site")

        # All breeds register the SAME 5 properties
        self.register_property("params", [0.0] * 15, neighbor_visible=True)
        self.register_property("state_db", [0.0] * 5, neighbor_visible=True)
        self.register_property("state", [0.0] * 20, neighbor_visible=True)
        self.register_property("output", [0.0] * 8, neighbor_visible=True)
        self.register_property("soil", [0.0] * 10, neighbor_visible=False)

        # Register soil decomposition step function (priority 2)
        # Reads litter from Gap neighbors, updates soil pools and avail_n
        self.register_step_func(
            site_soil_step,
            "site_step_func.py",
            priority=2,
            no_double_buffer=["params", "state", "output", "soil"],
        )
