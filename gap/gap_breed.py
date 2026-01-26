"""
GapBreed for GGap model.
Gap agent represents a forest gap, connects to its parent site and trees.

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
    import sagesim
except ImportError:
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    _sagesim_path = os.path.join(os.path.dirname(_current_dir), "SAGESim")
    if _sagesim_path not in sys.path:
        sys.path.insert(0, _sagesim_path)

from sagesim.breed import Breed
from gap.gap_step_func import gap_aggregate_step, gap_sync_step


class GapBreed(Breed):
    """
    GapBreed - registered second (breed_id = 1).

    Gap-specific data:
    - params[10:15]: site_params, gap_id, site_idx
    - state[12:15]: avail_n, total_n_demand, n_supply_ratio
    - output[3:5]: litter_accum
    """

    def __init__(self) -> None:
        super().__init__("Gap")

        # All breeds register the SAME 5 properties
        self.register_property("params", [0.0] * 15, neighbor_visible=True)
        self.register_property("state_db", [0.0] * 5, neighbor_visible=True)
        self.register_property("state", [0.0] * 20, neighbor_visible=True)
        self.register_property("output", [0.0] * 8, neighbor_visible=True)
        self.register_property("soil", [0.0] * 10, neighbor_visible=False)

        # Register aggregate step function (priority 1)
        # Reads output from trees, writes to own output/state
        self.register_step_func(
            gap_aggregate_step,
            "gap_step_func.py",
            priority=1,
            no_double_buffer=["params", "state", "output", "soil"],
        )

        # Register sync step function (priority 3)
        # Reads state from site, writes to own state
        self.register_step_func(
            gap_sync_step,
            "gap_step_func.py",
            priority=3,
            no_double_buffer=["params", "state", "output", "soil"],
        )
