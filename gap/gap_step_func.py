"""
Gap step functions for GGap model.
GPU kernels for Gap agent data relay between Trees and Site.

Two step functions:
- gap_aggregate_step (priority 1): Read litter from trees, store in own litter_accum
- gap_sync_step (priority 3): Read avail_n from site, store for trees to read

All step functions share same signature with 5 property tensors:
- params_tensor: static parameters
- state_db_tensor: state needing double buffer
- state_tensor: state NOT needing double buffer
- output_tensor: outputs
- soil_tensor: soil state
"""

import sys
import os

_current_dir = os.path.dirname(os.path.abspath(__file__))
_sagesim_path = os.path.join(os.path.dirname(_current_dir), "SAGESim")
if _sagesim_path not in sys.path:
    sys.path.insert(0, _sagesim_path)

import cupy as cp  # noqa: F401
from cupyx import jit

# === Index constants (shared across all breeds) ===

# state indices (20 floats)
S_AVAIL_N = 12
S_TOTAL_N_DEMAND = 13
S_N_SUPPLY_RATIO = 14

# output indices (8 floats)
O_LITTER_C = 0
O_LITTER_N = 1
O_N_DEMAND = 2
O_LITTER_ACCUM_C = 3
O_LITTER_ACCUM_N = 4

# Breed IDs (must match registration order in GAPModel)
BREED_TREE = 0
BREED_GAP = 1
BREED_SITE = 2


@jit.rawkernel(device="cuda")
def gap_aggregate_step(
    tick,
    agent_index,
    globals,
    agent_ids,
    breeds,
    locations,
    # 5 consolidated property tensors
    params_tensor,
    state_db_tensor,
    state_tensor,
    output_tensor,
    soil_tensor,
):
    """
    Gap aggregate step (priority 1).

    Reads from tree neighbors:
    - litter_out [C, N] from each tree (output[0:2])
    - n_demand from each tree (output[2])

    Writes to own state:
    - litter_accum: aggregated litter for Site to read (output[3:5])
    - total_n_demand: aggregated N demand for supply ratio calculation (state[13])
    """
    # Note: Breed check is done by SAGESim wrapper (stepfunc), no need to check here

    # Aggregate litter and N demand from tree neighbors
    total_litter_c = 0.0
    total_litter_n = 0.0
    total_n_dem = 0.0

    neighbor_indices = locations[agent_index]
    i = 0
    while i < len(neighbor_indices) and neighbor_indices[i] != -1:
        neighbor_idx = int(neighbor_indices[i])
        neighbor_breed = int(breeds[neighbor_idx])

        if neighbor_breed == BREED_TREE:
            # Read tree's litter output
            tree_litter_c = output_tensor[neighbor_idx][O_LITTER_C]
            tree_litter_n = output_tensor[neighbor_idx][O_LITTER_N]
            total_litter_c = total_litter_c + tree_litter_c
            total_litter_n = total_litter_n + tree_litter_n

            # Read tree's N demand
            tree_n_demand = output_tensor[neighbor_idx][O_N_DEMAND]
            total_n_dem = total_n_dem + tree_n_demand

        i = i + 1

    # Write to own state (Site will read this)
    output_tensor[agent_index][O_LITTER_ACCUM_C] = total_litter_c
    output_tensor[agent_index][O_LITTER_ACCUM_N] = total_litter_n
    state_tensor[agent_index][S_TOTAL_N_DEMAND] = total_n_dem


@jit.rawkernel(device="cuda")
def gap_sync_step(
    tick,
    agent_index,
    globals,
    agent_ids,
    breeds,
    locations,
    # 5 consolidated property tensors
    params_tensor,
    state_db_tensor,
    state_tensor,
    output_tensor,
    soil_tensor,
):
    """
    Gap sync step (priority 3).

    Reads from site neighbor:
    - avail_n: available nitrogen after decomposition (state[12])

    Writes to own state:
    - avail_n: stored for trees to read next tick (state[12])
    - n_supply_ratio: avail_n / total_n_demand (state[14])
    """
    # Note: Breed check is done by SAGESim wrapper (stepfunc), no need to check here

    # Find Site neighbor and read avail_n
    site_avail_n = 0.1  # Default

    neighbor_indices = locations[agent_index]
    i = 0
    while i < len(neighbor_indices) and neighbor_indices[i] != -1:
        neighbor_idx = int(neighbor_indices[i])
        neighbor_breed = int(breeds[neighbor_idx])

        if neighbor_breed == BREED_SITE:
            # Read Site's avail_n from state tensor at Site's index
            site_avail_n = state_tensor[neighbor_idx][S_AVAIL_N]

        i = i + 1

    # Store avail_n for trees to read (at Gap's own index)
    state_tensor[agent_index][S_AVAIL_N] = site_avail_n

    # Calculate N supply/demand ratio
    total_n_dem = state_tensor[agent_index][S_TOTAL_N_DEMAND]
    n_supply_ratio = 1.0
    if total_n_dem > 0.0001:
        n_supply_ratio = site_avail_n / total_n_dem
        if n_supply_ratio > 2.0:
            n_supply_ratio = 2.0  # Cap at 2x supply

    state_tensor[agent_index][S_N_SUPPLY_RATIO] = n_supply_ratio

    # Clear litter accumulator for next tick
    output_tensor[agent_index][O_LITTER_ACCUM_C] = 0.0
    output_tensor[agent_index][O_LITTER_ACCUM_N] = 0.0
