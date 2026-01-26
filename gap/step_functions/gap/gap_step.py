"""
Gap step functions for GGap model.
GPU kernels for Gap agent data relay between Trees and Site.

Two step functions:
- gap_aggregate_step (priority 1): Read litter from trees, store in own states
- gap_sync_step (priority 3): Read climate+avail_n from site, store for trees to read

Property scheme (3 properties):
- params[2]: gap_id, total_n_demand (private internal)
- states[7]: climate + nutrients + litter_pool (public, no buffer)
- states_db[1]: placeholder (public, double buffered but unused)
"""

import cupy as cp  # noqa: F401
from cupyx import jit

# === Breed IDs ===
BREED_TREE = 0
BREED_GAP = 1
BREED_SITE = 2

# === Gap params[2] (private) ===
GAP_P_GAP_ID = 0
GAP_P_TOTAL_N_DEMAND = 1

# === Gap states[7] (public, no buffer) ===
GAP_S_DEG_DAYS = 0
GAP_S_DRY_DAYS = 1
GAP_S_BASE_MORTALITY = 2
GAP_S_AVAIL_N = 3
GAP_S_N_SUPPLY_RATIO = 4
GAP_S_LITTER_ACCUM_C = 5
GAP_S_LITTER_ACCUM_N = 6

# === Tree states[3] (for reading from Tree neighbors) ===
TREE_S_LITTER_C = 0
TREE_S_LITTER_N = 1
TREE_S_N_DEMAND = 2

# === Site states[4] (for reading from Site neighbor) ===
SITE_S_DEG_DAYS = 0
SITE_S_DRY_DAYS = 1
SITE_S_BASE_MORTALITY = 2
SITE_S_AVAIL_N = 3


@jit.rawkernel(device="cuda")
def gap_aggregate_step(
    tick,
    agent_index,
    globals,
    agent_ids,
    breeds,
    locations,
    params_tensor,
    states_tensor,
    states_db_tensor,
):
    """
    Gap aggregate step (priority 1).

    Reads from tree neighbors:
    - states: litter_c, litter_n, n_demand

    Writes to own:
    - params: total_n_demand (internal)
    - states: litter_accum_c, litter_accum_n
    """
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
            # Read tree's litter from states
            tree_litter_c = states_tensor[neighbor_idx][TREE_S_LITTER_C]
            tree_litter_n = states_tensor[neighbor_idx][TREE_S_LITTER_N]
            total_litter_c = total_litter_c + tree_litter_c
            total_litter_n = total_litter_n + tree_litter_n

            # Read tree's N demand from states
            tree_n_demand = states_tensor[neighbor_idx][TREE_S_N_DEMAND]
            total_n_dem = total_n_dem + tree_n_demand

        i = i + 1

    # Write to own params (internal)
    params_tensor[agent_index][GAP_P_TOTAL_N_DEMAND] = total_n_dem

    # Write to own states (Site will read this)
    states_tensor[agent_index][GAP_S_LITTER_ACCUM_C] = total_litter_c
    states_tensor[agent_index][GAP_S_LITTER_ACCUM_N] = total_litter_n


@jit.rawkernel(device="cuda")
def gap_sync_step(
    tick,
    agent_index,
    globals,
    agent_ids,
    breeds,
    locations,
    params_tensor,
    states_tensor,
    states_db_tensor,
):
    """
    Gap sync step (priority 3).

    Reads from site neighbor:
    - states: deg_days, dry_days, base_mortality, avail_n

    Writes to own states:
    - climate: copied from Site (for dynamic climate support)
    - n_supply_ratio: avail_n / total_n_demand
    """
    # Find Site neighbor and read climate + avail_n
    site_deg_days = 2500.0  # Default
    site_dry_days = 30.0
    site_base_mortality = 0.02
    site_avail_n = 0.1

    neighbor_indices = locations[agent_index]
    i = 0
    while i < len(neighbor_indices) and neighbor_indices[i] != -1:
        neighbor_idx = int(neighbor_indices[i])
        neighbor_breed = int(breeds[neighbor_idx])

        if neighbor_breed == BREED_SITE:
            # Read Site's climate and avail_n from states
            site_deg_days = states_tensor[neighbor_idx][SITE_S_DEG_DAYS]
            site_dry_days = states_tensor[neighbor_idx][SITE_S_DRY_DAYS]
            site_base_mortality = states_tensor[neighbor_idx][SITE_S_BASE_MORTALITY]
            site_avail_n = states_tensor[neighbor_idx][SITE_S_AVAIL_N]

        i = i + 1

    # Copy climate to own states (for trees to read next tick)
    states_tensor[agent_index][GAP_S_DEG_DAYS] = site_deg_days
    states_tensor[agent_index][GAP_S_DRY_DAYS] = site_dry_days
    states_tensor[agent_index][GAP_S_BASE_MORTALITY] = site_base_mortality
    states_tensor[agent_index][GAP_S_AVAIL_N] = site_avail_n

    # Calculate N supply/demand ratio
    total_n_dem = params_tensor[agent_index][GAP_P_TOTAL_N_DEMAND]
    n_supply_ratio = 1.0
    if total_n_dem > 0.0001:
        n_supply_ratio = site_avail_n / total_n_dem
        if n_supply_ratio > 2.0:
            n_supply_ratio = 2.0  # Cap at 2x supply

    states_tensor[agent_index][GAP_S_N_SUPPLY_RATIO] = n_supply_ratio

    # Clear litter accumulator for next tick
    states_tensor[agent_index][GAP_S_LITTER_ACCUM_C] = 0.0
    states_tensor[agent_index][GAP_S_LITTER_ACCUM_N] = 0.0
