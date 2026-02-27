"""
Gap N demand aggregate step function for GGap model (Priority 4).
Aggregates nitrogen demand from living trees after tree_potential_growth_step (P3).

Recruitment count is now handled by gap_recruit_aggregate_step (P7),
which reads template regrowth after tree_template_renewal_step (P6).

Execution Flow:
    1. Loop through Tree neighbors
    2. Sum n_demand from living trees (written at P3, same tick)
    3. Write total_n_demand to Gap states (own Gap reads at P5 for per-gap ratio)
"""

import cupy as cp  # noqa: F401
from cupyx import jit

# === Breed IDs ===
BREED_TREE = 0
BREED_GAP = 1
BREED_SITE = 2

# === Gap params[2] (private) ===
GAP_P_TOTAL_N_DEMAND = 1

# === Gap states[16] (public, no buffer) ===
GAP_S_TOTAL_N_DEMAND = 11  # Public slot for Site to read

# === Tree states[5] (for reading n_demand) ===
TREE_S_N_DEMAND = 2

# === Tree states_db[5] (for checking alive status) ===
TREE_DB_IS_ALIVE = 0


@jit.rawkernel(device="cuda")
def gap_demand_aggregate_step(
    tick,
    agent_index,
    globals_data,
    agent_ids,
    breeds,
    locations,
    params_tensor,
    states_tensor,
    states_db_tensor,
):
    """
    Gap N demand aggregate step (priority 3).

    Reads n_demand from living tree neighbors (written at P3, same tick).
    Writes total_n_demand to Gap states (public, Site reads at P4).
    """
    total_n_dem = 0.0

    neighbor_indices = locations[agent_index]
    i = 0
    while i < len(neighbor_indices) and neighbor_indices[i] != -1:
        neighbor_idx = int(neighbor_indices[i])
        neighbor_breed = int(breeds[neighbor_idx])

        if neighbor_breed == BREED_TREE:
            tree_alive = states_db_tensor[neighbor_idx][TREE_DB_IS_ALIVE]
            if tree_alive > 0.5:
                # Living tree: sum N demand
                tree_n_demand = states_tensor[neighbor_idx][TREE_S_N_DEMAND]
                total_n_dem = total_n_dem + tree_n_demand

        i = i + 1

    # Write to params (internal)
    params_tensor[agent_index][GAP_P_TOTAL_N_DEMAND] = total_n_dem

    # Write to states (public, Site reads at P4)
    states_tensor[agent_index][GAP_S_TOTAL_N_DEMAND] = total_n_dem
