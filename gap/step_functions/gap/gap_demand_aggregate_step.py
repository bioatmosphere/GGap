"""
Gap N demand aggregate + sync step function for GGap model (Priority 4).
Aggregates nitrogen demand from living trees after tree_potential_growth_step (P3),
computes per-gap N supply ratio, and clears accumulators for next tick.

Recruitment count is now handled by gap_recruit_aggregate_step (P6),
which reads template regrowth after tree_template_renewal_step (P5).

Execution Flow:
    1. Loop through Tree neighbors
    2. Sum n_demand from living trees (written at P3, same tick)
    3. Write total_n_demand to Gap states
    4. Compute per-gap N supply ratio (avail_n from P2 / scaled demand)
    5. Clear accumulators consumed by P0/P1
"""

import cupy as cp  # noqa: F401
from cupyx import jit

# === Breed IDs ===
BREED_TREE = 0
BREED_GAP = 1
BREED_SITE = 2

# === Gap params[2] (private) ===
GAP_P_TOTAL_N_DEMAND = 1

# === Gap states (public, no buffer) ===
GAP_S_AVAIL_N = 2             # avail_n (copied from Site at P2 climate relay)
GAP_S_N_SUPPLY_RATIO = 3
GAP_S_LITTER_ACCUM_C = 4
GAP_S_LITTER_ACCUM_N = 5
GAP_S_TOTAL_N_DEMAND = 11     # Public slot for Site to read
GAP_S_TOTAL_LAI = 12          # Per-gap normalized LAI (from P0)
GAP_S_N_CONSUMED = 13         # N consumed by trees (written at P8, read at P9)

# === Tree states[5] (for reading n_demand) ===
TREE_S_N_DEMAND = 2

# === Tree states_db[5] (for checking alive status) ===
TREE_DB_IS_ALIVE = 0

# Unit conversion: kg (tree-level) → tn/ha (soil pools)
# = HEC_TO_M2 / plotsize / 1000 = 10000 / 500 / 1000
UNIT_CONV = 0.02


@jit.rawkernel(device="cuda")
def gap_demand_aggregate_step(
    tick,
    agent_index,
    species_traits, site_configs, rangelists,
    agent_ids,
    breeds,
    locations,
    params_tensor,
    states_tensor,
    states_db_tensor,
):
    """
    Gap N demand aggregate + sync step (priority 4).

    Reads n_demand from living tree neighbors (written at P3, same tick).
    Computes per-gap N supply ratio using avail_n (from P2 climate relay)
    and own N demand. Clears accumulators consumed by P0/P1.
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

    # Write to states (public)
    states_tensor[agent_index][GAP_S_TOTAL_N_DEMAND] = total_n_dem

    # --- Compute per-gap N supply ratio (was P5 gap_sync_step) ---
    # GAPpy computes N_supply_demand per-plot (model.py:475-488):
    #   N_req = max(N_req * HEC_TO_M2 / plotsize, 0.00001)
    #   N_supply_demand = site.soil.avail_N / N_req
    avail_n = states_tensor[agent_index][GAP_S_AVAIL_N]
    gap_n_demand_scaled = total_n_dem * UNIT_CONV
    gap_n_supply_ratio = 1.0
    if gap_n_demand_scaled > 0.00001:
        gap_n_supply_ratio = avail_n / gap_n_demand_scaled
        if gap_n_supply_ratio > 1.0:
            gap_n_supply_ratio = 1.0
    states_tensor[agent_index][GAP_S_N_SUPPLY_RATIO] = gap_n_supply_ratio

    # Clear accumulators (consumed by P1, P0 rewrites next tick)
    states_tensor[agent_index][GAP_S_LITTER_ACCUM_C] = 0.0
    states_tensor[agent_index][GAP_S_LITTER_ACCUM_N] = 0.0
    states_tensor[agent_index][GAP_S_TOTAL_LAI] = 0.0
    states_tensor[agent_index][GAP_S_N_CONSUMED] = 0.0
    # Note: NUM_TO_RECRUIT and RECRUIT_RAND_SEED are NOT cleared here.
    # Trees read them at P7 (free slot activation). P6 overwrites them each tick.
