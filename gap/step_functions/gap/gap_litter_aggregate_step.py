"""
Gap litter aggregate step function for GGap model (Priority 0).
First step in the tick - aggregates litter from previous tick's tree output.

Matches GAPpy where bio_geo_climate() (soil) runs first each year.
Litter must be aggregated before site_soil_step (P1) can decompose it.

Execution Flow:
    1. Loop through Tree neighbors
    2. Sum litter_c, litter_n from living trees (written at P6 of previous tick)
    3. Count living trees and dormant slots
    4. Read growmax from templates (env_stress = regrowth, written at P6 prev tick)
    5. Compute density-based nrenew (GAPpy model.py:833-837):
       nrenew = min(max(plotsize*growmax - numtrees, 3), plotsize - numtrees)
       capped by available dormant slots
    6. Generate random seed for species selection

    Writes: litter_accum_c/n (above-ground), litter_accum_c/n_bg (below-ground),
            num_to_recruit, recruit_rand_seed
"""

import cupy as cp  # noqa: F401
from cupyx import jit

# === Breed IDs ===
BREED_TREE = 0
BREED_GAP = 1
BREED_SITE = 2

# === Gap states[15] (public, no buffer) ===
GAP_S_LITTER_ACCUM_C = 5
GAP_S_LITTER_ACCUM_N = 6
GAP_S_NUM_TO_RECRUIT = 7
GAP_S_RECRUIT_RAND_SEED = 8
GAP_S_LITTER_ACCUM_C_BG = 13  # Below-ground litter carbon aggregate
GAP_S_LITTER_ACCUM_N_BG = 14  # Below-ground litter nitrogen aggregate

# === Tree params (for reading env_stress/regrowth from templates) ===
TREE_P_ENV_STRESS = 32  # Templates store regrowth here (written at P6 prev tick)

# === Tree states[5] (for reading litter from Tree neighbors) ===
TREE_S_LITTER_C = 0       # Above-ground litter carbon
TREE_S_LITTER_N = 1       # Above-ground litter nitrogen
TREE_S_LITTER_C_BG = 3    # Below-ground litter carbon
TREE_S_LITTER_N_BG = 4    # Below-ground litter nitrogen

# === Tree states_db[4] (for checking alive status) ===
TREE_DB_IS_ALIVE = 0


@jit.rawkernel(device="cuda")
def gap_litter_aggregate_step(
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
    Gap litter aggregate step (priority 0).

    Aggregates litter from tree neighbors (written at P6 of previous tick).
    Also handles seed bank and recruitment count.
    Must run before site_soil_step (P1) which reads aggregated litter.
    """
    total_litter_c = 0.0
    total_litter_n = 0.0
    total_litter_c_bg = 0.0
    total_litter_n_bg = 0.0
    living_tree_count = 0.0
    dormant_tree_count = 0.0
    growmax = 0.0

    neighbor_indices = locations[agent_index]
    i = 0
    while i < len(neighbor_indices) and neighbor_indices[i] != -1:
        neighbor_idx = int(neighbor_indices[i])
        neighbor_breed = int(breeds[neighbor_idx])

        if neighbor_breed == BREED_TREE:
            tree_alive = states_db_tensor[neighbor_idx][TREE_DB_IS_ALIVE]

            # Read litter from ALL trees (alive + recently dead)
            # When a tree dies at P6, is_alive=0 in states_db but litter is in states.
            # Dormant/template trees have litter=0, so reading is harmless.
            if tree_alive > -0.5:
                # Read above-ground litter (written at P6 of previous tick)
                tree_litter_c = states_tensor[neighbor_idx][TREE_S_LITTER_C]
                tree_litter_n = states_tensor[neighbor_idx][TREE_S_LITTER_N]
                total_litter_c = total_litter_c + tree_litter_c
                total_litter_n = total_litter_n + tree_litter_n

                # Read below-ground litter (roots from dead trees)
                tree_litter_c_bg = states_tensor[neighbor_idx][TREE_S_LITTER_C_BG]
                tree_litter_n_bg = states_tensor[neighbor_idx][TREE_S_LITTER_N_BG]
                total_litter_c_bg = total_litter_c_bg + tree_litter_c_bg
                total_litter_n_bg = total_litter_n_bg + tree_litter_n_bg

            if tree_alive > 0.5:
                living_tree_count = living_tree_count + 1.0
            elif tree_alive > -0.5:
                # Dormant slot (is_alive == 0), not template (is_alive == -1)
                dormant_tree_count = dormant_tree_count + 1.0
            else:
                # Template (is_alive == -1): read regrowth for growmax
                template_regrowth = params_tensor[neighbor_idx][TREE_P_ENV_STRESS]
                if template_regrowth > growmax:
                    growmax = template_regrowth

        i = i + 1

    # Density-based recruitment count (GAPpy model.py:833-837)
    # total_capacity = living + dormant (acts as plotsize equivalent)
    total_capacity = living_tree_count + dormant_tree_count
    num_to_recruit = 0.0

    if dormant_tree_count > 0.5 and total_capacity > 0.5:
        # max_renew = min(plotsize * growmax - numtrees, plotsize * 0.5)
        max_renew = total_capacity * growmax - living_tree_count
        half_cap = total_capacity * 0.5
        if max_renew > half_cap:
            max_renew = half_cap

        # nrenew = min(max(max_renew, 3), plotsize - numtrees)
        nrenew = max_renew
        if nrenew < 3.0:
            nrenew = 3.0
        cap = total_capacity - living_tree_count
        if nrenew > cap:
            nrenew = cap

        # Cap by available dormant slots
        if nrenew > dormant_tree_count:
            nrenew = dormant_tree_count
        if nrenew < 0.0:
            nrenew = 0.0

        num_to_recruit = nrenew

    # Generate pseudo-random seed for species selection
    recruit_rand_seed = float((tick * 997 + agent_index * 991) % 10000)

    # Write aggregated litter (Site reads at P1)
    states_tensor[agent_index][GAP_S_LITTER_ACCUM_C] = total_litter_c      # Above-ground -> A0
    states_tensor[agent_index][GAP_S_LITTER_ACCUM_N] = total_litter_n
    states_tensor[agent_index][GAP_S_LITTER_ACCUM_C_BG] = total_litter_c_bg  # Below-ground -> A layer
    states_tensor[agent_index][GAP_S_LITTER_ACCUM_N_BG] = total_litter_n_bg

    # Write recruitment info (Trees read at P2)
    states_tensor[agent_index][GAP_S_NUM_TO_RECRUIT] = num_to_recruit
    states_tensor[agent_index][GAP_S_RECRUIT_RAND_SEED] = recruit_rand_seed
