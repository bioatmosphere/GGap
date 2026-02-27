"""
Gap recruitment aggregate step function for GGap model (Priority 7).
Reads template regrowth (written at P6, same tick) to compute growmax
and density-based num_to_recruit.

Runs AFTER tree_template_renewal_step (P6) so it reads current-tick
regrowth from template params (no double buffer, immediately visible).

Execution Flow:
    1. Loop through Tree neighbors
    2. Count living trees and free slots
    3. Read growmax from templates (params[ENV_STRESS], written at P6 same tick)
    4. Compute density-based num_to_recruit (GAPpy model.py:833-837)
    5. Convert num_to_recruit to per-slot probability (nrenew / free_slots)
    6. Generate random seed for species selection
    7. Write recruit_prob, recruit_rand_seed to Gap states
"""

import cupy as cp  # noqa: F401
from cupyx import jit

# === Breed IDs ===
BREED_TREE = 0
BREED_GAP = 1
BREED_SITE = 2

# === Gap states[16] (public, no buffer) ===
GAP_S_NUM_TO_RECRUIT = 6
GAP_S_RECRUIT_RAND_SEED = 7

# === Tree params (for reading template regrowth) ===
TREE_P_ENV_STRESS = 32  # Templates store regrowth here (written at P6, same tick)

# === Constants ===
PLOTSIZE = 500.0  # GAPpy parameters.py:59 — plot area m² (also max trees per plot)

# === Tree states_db[5] (for checking alive status) ===
TREE_DB_IS_ALIVE = 0


@jit.rawkernel(device="cuda")
def gap_recruit_aggregate_step(
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
    Gap recruitment aggregate step (priority 7).

    Reads regrowth from template neighbors (written at P6, same tick).
    Computes num_to_recruit from growmax (GAPpy model.py:833-837).
    Writes recruitment info to Gap states for P8 free slots to read.
    """
    living_tree_count = 0.0
    free_slot_tree_count = 0.0
    growmax = 0.0

    neighbor_indices = locations[agent_index]
    i = 0
    while i < len(neighbor_indices) and neighbor_indices[i] != -1:
        neighbor_idx = int(neighbor_indices[i])
        neighbor_breed = int(breeds[neighbor_idx])

        if neighbor_breed == BREED_TREE:
            tree_alive = states_db_tensor[neighbor_idx][TREE_DB_IS_ALIVE]
            if tree_alive > 0.5:
                living_tree_count = living_tree_count + 1.0
            elif tree_alive > -0.5:
                # Free slot (is_alive == 0)
                free_slot_tree_count = free_slot_tree_count + 1.0
            else:
                # Template (is_alive == -1): read regrowth for growmax
                template_regrowth = params_tensor[neighbor_idx][TREE_P_ENV_STRESS]
                if template_regrowth > growmax:
                    growmax = template_regrowth

        i = i + 1

    # Density-based recruitment count (GAPpy model.py:819-823)
    # GAPpy uses int() truncation on plotsize*growmax and plotsize*0.5
    num_to_recruit = 0.0

    if free_slot_tree_count > 0.5:
        # max_renew = min(int(plotsize * growmax) - numtrees, int(plotsize * 0.5))
        max_renew = float(int(PLOTSIZE * growmax)) - living_tree_count
        half_cap = float(int(PLOTSIZE * 0.5))
        if max_renew > half_cap:
            max_renew = half_cap

        # nrenew = min(max(max_renew, 3), int(plotsize) - numtrees)
        nrenew = max_renew
        if nrenew < 3.0:
            nrenew = 3.0
        cap = float(int(PLOTSIZE)) - living_tree_count
        if nrenew > cap:
            nrenew = cap

        # Cap by available free slots (GPU guard: can't recruit more than allocated)
        if nrenew > free_slot_tree_count:
            nrenew = free_slot_tree_count
        if nrenew < 0.0:
            nrenew = 0.0

        num_to_recruit = nrenew

    # Convert count to per-slot activation probability for parallel GPU execution.
    # Expected activations = free_slots × (nrenew / free_slots) = nrenew.
    recruit_prob = 0.0
    if num_to_recruit > 0.5 and free_slot_tree_count > 0.5:
        recruit_prob = num_to_recruit / free_slot_tree_count

    # Generate pseudo-random seed for species selection
    recruit_rand_seed = float((tick * 997 + agent_index * 991) % 10000)

    # Write recruitment probability + seed (P8 free slots read from Gap)
    states_tensor[agent_index][GAP_S_NUM_TO_RECRUIT] = recruit_prob
    states_tensor[agent_index][GAP_S_RECRUIT_RAND_SEED] = recruit_rand_seed
