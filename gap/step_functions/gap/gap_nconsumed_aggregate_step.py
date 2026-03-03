"""
Gap N consumed aggregate step function for GGap model (Priority 8).
Aggregates actual N consumed from trees after tree_actual_growth_step (P7).

This runs AFTER P7 so it reads the CURRENT tick's n_consumed values,
enabling same-tick N balance at P9 (no 1-tick delay).

Execution Flow:
    1. Loop through Tree neighbors
    2. Sum n_consumed from living trees and newly recruited seedlings (written at P7, same tick)
    3. Write total to Gap states (Site reads at P9 for N balance)
"""

import cupy as cp  # noqa: F401
from cupyx import jit

# === Breed IDs ===
BREED_TREE = 0
BREED_GAP = 1
BREED_SITE = 2

# === Gap states[16] (public, no buffer) ===
GAP_S_N_CONSUMED = 13  # Total N consumed by trees (for P9 N balance)

# === Tree states[5] (for reading n_consumed) ===
TREE_S_N_CONSUMED = 3

# === Tree states_db[5] (for checking alive status) ===
TREE_DB_IS_ALIVE = 0


@jit.rawkernel(device="cuda")
def gap_nconsumed_aggregate_step(
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
    Gap N consumed aggregate step (priority 8).

    Reads n_consumed from tree neighbors (written at P7, same tick).
    Writes total to Gap states (Site reads at P9 for same-tick N balance).
    """
    total_n_consumed = 0.0

    neighbor_indices = locations[agent_index]
    i = 0
    while i < len(neighbor_indices) and neighbor_indices[i] != -1:
        neighbor_idx = int(neighbor_indices[i])
        neighbor_breed = int(breeds[neighbor_idx])

        if neighbor_breed == BREED_TREE:
            tree_alive = states_db_tensor[neighbor_idx][TREE_DB_IS_ALIVE]
            # Read from all non-template trees (alive + recently dead + newly recruited)
            # Templates (is_alive == -1) have n_consumed = 0, so reading is harmless
            if tree_alive > -0.5:
                tree_n_consumed = states_tensor[neighbor_idx][TREE_S_N_CONSUMED]
                total_n_consumed = total_n_consumed + tree_n_consumed

        i = i + 1

    states_tensor[agent_index][GAP_S_N_CONSUMED] = total_n_consumed
