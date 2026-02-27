"""
Gap litter aggregate step function for GGap model (Priority 0).
First step in the tick - aggregates litter from previous tick's tree output.

Matches GAPpy where bio_geo_climate() (soil) runs first each year.
Litter must be aggregated before site_soil_step (P1) can decompose it.

Execution Flow:
    1. Loop through Tree neighbors
    2. Sum litter_c, litter_n from living trees (written at P8 of previous tick)
    3. Count living trees and free slots
    4. Compute per-gap LAI from living tree canopy
    5. Sum seedling weights from templates (for proportional decrement at P6)

    Writes: litter_accum_c/n (above-ground), total_lai (for soil water balance),
            total_seedling_weight

Note: N consumed aggregation moved to gap_nconsumed_aggregate_step (P9)
which runs AFTER P8, enabling same-tick N balance at P10.
"""

import cupy as cp  # noqa: F401
from cupyx import jit

# === Breed IDs ===
BREED_TREE = 0
BREED_GAP = 1
BREED_SITE = 2

# === Gap states[16] (public, no buffer) ===
GAP_S_LITTER_ACCUM_C = 4
GAP_S_LITTER_ACCUM_N = 5
GAP_S_TOTAL_SEEDLING_WEIGHT = 9  # Sum of all templates' seedling weights (for proportional decrement)
GAP_S_TOTAL_LAI = 12  # Per-gap normalized LAI (sum of tree LAI / PLOTSIZE, GAPpy canopy())

# === Tree params (for reading from Tree neighbors) ===
TREE_P_LEAFDIAM_A = 38  # Species-specific leaf area coefficient

# === Tree states[5] (for reading litter from Tree neighbors) ===
TREE_S_LITTER_C = 0       # Above-ground litter carbon
TREE_S_LITTER_N = 1       # Above-ground litter nitrogen

# === Tree states_db[5] (for checking alive status + dimensions + seedling weight) ===
TREE_DB_IS_ALIVE = 0
TREE_DB_DIAM = 1
TREE_DB_HEIGHT = 2
TREE_DB_CANOPY_HT = 3
TREE_DB_SEEDLING_WEIGHT = 4

# === Constants ===
STD_HT = 1.3  # Standard measurement height (m)
PLOTSIZE = 500.0  # GAPpy parameters.py:59 — plot area m² (also max trees per plot)


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
    total_seedling_weight = 0.0
    total_lai = 0.0  # Sum of tree LAI for canopy water interception (GAPpy canopy())

    neighbor_indices = locations[agent_index]
    i = 0
    while i < len(neighbor_indices) and neighbor_indices[i] != -1:
        neighbor_idx = int(neighbor_indices[i])
        neighbor_breed = int(breeds[neighbor_idx])

        if neighbor_breed == BREED_TREE:
            tree_alive = states_db_tensor[neighbor_idx][TREE_DB_IS_ALIVE]

            # Read litter from ALL trees (alive + recently dead)
            # When a tree dies at P6, is_alive=0 in states_db but litter is in states.
            # Free slot/template trees have litter=0, so reading is harmless.
            if tree_alive > -0.5:
                # Read above-ground litter (written at P6 of previous tick)
                tree_litter_c = states_tensor[neighbor_idx][TREE_S_LITTER_C]
                tree_litter_n = states_tensor[neighbor_idx][TREE_S_LITTER_N]
                total_litter_c = total_litter_c + tree_litter_c
                total_litter_n = total_litter_n + tree_litter_n

            if tree_alive > 0.5:
                # Compute tree LAI for canopy water interception (GAPpy canopy():311)
                # LAI = dc² * leafdiam_a, where dc = (h-hc)/(h-STD_HT)*diam
                n_diam = states_db_tensor[neighbor_idx][TREE_DB_DIAM]
                n_height = states_db_tensor[neighbor_idx][TREE_DB_HEIGHT]
                n_canopy_ht = states_db_tensor[neighbor_idx][TREE_DB_CANOPY_HT]
                n_leafdiam_a = params_tensor[neighbor_idx][TREE_P_LEAFDIAM_A]
                if n_height > STD_HT + 0.1:
                    n_dc = (n_height - n_canopy_ht) / (n_height - STD_HT) * n_diam
                    total_lai = total_lai + n_dc * n_dc * n_leafdiam_a
            elif tree_alive < -0.5:
                # Template (is_alive == -1): read seedling weight for proportional decrement
                template_weight = states_db_tensor[neighbor_idx][TREE_DB_SEEDLING_WEIGHT]
                total_seedling_weight = total_seedling_weight + template_weight

        i = i + 1

    # Write aggregated litter (Site reads at P1)
    states_tensor[agent_index][GAP_S_LITTER_ACCUM_C] = total_litter_c      # Above-ground -> A0
    states_tensor[agent_index][GAP_S_LITTER_ACCUM_N] = total_litter_n

    # Write per-gap normalized LAI (GAPpy canopy():362 divides by numplots*plotsize)
    # PLOTSIZE matches GAPpy plotsize=500; Site averages across gaps (= /numplots)
    gap_lai = total_lai / PLOTSIZE
    states_tensor[agent_index][GAP_S_TOTAL_LAI] = gap_lai

    # Write total seedling weight for proportional decrement (templates read at P6)
    # GAPpy model.py:941: seedling[species] -= 1.0 per recruited tree
    # GPU approximation: each template subtracts its proportional share
    states_tensor[agent_index][GAP_S_TOTAL_SEEDLING_WEIGHT] = total_seedling_weight
