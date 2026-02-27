"""
Gap litter aggregate step function for GGap model (Priority 0).
First step in the tick - aggregates litter from previous tick's tree output.

Matches GAPpy where bio_geo_climate() (soil) runs first each year.
Litter must be aggregated before site_soil_step (P1) can decompose it.

Execution Flow:
    1. Zero cumulative LAI bins (50 dec + 50 con) and avail_spec flags (50)
    2. Loop through Tree neighbors:
       - Sum litter_c, litter_n from living trees (written at P8 of previous tick)
       - Compute per-tree LAI, distribute across height-layer bins (dec/con split)
       - Check avail_spec: set flag if mature tree of species exists
       - Sum seedling weights from templates (for proportional decrement at P6)
    3. Top-down cumulative prefix sum over LAI bins (trees read O(1) at P3/P6)
    4. Write aggregated litter, total_lai, seedling_weight

    Writes: litter_accum_c/n (above-ground), total_lai (for soil water balance),
            total_seedling_weight, cum_dec_lai[0..49], cum_con_lai[0..49],
            avail_spec[0..49]

Note: N consumed aggregation moved to gap_nconsumed_aggregate_step (P9)
which runs AFTER P8, enabling same-tick N balance at P10.
"""

import cupy as cp  # noqa: F401
from cupyx import jit

# === Breed IDs ===
BREED_TREE = 0
BREED_GAP = 1
BREED_SITE = 2

# === Gap states (public, no buffer) ===
GAP_S_LITTER_ACCUM_C = 4
GAP_S_LITTER_ACCUM_N = 5
GAP_S_TOTAL_SEEDLING_WEIGHT = 9  # Sum of all templates' seedling weights (for proportional decrement)
GAP_S_TOTAL_LAI = 12  # Per-gap normalized LAI (sum of tree LAI / PLOTSIZE, GAPpy canopy())

# Pre-aggregated light competition bins (P3/P6 read O(1))
GAP_S_CUM_DEC_LAI_BASE = 16   # cum_dec_lai[0..49] at slots 16-65
GAP_S_CUM_CON_LAI_BASE = 66   # cum_con_lai[0..49] at slots 66-115
GAP_S_AVAIL_SPEC_BASE = 116   # avail_spec[0..49] at slots 116-165
MAX_HEIGHT_BINS = 50
MAX_SPECIES = 50

# === Tree params (for reading from Tree neighbors) ===
TREE_P_SPECIES_ID = 0
TREE_P_MAX_DIAM = 2
TREE_P_EVERGREEN = 17
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

    Aggregates litter from tree neighbors (written at P8 of previous tick).
    Bins LAI by height layer and computes top-down cumulative sums.
    Sets avail_spec flags for species with mature trees.
    Must run before site_soil_step (P1) which reads aggregated litter.
    """
    total_litter_c = 0.0
    total_litter_n = 0.0
    total_seedling_weight = 0.0
    total_lai = 0.0  # Sum of tree LAI for canopy water interception (GAPpy canopy())

    # --- 1. Zero the 150 new accumulator slots (3 × 50) ---
    for k in range(MAX_HEIGHT_BINS):
        states_tensor[agent_index][GAP_S_CUM_DEC_LAI_BASE + k] = 0.0
        states_tensor[agent_index][GAP_S_CUM_CON_LAI_BASE + k] = 0.0
    for k in range(MAX_SPECIES):
        states_tensor[agent_index][GAP_S_AVAIL_SPEC_BASE + k] = 0.0

    # --- 2. Loop through Tree neighbors ---
    neighbor_indices = locations[agent_index]
    i = 0
    while i < len(neighbor_indices) and neighbor_indices[i] != -1:
        neighbor_idx = int(neighbor_indices[i])
        neighbor_breed = int(breeds[neighbor_idx])

        if neighbor_breed == BREED_TREE:
            tree_alive = states_db_tensor[neighbor_idx][TREE_DB_IS_ALIVE]

            # Read litter from ALL trees (alive + recently dead)
            # When a tree dies at P8, is_alive=0 in states_db but litter is in states.
            # Free slot/template trees have litter=0, so reading is harmless.
            if tree_alive > -0.5:
                tree_litter_c = states_tensor[neighbor_idx][TREE_S_LITTER_C]
                tree_litter_n = states_tensor[neighbor_idx][TREE_S_LITTER_N]
                total_litter_c = total_litter_c + tree_litter_c
                total_litter_n = total_litter_n + tree_litter_n

            if tree_alive > 0.5:
                # === Living tree: compute LAI and bin by height layer ===
                n_diam = states_db_tensor[neighbor_idx][TREE_DB_DIAM]
                n_height = states_db_tensor[neighbor_idx][TREE_DB_HEIGHT]
                n_canopy_ht = states_db_tensor[neighbor_idx][TREE_DB_CANOPY_HT]
                n_leafdiam_a = params_tensor[neighbor_idx][TREE_P_LEAFDIAM_A]
                n_evergreen = int(params_tensor[neighbor_idx][TREE_P_EVERGREEN])

                # Canopy diameter (GAPpy tree.py stem_shape: dc = (h-hc)/(h-STD_HT)*d)
                n_dc = n_diam
                if n_height > n_canopy_ht and n_height > STD_HT:
                    n_dc = (n_height - n_canopy_ht) / (n_height - STD_HT) * n_diam

                # LAI (GAPpy tree.py:174-181: lai = dc^2 * leafdiam_a)
                n_lai = n_dc * n_dc * n_leafdiam_a

                # Accumulate total LAI for soil water balance (existing behavior)
                if n_height > STD_HT + 0.1:
                    total_lai = total_lai + n_lai

                # Layer indices (GAPpy 0-based: canht_int, forht_int)
                canht_int = int(n_canopy_ht) - 1
                if canht_int < 0:
                    canht_int = 0
                forht_int = int(n_height) - 1
                if forht_int < 0:
                    forht_int = 0
                if forht_int > 49:
                    forht_int = 49

                # Number of canopy layers
                n_canopy_layers = forht_int - canht_int + 1
                if n_canopy_layers < 1:
                    n_canopy_layers = 1

                # Distribute LAI across height layers (GAPpy model.py:307-334)
                lai_per_layer = n_lai / float(n_canopy_layers)
                for layer in range(n_canopy_layers):
                    bin_idx = canht_int + layer
                    if bin_idx >= 0 and bin_idx < MAX_HEIGHT_BINS:
                        if n_evergreen > 0:
                            # Conifers: 100% to both arrays (GAPpy model.py:325-328)
                            states_tensor[agent_index][GAP_S_CUM_DEC_LAI_BASE + bin_idx] = states_tensor[agent_index][GAP_S_CUM_DEC_LAI_BASE + bin_idx] + lai_per_layer
                            states_tensor[agent_index][GAP_S_CUM_CON_LAI_BASE + bin_idx] = states_tensor[agent_index][GAP_S_CUM_CON_LAI_BASE + bin_idx] + lai_per_layer
                        else:
                            # Deciduous: 100% to dec, 80% to con (GAPpy model.py:331-334)
                            states_tensor[agent_index][GAP_S_CUM_DEC_LAI_BASE + bin_idx] = states_tensor[agent_index][GAP_S_CUM_DEC_LAI_BASE + bin_idx] + lai_per_layer
                            states_tensor[agent_index][GAP_S_CUM_CON_LAI_BASE + bin_idx] = states_tensor[agent_index][GAP_S_CUM_CON_LAI_BASE + bin_idx] + lai_per_layer * 0.8

                # Check avail_spec: mature tree of this species
                n_species_id = int(params_tensor[neighbor_idx][TREE_P_SPECIES_ID])
                n_max_diam = params_tensor[neighbor_idx][TREE_P_MAX_DIAM]
                if n_species_id >= 0 and n_species_id < MAX_SPECIES:
                    if n_diam > n_max_diam * 0.05:
                        states_tensor[agent_index][GAP_S_AVAIL_SPEC_BASE + n_species_id] = 1.0

            elif tree_alive < -0.5:
                # Template (is_alive == -1): read seedling weight for proportional decrement
                template_weight = states_db_tensor[neighbor_idx][TREE_DB_SEEDLING_WEIGHT]
                total_seedling_weight = total_seedling_weight + template_weight

        i = i + 1

    # --- 3. Top-down cumulative prefix sum (GAPpy model.py:337-345) ---
    # After this, cum_lai[layer] = total LAI at layer and above (what trees need for Beer-Lambert)
    for k in range(49):
        layer = 48 - k
        states_tensor[agent_index][GAP_S_CUM_DEC_LAI_BASE + layer] = states_tensor[agent_index][GAP_S_CUM_DEC_LAI_BASE + layer] + states_tensor[agent_index][GAP_S_CUM_DEC_LAI_BASE + layer + 1]
        states_tensor[agent_index][GAP_S_CUM_CON_LAI_BASE + layer] = states_tensor[agent_index][GAP_S_CUM_CON_LAI_BASE + layer] + states_tensor[agent_index][GAP_S_CUM_CON_LAI_BASE + layer + 1]

    # --- 4. Write aggregated outputs ---
    # Litter (Site reads at P1)
    states_tensor[agent_index][GAP_S_LITTER_ACCUM_C] = total_litter_c
    states_tensor[agent_index][GAP_S_LITTER_ACCUM_N] = total_litter_n

    # Per-gap normalized LAI (GAPpy canopy():362 divides by numplots*plotsize)
    # PLOTSIZE matches GAPpy plotsize=500; Site averages across gaps (= /numplots)
    gap_lai = total_lai / PLOTSIZE
    states_tensor[agent_index][GAP_S_TOTAL_LAI] = gap_lai

    # Total seedling weight for proportional decrement (templates read at P6)
    states_tensor[agent_index][GAP_S_TOTAL_SEEDLING_WEIGHT] = total_seedling_weight
