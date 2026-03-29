"""
Gap litter aggregate step function for GGap model (Priority 0).
First step in the tick - aggregates litter from previous tick's tree output.

Species traits (EVERGREEN, LEAFDIAM_A, MAX_DIAM) now read from species_traits
via the tree's species_id, instead of from tree params.

Execution Flow:
    1. Zero cumulative LAI bins in gap_lai and avail_spec flags in gap_species
    2. Loop through Tree neighbors:
       - Sum litter_c, litter_n from living trees (written at P7 of previous tick)
       - Read species_id from tree params → look up EVERGREEN, LEAFDIAM_A, MAX_DIAM from globals
       - Compute per-tree LAI, distribute across height-layer bins (dec/con split)
       - Check avail_spec: set flag if mature tree of species exists
       - Sum seedling weights from templates (for proportional decrement at P5)
    3. Top-down cumulative prefix sum over LAI bins (trees read O(1) at P3/P5)
    4. Write aggregated litter, total_lai, seedling_weight
"""

import cupy as cp  # noqa: F401
from cupyx import jit

from gap.constants import (
    Breed, Trait, TreeS, GapS,
    STD_HT, PLOTSIZE, MAX_HEIGHT_BINS,
)



@jit.rawkernel(device="cuda")
def gap_litter_aggregate_step(
    tick,
    agent_index,
    species_traits, site_configs, rangelists, site_distances,
    agent_ids,
    breeds,
    locations,
    params_tensor,
    states_tensor,
    gap_lai, gap_species, site_species,
    gap_lai_idx, gap_species_idx, site_species_idx,
):
    """
    Gap litter aggregate step (priority 0).

    Aggregates litter from tree neighbors (written at P7 of previous tick).
    Bins LAI by height layer and computes top-down cumulative sums.
    Sets avail_spec flags for species with mature trees.
    Species traits (EVERGREEN, LEAFDIAM_A, MAX_DIAM) read from globals.
    """
    total_litter_c = 0.0
    total_litter_n = 0.0
    total_seedling_weight = 0.0
    total_lai = 0.0

    # Get breed-local array rows for this gap
    lai_row = gap_lai_idx[agent_index]
    sp_row = gap_species_idx[agent_index]

    # --- 1. Zero the LAI bins and avail_spec flags ---
    for k in range(MAX_HEIGHT_BINS):
        gap_lai[lai_row][k][0] = 0.0
        gap_lai[lai_row][k][1] = 0.0
    sp = 0
    while sp < len(species_traits):
        gap_species[sp_row][sp] = 0.0
        sp = sp + 1

    # --- 2. Loop through Tree neighbors ---
    neighbor_indices = locations[agent_index]
    i = 0
    while i < len(neighbor_indices) and neighbor_indices[i] != -1:
        neighbor_idx = int(neighbor_indices[i])
        neighbor_breed = int(breeds[neighbor_idx])

        if neighbor_breed == Breed.TREE:
            tree_alive = states_tensor[neighbor_idx][TreeS.IS_ALIVE]

            # Read litter from ALL trees (alive + recently dead)
            if tree_alive > -0.5:
                tree_litter_c = states_tensor[neighbor_idx][TreeS.LITTER_C]
                tree_litter_n = states_tensor[neighbor_idx][TreeS.LITTER_N]
                total_litter_c = total_litter_c + tree_litter_c
                total_litter_n = total_litter_n + tree_litter_n

            if tree_alive > 0.5:
                # === Living tree: compute LAI and bin by height layer ===
                n_diam = states_tensor[neighbor_idx][TreeS.DIAM]
                n_height = states_tensor[neighbor_idx][TreeS.HEIGHT]
                n_canopy_ht = states_tensor[neighbor_idx][TreeS.CANOPY_HT]

                # Read species traits from globals
                n_species_id = int(states_tensor[neighbor_idx][TreeS.SPECIES_ID])
                n_leafdiam_a = species_traits[int(n_species_id)][Trait.LEAFDIAM_A]
                n_evergreen = int(species_traits[int(n_species_id)][Trait.EVERGREEN])
                n_max_diam = species_traits[int(n_species_id)][Trait.MAX_DIAM]

                # Canopy diameter
                n_dc = n_diam
                if n_height > n_canopy_ht and n_height > STD_HT:
                    n_dc = (n_height - n_canopy_ht) / (n_height - STD_HT) * n_diam

                # LAI
                n_lai = n_dc * n_dc * n_leafdiam_a

                # Accumulate total LAI for soil water balance
                if n_height > STD_HT + 0.1:
                    total_lai = total_lai + n_lai

                # Layer indices
                canht_int = int(n_canopy_ht) - 1
                if canht_int < 0:
                    canht_int = 0
                forht_int = int(n_height) - 1
                if forht_int < 0:
                    forht_int = 0
                if forht_int > 49:
                    forht_int = 49

                n_canopy_layers = forht_int - canht_int + 1
                if n_canopy_layers < 1:
                    n_canopy_layers = 1

                # Distribute LAI across height layers
                lai_per_layer = n_lai / float(n_canopy_layers)
                for layer in range(n_canopy_layers):
                    bin_idx = canht_int + layer
                    if bin_idx >= 0 and bin_idx < MAX_HEIGHT_BINS:
                        if n_evergreen > 0:
                            gap_lai[lai_row][bin_idx][0] = gap_lai[lai_row][bin_idx][0] + lai_per_layer
                            gap_lai[lai_row][bin_idx][1] = gap_lai[lai_row][bin_idx][1] + lai_per_layer
                        else:
                            gap_lai[lai_row][bin_idx][0] = gap_lai[lai_row][bin_idx][0] + lai_per_layer
                            gap_lai[lai_row][bin_idx][1] = gap_lai[lai_row][bin_idx][1] + lai_per_layer * 0.8

                # Check avail_spec: mature tree of this species
                if n_species_id >= 0 and n_species_id < len(species_traits):
                    if n_diam > n_max_diam * 0.05:
                        gap_species[sp_row][n_species_id] = 1.0

            elif tree_alive < -0.5:
                # Template: read seedling weight for proportional decrement
                template_weight = states_tensor[neighbor_idx][TreeS.SEEDLING_WEIGHT]
                total_seedling_weight = total_seedling_weight + template_weight

        i = i + 1

    # --- 3. Top-down cumulative prefix sum ---
    for k in range(49):
        layer = 48 - k
        gap_lai[lai_row][layer][0] = gap_lai[lai_row][layer][0] + gap_lai[lai_row][layer + 1][0]
        gap_lai[lai_row][layer][1] = gap_lai[lai_row][layer][1] + gap_lai[lai_row][layer + 1][1]

    # --- 4. Write aggregated outputs ---
    states_tensor[agent_index][GapS.LITTER_ACCUM_C] = total_litter_c
    states_tensor[agent_index][GapS.LITTER_ACCUM_N] = total_litter_n

    total_lai_scaled = total_lai / PLOTSIZE
    states_tensor[agent_index][GapS.TOTAL_LAI] = total_lai_scaled

    states_tensor[agent_index][GapS.TOTAL_SEEDLING_WEIGHT] = total_seedling_weight
