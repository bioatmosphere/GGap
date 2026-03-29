"""
Site seed dispersal step function for GGap model (Priority 10).
Implements inter-site distance-based seed dispersal matching GAPpy dispersal.py.

Runs at end of tick after all tree dynamics are complete.
Aggregates avail_spec from own gaps (for ghost export to neighbor sites),
then reads neighbor sites' avail_spec (from ghost, previous tick) to
compute imported seeds using a negative exponential kernel.

Execution Flow:
    1. Iterate neighbors: aggregate avail_spec from gap neighbors,
       collect site neighbor indices
    2. Write own averaged avail_spec to SiteS (ghosted to other ranks)
    3. For each neighbor site (ghost data from previous tick):
       - Read pre-computed distance from site_distances global
       - Read neighbor's avail_spec
       - For each species: weight = exp(-distance / max_dispersal_dist)
       - imported_seeds += seed_num * avail * weight
    4. Divide by gap count (distribute per gap, matches GAPpy per_plot)
    5. Write imported_seeds to SiteS (P2 relays to GapS next tick)
"""

import cupy as cp
from cupyx import jit

from gap.constants import (
    Breed, Trait, SiteP,
)


@jit.rawkernel(device="cuda")
def site_seed_dispersal_step(
    tick,
    agent_index,
    species_traits, site_configs, rangelists, site_distances,
    agent_ids,
    breeds,
    locations,
    params_tensor,
    states_tensor,
    states_db_tensor,
    gap_lai, gap_species, site_species,
    gap_lai_idx, gap_species_idx, site_species_idx,
):
    """
    Site seed dispersal step (priority 10).

    Aggregates own avail_spec from gap neighbors (for ghost export).
    Reads neighbor site ghost avail_spec (previous tick) and computes
    dispersal using negative exponential kernel.
    Writes imported_seeds to site_species for P2 gap relay next tick.
    """
    num_species = len(species_traits)
    srow = site_species_idx[agent_index]

    own_site_id = int(site_species_idx[agent_index])

    # --- 1. Iterate neighbors: aggregate avail_spec from gaps, collect site neighbors ---
    gap_count = 0.0

    # We can have at most a few site neighbors; store indices (CuPy JIT: no lists)
    # Use fixed slots for neighbor sites (practical limit)
    site_neighbor_0 = -1
    site_neighbor_1 = -1
    site_neighbor_2 = -1
    site_neighbor_3 = -1
    num_site_neighbors = 0

    # Zero avail_spec accumulator in site_species
    sp = 0
    while sp < num_species:
        site_species[srow][sp] = 0.0
        sp = sp + 1

    neighbor_indices = locations[agent_index]
    i = 0
    while i < len(neighbor_indices) and neighbor_indices[i] != -1:
        neighbor_idx = int(neighbor_indices[i])
        neighbor_breed = int(breeds[neighbor_idx])

        if neighbor_breed == Breed.GAP:
            gap_count = gap_count + 1.0
            # Accumulate avail_spec from this gap
            sp = 0
            while sp < num_species:
                gap_avail = gap_species[gap_species_idx[neighbor_idx]][sp]
                site_species[srow][sp] = \
                    site_species[srow][sp] + gap_avail
                sp = sp + 1

        elif neighbor_breed == Breed.SITE:
            # Store neighbor site index
            if num_site_neighbors == 0:
                site_neighbor_0 = neighbor_idx
            elif num_site_neighbors == 1:
                site_neighbor_1 = neighbor_idx
            elif num_site_neighbors == 2:
                site_neighbor_2 = neighbor_idx
            elif num_site_neighbors == 3:
                site_neighbor_3 = neighbor_idx
            num_site_neighbors = num_site_neighbors + 1

        i = i + 1

    # --- 2. Average avail_spec across gaps ---
    if gap_count > 0.5:
        sp = 0
        while sp < num_species:
            site_species[srow][sp] = \
                site_species[srow][sp] / gap_count
            sp = sp + 1

    # --- 3. Compute dispersal from each neighbor site ---
    # Zero imported_seeds (second half of site_species row)
    sp = 0
    while sp < num_species:
        site_species[srow][num_species + sp] = 0.0
        sp = sp + 1

    # Process each neighbor site
    ns = 0
    while ns < num_site_neighbors:
        neighbor_idx = site_neighbor_0
        if ns == 1:
            neighbor_idx = site_neighbor_1
        elif ns == 2:
            neighbor_idx = site_neighbor_2
        elif ns == 3:
            neighbor_idx = site_neighbor_3

        if neighbor_idx >= 0:
            neighbor_site_id = int(site_species_idx[neighbor_idx])
            distance = site_distances[own_site_id][neighbor_site_id]

            sp = 0
            while sp < num_species:
                neighbor_avail = site_species[site_species_idx[neighbor_idx]][sp]

                if neighbor_avail > 0.0:
                    # Check if species is in our rangelist
                    in_range = rangelists[own_site_id][sp]
                    if in_range > 0.5:
                        max_disp = species_traits[sp][Trait.MAX_DISPERSAL_DIST]
                        if max_disp > 0.0:
                            weight = cp.exp(-distance / max_disp)
                            seed_num = species_traits[sp][Trait.SEED]
                            seed_import = seed_num * neighbor_avail * weight
                            site_species[srow][num_species + sp] = \
                                site_species[srow][num_species + sp] + seed_import

                sp = sp + 1

        ns = ns + 1

    # --- 4. Divide by gap count (distribute per gap, matches GAPpy per_plot) ---
    if gap_count > 0.5:
        sp = 0
        while sp < num_species:
            site_species[srow][num_species + sp] = \
                site_species[srow][num_species + sp] / gap_count
            sp = sp + 1
