"""
Site final step function for GGap model (Priority 9).
Merges N balance (formerly P9) and seed dispersal (formerly P10) into one
step, eliminating one grid barrier. Both operations are independent: N balance
reads Gap N_CONSUMED (P8) while dispersal reads Gap avail_spec (P0) and
neighbor Site ghost data (previous tick).

N Balance (from site_nbalance_step):
    1. Read avail_N from own states (computed at P1, same tick)
    2. Read annual_runoff from own params (computed at P1, same tick)
    3. Read total N consumed from all Gap neighbors (aggregated at P8, same tick)
    4. Apply unit conversion (kg -> tn/ha)
    5. Compute surplus = avail_N - total_N_consumed
    6. If surplus > 0: return to A layer minus leach fraction
    7. If surplus <= 0: debit A layer
    8. Apply runoff leaching (always)
    9. Transfer leached N*20 C to base layer

Seed Dispersal (from site_seed_dispersal_step):
    1. Aggregate avail_spec from gap neighbors (for ghost export)
    2. Average avail_spec across gaps
    3. Compute dispersal from each neighbor site using negative exponential kernel
    4. Divide imported seeds by gap count (distribute per gap)
"""

import cupy as cp
from cupyx import jit

from gap.constants import (
    Breed, Trait, SiteP, SiteS, GapS,
    UNIT_CONV,
)


@jit.rawkernel(device="cuda")
def site_final_step(
    tick,
    agent_index,
    species_traits, site_configs, rangelists, site_distances,
    agent_ids,
    breeds,
    locations,
    params_tensor,
    states_tensor,
    gap_lai, gap_lai_idx,
    gap_avail_spec, gap_avail_spec_idx,
    gap_imported_seeds, gap_imported_seeds_idx,
    site_avail_spec, site_avail_spec_idx,
    site_imported_seeds, site_imported_seeds_idx,
):
    """
    Site final step (priority 9).

    Part A - N Balance:
      Reads avail_N (P1 same tick), N consumed (P8 same tick), annual_runoff (P1 same tick).
      Applies surplus/deficit to A layer, leaching to base layer.
      Matches GAPpy model.py:993-1005.

    Part B - Seed Dispersal:
      Aggregates avail_spec from gap neighbors (for ghost export).
      Reads neighbor site ghost avail_spec (previous tick) and computes
      dispersal using negative exponential kernel.
      Writes imported_seeds to site_species for P2 gap relay next tick.
    """
    num_species = len(species_traits)
    own_site_id = int(params_tensor[agent_index][SiteP.SITE_ID])

    # ================================================================
    # SHARED NEIGHBOR LOOP: collect Gap data (N consumed + avail_spec)
    # and Site neighbor indices in a single pass
    # ================================================================
    total_n_consumed = 0.0
    gap_count = 0.0

    # Fixed slots for neighbor sites (CuPy JIT: no lists)
    site_neighbor_0 = -1
    site_neighbor_1 = -1
    site_neighbor_2 = -1
    site_neighbor_3 = -1
    num_site_neighbors = 0

    # Zero avail_spec accumulator
    srow_avail = site_avail_spec_idx[agent_index]
    sp = 0
    while sp < num_species:
        site_avail_spec[srow_avail][sp] = 0.0
        sp = sp + 1

    neighbor_indices = locations[agent_index]
    i = 0
    while i < len(neighbor_indices) and neighbor_indices[i] != -1:
        neighbor_idx = int(neighbor_indices[i])
        neighbor_breed = int(breeds[neighbor_idx])

        if neighbor_breed == Breed.GAP:
            gap_count = gap_count + 1.0

            # N balance: sum N consumed from gaps (P8 same tick)
            gap_n_consumed = states_tensor[neighbor_idx][GapS.N_CONSUMED]
            total_n_consumed = total_n_consumed + gap_n_consumed

            # Dispersal: accumulate avail_spec from this gap
            sp = 0
            while sp < num_species:
                gap_avail = gap_avail_spec[gap_avail_spec_idx[neighbor_idx]][sp]
                site_avail_spec[srow_avail][sp] = \
                    site_avail_spec[srow_avail][sp] + gap_avail
                sp = sp + 1

        elif neighbor_breed == Breed.SITE:
            # Store neighbor site index for dispersal
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

    # ================================================================
    # PART A: N BALANCE
    # ================================================================

    # Convert N consumed from kg to tn/ha (GAPpy uconvert)
    if gap_count > 0.5:
        total_n_consumed = total_n_consumed * UNIT_CONV / gap_count

    # Read avail_N computed by P1 this tick
    avail_n = states_tensor[agent_index][SiteS.AVAIL_N]

    # Read annual runoff computed by P1 this tick
    annual_runoff = params_tensor[agent_index][SiteP.ANNUAL_RUNOFF]

    # Read current soil pools (written by P1 same tick, params has no double buffer)
    sa_n0 = params_tensor[agent_index][SiteP.A_N]
    sa_c0 = params_tensor[agent_index][SiteP.A_C]
    sb_c0 = params_tensor[agent_index][SiteP.BL_C]
    sb_n0 = params_tensor[agent_index][SiteP.BL_N]

    # Surplus = available N - consumed N (GAPpy model.py:993)
    surplus = avail_n - total_n_consumed
    net_n_leach = 0.0

    if surplus > 0.0:
        # Fraction leached via runoff (capped at 10%, GAPpy model.py:994)
        leach_frac = annual_runoff / 1000.0
        if leach_frac > 0.1:
            leach_frac = 0.1
        net_n_leach = surplus * leach_frac
        # Return remainder to A layer (GAPpy model.py:995)
        sa_n0 = sa_n0 + surplus - net_n_leach
    else:
        # Deficit: debit A layer (GAPpy model.py:997)
        sa_n0 = sa_n0 + surplus  # surplus is negative
        net_n_leach = 0.0

    # Runoff leaching from A layer (always applied, GAPpy model.py:1000)
    sa_n0 = sa_n0 - 0.00002 * annual_runoff

    # Transfer leached N and C to base layer (GAPpy model.py:1002-1005)
    sa_c0 = sa_c0 - net_n_leach * 20.0
    sb_c0 = sb_c0 + net_n_leach * 20.0
    sb_n0 = sb_n0 + net_n_leach

    # Write adjusted soil pools back
    params_tensor[agent_index][SiteP.A_N] = sa_n0
    params_tensor[agent_index][SiteP.A_C] = sa_c0
    params_tensor[agent_index][SiteP.BL_C] = sb_c0
    params_tensor[agent_index][SiteP.BL_N] = sb_n0

    # Export net N leached for CSV output (output-only in params)
    params_tensor[agent_index][SiteP.NET_N_INTO_A0] = net_n_leach

    # ================================================================
    # PART B: SEED DISPERSAL
    # ================================================================

    # Average avail_spec across gaps
    if gap_count > 0.5:
        sp = 0
        while sp < num_species:
            site_avail_spec[srow_avail][sp] = \
                site_avail_spec[srow_avail][sp] / gap_count
            sp = sp + 1

    # Zero imported_seeds
    srow_imported = site_imported_seeds_idx[agent_index]
    sp = 0
    while sp < num_species:
        site_imported_seeds[srow_imported][sp] = 0.0
        sp = sp + 1

    # Compute dispersal from each neighbor site
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
            neighbor_site_id = int(states_tensor[neighbor_idx][SiteS.SITE_ID])
            distance = site_distances[own_site_id][neighbor_site_id]

            sp = 0
            while sp < num_species:
                neighbor_avail = site_avail_spec[site_avail_spec_idx[neighbor_idx]][sp]

                if neighbor_avail > 0.0:
                    # Check if species is in our rangelist
                    in_range = rangelists[own_site_id][sp]
                    if in_range > 0.5:
                        max_disp = species_traits[sp][Trait.MAX_DISPERSAL_DIST]
                        if max_disp > 0.0:
                            weight = cp.exp(-distance / max_disp)
                            seed_num = species_traits[sp][Trait.SEED]
                            seed_import = seed_num * neighbor_avail * weight
                            site_imported_seeds[srow_imported][sp] = \
                                site_imported_seeds[srow_imported][sp] + seed_import

                sp = sp + 1

        ns = ns + 1

    # Divide imported seeds by gap count (distribute per gap, matches GAPpy per_plot)
    if gap_count > 0.5:
        sp = 0
        while sp < num_species:
            site_imported_seeds[srow_imported][sp] = \
                site_imported_seeds[srow_imported][sp] / gap_count
            sp = sp + 1
