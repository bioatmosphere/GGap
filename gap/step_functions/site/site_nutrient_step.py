"""
Site nutrient allocation step function for GGap model (Priority 4).
Computes n_supply_ratio from soil's avail_n and total tree N demand.

Matches GAPpy's between-loops N ratio calculation:
    n_supply_ratio = avail_N / total_N_demand

This is a Site breed step because avail_n is soil info. The ratio is
computed at the site level (summing demand from ALL gaps in the site),
matching UVAFME where N supply ratio is per-site, not per-gap.

Execution Flow:
    1. Read own avail_n from states (computed at P1 by site_soil_step)
    2. Read total_n_demand from all Gap neighbors (written at P3)
    3. Compute n_supply_ratio = avail_n / sum(total_n_demand)
    4. Write n_supply_ratio to own states (Gap reads at P5 to relay)
"""

import cupy as cp  # noqa: F401
from cupyx import jit

from gap.constants import (
    Breed, GapS, SiteS,
    UNIT_CONV,
)


@jit.rawkernel(device="cuda")
def site_nutrient_step(
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
    Site nutrient allocation step (priority 4).

    Reads avail_n from own states (written at P1).
    Reads total_n_demand from all Gap neighbors (written at P3).
    Computes site-level n_supply_ratio and writes to own states.
    Gap sync step (P5) relays this to Gap states for trees to read.
    """
    # Read avail_n (computed at P1 by site_soil_step, same tick)
    avail_n = states_tensor[agent_index][SiteS.AVAIL_N]

    # Sum total N demand from all Gap neighbors
    site_total_n_demand = 0.0
    gap_count = 0.0

    neighbor_indices = locations[agent_index]
    i = 0
    while i < len(neighbor_indices) and neighbor_indices[i] != -1:
        neighbor_idx = int(neighbor_indices[i])
        neighbor_breed = int(breeds[neighbor_idx])

        if neighbor_breed == Breed.GAP:
            gap_n_demand = states_tensor[neighbor_idx][GapS.TOTAL_N_DEMAND]
            site_total_n_demand = site_total_n_demand + gap_n_demand
            gap_count = gap_count + 1.0

        i = i + 1

    # Convert N demand from kg to tn/ha (GAPpy uconvert)
    if gap_count > 0.5:
        site_total_n_demand = site_total_n_demand * UNIT_CONV / gap_count

    # Compute n_supply_ratio = avail_n / total_n_demand
    n_supply_ratio = 1.0
    if site_total_n_demand > 0.00001:
        n_supply_ratio = avail_n / site_total_n_demand
        if n_supply_ratio > 2.0:
            n_supply_ratio = 2.0  # Cap at 2x supply

    # Write to own states (Gap reads at P5)
    states_tensor[agent_index][SiteS.N_SUPPLY_RATIO] = n_supply_ratio
