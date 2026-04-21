"""
Gap climate relay step function for GGap model (Priority 2).
Copies climate from Site to Gap states so trees can read current-tick climate at P3.
Also relays imported_seeds from site_species to gap_species (breed-local arrays)
for P5 templates to read.

This runs AFTER P1 (site soil) but BEFORE P3 (tree potential growth),
eliminating the 1-tick climate lag for living trees.

Execution Flow:
    1. Find Site neighbor
    2. Copy deg_days, dry_days, flood_days, fire/wind intensity, dry_days_base
    3. Copy avail_n (for P5 templates and P7 disturbance gate)
    4. Copy imported_seeds from site_species to gap_species (for P5 template seedbank)
"""

import cupy as cp  # noqa: F401
from cupyx import jit

from gap.constants import (
    Breed, GapS, SiteS,
)


@jit.rawkernel(device="cuda")
def gap_climate_relay_step(
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
    gap_seedling_weights, gap_seedling_weights_idx,
    site_avail_spec, site_avail_spec_idx,
    site_imported_seeds, site_imported_seeds_idx,
):
    """
    Gap climate relay step (priority 2).

    Copies current-tick climate from Site to Gap states.
    Trees read these at P3 (potential growth) for same-tick climate.
    """
    site_deg_days = 2500.0
    site_dry_days = 0.0
    site_avail_n = 0.1
    site_flood_days = 0.0
    site_fire_intensity = 0.0
    site_wind_intensity = 0.0
    site_dry_days_base = 0.0
    site_idx = -1

    neighbor_indices = locations[agent_index]
    i = 0
    while i < len(neighbor_indices) and neighbor_indices[i] != -1:
        neighbor_idx = int(neighbor_indices[i])
        neighbor_breed = int(breeds[neighbor_idx])

        if neighbor_breed == Breed.SITE:
            site_idx = neighbor_idx
            site_deg_days = states_tensor[neighbor_idx][SiteS.DEG_DAYS]
            site_dry_days = states_tensor[neighbor_idx][SiteS.DRY_DAYS]
            site_avail_n = states_tensor[neighbor_idx][SiteS.AVAIL_N]
            site_flood_days = states_tensor[neighbor_idx][SiteS.FLOOD_DAYS]
            site_fire_intensity = states_tensor[neighbor_idx][SiteS.FIRE_INTENSITY]
            site_wind_intensity = states_tensor[neighbor_idx][SiteS.WIND_INTENSITY]
            site_dry_days_base = states_tensor[neighbor_idx][SiteS.DRY_DAYS_BASE]

        i = i + 1

    states_tensor[agent_index][GapS.DEG_DAYS] = site_deg_days
    states_tensor[agent_index][GapS.DRY_DAYS] = site_dry_days
    states_tensor[agent_index][GapS.AVAIL_N] = site_avail_n
    states_tensor[agent_index][GapS.FLOOD_DAYS] = site_flood_days
    states_tensor[agent_index][GapS.FIRE_INTENSITY] = site_fire_intensity
    states_tensor[agent_index][GapS.WIND_INTENSITY] = site_wind_intensity
    states_tensor[agent_index][GapS.DRY_DAYS_BASE] = site_dry_days_base

    # Set recovery countdown when fire/wind occurs (GAPpy: fire=5, wind=3).
    # Counter persists across ticks; P6 decrements it each tick.
    if site_fire_intensity > 0.01:
        states_tensor[agent_index][GapS.RECOVERY_YEARS] = 5.0
    elif site_wind_intensity > 0.01:
        states_tensor[agent_index][GapS.RECOVERY_YEARS] = 3.0

    # Relay imported_seeds from site_imported_seeds → gap_imported_seeds (written at P10 previous tick)
    # P5 templates read from gap_imported_seeds for seedbank accumulation
    num_species = len(species_traits)
    if site_idx >= 0:
        sp = 0
        while sp < num_species:
            gap_imported_seeds[gap_imported_seeds_idx[agent_index]][sp] = site_imported_seeds[site_imported_seeds_idx[site_idx]][sp]
            sp = sp + 1
