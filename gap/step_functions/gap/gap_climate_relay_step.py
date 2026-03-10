"""
Gap climate relay step function for GGap model (Priority 2).
Copies climate from Site to Gap states so trees can read current-tick climate at P3.

This runs AFTER P1 (site soil) but BEFORE P3 (tree potential growth),
eliminating the 1-tick climate lag for living trees.

GAPpy computes climate in bio_geo_climate() then uses it immediately in growth().
This step matches that by relaying climate before tree growth.

Execution Flow:
    1. Find Site neighbor
    2. Copy deg_days, dry_days, flood_days, fire/wind intensity, dry_days_base
    3. Copy avail_n (for P5 templates and P7 disturbance gate)
"""

import cupy as cp  # noqa: F401
from cupyx import jit

# === Breed IDs ===
BREED_TREE = 0
BREED_GAP = 1
BREED_SITE = 2

# === Gap states[16] (public, no buffer) ===
GAP_S_DEG_DAYS = 0
GAP_S_DRY_DAYS = 1
GAP_S_AVAIL_N = 2
GAP_S_FLOOD_DAYS = 8
GAP_S_FIRE_INTENSITY = 10
GAP_S_DRY_DAYS_BASE = 14
GAP_S_WIND_INTENSITY = 15
GAP_S_RECOVERY_YEARS = 166     # Fire/wind recovery countdown (GAPpy fire=5, wind=3)

# === Site states[8] (for reading from Site neighbor) ===
SITE_S_DEG_DAYS = 0
SITE_S_DRY_DAYS = 1
SITE_S_AVAIL_N = 2
SITE_S_FLOOD_DAYS = 3
SITE_S_FIRE_INTENSITY = 4
SITE_S_DRY_DAYS_BASE = 6
SITE_S_WIND_INTENSITY = 7


@jit.rawkernel(device="cuda")
def gap_climate_relay_step(
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

    neighbor_indices = locations[agent_index]
    i = 0
    while i < len(neighbor_indices) and neighbor_indices[i] != -1:
        neighbor_idx = int(neighbor_indices[i])
        neighbor_breed = int(breeds[neighbor_idx])

        if neighbor_breed == BREED_SITE:
            site_deg_days = states_tensor[neighbor_idx][SITE_S_DEG_DAYS]
            site_dry_days = states_tensor[neighbor_idx][SITE_S_DRY_DAYS]
            site_avail_n = states_tensor[neighbor_idx][SITE_S_AVAIL_N]
            site_flood_days = states_tensor[neighbor_idx][SITE_S_FLOOD_DAYS]
            site_fire_intensity = states_tensor[neighbor_idx][SITE_S_FIRE_INTENSITY]
            site_wind_intensity = states_tensor[neighbor_idx][SITE_S_WIND_INTENSITY]
            site_dry_days_base = states_tensor[neighbor_idx][SITE_S_DRY_DAYS_BASE]

        i = i + 1

    states_tensor[agent_index][GAP_S_DEG_DAYS] = site_deg_days
    states_tensor[agent_index][GAP_S_DRY_DAYS] = site_dry_days
    states_tensor[agent_index][GAP_S_AVAIL_N] = site_avail_n
    states_tensor[agent_index][GAP_S_FLOOD_DAYS] = site_flood_days
    states_tensor[agent_index][GAP_S_FIRE_INTENSITY] = site_fire_intensity
    states_tensor[agent_index][GAP_S_WIND_INTENSITY] = site_wind_intensity
    states_tensor[agent_index][GAP_S_DRY_DAYS_BASE] = site_dry_days_base

    # Set recovery countdown when fire/wind occurs (GAPpy: fire=5, wind=3).
    # Counter persists across ticks; P6 decrements it each tick.
    if site_fire_intensity > 0.01:
        states_tensor[agent_index][GAP_S_RECOVERY_YEARS] = 5.0
    elif site_wind_intensity > 0.01:
        states_tensor[agent_index][GAP_S_RECOVERY_YEARS] = 3.0
