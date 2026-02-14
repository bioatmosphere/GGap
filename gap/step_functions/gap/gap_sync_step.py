"""
Gap sync step function for GGap model (Priority 5).
Relays climate and n_supply_ratio from Site to Gap states for trees to read.

Execution Flow:
    1. Read climate from Site neighbor (deg_days, dry_days, etc.)
    2. Read n_supply_ratio from Site neighbor (computed at P4 by site_nutrient_step)
    3. Copy all values to own states (Trees read at P2 and P6)
    4. Clear accumulators (litter consumed by site_soil_step at P1)

Note: n_supply_ratio is no longer computed here. It's computed by
site_nutrient_step (P4) at the site level, then relayed through Gap
to trees. This matches GAPpy where N ratio is per-site.
"""

import cupy as cp  # noqa: F401
from cupyx import jit

# === Breed IDs ===
BREED_TREE = 0
BREED_GAP = 1
BREED_SITE = 2

# === Gap states[15] (public, no buffer) ===
GAP_S_DEG_DAYS = 0
GAP_S_DRY_DAYS = 1
GAP_S_BASE_MORTALITY = 2
GAP_S_AVAIL_N = 3
GAP_S_N_SUPPLY_RATIO = 4
GAP_S_LITTER_ACCUM_C = 5
GAP_S_LITTER_ACCUM_N = 6
GAP_S_NUM_TO_RECRUIT = 7
GAP_S_RECRUIT_RAND_SEED = 8
GAP_S_FLOOD_DAYS = 9
GAP_S_FIRE_INTENSITY = 11
GAP_S_LITTER_ACCUM_C_BG = 13
GAP_S_LITTER_ACCUM_N_BG = 14

# === Site states[7] (for reading from Site neighbor) ===
SITE_S_DEG_DAYS = 0
SITE_S_DRY_DAYS = 1
SITE_S_BASE_MORTALITY = 2
SITE_S_AVAIL_N = 3
SITE_S_FLOOD_DAYS = 4
SITE_S_FIRE_INTENSITY = 5
SITE_S_N_SUPPLY_RATIO = 6


@jit.rawkernel(device="cuda")
def gap_sync_step(
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
    Gap sync step (priority 5).

    Reads climate + n_supply_ratio from Site neighbor.
    Copies to own states for trees to read at P2 (climate) and P6 (n_supply_ratio).
    Clears litter accumulators (consumed by site_soil_step at P1).
    """
    # Find Site neighbor and read climate + n_supply_ratio
    site_deg_days = 2500.0
    site_dry_days = 30.0
    site_base_mortality = 0.02
    site_avail_n = 0.1
    site_flood_days = 0.0
    site_fire_intensity = 0.0
    site_n_supply_ratio = 1.0

    neighbor_indices = locations[agent_index]
    i = 0
    while i < len(neighbor_indices) and neighbor_indices[i] != -1:
        neighbor_idx = int(neighbor_indices[i])
        neighbor_breed = int(breeds[neighbor_idx])

        if neighbor_breed == BREED_SITE:
            site_deg_days = states_tensor[neighbor_idx][SITE_S_DEG_DAYS]
            site_dry_days = states_tensor[neighbor_idx][SITE_S_DRY_DAYS]
            site_base_mortality = states_tensor[neighbor_idx][SITE_S_BASE_MORTALITY]
            site_avail_n = states_tensor[neighbor_idx][SITE_S_AVAIL_N]
            site_flood_days = states_tensor[neighbor_idx][SITE_S_FLOOD_DAYS]
            site_fire_intensity = states_tensor[neighbor_idx][SITE_S_FIRE_INTENSITY]
            site_n_supply_ratio = states_tensor[neighbor_idx][SITE_S_N_SUPPLY_RATIO]

        i = i + 1

    # Copy climate to own states (Trees read at P2)
    states_tensor[agent_index][GAP_S_DEG_DAYS] = site_deg_days
    states_tensor[agent_index][GAP_S_DRY_DAYS] = site_dry_days
    states_tensor[agent_index][GAP_S_BASE_MORTALITY] = site_base_mortality
    states_tensor[agent_index][GAP_S_AVAIL_N] = site_avail_n
    states_tensor[agent_index][GAP_S_FLOOD_DAYS] = site_flood_days
    states_tensor[agent_index][GAP_S_FIRE_INTENSITY] = site_fire_intensity

    # Copy n_supply_ratio (computed by site_nutrient_step at P4)
    states_tensor[agent_index][GAP_S_N_SUPPLY_RATIO] = site_n_supply_ratio

    # Clear litter accumulators (consumed by site_soil_step at P1)
    states_tensor[agent_index][GAP_S_LITTER_ACCUM_C] = 0.0
    states_tensor[agent_index][GAP_S_LITTER_ACCUM_N] = 0.0
    states_tensor[agent_index][GAP_S_LITTER_ACCUM_C_BG] = 0.0
    states_tensor[agent_index][GAP_S_LITTER_ACCUM_N_BG] = 0.0
    # Note: NUM_TO_RECRUIT and RECRUIT_RAND_SEED are NOT cleared here.
    # Trees read them at P6 (dormant activation). P0 overwrites them each tick.
