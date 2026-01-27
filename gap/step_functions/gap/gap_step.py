"""
Gap step functions for GGap model.
GPU kernels for Gap agent data relay between Trees and Site.

Two step functions:
- gap_aggregate_step (priority 1): Read litter from trees, calculate recruitment, store in own states
- gap_sync_step (priority 3): Read climate+avail_n from site, store for trees to read

Property scheme (3 properties):
- params[2]: gap_id, total_n_demand (private internal)
- states[12]: climate + nutrients + litter_pool + recruitment + flood_days + seed_bank + fire (public, no buffer)
- states_db[1]: placeholder (public, double buffered but unused)
"""

import cupy as cp  # noqa: F401
from cupyx import jit

# === Breed IDs ===
BREED_TREE = 0
BREED_GAP = 1
BREED_SITE = 2

# === Gap params[2] (private) ===
GAP_P_GAP_ID = 0
GAP_P_TOTAL_N_DEMAND = 1

# === Gap states[12] (public, no buffer) ===
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
GAP_S_SEED_BANK = 10
GAP_S_FIRE_INTENSITY = 11

# === Tree params[23] (for reading invader/seed) ===
TREE_P_INVADER = 10
TREE_P_SEED = 11

# === Tree states[3] (for reading from Tree neighbors) ===
TREE_S_LITTER_C = 0
TREE_S_LITTER_N = 1
TREE_S_N_DEMAND = 2

# === Tree states_db[4] (for checking alive status) ===
TREE_DB_IS_ALIVE = 0

# === Site states[6] (for reading from Site neighbor) ===
SITE_S_DEG_DAYS = 0
SITE_S_DRY_DAYS = 1
SITE_S_BASE_MORTALITY = 2
SITE_S_AVAIL_N = 3
SITE_S_FLOOD_DAYS = 4
SITE_S_FIRE_INTENSITY = 5


@jit.rawkernel(device="cuda")
def gap_aggregate_step(
    tick,
    agent_index,
    globals,
    agent_ids,
    breeds,
    locations,
    params_tensor,
    states_tensor,
    states_db_tensor,
):
    """
    Gap aggregate step (priority 1).

    Reads from tree neighbors:
    - states_db: is_alive (to filter alive vs dormant)
    - states: litter_c, litter_n, n_demand
    - params: invader, seed (for recruitment calculation)

    Writes to own:
    - params: total_n_demand (internal)
    - states: litter_accum_c, litter_accum_n, num_to_recruit, recruit_rand_seed
    """
    # Aggregate litter and N demand from tree neighbors
    # Also count living/dormant trees and calculate seed production
    total_litter_c = 0.0
    total_litter_n = 0.0
    total_n_dem = 0.0
    living_tree_count = 0.0
    dormant_tree_count = 0.0
    total_seed_production = 0.0

    neighbor_indices = locations[agent_index]
    i = 0
    while i < len(neighbor_indices) and neighbor_indices[i] != -1:
        neighbor_idx = int(neighbor_indices[i])
        neighbor_breed = int(breeds[neighbor_idx])

        if neighbor_breed == BREED_TREE:
            tree_alive = states_db_tensor[neighbor_idx][TREE_DB_IS_ALIVE]
            if tree_alive > 0.5:
                # Living tree: aggregate litter and count for recruitment
                living_tree_count = living_tree_count + 1.0

                # Read tree's litter from states
                tree_litter_c = states_tensor[neighbor_idx][TREE_S_LITTER_C]
                tree_litter_n = states_tensor[neighbor_idx][TREE_S_LITTER_N]
                total_litter_c = total_litter_c + tree_litter_c
                total_litter_n = total_litter_n + tree_litter_n

                # Read tree's N demand from states
                tree_n_demand = states_tensor[neighbor_idx][TREE_S_N_DEMAND]
                total_n_dem = total_n_dem + tree_n_demand

                # Accumulate seed production for recruitment
                tree_invader = params_tensor[neighbor_idx][TREE_P_INVADER]
                tree_seed = params_tensor[neighbor_idx][TREE_P_SEED]
                total_seed_production = total_seed_production + tree_invader * tree_seed
            else:
                # Dormant tree slot available for recruitment
                dormant_tree_count = dormant_tree_count + 1.0

        i = i + 1

    # Read existing seed bank (accumulated from previous years)
    seed_bank = states_tensor[agent_index][GAP_S_SEED_BANK]

    # Total available seeds = current production + seed bank
    total_available_seeds = total_seed_production + seed_bank

    # Calculate number of seedlings to recruit this tick
    # Based on UVAFME: seedling establishment depends on seed availability and space
    num_to_recruit = 0.0
    seeds_used = 0.0
    if dormant_tree_count > 0.5 and total_available_seeds > 0.1:
        # Recruitment rate: proportional to seeds, limited by available slots
        # UVAFME uses: seedling = invader * 10 + sprout * avail_spec
        # Simplified: recruit based on seed production, capped by dormant slots
        potential_recruits = total_available_seeds * 0.3  # 30% germination rate
        if potential_recruits > dormant_tree_count:
            potential_recruits = dormant_tree_count
        if potential_recruits > 10.0:
            potential_recruits = 10.0  # Cap per tick to avoid explosive growth
        num_to_recruit = potential_recruits
        seeds_used = num_to_recruit * 2.0  # Each recruit uses ~2 seeds worth

    # Update seed bank: add new production, subtract used, apply decay
    # Seeds decay at 30% per year (seed longevity)
    new_seed_bank = (total_available_seeds - seeds_used) * 0.7  # 30% decay
    if new_seed_bank < 0.0:
        new_seed_bank = 0.0
    if new_seed_bank > 100.0:
        new_seed_bank = 100.0  # Cap seed bank to prevent unbounded accumulation

    # Generate a pseudo-random seed for species selection in tree_step
    recruit_rand_seed = float((tick * 997 + agent_index * 991) % 10000)

    # Write to own params (internal)
    params_tensor[agent_index][GAP_P_TOTAL_N_DEMAND] = total_n_dem

    # Write to own states (Site will read litter, Trees will read recruitment)
    states_tensor[agent_index][GAP_S_LITTER_ACCUM_C] = total_litter_c
    states_tensor[agent_index][GAP_S_LITTER_ACCUM_N] = total_litter_n
    states_tensor[agent_index][GAP_S_NUM_TO_RECRUIT] = num_to_recruit
    states_tensor[agent_index][GAP_S_RECRUIT_RAND_SEED] = recruit_rand_seed
    states_tensor[agent_index][GAP_S_SEED_BANK] = new_seed_bank


@jit.rawkernel(device="cuda")
def gap_sync_step(
    tick,
    agent_index,
    globals,
    agent_ids,
    breeds,
    locations,
    params_tensor,
    states_tensor,
    states_db_tensor,
):
    """
    Gap sync step (priority 3).

    Reads from site neighbor:
    - states: deg_days, dry_days, base_mortality, avail_n, flood_days

    Writes to own states:
    - climate: copied from Site (for dynamic climate support)
    - n_supply_ratio: avail_n / total_n_demand
    - flood_days: relayed from Site
    """
    # Find Site neighbor and read climate + avail_n + flood_days + fire
    site_deg_days = 2500.0  # Default
    site_dry_days = 30.0
    site_base_mortality = 0.02
    site_avail_n = 0.1
    site_flood_days = 0.0
    site_fire_intensity = 0.0

    neighbor_indices = locations[agent_index]
    i = 0
    while i < len(neighbor_indices) and neighbor_indices[i] != -1:
        neighbor_idx = int(neighbor_indices[i])
        neighbor_breed = int(breeds[neighbor_idx])

        if neighbor_breed == BREED_SITE:
            # Read Site's climate and avail_n from states
            site_deg_days = states_tensor[neighbor_idx][SITE_S_DEG_DAYS]
            site_dry_days = states_tensor[neighbor_idx][SITE_S_DRY_DAYS]
            site_base_mortality = states_tensor[neighbor_idx][SITE_S_BASE_MORTALITY]
            site_avail_n = states_tensor[neighbor_idx][SITE_S_AVAIL_N]
            site_flood_days = states_tensor[neighbor_idx][SITE_S_FLOOD_DAYS]
            site_fire_intensity = states_tensor[neighbor_idx][SITE_S_FIRE_INTENSITY]

        i = i + 1

    # Copy climate to own states (for trees to read next tick)
    states_tensor[agent_index][GAP_S_DEG_DAYS] = site_deg_days
    states_tensor[agent_index][GAP_S_DRY_DAYS] = site_dry_days
    states_tensor[agent_index][GAP_S_BASE_MORTALITY] = site_base_mortality
    states_tensor[agent_index][GAP_S_AVAIL_N] = site_avail_n
    states_tensor[agent_index][GAP_S_FLOOD_DAYS] = site_flood_days
    states_tensor[agent_index][GAP_S_FIRE_INTENSITY] = site_fire_intensity

    # Calculate N supply/demand ratio
    total_n_dem = params_tensor[agent_index][GAP_P_TOTAL_N_DEMAND]
    n_supply_ratio = 1.0
    if total_n_dem > 0.0001:
        n_supply_ratio = site_avail_n / total_n_dem
        if n_supply_ratio > 2.0:
            n_supply_ratio = 2.0  # Cap at 2x supply

    states_tensor[agent_index][GAP_S_N_SUPPLY_RATIO] = n_supply_ratio

    # Clear accumulators for next tick
    states_tensor[agent_index][GAP_S_LITTER_ACCUM_C] = 0.0
    states_tensor[agent_index][GAP_S_LITTER_ACCUM_N] = 0.0
    # Clear recruitment counters (trees have already read them at P0)
    states_tensor[agent_index][GAP_S_NUM_TO_RECRUIT] = 0.0
    states_tensor[agent_index][GAP_S_RECRUIT_RAND_SEED] = 0.0
