"""
Tree step function for GGap model.
Combined GPU kernel implementing light, growth, and mortality.

Property scheme (3 properties):
- params[20]: species traits (static) + physiology (dynamic internal) - private
- states[3]: litter output (litter_c, litter_n, n_demand) - public, no buffer
- states_db[4]: structure (is_alive, diam, height, canopy_ht) - public, double buffered
"""

import cupy as cp
from cupyx import jit

# === Breed IDs ===
BREED_TREE = 0
BREED_GAP = 1
BREED_SITE = 2

# === Tree params[20] (private) ===
# Species traits [0-9]:
TREE_P_SPECIES_ID = 0
TREE_P_MAX_AGE = 1
TREE_P_MAX_DIAM = 2
TREE_P_MAX_HT = 3
TREE_P_ARFA_0 = 4
TREE_P_G = 5
TREE_P_SHADE_TOL = 6
TREE_P_DEG_DAY_MIN = 7
TREE_P_DEG_DAY_OPT = 8
TREE_P_DEG_DAY_MAX = 9
# Physiology [10-19]:
TREE_P_AGE = 10
TREE_P_BIOMC = 11
TREE_P_BIOMN = 12
TREE_P_LEAF_BM = 13
TREE_P_X = 14
TREE_P_Y = 15
TREE_P_LIGHT_AVAIL = 16
TREE_P_FC_DEGDAY = 17
TREE_P_FC_DROUGHT = 18
TREE_P_FC_FLOOD = 19

# === Tree states[3] (public, no buffer) ===
TREE_S_LITTER_C = 0
TREE_S_LITTER_N = 1
TREE_S_N_DEMAND = 2

# === Tree states_db[4] (public, double buffered) ===
TREE_DB_IS_ALIVE = 0
TREE_DB_DIAM = 1
TREE_DB_HEIGHT = 2
TREE_DB_CANOPY_HT = 3

# === Gap states[7] (for reading from Gap neighbor) ===
GAP_S_DEG_DAYS = 0
GAP_S_DRY_DAYS = 1
GAP_S_BASE_MORTALITY = 2
GAP_S_AVAIL_N = 3
GAP_S_N_SUPPLY_RATIO = 4

# === Constants ===
PI = 3.14159265359
STD_HT = 1.3  # Breast height in meters
XT = -0.40   # Light extinction coefficient

# C/N ratios
STEM_C_N = 450.0
LEAF_C_N = 50.0  # Average of conifer (60) and deciduous (40)


@jit.rawkernel(device="cuda")
def tree_step(
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
    Combined tree step function: light, growth, mortality, litter.

    Reads site params (deg_days, dry_days, base_mortality) from Gap neighbor states.
    Calculates light availability from neighbor tree states_db (heights).
    Applies environmental stress and grows tree.
    Checks mortality and outputs litter.
    """
    # ===== GET CURRENT STATE =====
    is_alive = states_db_tensor[agent_index][TREE_DB_IS_ALIVE]

    # Initialize outputs
    litter_c = 0.0
    litter_n = 0.0
    n_demand = 0.0

    # Only process living trees
    if is_alive > 0.5:
        # Get species parameters from params
        max_age = params_tensor[agent_index][TREE_P_MAX_AGE]
        max_diam = params_tensor[agent_index][TREE_P_MAX_DIAM]
        max_ht = params_tensor[agent_index][TREE_P_MAX_HT]
        arfa_0 = params_tensor[agent_index][TREE_P_ARFA_0]
        g = params_tensor[agent_index][TREE_P_G]
        shade_tol = int(params_tensor[agent_index][TREE_P_SHADE_TOL])
        deg_day_min = params_tensor[agent_index][TREE_P_DEG_DAY_MIN]
        deg_day_opt = params_tensor[agent_index][TREE_P_DEG_DAY_OPT]
        deg_day_max = params_tensor[agent_index][TREE_P_DEG_DAY_MAX]

        # Get current tree structure from states_db
        diam = states_db_tensor[agent_index][TREE_DB_DIAM]
        height = states_db_tensor[agent_index][TREE_DB_HEIGHT]
        canopy_ht = states_db_tensor[agent_index][TREE_DB_CANOPY_HT]

        # Get internal physiology from params
        age = params_tensor[agent_index][TREE_P_AGE]
        biomC = params_tensor[agent_index][TREE_P_BIOMC]
        leaf_bm = params_tensor[agent_index][TREE_P_LEAF_BM]

        # ===== READ SITE PARAMS FROM GAP NEIGHBOR STATES =====
        deg_days = 2500.0  # Default
        dry_days = 30.0
        base_mortality = 0.02
        n_supply_ratio = 1.0

        neighbor_indices = locations[agent_index]
        i = 0
        while i < len(neighbor_indices) and neighbor_indices[i] != -1:
            neighbor_idx = int(neighbor_indices[i])
            neighbor_breed = int(breeds[neighbor_idx])
            if neighbor_breed == BREED_GAP:
                # Read site params from Gap states
                deg_days = states_tensor[neighbor_idx][GAP_S_DEG_DAYS]
                dry_days = states_tensor[neighbor_idx][GAP_S_DRY_DAYS]
                base_mortality = states_tensor[neighbor_idx][GAP_S_BASE_MORTALITY]
                # Read N supply ratio from Gap states
                n_supply_ratio = states_tensor[neighbor_idx][GAP_S_N_SUPPLY_RATIO]
            i = i + 1

        # ===== CALCULATE LIGHT AVAILABILITY =====
        total_lai_above = 0.0

        i = 0
        while i < len(neighbor_indices) and neighbor_indices[i] != -1:
            neighbor_idx = int(neighbor_indices[i])
            neighbor_breed = int(breeds[neighbor_idx])

            # Only count tree neighbors
            if neighbor_breed == BREED_TREE:
                neighbor_alive = states_db_tensor[neighbor_idx][TREE_DB_IS_ALIVE]
                if neighbor_alive > 0.5:
                    neighbor_height = states_db_tensor[neighbor_idx][TREE_DB_HEIGHT]
                    neighbor_canopy_ht = states_db_tensor[neighbor_idx][TREE_DB_CANOPY_HT]
                    neighbor_diam = states_db_tensor[neighbor_idx][TREE_DB_DIAM]

                    # Only count taller neighbors
                    if neighbor_height > height:
                        canopy_depth = neighbor_height - neighbor_canopy_ht
                        if canopy_depth < 1.0:
                            canopy_depth = 1.0

                        # LAI contribution
                        neighbor_lai = (neighbor_diam * neighbor_diam * 0.01) / canopy_depth
                        overlap = neighbor_height - height
                        if overlap > canopy_depth:
                            overlap = canopy_depth

                        total_lai_above = total_lai_above + neighbor_lai * overlap
            i = i + 1

        # Beer-Lambert light attenuation
        light_avail = cp.exp(XT * total_lai_above)
        if light_avail < 0.01:
            light_avail = 0.01
        if light_avail > 1.0:
            light_avail = 1.0

        # ===== CALCULATE ENVIRONMENTAL STRESS FACTORS =====

        # Temperature response (parabolic)
        fc_degday = 0.0
        if deg_days > deg_day_min and deg_days < deg_day_max:
            a = (deg_day_opt - deg_day_min) / (deg_day_max - deg_day_min)
            b = (deg_day_max - deg_day_opt) / (deg_day_max - deg_day_min)
            tmp1 = (deg_days - deg_day_min) / (deg_day_opt - deg_day_min)
            tmp2 = (deg_day_max - deg_days) / (deg_day_max - deg_day_opt)
            if tmp1 > 0.0 and tmp2 > 0.0:
                fc_degday = (tmp1 ** a) * (tmp2 ** b)

        # Drought response (inverse sqrt)
        fc_drought = 1.0
        shade_idx = shade_tol - 1
        if shade_idx < 0:
            shade_idx = 0
        if shade_idx > 4:
            shade_idx = 4

        gamma = 0.35  # Default for shade_tol=3
        if shade_idx == 0:
            gamma = 0.50
        elif shade_idx == 1:
            gamma = 0.45
        elif shade_idx == 2:
            gamma = 0.35
        elif shade_idx == 3:
            gamma = 0.25
        elif shade_idx == 4:
            gamma = 0.15

        if dry_days < gamma * 365.0:
            tmp = (gamma * 365.0 - dry_days) / (gamma * 365.0)
            if tmp > 0.0:
                fc_drought = tmp ** 0.5
        else:
            fc_drought = 0.0

        # Light response
        c1 = 1.11
        c2 = 2.52
        c3 = 0.07
        if shade_idx == 0:
            c1 = 1.01
            c2 = 4.62
            c3 = 0.05
        elif shade_idx == 1:
            c1 = 1.04
            c2 = 3.44
            c3 = 0.06
        elif shade_idx == 2:
            c1 = 1.11
            c2 = 2.52
            c3 = 0.07
        elif shade_idx == 3:
            c1 = 1.24
            c2 = 1.78
            c3 = 0.08
        elif shade_idx == 4:
            c1 = 1.49
            c2 = 1.23
            c3 = 0.09

        fc_light = c1 * (1.0 - cp.exp(-c2 * (light_avail - c3)))
        if fc_light < 0.0:
            fc_light = 0.0
        if fc_light > 1.0:
            fc_light = 1.0

        # Combined growth factor
        growth_factor = fc_degday * fc_drought * fc_light
        if growth_factor < 0.0:
            growth_factor = 0.0
        if growth_factor > 1.0:
            growth_factor = 1.0

        # Nutrient limitation from N supply
        nutrient_factor = n_supply_ratio
        if nutrient_factor > 1.0:
            nutrient_factor = 1.0
        if nutrient_factor < 0.1:
            nutrient_factor = 0.1

        # ===== CALCULATE GROWTH =====

        # Maximum diameter increment (UVAFME equation)
        diam_max = 0.0
        if diam < max_diam and height < max_ht:
            delta_ht = max_ht - STD_HT
            exp_term = cp.exp(-arfa_0 * diam / delta_ht)
            denom = 2.0 * height + arfa_0 * exp_term * diam
            if denom > 0.001:
                diam_max = g * diam * (1.0 - diam * height / (max_diam * max_ht)) / denom

        # Apply stress factors
        diam_increment = diam_max * growth_factor * nutrient_factor
        if diam_increment < 0.0:
            diam_increment = 0.0

        # Minimum viable growth threshold
        pp = max_diam / max_age * 0.1
        if pp > 0.05:
            pp = 0.05

        # Check for growth stress mortality marker
        mort_marker = 0.0
        if diam_increment <= pp or growth_factor * nutrient_factor <= 0.05:
            mort_marker = 1.0

        # Update diameter
        new_diam = diam + diam_increment
        if new_diam > max_diam:
            new_diam = max_diam

        # Update height (Forska equation)
        delta_ht = max_ht - STD_HT
        new_height = STD_HT + delta_ht * (1.0 - cp.exp(-arfa_0 * new_diam / delta_ht))

        # Update biomass
        radius_m = new_diam / 200.0
        volume_m3 = PI * radius_m * radius_m * new_height
        wood_bulk_dens = 0.54
        new_biomC = volume_m3 * wood_bulk_dens * 1000.0 * 0.5

        # Leaf biomass (10% of stem)
        new_leaf_bm = new_biomC * 0.1

        # N demand for growth
        biomC_increment = new_biomC - biomC
        if biomC_increment < 0.0:
            biomC_increment = 0.0
        leaf_increment = new_leaf_bm - leaf_bm
        if leaf_increment < 0.0:
            leaf_increment = 0.0

        n_demand = biomC_increment / STEM_C_N + leaf_increment / LEAF_C_N

        # ===== CHECK MORTALITY =====

        # Age-based mortality
        age_mort_prob = 4.605 / max_age  # From UVAFME age_tol=1

        # Stress-based mortality (only if mort_marker set)
        stress_mort_prob = 0.0
        if mort_marker > 0.5:
            stress_mort_prob = 0.37  # From UVAFME stress_tol=3

        # Combined mortality probability
        total_mort_prob = age_mort_prob + stress_mort_prob * (1.0 - age_mort_prob)
        if total_mort_prob > 1.0:
            total_mort_prob = 1.0

        # Pseudo-random check (deterministic based on tick and agent_index)
        rand_val = ((tick * 997 + agent_index * 991) % 1000) / 1000.0

        # ===== APPLY UPDATES =====

        if rand_val < total_mort_prob:
            # Tree dies - all biomass becomes litter
            is_alive = 0.0
            litter_c = new_biomC
            litter_n = new_biomC / STEM_C_N
        else:
            # Tree survives - annual leaf litter (assume deciduous, 100% leaf drop)
            is_alive = 1.0
            litter_c = new_leaf_bm * 0.5  # 50% of leaves as litter
            litter_n = litter_c / LEAF_C_N

        # Update age
        new_age = age + 1.0

        # ===== WRITE TO params (internal physiology) =====
        params_tensor[agent_index][TREE_P_AGE] = new_age
        params_tensor[agent_index][TREE_P_BIOMC] = new_biomC
        params_tensor[agent_index][TREE_P_BIOMN] = new_biomC / STEM_C_N
        params_tensor[agent_index][TREE_P_LEAF_BM] = new_leaf_bm
        params_tensor[agent_index][TREE_P_LIGHT_AVAIL] = light_avail
        params_tensor[agent_index][TREE_P_FC_DEGDAY] = fc_degday
        params_tensor[agent_index][TREE_P_FC_DROUGHT] = fc_drought

        # ===== WRITE TO states_db (structure - double buffered) =====
        states_db_tensor[agent_index][TREE_DB_IS_ALIVE] = is_alive
        states_db_tensor[agent_index][TREE_DB_DIAM] = new_diam
        states_db_tensor[agent_index][TREE_DB_HEIGHT] = new_height
        states_db_tensor[agent_index][TREE_DB_CANOPY_HT] = canopy_ht

    # ===== WRITE TO states (litter output - always, even for dead trees) =====
    states_tensor[agent_index][TREE_S_LITTER_C] = litter_c
    states_tensor[agent_index][TREE_S_LITTER_N] = litter_n
    states_tensor[agent_index][TREE_S_N_DEMAND] = n_demand
