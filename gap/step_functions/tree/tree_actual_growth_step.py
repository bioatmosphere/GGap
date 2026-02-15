"""
Tree actual growth step function for GGap model (Priority 6).
Phase B of the two-phase tree growth pattern matching GAPpy's two-loop structure.

This step reads the same-tick n_supply_ratio (computed at P4, relayed at P5) and
applies nutrient feedback to finalize growth, check mortality, and output litter.

Execution Flow:
    1. FINAL GROWTH (living trees, is_alive > 0.5):
       - Read env_stress and diam_max from own params (written at P2, same tick)
       - Read n_supply_ratio from Gap neighbor states (written at P5, same tick)
       - Compute fc_nutrient from n_supply_ratio + lownutr_tol
       - Compute growth_factor = env_stress * fc_nutrient
       - Compute final diam_increment = diam_max * growth_factor
       - Update diameter, height, biomass
       - Canopy self-pruning, mortality, litter output

    2. TEMPLATE RENEWAL (templates, is_alive < -0.5):
       - Compute per-species environmental response (regrowth)
       - Count same-species living trees (avail_spec)
       - Update seedbank/seedling populations
       - Write seedling_weight to states_db for free slot selection

    3. DORMANT ACTIVATION (free slots, is_alive == 0):
       - Read num_to_recruit from Gap (written at P0, same tick)
       - Select species from templates weighted by seedling_weight
       - Copy traits, initialize as seedling, set is_alive = 1.0
       - Visible as alive next tick (double-buffered states_db)
       - Matches GAPpy: renewal is last, seedlings don't grow until next year

Key improvement: n_supply_ratio is computed and consumed within the same tick.
Previous single-pass model had a one-tick lag on nutrient feedback.

Property scheme:
- params[40]: reads env_stress, diam_max (from P2), writes age, biomC, etc.
              templates write seedbank/seedling/env_stress for renewal
- states[5]: writes litter_c/n (above-ground), litter_c/n_bg (below-ground) (consumed at P0 next tick)
- states_db[5]: writes is_alive, diam, height, canopy_ht
                templates write seedling_weight (read by free slot activation, same priority)
                free slot activation writes is_alive=1 (visible next tick via double buffer)
"""

import cupy as cp
from cupyx import jit

# === Breed IDs ===
BREED_TREE = 0
BREED_GAP = 1
BREED_SITE = 2

# === Tree params[40] (private) ===
# Species traits [0-21]:
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
TREE_P_INVADER = 10
TREE_P_SEED = 11
TREE_P_SPROUT = 12
TREE_P_WOOD_BULK_DENS = 13
TREE_P_LOWNUTR_TOL = 14
TREE_P_FLOOD_TOL = 15
TREE_P_DROUGHT_TOL = 16
TREE_P_EVERGREEN = 17
TREE_P_FIRE_TOL = 18
TREE_P_ROOTDEPTH = 19
TREE_P_STRESS_TOL = 20
TREE_P_AGE_TOL = 21
# Physiology [22-31]:
TREE_P_AGE = 22
TREE_P_BIOMC = 23
TREE_P_BIOMN = 24
TREE_P_LEAF_BM = 25
TREE_P_X = 26
TREE_P_Y = 27
TREE_P_LIGHT_AVAIL = 28
TREE_P_FC_DEGDAY = 29
TREE_P_FC_DROUGHT = 30
TREE_P_FC_FLOOD = 31
# Intermediates [32-33] (written at P2, read here at P6):
TREE_P_ENV_STRESS = 32
TREE_P_DIAM_MAX_CALC = 33
# Renewal params [34-37] (template-only):
TREE_P_SEED_SURV = 34
TREE_P_SEEDLING_LG = 35
TREE_P_SEEDBANK = 36
TREE_P_SEEDLING = 37
# Leaf area params [38-39]:
TREE_P_LEAFDIAM_A = 38
TREE_P_LEAFAREA_C = 39

# === Tree states[5] (public, no buffer) ===
TREE_S_LITTER_C = 0       # Above-ground litter carbon
TREE_S_LITTER_N = 1       # Above-ground litter nitrogen
TREE_S_N_DEMAND = 2
TREE_S_LITTER_C_BG = 3    # Below-ground litter carbon (roots)
TREE_S_LITTER_N_BG = 4    # Below-ground litter nitrogen (roots)

# === Tree states_db[5] (public, double buffered) ===
TREE_DB_IS_ALIVE = 0
TREE_DB_DIAM = 1
TREE_DB_HEIGHT = 2
TREE_DB_CANOPY_HT = 3
TREE_DB_SEEDLING_WEIGHT = 4

# === Gap states[14] (for reading from Gap neighbor) ===
GAP_S_DEG_DAYS = 0
GAP_S_DRY_DAYS = 1
GAP_S_AVAIL_N = 2
GAP_S_N_SUPPLY_RATIO = 3
GAP_S_LITTER_ACCUM_C = 4
GAP_S_LITTER_ACCUM_N = 5
GAP_S_NUM_TO_RECRUIT = 6
GAP_S_RECRUIT_RAND_SEED = 7
GAP_S_FLOOD_DAYS = 8
GAP_S_SEED_BANK = 9
GAP_S_FIRE_INTENSITY = 10

# === Constants ===
PI = 3.14159265359
STD_HT = 1.3

STEM_C_N = 400.0
CON_LEAF_C_N = 50.0   # Conifer leaf C/N ratio (GAPpy)
DEC_LEAF_C_N = 25.0   # Deciduous leaf C/N ratio (GAPpy)
CON_LEAF_B = 1.32      # Conifer leaf biomass multiplier (GAPpy: needles persist ~1.3 years)
TC_KG = 0.039269908  # PI / 80 - stem volume constant, biomC in kg (cm, m, g/cm³)

SEEDLING_AGE = 1.0


@jit.rawkernel(device="cuda")
def tree_actual_growth_step(
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
    Phase B (P6): Nutrient response + final growth + mortality + litter.

    Reads env_stress and diam_max from params (written at P2).
    Reads n_supply_ratio from Gap neighbor (written at P5, same tick).
    Applies nutrient factor and computes final diameter, height, biomass.
    """
    # ===== GET CURRENT STATE =====
    is_alive = states_db_tensor[agent_index][TREE_DB_IS_ALIVE]

    # Initialize outputs (above-ground and below-ground litter)
    litter_c = 0.0
    litter_n = 0.0
    litter_c_bg = 0.0
    litter_n_bg = 0.0

    # ===== ACTUAL GROWTH + MORTALITY: Process living trees only =====
    if is_alive > 0.5:
        # Get species parameters
        max_age = params_tensor[agent_index][TREE_P_MAX_AGE]
        max_diam = params_tensor[agent_index][TREE_P_MAX_DIAM]
        max_ht = params_tensor[agent_index][TREE_P_MAX_HT]
        arfa_0 = params_tensor[agent_index][TREE_P_ARFA_0]
        wood_bulk_dens = params_tensor[agent_index][TREE_P_WOOD_BULK_DENS]
        lownutr_tol = int(params_tensor[agent_index][TREE_P_LOWNUTR_TOL])
        evergreen = int(params_tensor[agent_index][TREE_P_EVERGREEN])
        rootdepth = params_tensor[agent_index][TREE_P_ROOTDEPTH]

        # Get current tree structure from states_db (read buffer = previous tick values)
        diam = states_db_tensor[agent_index][TREE_DB_DIAM]
        height = states_db_tensor[agent_index][TREE_DB_HEIGHT]
        canopy_ht = states_db_tensor[agent_index][TREE_DB_CANOPY_HT]

        # Get internal physiology from params
        age = params_tensor[agent_index][TREE_P_AGE]

        # Read intermediates from P2 (same tick, params has no double buffer)
        env_stress = params_tensor[agent_index][TREE_P_ENV_STRESS]
        diam_max = params_tensor[agent_index][TREE_P_DIAM_MAX_CALC]

        # ===== READ N SUPPLY RATIO FROM GAP (written at P5, same tick) =====
        n_supply_ratio = 1.0

        neighbor_indices = locations[agent_index]
        i = 0
        while i < len(neighbor_indices) and neighbor_indices[i] != -1:
            neighbor_idx = int(neighbor_indices[i])
            neighbor_breed = int(breeds[neighbor_idx])
            if neighbor_breed == BREED_GAP:
                n_supply_ratio = states_tensor[neighbor_idx][GAP_S_N_SUPPLY_RATIO]
            i = i + 1

        # ===== COMPUTE NUTRIENT RESPONSE (UVAFME poor_soil_rsp) =====
        lownutr_idx = lownutr_tol
        if lownutr_idx < 1:
            lownutr_idx = 1
        if lownutr_idx > 3:
            lownutr_idx = 3

        gamma_n = 0.25
        if lownutr_idx == 1:
            gamma_n = 0.5
        elif lownutr_idx == 2:
            gamma_n = 0.25
        elif lownutr_idx == 3:
            gamma_n = 0.05

        fc_nutrient = 1.0
        if n_supply_ratio < gamma_n:
            fc_nutrient = n_supply_ratio / gamma_n
        if fc_nutrient < 0.0:
            fc_nutrient = 0.0
        if fc_nutrient > 1.0:
            fc_nutrient = 1.0

        # ===== COMPUTE FINAL GROWTH =====
        growth_factor = env_stress * fc_nutrient
        if growth_factor < 0.0:
            growth_factor = 0.0
        if growth_factor > 1.0:
            growth_factor = 1.0

        # Final diameter increment
        diam_increment = diam_max * growth_factor
        if diam_increment < 0.0:
            diam_increment = 0.0

        # Minimum viable growth threshold
        pp = max_diam / max_age * 0.1
        if pp > 0.05:
            pp = 0.05

        # Check for growth stress mortality marker
        mort_marker = 0.0
        if diam_increment <= pp or growth_factor <= 0.05:
            mort_marker = 1.0

        # Update diameter
        new_diam = diam + diam_increment
        if new_diam > max_diam:
            new_diam = max_diam

        # Update height (Forska equation)
        delta_ht = max_ht - STD_HT
        new_height = STD_HT + delta_ht * (1.0 - cp.exp(-arfa_0 * new_diam / delta_ht))

        # Update biomass (GAPpy: stem + twig + root formula)
        # Step 1: Compute growth biomass with OLD canopy_ht
        # Basal diameter
        growth_bd = new_diam
        if new_height > STD_HT:
            growth_bd = new_height / (new_height - STD_HT) * new_diam

        # Canopy diameter (uses OLD canopy_ht)
        growth_dc = new_diam
        if new_height > canopy_ht and new_height > STD_HT:
            growth_dc = (new_height - canopy_ht) / (new_height - STD_HT) * new_diam

        # Stem biomass (trunk)
        growth_stembc = TC_KG * wood_bulk_dens * 0.3 * growth_bd * growth_bd * new_height

        # Twig biomass (crown with OLD canopy_ht)
        growth_crown_depth = new_height - canopy_ht
        if growth_crown_depth < 0.0:
            growth_crown_depth = 0.0
        growth_twigbc = TC_KG * wood_bulk_dens * 0.337 * growth_dc * growth_dc * growth_crown_depth

        # Root biomass
        growth_root_c = 0.0
        if new_height > 0.01:
            growth_root_c = growth_stembc * rootdepth / new_height + growth_twigbc * 0.5

        growth_biomC = growth_stembc + growth_twigbc + growth_root_c

        # Leaf biomass from growth
        growth_leaf_bm = growth_biomC * 0.1

        # ===== CANOPY HEIGHT SELF-PRUNING (GAPpy third tree loop) =====
        # When growth is very poor, crown base rises (self-pruning)
        new_canopy_ht = canopy_ht
        if growth_factor <= 0.05:
            new_canopy_ht = canopy_ht + 1.0
            if new_canopy_ht >= new_height:
                new_canopy_ht = new_height - 0.01
            if new_canopy_ht < STD_HT:
                new_canopy_ht = STD_HT

        # Step 2: Recompute biomass with NEW canopy_ht (after pruning)
        # Stem biomass is unchanged (independent of canopy_ht)
        new_stembc = growth_stembc

        # Twig biomass (crown with NEW canopy_ht)
        new_dc = new_diam
        if new_height > new_canopy_ht and new_height > STD_HT:
            new_dc = (new_height - new_canopy_ht) / (new_height - STD_HT) * new_diam

        new_crown_depth = new_height - new_canopy_ht
        if new_crown_depth < 0.0:
            new_crown_depth = 0.0
        new_twigbc = TC_KG * wood_bulk_dens * 0.337 * new_dc * new_dc * new_crown_depth

        # Root biomass with new twig
        new_root_c = 0.0
        if new_height > 0.01:
            new_root_c = new_stembc * rootdepth / new_height + new_twigbc * 0.5

        new_biomC = new_stembc + new_twigbc + new_root_c

        # Leaf biomass
        new_leaf_bm = new_biomC * 0.1

        # ===== CHECK MORTALITY (GAPpy: two independent survival checks) =====

        # Read species-specific mortality tolerances
        stress_tol = int(params_tensor[agent_index][TREE_P_STRESS_TOL])
        age_tol = int(params_tensor[agent_index][TREE_P_AGE_TOL])

        # --- Age survival (GAPpy tree.py:178-202) ---
        # age_tol (1-3) selects from [4.605, 6.908, 11.51]
        age_k = age_tol - 1
        if age_k < 0:
            age_k = 0
        if age_k > 2:
            age_k = 2
        age_check = 4.605
        if age_k == 1:
            age_check = 6.908
        if age_k == 2:
            age_check = 11.51
        age_mort_prob = age_check / max_age
        age_rand = ((tick * 997 + agent_index * 991) % 10000) / 10000.0
        age_dies = 0
        if age_rand < age_mort_prob:
            age_dies = 1

        # --- Growth survival (GAPpy tree.py:204-222) ---
        # stress_tol (1-5) selects from [0.31, 0.34, 0.37, 0.40, 0.43]
        stress_k = stress_tol - 1
        if stress_k < 0:
            stress_k = 0
        if stress_k > 4:
            stress_k = 4
        stress_check = 0.37
        if stress_k == 0:
            stress_check = 0.31
        if stress_k == 1:
            stress_check = 0.34
        if stress_k == 2:
            stress_check = 0.37
        if stress_k == 3:
            stress_check = 0.40
        if stress_k == 4:
            stress_check = 0.43
        stress_rand = ((tick * 1013 + agent_index * 1009) % 10000) / 10000.0
        growth_dies = 0
        if mort_marker > 0.5 and stress_rand < stress_check:
            growth_dies = 1

        # --- Combined: tree survives only if BOTH checks pass ---
        tree_dies = 0
        if age_dies > 0 or growth_dies > 0:
            tree_dies = 1

        # ===== CANOPY HEIGHT LITTER (GAPpy growth() third tree loop) =====
        # Biomass lost from canopy pruning: growth_biomC (before pruning) - new_biomC (after)
        canopy_litter_c = 0.0
        canopy_litter_n = 0.0
        d_bc = growth_biomC - new_biomC  # Positive when pruning shrinks crown
        if d_bc > 0.0:
            canopy_litter_c = canopy_litter_c + d_bc
            canopy_litter_n = canopy_litter_n + d_bc / STEM_C_N

        # Leaf biomass lost from crown reduction
        d_leafb = growth_leaf_bm - new_leaf_bm
        if d_leafb > 0.0:
            if evergreen > 0:
                canopy_litter_c = canopy_litter_c + d_leafb * CON_LEAF_B
                canopy_litter_n = canopy_litter_n + d_leafb / CON_LEAF_C_N * CON_LEAF_B
            else:
                canopy_litter_c = canopy_litter_c + d_leafb
                canopy_litter_n = canopy_litter_n + d_leafb / DEC_LEAF_C_N

        # ===== APPLY UPDATES =====

        if tree_dies > 0:
            # Tree dies - all biomass becomes litter
            # GAPpy: 70% above-ground (A0), 30% below-ground (A layer roots)
            is_alive = 0.0
            total_c = 0.0
            total_n = 0.0
            if evergreen > 0:
                total_c = new_biomC + new_leaf_bm * CON_LEAF_B
                total_n = new_biomC / STEM_C_N + new_leaf_bm / CON_LEAF_C_N * CON_LEAF_B
            else:
                total_c = new_biomC + new_leaf_bm
                total_n = new_biomC / STEM_C_N + new_leaf_bm / DEC_LEAF_C_N
            litter_c = total_c * 0.7
            litter_n = total_n * 0.7
            litter_c_bg = total_c * 0.3
            litter_n_bg = total_n * 0.3
        else:
            # Tree survives - annual leaf litter (100% above-ground)
            # GAPpy: deciduous=100% leaf drop, conifer=~32% needle drop (leaf_b - 1.0)
            is_alive = 1.0
            if evergreen > 0:
                litter_c = new_leaf_bm * (CON_LEAF_B - 1.0)
                litter_n = litter_c / CON_LEAF_C_N
            else:
                litter_c = new_leaf_bm
                litter_n = litter_c / DEC_LEAF_C_N
            # No below-ground litter from annual leaf drop

        # Add canopy height adjustment litter (100% above-ground)
        litter_c = litter_c + canopy_litter_c
        litter_n = litter_n + canopy_litter_n

        # Update age
        new_age = age + 1.0

        # ===== WRITE TO params (internal physiology) =====
        params_tensor[agent_index][TREE_P_AGE] = new_age
        params_tensor[agent_index][TREE_P_BIOMC] = new_biomC
        params_tensor[agent_index][TREE_P_BIOMN] = new_biomC / STEM_C_N
        params_tensor[agent_index][TREE_P_LEAF_BM] = new_leaf_bm

        # ===== WRITE TO states_db (structure - double buffered) =====
        states_db_tensor[agent_index][TREE_DB_IS_ALIVE] = is_alive
        states_db_tensor[agent_index][TREE_DB_DIAM] = new_diam
        states_db_tensor[agent_index][TREE_DB_HEIGHT] = new_height
        states_db_tensor[agent_index][TREE_DB_CANOPY_HT] = new_canopy_ht

    # ===== TEMPLATE RENEWAL (GAPpy renewal(), model.py:792-982) =====
    # Templates (is_alive = -1) compute per-species seedbank/seedling dynamics.
    # Each template represents one species and tracks its renewal state.
    elif is_alive < -0.5:
        # --- 1. Read climate from Gap neighbor (same-tick, from P5 gap_sync) ---
        deg_days = 2500.0
        dry_days = 30.0
        flood_days = 0.0
        n_supply_ratio = 1.0

        neighbor_indices = locations[agent_index]
        i = 0
        while i < len(neighbor_indices) and neighbor_indices[i] != -1:
            neighbor_idx = int(neighbor_indices[i])
            neighbor_breed = int(breeds[neighbor_idx])
            if neighbor_breed == BREED_GAP:
                deg_days = states_tensor[neighbor_idx][GAP_S_DEG_DAYS]
                dry_days = states_tensor[neighbor_idx][GAP_S_DRY_DAYS]
                flood_days = states_tensor[neighbor_idx][GAP_S_FLOOD_DAYS]
                n_supply_ratio = states_tensor[neighbor_idx][GAP_S_N_SUPPLY_RATIO]
            i = i + 1

        # Read species params
        deg_day_min = params_tensor[agent_index][TREE_P_DEG_DAY_MIN]
        deg_day_opt = params_tensor[agent_index][TREE_P_DEG_DAY_OPT]
        deg_day_max = params_tensor[agent_index][TREE_P_DEG_DAY_MAX]
        shade_tol = int(params_tensor[agent_index][TREE_P_SHADE_TOL])
        drought_tol = int(params_tensor[agent_index][TREE_P_DROUGHT_TOL])
        flood_tol = int(params_tensor[agent_index][TREE_P_FLOOD_TOL])
        lownutr_tol = int(params_tensor[agent_index][TREE_P_LOWNUTR_TOL])
        evergreen = int(params_tensor[agent_index][TREE_P_EVERGREEN])
        species_id = params_tensor[agent_index][TREE_P_SPECIES_ID]
        invader_val = params_tensor[agent_index][TREE_P_INVADER]
        seed_val = params_tensor[agent_index][TREE_P_SEED]
        sprout_val = params_tensor[agent_index][TREE_P_SPROUT]
        seed_surv = params_tensor[agent_index][TREE_P_SEED_SURV]
        seedling_lg = params_tensor[agent_index][TREE_P_SEEDLING_LG]
        seedbank = params_tensor[agent_index][TREE_P_SEEDBANK]
        seedling = params_tensor[agent_index][TREE_P_SEEDLING]

        # --- 2. Compute species-specific environmental response ---

        # Temperature response (parabolic) - same as living trees
        fc_degday = 0.0
        if deg_days > deg_day_min and deg_days < deg_day_max:
            a = (deg_day_opt - deg_day_min) / (deg_day_max - deg_day_min)
            b = (deg_day_max - deg_day_opt) / (deg_day_max - deg_day_min)
            tmp1 = (deg_days - deg_day_min) / (deg_day_opt - deg_day_min)
            tmp2 = (deg_day_max - deg_days) / (deg_day_max - deg_day_opt)
            if tmp1 > 0.0 and tmp2 > 0.0:
                fc_degday = (tmp1 ** a) * (tmp2 ** b)

        # Drought response - same as living trees
        fc_drought = 1.0
        drought_idx = drought_tol - 1
        if drought_idx < 0:
            drought_idx = 0
        if drought_idx > 4:
            drought_idx = 4
        gamma_d = 0.35
        if drought_idx == 0:
            gamma_d = 0.50
        elif drought_idx == 1:
            gamma_d = 0.45
        elif drought_idx == 2:
            gamma_d = 0.35
        elif drought_idx == 3:
            gamma_d = 0.25
        elif drought_idx == 4:
            gamma_d = 0.15
        if dry_days < gamma_d * 365.0:
            tmp = (gamma_d * 365.0 - dry_days) / (gamma_d * 365.0)
            if tmp > 0.0:
                fc_drought = tmp ** 0.5
        else:
            fc_drought = 0.0

        # Flood response - same as living trees
        flood_idx = flood_tol
        if flood_idx < 1:
            flood_idx = 1
        if flood_idx > 6:
            flood_idx = 6
        gamma_f = 1.0 - (flood_idx - 1) * 0.1
        flood_threshold = gamma_f * 365.0
        fc_flood = 1.0
        if flood_days >= flood_threshold:
            fc_flood = 0.0
        elif flood_days > 0.0 and flood_threshold > 0.0:
            fc_flood = 1.0 - (flood_days / flood_threshold)
        if fc_flood < 0.0:
            fc_flood = 0.0
        if fc_flood > 1.0:
            fc_flood = 1.0

        # Nutrient response - same as living trees
        lownutr_idx = lownutr_tol
        if lownutr_idx < 1:
            lownutr_idx = 1
        if lownutr_idx > 3:
            lownutr_idx = 3
        gamma_n = 0.25
        if lownutr_idx == 1:
            gamma_n = 0.5
        elif lownutr_idx == 2:
            gamma_n = 0.25
        elif lownutr_idx == 3:
            gamma_n = 0.05
        fc_nutrient = 1.0
        if n_supply_ratio < gamma_n:
            fc_nutrient = n_supply_ratio / gamma_n
        if fc_nutrient < 0.0:
            fc_nutrient = 0.0
        if fc_nutrient > 1.0:
            fc_nutrient = 1.0

        # Light response at ground level (template height=0, all neighbors shade it)
        # Beer-Lambert: compute total LAI from all living neighbors above ground
        total_lai_above = 0.0
        i = 0
        while i < len(neighbor_indices) and neighbor_indices[i] != -1:
            neighbor_idx = int(neighbor_indices[i])
            neighbor_breed = int(breeds[neighbor_idx])
            if neighbor_breed == BREED_TREE:
                neighbor_alive = states_db_tensor[neighbor_idx][TREE_DB_IS_ALIVE]
                if neighbor_alive > 0.5:
                    neighbor_height = states_db_tensor[neighbor_idx][TREE_DB_HEIGHT]
                    neighbor_canopy_ht = states_db_tensor[neighbor_idx][TREE_DB_CANOPY_HT]
                    neighbor_diam = states_db_tensor[neighbor_idx][TREE_DB_DIAM]
                    neighbor_evergreen = int(params_tensor[neighbor_idx][TREE_P_EVERGREEN])
                    # Template is at ground level, so all trees with height > 0 shade it
                    if neighbor_height > 0.01:
                        canopy_depth = neighbor_height - neighbor_canopy_ht
                        if canopy_depth < 1.0:
                            canopy_depth = 1.0
                        neighbor_lai = (neighbor_diam * neighbor_diam * 0.01) / canopy_depth
                        # Full overlap since template is at ground
                        overlap = neighbor_height
                        if overlap > canopy_depth:
                            overlap = canopy_depth
                        if neighbor_evergreen > 0:
                            neighbor_xt = -0.70
                        else:
                            neighbor_xt = -0.80
                        weighted_lai = neighbor_lai * overlap * (neighbor_xt / (-0.80))
                        total_lai_above = total_lai_above + weighted_lai
            i = i + 1

        light_avail = cp.exp(-0.80 * total_lai_above)
        if light_avail < 0.01:
            light_avail = 0.01
        if light_avail > 1.0:
            light_avail = 1.0

        # Light tolerance response - same as living trees
        shade_idx = shade_tol - 1
        if shade_idx < 0:
            shade_idx = 0
        if shade_idx > 4:
            shade_idx = 4
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

        # --- 3. Compute regrowth (GAPpy model.py:816-825) ---
        regrowth = fc_degday * fc_drought * fc_flood * fc_nutrient * fc_light
        if regrowth <= 0.05:
            regrowth = 0.0

        # --- 4. Count living trees of same species (avail_spec) ---
        avail_spec = 0.0
        i = 0
        while i < len(neighbor_indices) and neighbor_indices[i] != -1:
            neighbor_idx = int(neighbor_indices[i])
            neighbor_breed = int(breeds[neighbor_idx])
            if neighbor_breed == BREED_TREE:
                neighbor_alive = states_db_tensor[neighbor_idx][TREE_DB_IS_ALIVE]
                if neighbor_alive > 0.5:
                    neighbor_species = params_tensor[neighbor_idx][TREE_P_SPECIES_ID]
                    diff = neighbor_species - species_id
                    if diff < 0.5 and diff > -0.5:
                        avail_spec = avail_spec + 1.0
            i = i + 1

        # --- 5. Update seedbank (GAPpy model.py:843-856) ---
        seedbank = seedbank + invader_val + seed_val * avail_spec + sprout_val * avail_spec
        if regrowth >= 0.05:
            seedling = seedling + seedbank
            seedbank = 0.0
        else:
            seedbank = seedbank * seed_surv

        # --- 6. Compute recruitment weight ---
        weight = seedling * regrowth

        # --- 7. Apply annual seedling survival (GAPpy model.py:969-972) ---
        seedling = seedling * seedling_lg

        # --- 8. Write outputs ---
        params_tensor[agent_index][TREE_P_SEEDBANK] = seedbank
        params_tensor[agent_index][TREE_P_SEEDLING] = seedling
        params_tensor[agent_index][TREE_P_ENV_STRESS] = regrowth  # Gap reads at P0 for growmax
        states_db_tensor[agent_index][TREE_DB_SEEDLING_WEIGHT] = weight

    # ===== DORMANT SLOT ACTIVATION (moved from P2 to match GAPpy renewal-last ordering) =====
    # Free slots (is_alive == 0) check if Gap signals recruitment, select species
    # from templates weighted by SEEDLING_WEIGHT, and activate as seedlings.
    # By activating at P6, seedlings don't grow until next tick (matching GAPpy).
    else:
        # Read recruitment info from Gap neighbor
        num_to_recruit = 0.0
        recruit_rand_seed = 0.0

        neighbor_indices = locations[agent_index]
        i = 0
        while i < len(neighbor_indices) and neighbor_indices[i] != -1:
            neighbor_idx = int(neighbor_indices[i])
            neighbor_breed = int(breeds[neighbor_idx])
            if neighbor_breed == BREED_GAP:
                num_to_recruit = states_tensor[neighbor_idx][GAP_S_NUM_TO_RECRUIT]
                recruit_rand_seed = states_tensor[neighbor_idx][GAP_S_RECRUIT_RAND_SEED]
            i = i + 1

        # Determine if this free slot should be recruited
        if num_to_recruit > 0.5:
            slot_priority = ((agent_index * 997 + int(recruit_rand_seed)) % 10000) / 10000.0
            recruit_threshold = num_to_recruit / 100.0
            if recruit_threshold > 1.0:
                recruit_threshold = 1.0

            if slot_priority < recruit_threshold:
                # Select species from templates weighted by SEEDLING_WEIGHT
                total_weight = 0.0
                selected_neighbor_idx = -1

                i = 0
                while i < len(neighbor_indices) and neighbor_indices[i] != -1:
                    neighbor_idx = int(neighbor_indices[i])
                    neighbor_breed = int(breeds[neighbor_idx])
                    if neighbor_breed == BREED_TREE:
                        neighbor_alive = states_db_tensor[neighbor_idx][TREE_DB_IS_ALIVE]
                        # Only select from templates (is_alive < -0.5)
                        if neighbor_alive < -0.5:
                            weight = states_db_tensor[neighbor_idx][TREE_DB_SEEDLING_WEIGHT]
                            if weight < 0.01:
                                weight = 0.01
                            total_weight = total_weight + weight

                            rand_val = ((agent_index * 991 + int(recruit_rand_seed) * 7 + i) % 1000) / 1000.0
                            select_prob = weight / (total_weight + 0.001)
                            if rand_val < select_prob:
                                selected_neighbor_idx = neighbor_idx
                    i = i + 1

                if selected_neighbor_idx >= 0:
                    # Copy species traits [0-21] from template
                    params_tensor[agent_index][TREE_P_SPECIES_ID] = params_tensor[selected_neighbor_idx][TREE_P_SPECIES_ID]
                    params_tensor[agent_index][TREE_P_MAX_AGE] = params_tensor[selected_neighbor_idx][TREE_P_MAX_AGE]
                    params_tensor[agent_index][TREE_P_MAX_DIAM] = params_tensor[selected_neighbor_idx][TREE_P_MAX_DIAM]
                    params_tensor[agent_index][TREE_P_MAX_HT] = params_tensor[selected_neighbor_idx][TREE_P_MAX_HT]
                    params_tensor[agent_index][TREE_P_ARFA_0] = params_tensor[selected_neighbor_idx][TREE_P_ARFA_0]
                    params_tensor[agent_index][TREE_P_G] = params_tensor[selected_neighbor_idx][TREE_P_G]
                    params_tensor[agent_index][TREE_P_SHADE_TOL] = params_tensor[selected_neighbor_idx][TREE_P_SHADE_TOL]
                    params_tensor[agent_index][TREE_P_DEG_DAY_MIN] = params_tensor[selected_neighbor_idx][TREE_P_DEG_DAY_MIN]
                    params_tensor[agent_index][TREE_P_DEG_DAY_OPT] = params_tensor[selected_neighbor_idx][TREE_P_DEG_DAY_OPT]
                    params_tensor[agent_index][TREE_P_DEG_DAY_MAX] = params_tensor[selected_neighbor_idx][TREE_P_DEG_DAY_MAX]
                    params_tensor[agent_index][TREE_P_INVADER] = params_tensor[selected_neighbor_idx][TREE_P_INVADER]
                    params_tensor[agent_index][TREE_P_SEED] = params_tensor[selected_neighbor_idx][TREE_P_SEED]
                    params_tensor[agent_index][TREE_P_SPROUT] = params_tensor[selected_neighbor_idx][TREE_P_SPROUT]
                    params_tensor[agent_index][TREE_P_WOOD_BULK_DENS] = params_tensor[selected_neighbor_idx][TREE_P_WOOD_BULK_DENS]
                    params_tensor[agent_index][TREE_P_LOWNUTR_TOL] = params_tensor[selected_neighbor_idx][TREE_P_LOWNUTR_TOL]
                    params_tensor[agent_index][TREE_P_FLOOD_TOL] = params_tensor[selected_neighbor_idx][TREE_P_FLOOD_TOL]
                    params_tensor[agent_index][TREE_P_DROUGHT_TOL] = params_tensor[selected_neighbor_idx][TREE_P_DROUGHT_TOL]
                    params_tensor[agent_index][TREE_P_EVERGREEN] = params_tensor[selected_neighbor_idx][TREE_P_EVERGREEN]
                    params_tensor[agent_index][TREE_P_FIRE_TOL] = params_tensor[selected_neighbor_idx][TREE_P_FIRE_TOL]
                    params_tensor[agent_index][TREE_P_STRESS_TOL] = params_tensor[selected_neighbor_idx][TREE_P_STRESS_TOL]
                    params_tensor[agent_index][TREE_P_AGE_TOL] = params_tensor[selected_neighbor_idx][TREE_P_AGE_TOL]
                    # Copy renewal params from template
                    params_tensor[agent_index][TREE_P_SEED_SURV] = params_tensor[selected_neighbor_idx][TREE_P_SEED_SURV]
                    params_tensor[agent_index][TREE_P_SEEDLING_LG] = params_tensor[selected_neighbor_idx][TREE_P_SEEDLING_LG]
                    # Copy leaf area params from template
                    params_tensor[agent_index][TREE_P_LEAFDIAM_A] = params_tensor[selected_neighbor_idx][TREE_P_LEAFDIAM_A]
                    params_tensor[agent_index][TREE_P_LEAFAREA_C] = params_tensor[selected_neighbor_idx][TREE_P_LEAFAREA_C]

                    # Variable seedling diameter: uniform [0.5, 2.5] (approximates GAPpy 1.5+N(0,1))
                    seedling_diam = 0.5 + ((agent_index * 1013 + tick * 997 + int(recruit_rand_seed)) % 2000) / 1000.0

                    # Initialize physiology as seedling
                    params_tensor[agent_index][TREE_P_AGE] = SEEDLING_AGE
                    params_tensor[agent_index][TREE_P_BIOMC] = 0.1
                    params_tensor[agent_index][TREE_P_BIOMN] = 0.1 / STEM_C_N
                    params_tensor[agent_index][TREE_P_LEAF_BM] = 0.01
                    params_tensor[agent_index][TREE_P_LIGHT_AVAIL] = 1.0
                    params_tensor[agent_index][TREE_P_FC_DEGDAY] = 1.0
                    params_tensor[agent_index][TREE_P_FC_DROUGHT] = 1.0
                    params_tensor[agent_index][TREE_P_FC_FLOOD] = 1.0

                    # Calculate initial height from seedling diameter
                    max_ht = params_tensor[agent_index][TREE_P_MAX_HT]
                    arfa_0 = params_tensor[agent_index][TREE_P_ARFA_0]
                    delta_ht = max_ht - STD_HT
                    seedling_ht = STD_HT + delta_ht * (1.0 - cp.exp(-arfa_0 * seedling_diam / delta_ht))

                    # Set structure for seedling
                    states_db_tensor[agent_index][TREE_DB_DIAM] = seedling_diam
                    states_db_tensor[agent_index][TREE_DB_HEIGHT] = seedling_ht
                    states_db_tensor[agent_index][TREE_DB_CANOPY_HT] = STD_HT

                    # Activate the seedling (visible next tick via double buffer)
                    states_db_tensor[agent_index][TREE_DB_IS_ALIVE] = 1.0

    # ===== WRITE LITTER TO states (for Gap to aggregate at P0 next tick) =====
    states_tensor[agent_index][TREE_S_LITTER_C] = litter_c          # Above-ground -> A0 layer
    states_tensor[agent_index][TREE_S_LITTER_N] = litter_n
    states_tensor[agent_index][TREE_S_LITTER_C_BG] = litter_c_bg    # Below-ground -> A layer
    states_tensor[agent_index][TREE_S_LITTER_N_BG] = litter_n_bg
