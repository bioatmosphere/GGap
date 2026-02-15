"""
Tree potential growth step function for GGap model (Priority 2).
Phase A of the two-phase tree growth pattern matching GAPpy's two-loop structure.

This step computes environmental stress and potential growth WITHOUT nutrient feedback.
The nutrient factor is applied later in tree_actual_growth_step (Priority 6) after
the soil N cycle computes the same-tick n_supply_ratio.

Execution Flow:
    1. POTENTIAL GROWTH (living trees only):
       - Read climate from Gap (deg_days, dry_days, etc.)
       - Calculate light availability from neighbor heights
       - Calculate env stress: fc_degday * fc_drought * fc_light * fc_flood (NO fc_nutrient)
       - Calculate diam_max (maximum diameter increment for current size)
       - Compute potential diameter/height/biomass from diam_max * env_stress
       - Compute n_demand from potential biomass change

    2. OUTPUTS (written to params for same-tick P4 consumption):
       - params[ENV_STRESS]: composite env stress (no nutrient)
       - params[DIAM_MAX_CALC]: max diameter increment
       - params[FC_DEGDAY/DROUGHT/FLOOD/LIGHT_AVAIL]: individual factors
       - states[N_DEMAND]: nitrogen demand from potential growth

Note: Recruitment (free slot activation) moved to P6 (tree_actual_growth_step)
to match GAPpy ordering where renewal is the last annual operation.

Property scheme:
- params[40]: species traits (static) + physiology (dynamic) + intermediates + renewal + leaf_area - private, no buffer
- states[5]: n_demand output (for Gap to aggregate at P3) - public, no buffer
- states_db[5]: structure (is_alive, diam, height, canopy_ht) + seedling_weight - public, double buffered
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
# Intermediates [32-33] (P2 writes, P6 reads, same tick via no_double_buffer):
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
TC_KG = 0.039269908  # PI / 80 - stem volume constant, biomC in kg (cm, m, g/cm³)
XT_CONIFER = -0.70
XT_DECIDUOUS = -0.80

STEM_C_N = 400.0
CON_LEAF_C_N = 50.0
DEC_LEAF_C_N = 25.0
CON_LEAF_B = 1.32

SEEDLING_DIAM = 1.0
SEEDLING_AGE = 1.0


@jit.rawkernel(device="cuda")
def tree_potential_growth_step(
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
    Phase A (P2): Recruitment + environmental stress + potential growth + N demand.

    Computes env_stress = fc_degday * fc_drought * fc_light * fc_flood (no nutrient).
    Stores env_stress and diam_max in params for P6 to consume same-tick.
    """
    # ===== GET CURRENT STATE =====
    is_alive = states_db_tensor[agent_index][TREE_DB_IS_ALIVE]

    # Initialize outputs
    n_demand = 0.0

    # ===== POTENTIAL GROWTH: Process living trees only =====
    if is_alive > 0.5:
        # Get species parameters
        max_diam = params_tensor[agent_index][TREE_P_MAX_DIAM]
        max_ht = params_tensor[agent_index][TREE_P_MAX_HT]
        arfa_0 = params_tensor[agent_index][TREE_P_ARFA_0]
        g = params_tensor[agent_index][TREE_P_G]
        shade_tol = int(params_tensor[agent_index][TREE_P_SHADE_TOL])
        deg_day_min = params_tensor[agent_index][TREE_P_DEG_DAY_MIN]
        deg_day_opt = params_tensor[agent_index][TREE_P_DEG_DAY_OPT]
        deg_day_max = params_tensor[agent_index][TREE_P_DEG_DAY_MAX]
        wood_bulk_dens = params_tensor[agent_index][TREE_P_WOOD_BULK_DENS]
        flood_tol = int(params_tensor[agent_index][TREE_P_FLOOD_TOL])
        drought_tol = int(params_tensor[agent_index][TREE_P_DROUGHT_TOL])
        evergreen = int(params_tensor[agent_index][TREE_P_EVERGREEN])

        rootdepth = params_tensor[agent_index][TREE_P_ROOTDEPTH]

        # Get current tree structure from states_db
        diam = states_db_tensor[agent_index][TREE_DB_DIAM]
        height = states_db_tensor[agent_index][TREE_DB_HEIGHT]
        canopy_ht = states_db_tensor[agent_index][TREE_DB_CANOPY_HT]

        # Get internal physiology from params
        biomC = params_tensor[agent_index][TREE_P_BIOMC]
        leaf_bm = params_tensor[agent_index][TREE_P_LEAF_BM]

        # ===== READ CLIMATE FROM GAP NEIGHBOR =====
        deg_days = 2500.0
        dry_days = 30.0
        flood_days = 0.0

        neighbor_indices = locations[agent_index]
        i = 0
        while i < len(neighbor_indices) and neighbor_indices[i] != -1:
            neighbor_idx = int(neighbor_indices[i])
            neighbor_breed = int(breeds[neighbor_idx])
            if neighbor_breed == BREED_GAP:
                deg_days = states_tensor[neighbor_idx][GAP_S_DEG_DAYS]
                dry_days = states_tensor[neighbor_idx][GAP_S_DRY_DAYS]
                flood_days = states_tensor[neighbor_idx][GAP_S_FLOOD_DAYS]
            i = i + 1

        # ===== CALCULATE LIGHT AVAILABILITY =====
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

                    if neighbor_height > height:
                        canopy_depth = neighbor_height - neighbor_canopy_ht
                        if canopy_depth < 1.0:
                            canopy_depth = 1.0

                        neighbor_lai = (neighbor_diam * neighbor_diam * 0.01) / canopy_depth
                        overlap = neighbor_height - height
                        if overlap > canopy_depth:
                            overlap = canopy_depth

                        if neighbor_evergreen > 0:
                            neighbor_xt = XT_CONIFER
                        else:
                            neighbor_xt = XT_DECIDUOUS

                        weighted_lai = neighbor_lai * overlap * (neighbor_xt / XT_DECIDUOUS)
                        total_lai_above = total_lai_above + weighted_lai
            i = i + 1

        # Beer-Lambert light attenuation
        light_avail = cp.exp(XT_DECIDUOUS * total_lai_above)
        if light_avail < 0.01:
            light_avail = 0.01
        if light_avail > 1.0:
            light_avail = 1.0

        # ===== CALCULATE ENVIRONMENTAL STRESS FACTORS (NO NUTRIENT) =====

        # Temperature response (parabolic)
        fc_degday = 0.0
        if deg_days > deg_day_min and deg_days < deg_day_max:
            a = (deg_day_opt - deg_day_min) / (deg_day_max - deg_day_min)
            b = (deg_day_max - deg_day_opt) / (deg_day_max - deg_day_min)
            tmp1 = (deg_days - deg_day_min) / (deg_day_opt - deg_day_min)
            tmp2 = (deg_day_max - deg_days) / (deg_day_max - deg_day_opt)
            if tmp1 > 0.0 and tmp2 > 0.0:
                fc_degday = (tmp1 ** a) * (tmp2 ** b)

        # Drought response
        fc_drought = 1.0
        drought_idx = drought_tol - 1
        if drought_idx < 0:
            drought_idx = 0
        if drought_idx > 4:
            drought_idx = 4

        gamma = 0.35
        if drought_idx == 0:
            gamma = 0.50
        elif drought_idx == 1:
            gamma = 0.45
        elif drought_idx == 2:
            gamma = 0.35
        elif drought_idx == 3:
            gamma = 0.25
        elif drought_idx == 4:
            gamma = 0.15

        if dry_days < gamma * 365.0:
            tmp = (gamma * 365.0 - dry_days) / (gamma * 365.0)
            if tmp > 0.0:
                fc_drought = tmp ** 0.5
        else:
            fc_drought = 0.0

        # Light response
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

        # Flood response
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

        # Combined env stress (NO nutrient factor - that comes at P4)
        env_stress = fc_degday * fc_drought * fc_light * fc_flood
        if env_stress < 0.0:
            env_stress = 0.0
        if env_stress > 1.0:
            env_stress = 1.0

        # ===== CALCULATE POTENTIAL GROWTH (used for N demand) =====

        # Maximum diameter increment (UVAFME equation)
        diam_max = 0.0
        if diam < max_diam and height < max_ht:
            delta_ht = max_ht - STD_HT
            exp_term = cp.exp(-arfa_0 * diam / delta_ht)
            denom = 2.0 * height + arfa_0 * exp_term * diam
            if denom > 0.001:
                diam_max = g * diam * (1.0 - diam * height / (max_diam * max_ht)) / denom

        # Potential diameter increment (env stress only, no nutrient)
        pot_diam_increment = diam_max * env_stress
        if pot_diam_increment < 0.0:
            pot_diam_increment = 0.0

        # Potential new diameter
        pot_new_diam = diam + pot_diam_increment
        if pot_new_diam > max_diam:
            pot_new_diam = max_diam

        # Potential new height
        delta_ht = max_ht - STD_HT
        pot_new_height = STD_HT + delta_ht * (1.0 - cp.exp(-arfa_0 * pot_new_diam / delta_ht))

        # Potential new biomass (GAPpy: stem + twig + root)
        # Basal diameter (canopy at ground)
        pot_bd = pot_new_diam
        if pot_new_height > STD_HT:
            pot_bd = pot_new_height / (pot_new_height - STD_HT) * pot_new_diam

        # Canopy diameter (uses current canopy_ht)
        pot_dc = pot_new_diam
        if pot_new_height > canopy_ht and pot_new_height > STD_HT:
            pot_dc = (pot_new_height - canopy_ht) / (pot_new_height - STD_HT) * pot_new_diam

        # Stem biomass (trunk)
        pot_stembc = TC_KG * wood_bulk_dens * 0.3 * pot_bd * pot_bd * pot_new_height

        # Twig biomass (crown)
        pot_crown_depth = pot_new_height - canopy_ht
        if pot_crown_depth < 0.0:
            pot_crown_depth = 0.0
        pot_twigbc = TC_KG * wood_bulk_dens * 0.337 * pot_dc * pot_dc * pot_crown_depth

        # Root biomass
        pot_root_c = 0.0
        if pot_new_height > 0.01:
            pot_root_c = pot_stembc * rootdepth / pot_new_height + pot_twigbc * 0.5

        pot_new_biomC = pot_stembc + pot_twigbc + pot_root_c

        # Potential leaf biomass
        pot_new_leaf_bm = pot_new_biomC * 0.1

        # N demand from potential growth (GAPpy: species-specific leaf C/N)
        biomC_increment = pot_new_biomC - biomC
        if biomC_increment < 0.0:
            biomC_increment = 0.0

        # Stem N demand
        stem_n_demand = biomC_increment / STEM_C_N

        # Leaf N demand (GAPpy differentiates conifer vs deciduous)
        leaf_n_demand = 0.0
        if evergreen > 0:
            # Conifer: maintain CON_LEAF_B × leaf mass, minus old retained needles
            leaf_n_demand = (CON_LEAF_B * pot_new_leaf_bm - leaf_bm) / CON_LEAF_C_N
        else:
            # Deciduous: regrow all leaves each year
            leaf_n_demand = pot_new_leaf_bm / DEC_LEAF_C_N
        if leaf_n_demand < 0.0:
            leaf_n_demand = 0.0

        n_demand = stem_n_demand + leaf_n_demand

        # ===== WRITE INTERMEDIATES TO params (for P4 to read same-tick) =====
        params_tensor[agent_index][TREE_P_LIGHT_AVAIL] = light_avail
        params_tensor[agent_index][TREE_P_FC_DEGDAY] = fc_degday
        params_tensor[agent_index][TREE_P_FC_DROUGHT] = fc_drought
        params_tensor[agent_index][TREE_P_FC_FLOOD] = fc_flood
        params_tensor[agent_index][TREE_P_ENV_STRESS] = env_stress
        params_tensor[agent_index][TREE_P_DIAM_MAX_CALC] = diam_max

    # ===== WRITE N DEMAND TO states (for Gap to aggregate at P1) =====
    states_tensor[agent_index][TREE_S_N_DEMAND] = n_demand
