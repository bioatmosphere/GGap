"""
Tree potential growth step function for GGap model (Priority 2).
Phase A: computes environmental stress and potential growth for LIVING TREES only.

The nutrient factor is applied later in tree_actual_growth_step (Priority 8) after
the soil N cycle computes the same-tick n_supply_ratio.

Templates are handled by tree_template_renewal_step (P6) after P5 syncs current-tick
climate. Free slots are handled by tree_actual_growth_step (P8) after P7 computes
num_to_recruit.

Execution Flow:
    1. LIVING TREES (is_alive > 0.5):
       - Read climate from Gap (deg_days, dry_days, etc.)
       - Calculate light availability from neighbor heights
       - Calculate env stress: fc_degday * fc_drought * fc_light * fc_flood (NO fc_nutrient)
       - Calculate diam_max (maximum diameter increment for current size)
       - Compute potential diameter/height/biomass from diam_max * env_stress
       - Compute n_demand from potential biomass change

    2. TEMPLATES + FREE SLOTS: no-op (n_demand = 0.0)

    3. OUTPUTS:
       - params[ENV_STRESS]: env stress (living only)
       - params[DIAM_MAX_CALC]: max diameter increment (living only)
       - params[FC_DEGDAY/DROUGHT/FLOOD/LIGHT_AVAIL]: individual factors (living only)
       - states[N_DEMAND]: nitrogen demand from potential growth (living only, 0 for others)

Property scheme:
- params[42]: species traits (static) + physiology (dynamic) + intermediates + renewal + leaf_area + seedling_weight - private, no buffer
- states[5]: n_demand output (for Gap to aggregate at P4) - public, no buffer
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
TREE_P_FORSKA_SHADE = 40  # Light response at canopy base (P2->P6 for self-pruning)

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

# === Gap states (for reading from Gap neighbor) ===
GAP_S_DEG_DAYS = 0
GAP_S_DRY_DAYS = 1
GAP_S_FLOOD_DAYS = 8
GAP_S_DRY_DAYS_BASE = 14
# Pre-aggregated cumulative LAI bins (computed at P0)
GAP_S_CUM_DEC_LAI_BASE = 16   # cum_dec_lai[0..49] at slots 16-65
GAP_S_CUM_CON_LAI_BASE = 66   # cum_con_lai[0..49] at slots 66-115

# === Constants ===
PI = 3.14159265359
STD_HT = 1.3
TC_KG = 0.039269908  # PI / 80 - stem volume constant, biomC in kg (cm, m, g/cm³)
XT = -0.40  # GAPpy model.py:280 — universal light extinction coefficient
PLOTSIZE = 500.0  # GAPpy parameters.py:59 — plot area m² (also max trees per plot)

STEM_C_N = 450.0        # GAPpy constants.py:77
CON_LEAF_C_N = 60.0     # GAPpy constants.py:71
DEC_LEAF_C_N = 40.0     # GAPpy constants.py:74
CON_LEAF_B = 1.3        # GAPpy: 1.0 + CON_LEAF_RATIO (0.3)

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
        leafdiam_a = params_tensor[agent_index][TREE_P_LEAFDIAM_A]
        leafarea_c = params_tensor[agent_index][TREE_P_LEAFAREA_C]

        # Get current tree structure from states_db
        diam = states_db_tensor[agent_index][TREE_DB_DIAM]
        height = states_db_tensor[agent_index][TREE_DB_HEIGHT]
        canopy_ht = states_db_tensor[agent_index][TREE_DB_CANOPY_HT]

        # Get internal physiology from params
        biomC = params_tensor[agent_index][TREE_P_BIOMC]
        leaf_bm = params_tensor[agent_index][TREE_P_LEAF_BM]

        # ===== READ CLIMATE + CUMULATIVE LAI FROM GAP NEIGHBOR =====
        deg_days = 2500.0
        dry_days = 0.0
        dry_days_base = 0.0
        flood_days = 0.0
        cum_dec_lai = 0.0
        cum_con_lai = 0.0
        cum_dec_lai_base = 0.0
        cum_con_lai_base = 0.0

        # Height layer indices for cumulative LAI lookup
        # GAPpy: light[h] = exp(-0.40 * cumLAI[h+1] / plotsize), h = int(forht)
        tree_height_layer = int(height)
        if tree_height_layer < 0:
            tree_height_layer = 0
        if tree_height_layer > 49:
            tree_height_layer = 49
        tree_base_layer = int(canopy_ht)
        if tree_base_layer < 0:
            tree_base_layer = 0
        if tree_base_layer > 49:
            tree_base_layer = 49

        neighbor_indices = locations[agent_index]
        i = 0
        while i < len(neighbor_indices) and neighbor_indices[i] != -1:
            neighbor_idx = int(neighbor_indices[i])
            neighbor_breed = int(breeds[neighbor_idx])
            if neighbor_breed == BREED_GAP:
                deg_days = states_tensor[neighbor_idx][GAP_S_DEG_DAYS]
                dry_days = states_tensor[neighbor_idx][GAP_S_DRY_DAYS]
                dry_days_base = states_tensor[neighbor_idx][GAP_S_DRY_DAYS_BASE]
                flood_days = states_tensor[neighbor_idx][GAP_S_FLOOD_DAYS]
                # O(1) cumulative LAI reads (pre-aggregated at P0)
                cum_dec_lai = states_tensor[neighbor_idx][GAP_S_CUM_DEC_LAI_BASE + tree_height_layer]
                cum_con_lai = states_tensor[neighbor_idx][GAP_S_CUM_CON_LAI_BASE + tree_height_layer]
                cum_dec_lai_base = states_tensor[neighbor_idx][GAP_S_CUM_DEC_LAI_BASE + tree_base_layer]
                cum_con_lai_base = states_tensor[neighbor_idx][GAP_S_CUM_CON_LAI_BASE + tree_base_layer]
            i = i + 1

        # Beer-Lambert (GAPpy model.py:347-348): exp(xt * cumLAI / plotsize)
        light_avail = 1.0
        if evergreen > 0:
            light_avail = cp.exp(XT * cum_con_lai / PLOTSIZE)
        else:
            light_avail = cp.exp(XT * cum_dec_lai / PLOTSIZE)
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

        # Drought response (GAPpy fdry + dual-metric for drought_tol==1)
        fc_drought = 1.0
        drought_idx = drought_tol - 1
        if drought_idx < 0:
            drought_idx = 0
        if drought_idx > 5:
            drought_idx = 5

        # GAPpy gama = [0.50, 0.45, 0.35, 0.25, 0.15, 0.05] (species.py:219)
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
        elif drought_idx == 5:
            gamma = 0.05

        # fdry(dry_days, drought_tol): sqrt((gamma - dryday) / gamma)
        if dry_days < gamma:
            tmp_d = (gamma - dry_days) / gamma
            fc_drought = tmp_d ** 0.5
        else:
            fc_drought = 0.0

        # Dual metric for intolerant species (GAPpy species.py:140-151)
        if drought_tol == 1:
            # Base layer response using gamma=0.50 (drought_tol=1)
            fcdry_base = 0.0
            if dry_days_base < 0.50:
                tmp_b = (0.50 - dry_days_base) / 0.50
                fcdry_base = tmp_b ** 0.5
            # Conifer/deciduous multiplier
            if evergreen > 0:
                fcdry_base = fcdry_base * 0.33
            else:
                fcdry_base = fcdry_base * 0.2
            # Take maximum of upper and base responses
            if fcdry_base > fc_drought:
                fc_drought = fcdry_base

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

        # Forska shade: light response at canopy BASE (GAPpy model.py:428-431)
        # Used for self-pruning check in P6 (GAPpy third tree loop, model.py:564-565)
        light_at_base = 1.0
        if evergreen > 0:
            light_at_base = cp.exp(XT * cum_con_lai_base / PLOTSIZE)
        else:
            light_at_base = cp.exp(XT * cum_dec_lai_base / PLOTSIZE)
        if light_at_base < 0.01:
            light_at_base = 0.01
        if light_at_base > 1.0:
            light_at_base = 1.0
        forska_shade = c1 * (1.0 - cp.exp(-c2 * (light_at_base - c3)))
        if forska_shade < 0.0:
            forska_shade = 0.0
        if forska_shade > 1.0:
            forska_shade = 1.0

        # GAPpy flood_rsp always returns 1.0 (dead code, species.py:153-163)
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
        pot_twigbc = TC_KG * wood_bulk_dens * 0.33667 * pot_dc * pot_dc * pot_crown_depth

        # Root biomass
        pot_root_c = 0.0
        if pot_new_height > 0.01:
            pot_root_c = pot_stembc * rootdepth / pot_new_height + pot_twigbc * 0.5

        pot_new_biomC = pot_stembc + pot_twigbc + pot_root_c

        # Potential leaf biomass (GAPpy: dc² * leafdiam_a * leafarea_c * 2.0)
        # * 1000.0 converts GAPpy tonnes to GGap kg (TC_KG = TC * 1000)
        pot_new_leaf_bm = pot_dc * pot_dc * leafdiam_a * leafarea_c * 2.0 * 1000.0

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
        params_tensor[agent_index][TREE_P_FORSKA_SHADE] = forska_shade

    # ===== WRITE N DEMAND TO states (for Gap to aggregate at P4) =====
    # Templates and free slots: n_demand stays 0.0 (initialized above)
    states_tensor[agent_index][TREE_S_N_DEMAND] = n_demand
