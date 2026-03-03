"""
Tree template renewal step function for GGap model (Priority 6).
Computes per-species seedbank/seedling dynamics using current-tick climate.

Runs AFTER gap_demand_aggregate_step (P4) computes per-gap N supply ratio,
so templates read current-tick deg_days, dry_days, n_supply_ratio, etc.

P7 (gap_recruit_aggregate_step) reads template regrowth from params
(same tick, no double buffer) to compute growmax and num_to_recruit.

P8 (tree_actual_growth_step) free slots read seedling_weight from params
(same tick, no double buffer) for species selection.

Execution Flow:
    1. Read current-tick climate + disturbance from Gap (written by P2/P4)
    2. Compute fc_degday, fc_drought, fc_flood, fc_nutrient, fc_light
    3. Compute regrowth = product of factors (if avail_n > 0)
    4. Detect avail_spec (mature trees of same species)
    5. Seedbank/seedling pipeline (fire reset / wind accumulate / normal)
    6. Seedling decrement (proportional allocation, skipped during fire/wind)
    7. Annual survival (seedling *= seedling_lg, inside avail_N gate)
    8. Write regrowth to params (P7 reads same tick for growmax)
    9. Write seedling_weight to params (P8 free slots read same tick)

Property scheme:
- params[42]: reads species traits, writes seedbank/seedling/regrowth/weight
- states[5]: not used by templates
- states_db[5]: reads is_alive, reads old seedling_weight for decrement
"""

import cupy as cp
from cupyx import jit

# === Breed IDs ===
BREED_TREE = 0
BREED_GAP = 1
BREED_SITE = 2

# === Tree params[42] (private, no buffer) ===
TREE_P_SPECIES_ID = 0
TREE_P_SHADE_TOL = 6
TREE_P_DEG_DAY_MIN = 7
TREE_P_DEG_DAY_OPT = 8
TREE_P_DEG_DAY_MAX = 9
TREE_P_INVADER = 10
TREE_P_SEED = 11
TREE_P_SPROUT = 12
TREE_P_LOWNUTR_TOL = 14
TREE_P_FLOOD_TOL = 15
TREE_P_DROUGHT_TOL = 16
TREE_P_EVERGREEN = 17
TREE_P_FIRE_TOL = 18
TREE_P_ENV_STRESS = 32       # Regrowth output (P7 reads same tick for growmax)
TREE_P_SEED_SURV = 34
TREE_P_SEEDLING_LG = 35
TREE_P_SEEDBANK = 36
TREE_P_SEEDLING = 37
TREE_P_SEEDLING_WEIGHT = 41  # Non-buffered seedling weight (P8 free slots read same tick)

# === Tree states_db[5] (public, double buffered) ===
TREE_DB_IS_ALIVE = 0
TREE_DB_SEEDLING_WEIGHT = 4  # Previous-tick weight (read buffer, for decrement calculation)

# === Gap states (for reading from Gap neighbor) ===
GAP_S_DEG_DAYS = 0
GAP_S_DRY_DAYS = 1
GAP_S_AVAIL_N = 2
GAP_S_N_SUPPLY_RATIO = 3
GAP_S_NUM_TO_RECRUIT = 6
GAP_S_FLOOD_DAYS = 8
GAP_S_TOTAL_SEEDLING_WEIGHT = 9
GAP_S_FIRE_INTENSITY = 10
GAP_S_DRY_DAYS_BASE = 14
GAP_S_WIND_INTENSITY = 15
# Pre-aggregated cumulative LAI bins + avail_spec (computed at P0)
GAP_S_CUM_DEC_LAI_BASE = 16   # cum_dec_lai[0..49] at slots 16-65
GAP_S_CUM_CON_LAI_BASE = 66   # cum_con_lai[0..49] at slots 66-115
GAP_S_AVAIL_SPEC_BASE = 116   # avail_spec[0..49] at slots 116-165

# === Constants ===
XT = -0.40
PLOTSIZE = 500.0  # GAPpy parameters.py:59 — plot area m² (also max trees per plot)


@jit.rawkernel(device="cuda")
def tree_template_renewal_step(
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
    Template renewal step (priority 6).

    Computes per-species seedbank/seedling dynamics using current-tick climate
    (relayed by P2/P4). Writes regrowth to params for P7 growmax aggregation,
    and seedling_weight to params for P8 free slot species selection.
    Only processes templates (is_alive < -0.5); living/free trees skip.
    """
    is_alive = states_db_tensor[agent_index][TREE_DB_IS_ALIVE]

    if is_alive < -0.5:
        # --- 1. Read climate + disturbance + recruitment + LAI + avail_spec from Gap ---
        deg_days = 2500.0
        dry_days = 0.0
        dry_days_base = 0.0
        flood_days = 0.0
        n_supply_ratio = 1.0
        avail_n = 0.0
        fire_intensity = 0.0
        wind_intensity = 0.0
        num_to_recruit = 0.0
        total_seedling_weight = 0.0
        cum_dec_lai = 0.0
        cum_con_lai = 0.0
        gap_idx = -1

        neighbor_indices = locations[agent_index]
        i = 0
        while i < len(neighbor_indices) and neighbor_indices[i] != -1:
            neighbor_idx = int(neighbor_indices[i])
            neighbor_breed = int(breeds[neighbor_idx])
            if neighbor_breed == BREED_GAP:
                gap_idx = neighbor_idx
                deg_days = states_tensor[neighbor_idx][GAP_S_DEG_DAYS]
                dry_days = states_tensor[neighbor_idx][GAP_S_DRY_DAYS]
                dry_days_base = states_tensor[neighbor_idx][GAP_S_DRY_DAYS_BASE]
                flood_days = states_tensor[neighbor_idx][GAP_S_FLOOD_DAYS]
                n_supply_ratio = states_tensor[neighbor_idx][GAP_S_N_SUPPLY_RATIO]
                avail_n = states_tensor[neighbor_idx][GAP_S_AVAIL_N]
                fire_intensity = states_tensor[neighbor_idx][GAP_S_FIRE_INTENSITY]
                wind_intensity = states_tensor[neighbor_idx][GAP_S_WIND_INTENSITY]
                # For seedling decrement (GAPpy model.py:941)
                num_to_recruit = states_tensor[neighbor_idx][GAP_S_NUM_TO_RECRUIT]
                total_seedling_weight = states_tensor[neighbor_idx][GAP_S_TOTAL_SEEDLING_WEIGHT]
                # O(1) ground-level cumulative LAI (layer 0 = total, pre-aggregated at P0)
                cum_dec_lai = states_tensor[neighbor_idx][GAP_S_CUM_DEC_LAI_BASE]
                cum_con_lai = states_tensor[neighbor_idx][GAP_S_CUM_CON_LAI_BASE]
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
        fire_tol = int(params_tensor[agent_index][TREE_P_FIRE_TOL])

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

        # Drought response (GAPpy fdry + dual-metric for drought_tol==1)
        fc_drought = 1.0
        drought_idx = drought_tol - 1
        if drought_idx < 0:
            drought_idx = 0
        if drought_idx > 5:
            drought_idx = 5
        # GAPpy gama = [0.50, 0.45, 0.35, 0.25, 0.15, 0.05] (species.py:219)
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
        elif drought_idx == 5:
            gamma_d = 0.05
        if dry_days < gamma_d:
            tmp_d = (gamma_d - dry_days) / gamma_d
            fc_drought = tmp_d ** 0.5
        else:
            fc_drought = 0.0
        if drought_tol == 1:
            fcdry_base = 0.0
            if dry_days_base < 0.50:
                tmp_b = (0.50 - dry_days_base) / 0.50
                fcdry_base = tmp_b ** 0.5
            if evergreen > 0:
                fcdry_base = fcdry_base * 0.33
            else:
                fcdry_base = fcdry_base * 0.2
            if fcdry_base > fc_drought:
                fc_drought = fcdry_base

        # GAPpy flood_rsp always returns 1.0 (dead code, species.py:153-163)
        fc_flood = 1.0

        # Nutrient response (GAPpy poor_soil_rsp quadratic)
        lownutr_idx = lownutr_tol
        if lownutr_idx < 1:
            lownutr_idx = 1
        if lownutr_idx > 3:
            lownutr_idx = 3
        nrc = 4 - lownutr_idx
        nutr_c1 = -0.6274
        nutr_c2 = 3.600
        nutr_c3 = -1.994
        if nrc == 2:
            nutr_c1 = -0.2352
            nutr_c2 = 2.771
            nutr_c3 = -1.550
        elif nrc == 3:
            nutr_c1 = 0.2133
            nutr_c2 = 1.789
            nutr_c3 = -1.014
        sf = n_supply_ratio
        if sf < 0.0:
            sf = 0.0
        if sf > 1.0:
            sf = 1.0
        fpoor = nutr_c1 + nutr_c2 * sf + nutr_c3 * sf * sf
        if fpoor < 0.0:
            fpoor = 0.0
        if fpoor > 1.0:
            fpoor = 1.0
        fc_nutrient = fpoor * sf

        # Light at ground level: O(1) read from Gap (cum_dec_lai/cum_con_lai already read above)
        light_avail = 1.0
        if evergreen > 0:
            light_avail = cp.exp(XT * cum_con_lai / PLOTSIZE)
        else:
            light_avail = cp.exp(XT * cum_dec_lai / PLOTSIZE)
        if light_avail < 0.01:
            light_avail = 0.01
        if light_avail > 1.0:
            light_avail = 1.0

        # Light tolerance response
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

        # --- 3. Compute regrowth (GAPpy model.py:804-816) ---
        raw_regrowth = 0.0
        regrowth = 0.0
        if avail_n > 0.0:
            raw_regrowth = fc_degday * fc_drought * fc_flood * fc_nutrient * fc_light
            regrowth = raw_regrowth
            if regrowth <= 0.05:
                regrowth = 0.0

        # --- 4. Read avail_spec from Gap (pre-aggregated at P0) ---
        avail_spec = 0.0
        species_idx = int(species_id)
        if gap_idx >= 0 and species_idx >= 0 and species_idx < 50:
            avail_spec = states_tensor[gap_idx][GAP_S_AVAIL_SPEC_BASE + species_idx]

        # --- 5. Seedbank/seedling pipeline ---
        # Fire/wind: seedling reset/accumulate happens in GAPpy's mortality(),
        # BEFORE renewal(), OUTSIDE avail_N gate. No recruitment in disturbance year.
        # Normal: pipeline runs in GAPpy's renewal(), inside avail_N > 0 gate.
        weight = 0.0

        if fire_intensity > 0.01:
            # FIRE: Reset seedlings (GAPpy mortality() model.py:635-640).
            # fc_fire = gama[fire_tol-1] applied to reset formula.
            fc_fire = 1.0
            ft = fire_tol
            if ft < 1:
                ft = 1
            if ft > 6:
                ft = 6
            if ft == 1:
                fc_fire = 100.0
            elif ft == 2:
                fc_fire = 10.0
            elif ft == 3:
                fc_fire = 1.0
            elif ft == 4:
                fc_fire = 0.1
            elif ft == 5:
                fc_fire = 0.01
            elif ft == 6:
                fc_fire = 0.001
            seedling = (invader_val * 10.0 + sprout_val * avail_spec) * fc_fire
            # Seedbank untouched. Weight=0 (no recruitment during fire year).
            # GAPpy: can_recruit=False during fire, line 985 resets fire=0,
            # so recruitment resumes next tick via normal pipeline.

        elif wind_intensity > 0.01:
            # WIND: Accumulate seedlings (GAPpy mortality() model.py:648-653).
            # Wind keeps existing seedling pool + adds invader + sprouting.
            seedling = invader_val + seedling + sprout_val * avail_spec
            # Seedbank untouched. Weight=0 (no recruitment during wind year).

        elif avail_n > 0.0:
            # Normal renewal pipeline (GAPpy renewal(), inside avail_N > 0 gate).
            # GAPpy first cycle (seedling_number==0, model.py:828-858):
            #   pipeline first, then prob from UPDATED seedling
            # GAPpy subsequent (seedling_number!=0, model.py:860-891):
            #   prob from CURRENT seedling first, then pipeline update
            # (tick>0 is GGap proxy for seedling_number!=0)
            if tick > 0:
                weight = seedling * regrowth

            convert_threshold = 0.05
            if tick > 0:
                convert_threshold = 0.01
            seedbank = seedbank + invader_val + seed_val * avail_spec + sprout_val * avail_spec
            if raw_regrowth >= convert_threshold:
                seedling = seedling + seedbank
                seedbank = 0.0
            else:
                seedbank = seedbank * seed_surv
            seedling = seedling + sprout_val * avail_spec

            if tick < 1:
                weight = seedling * regrowth

        # avail_n <= 0 and no disturbance: seedling/seedbank frozen (unchanged)

        # --- 7. Seedling decrement ---
        # Corrects 1-tick lag from previous tick's recruitment.
        # Skipped during fire/wind: seedling reset/accumulate replaces prior state.
        if fire_intensity < 0.01 and wind_intensity < 0.01:
            old_weight = states_db_tensor[agent_index][TREE_DB_SEEDLING_WEIGHT]
            if total_seedling_weight > 0.01 and num_to_recruit > 0.5:
                my_share = num_to_recruit * old_weight / total_seedling_weight
                seedling = seedling - my_share / PLOTSIZE
                if seedling < 0.0:
                    seedling = 0.0

        # --- 7b. Annual survival ---
        # Only when avail_n > 0 (GAPpy model.py:981 is inside the renewal gate)
        if avail_n > 0.0:
            seedling = seedling * seedling_lg

        # --- 8. Write outputs ---
        params_tensor[agent_index][TREE_P_SEEDBANK] = seedbank
        params_tensor[agent_index][TREE_P_SEEDLING] = seedling
        params_tensor[agent_index][TREE_P_ENV_STRESS] = regrowth         # P7 reads same tick
        params_tensor[agent_index][TREE_P_SEEDLING_WEIGHT] = weight      # P8 free slots read same tick
        states_db_tensor[agent_index][TREE_DB_SEEDLING_WEIGHT] = weight  # P0 reads next tick via read buffer
