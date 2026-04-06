"""
Tree template renewal step function for GGap model (Priority 5).
Computes per-species seedbank/seedling dynamics using current-tick climate.

Runs AFTER gap_demand_aggregate_step (P4) computes per-gap N supply ratio,
so templates read current-tick deg_days, dry_days, n_supply_ratio, etc.

P6 (gap_recruit_aggregate_step) reads template regrowth from states[ENV_STRESS]
(same tick) to compute growmax and num_to_recruit.

P7 (tree_actual_growth_step) free slots read seedling_weight from states[SEEDLING_WEIGHT]
(same tick) for species selection.

Execution Flow:
    1. Read current-tick climate + disturbance + recovery_years from Gap (P2/P4)
    2. Compute fc_degday, fc_drought, fc_flood, fc_nutrient, fc_light
    3. Compute regrowth = product of factors (if avail_n > 0)
    4. Detect avail_spec (mature trees of same species)
    5. Seedbank/seedling pipeline (fire / wind / recovery / post-disturbance / normal)
    6. PLOTSIZE scaling + seedling decrement (D1/D2: matches GAPpy plotsize pattern)
    7. Annual survival: seedling = seedling * seedling_lg / PLOTSIZE (D5: always runs)
    8. Write regrowth to states[ENV_STRESS] (P6 reads same tick for growmax)
    9. Write seedling_weight to states[SEEDLING_WEIGHT] (P7 free slots read same tick)

Property scheme:
- params[14]: mutable seedbank/seedling
- states[11]: reads is_alive, species_id, old seedling_weight; writes env_stress, seedling_weight
- species traits read from species_traits[species_id][trait]
"""

import cupy as cp
from cupyx import jit

from gap.constants import (
    Breed, Trait, TreeP, TreeS, GapS,
    XT, PLOTSIZE,
)


@jit.rawkernel(device="cuda")
def tree_template_renewal_step(
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
    Template renewal step (priority 5).

    Computes per-species seedbank/seedling dynamics using current-tick climate
    (relayed by P2/P4). Writes regrowth to states[ENV_STRESS] for P6 growmax
    aggregation, and seedling_weight to states[SEEDLING_WEIGHT] for P7 free
    slot species selection.
    Only processes templates (is_alive < -0.5); living/free trees skip.
    """
    is_alive = states_tensor[agent_index][TreeS.IS_ALIVE]

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
        recovery_years = 0.0
        gap_idx = -1

        neighbor_indices = locations[agent_index]
        i = 0
        while i < len(neighbor_indices) and neighbor_indices[i] != -1:
            neighbor_idx = int(neighbor_indices[i])
            neighbor_breed = int(breeds[neighbor_idx])
            if neighbor_breed == Breed.GAP:
                gap_idx = neighbor_idx
                deg_days = states_tensor[neighbor_idx][GapS.DEG_DAYS]
                dry_days = states_tensor[neighbor_idx][GapS.DRY_DAYS]
                dry_days_base = states_tensor[neighbor_idx][GapS.DRY_DAYS_BASE]
                flood_days = states_tensor[neighbor_idx][GapS.FLOOD_DAYS]
                n_supply_ratio = states_tensor[neighbor_idx][GapS.N_SUPPLY_RATIO]
                avail_n = states_tensor[neighbor_idx][GapS.AVAIL_N]
                fire_intensity = states_tensor[neighbor_idx][GapS.FIRE_INTENSITY]
                wind_intensity = states_tensor[neighbor_idx][GapS.WIND_INTENSITY]
                recovery_years = states_tensor[neighbor_idx][GapS.RECOVERY_YEARS]
                # For seedling decrement (GAPpy model.py:941)
                num_to_recruit = states_tensor[neighbor_idx][GapS.NUM_TO_RECRUIT]
                total_seedling_weight = states_tensor[neighbor_idx][GapS.TOTAL_SEEDLING_WEIGHT]
                # D4 fix: ground-level light reads layer 1 (matches GAPpy dec_light[0]
                # = exp(xt * lvd_c3[1] / plotsize) which uses cumulative LAI from
                # layer 1+, excluding the ground layer itself)
                cum_dec_lai = gap_lai[gap_lai_idx[neighbor_idx]][1][0]
                cum_con_lai = gap_lai[gap_lai_idx[neighbor_idx]][1][1]
            i = i + 1

        # Read species_id from states, then look up traits from species_traits tensor
        species_id = states_tensor[agent_index][TreeS.SPECIES_ID]

        shade_tol = int(species_traits[int(species_id)][Trait.SHADE_TOL])
        deg_day_min = species_traits[int(species_id)][Trait.DEG_DAY_MIN]
        deg_day_opt = species_traits[int(species_id)][Trait.DEG_DAY_OPT]
        deg_day_max = species_traits[int(species_id)][Trait.DEG_DAY_MAX]
        invader_val = species_traits[int(species_id)][Trait.INVADER]
        seed_val = species_traits[int(species_id)][Trait.SEED]
        sprout_val = species_traits[int(species_id)][Trait.SPROUT]
        lownutr_tol = int(species_traits[int(species_id)][Trait.LOWNUTR_TOL])
        flood_tol = int(species_traits[int(species_id)][Trait.FLOOD_TOL])
        drought_tol = int(species_traits[int(species_id)][Trait.DROUGHT_TOL])
        evergreen = int(species_traits[int(species_id)][Trait.EVERGREEN])
        fire_tol = int(species_traits[int(species_id)][Trait.FIRE_TOL])
        seed_surv = species_traits[int(species_id)][Trait.SEED_SURV]
        seedling_lg = species_traits[int(species_id)][Trait.SEEDLING_LG]

        # Read mutable renewal state from params
        seedbank = params_tensor[agent_index][TreeP.SEEDBANK]
        seedling = params_tensor[agent_index][TreeP.SEEDLING]

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

        # --- 4. Read avail_spec and imported_seeds from Gap ---
        num_species = len(species_traits)
        avail_spec = 0.0
        imported_seeds = 0.0
        species_idx = int(species_id)
        if gap_idx >= 0 and species_idx >= 0 and species_idx < num_species:
            avail_spec = gap_avail_spec[gap_avail_spec_idx[gap_idx]][species_idx]
            imported_seeds = gap_imported_seeds[gap_imported_seeds_idx[gap_idx]][species_idx]

        # --- 5. Seedbank/seedling pipeline ---
        # Branches: fire / wind / recovery(>1) / post-disturbance(==1) / normal / frozen
        # in_pipeline tracks whether PLOTSIZE scaling should happen (matches GAPpy
        # pattern: scale up by plotsize during active renewal, scale down at end).
        # Recovery years (counter>1) intentionally skip scaling → seedling_lg/PLOTSIZE
        # causes rapid depletion (matching GAPpy's behavior when fire/wind > 1).
        weight = 0.0
        in_pipeline = 0

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
            # Seedbank untouched. Weight=0, in_pipeline=0 (no scaling, no recruitment).
            # GAPpy: can_recruit=False during fire year (counter=5>1), no plotsize scaling.

        elif wind_intensity > 0.01:
            # WIND: Accumulate seedlings (GAPpy mortality() model.py:648-653).
            seedling = invader_val + seedling + sprout_val * avail_spec
            # Weight=0, in_pipeline=0 (no scaling, no recruitment).

        elif recovery_years > 1.5:
            # Recovery countdown (GAPpy: fire>1 or wind>1, just decrement counter).
            # No pipeline, no scaling. seedling_lg/PLOTSIZE below causes rapid depletion
            # (matching GAPpy: seedling never scaled up, but divided by plotsize each year).
            regrowth = 0.0
            # weight stays 0, in_pipeline stays 0

        elif recovery_years > 0.5:
            # Post-disturbance year (GAPpy: fire==1 or wind==1, model.py:893-911).
            # GAPpy: prob = seedling * grow_cap (no nutrient/light), nrenew=0.
            # Scale seedlings by PLOTSIZE so seedling_lg/PLOTSIZE gives net seedling_lg.
            # P6 sets num_to_recruit=0 during recovery, so no actual recruitment.
            regrowth = 0.0
            in_pipeline = 1

        else:
            # Normal renewal pipeline (GAPpy renewal(), can_recruit=True).
            # Runs regardless of avail_n (GAPpy: can_recruit depends on numtrees
            # and fire/wind, not avail_n). When avail_n<=0, regrowth=0 so
            # weight=0 (no recruitment) and seedbank doesn't convert, but
            # seedbank still accumulates and PLOTSIZE scaling still applies.
            if tick > 0:
                weight = seedling * regrowth

            convert_threshold = 0.05
            if tick > 0:
                convert_threshold = 0.01
            seedbank = seedbank + invader_val + seed_val * avail_spec + sprout_val * avail_spec + imported_seeds
            if raw_regrowth >= convert_threshold:
                seedling = seedling + seedbank
                seedbank = 0.0
            else:
                seedbank = seedbank * seed_surv
            seedling = seedling + sprout_val * avail_spec

            if tick < 1:
                weight = seedling * regrowth

            in_pipeline = 1

        # --- 6. PLOTSIZE scaling + seedling decrement (D1, D2) ---
        # GAPpy scales seedlings up by plotsize during active renewal, making the
        # per-recruit decrement of 1.0 meaningful against a pool of ~500.
        if in_pipeline > 0:
            seedling = seedling * PLOTSIZE

            # Seedling decrement: corrects 1-tick lag from previous tick's recruitment.
            # my_share ≈ num_recruits × (species_weight / total_weight) = recruits for this species.
            # With PLOTSIZE-scaled seedling, each recruit decrements ~1.0 (matches GAPpy model.py:941).
            if fire_intensity < 0.01 and wind_intensity < 0.01:
                old_weight = gap_seedling_weights[gap_seedling_weights_idx[gap_idx]][int(species_id)]
                if total_seedling_weight > 0.01 and num_to_recruit > 0.5:
                    my_share = num_to_recruit * old_weight / total_seedling_weight
                    seedling = seedling - my_share
                    if seedling < 0.0:
                        seedling = 0.0

        # --- 7. Annual survival (D5: always runs, matches GAPpy model.py:980-982) ---
        # GAPpy: seedling[i] = seedling[i] * seedling_lg / plotsize (unconditional).
        # When in_pipeline=1: net effect = seedling * seedling_lg (plotsize cancels).
        # When in_pipeline=0: net effect = seedling * seedling_lg / plotsize (rapid depletion
        # during recovery years, matching GAPpy's no-scaling + divide pattern).
        seedling = seedling * seedling_lg / PLOTSIZE

        # --- 8. Write outputs ---
        params_tensor[agent_index][TreeP.SEEDBANK] = seedbank
        params_tensor[agent_index][TreeP.SEEDLING] = seedling
        states_tensor[agent_index][TreeS.ENV_STRESS] = regrowth         # P6 reads same tick
        gap_seedling_weights[gap_seedling_weights_idx[gap_idx]][int(species_id)] = weight  # P7 free slots read same tick
