"""
Tree actual growth step function for GGap model (Priority 7).
Handles living tree growth + mortality + litter and free slot activation.

Species traits are now read from species_traits tensor instead of params_tensor.

Execution Flow:
    1. FINAL GROWTH (living trees, is_alive > 0.5):
       - Read species_id from states[SPECIES_ID], look up traits in species_traits
       - Read env_stress from states[ENV_STRESS] and diam_max from params (written at P2, same tick)
       - Read n_supply_ratio from Gap neighbor states (written at P4, same tick)
       - Compute fc_nutrient from n_supply_ratio + lownutr_tol
       - Compute growth_factor = env_stress * fc_nutrient
       - Compute final diam_increment = diam_max * growth_factor
       - Update diameter, height, biomass
       - Canopy self-pruning, mortality, litter output

    2. DORMANT ACTIVATION (free slots, is_alive == 0):
       - Read recruit_prob from Gap (written at P6, same tick)
       - Select species from templates weighted by seedling_weight
       - Copy SPECIES_ID to states, read traits from species_traits for initialization
       - Set is_alive = 1.0

    3. TEMPLATES (is_alive < -0.5): skipped

Property scheme:
- params[14]: mutable state
- states[11]: is_alive, diam, height, canopy_ht, seedling_weight, litter, n_demand/consumed, species_id, env_stress
- species_traits[species_id][trait]: 2D species traits tensor
"""

import cupy as cp
from cupyx import jit
from sagesim.math_utils import rand_uniform_philox, rand_normal_bounded

from gap.constants import (
    Breed, Trait, TreeP, TreeS, GapS,
    STD_HT, TC_KG, SEEDLING_AGE,
    STEM_C_N, CON_LEAF_C_N, DEC_LEAF_C_N, CON_LEAF_B,
)


@jit.rawkernel(device="cuda")
def tree_actual_growth_step(
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
    Phase C (P7): Nutrient response + final growth + mortality + litter + free slot activation.

    Living trees: reads env_stress/diam_max from P2, n_supply_ratio from P4.
    Free slots: reads num_to_recruit from P6, seedling_weight from P5 templates.
    Templates: skipped.
    """
    # ===== GET CURRENT STATE =====
    is_alive = states_tensor[agent_index][TreeS.IS_ALIVE]

    # Initialize outputs (above-ground litter + N consumed for balance)
    litter_c = 0.0
    litter_n = 0.0
    n_consumed = 0.0  # Actual N consumed this tick (GAPpy model.py:539-552, 962-977)

    # ===== ACTUAL GROWTH + MORTALITY: Process living trees only =====
    if is_alive > 0.5:
        # Get species ID
        species_id = int(states_tensor[agent_index][TreeS.SPECIES_ID])

        # Get species parameters from species_traits tensor
        max_age = species_traits[species_id][Trait.MAX_AGE]
        max_diam = species_traits[species_id][Trait.MAX_DIAM]
        max_ht = species_traits[species_id][Trait.MAX_HT]
        arfa_0 = species_traits[species_id][Trait.ARFA_0]
        wood_bulk_dens = species_traits[species_id][Trait.WOOD_BULK_DENS]
        lownutr_tol = int(species_traits[species_id][Trait.LOWNUTR_TOL])
        evergreen = int(species_traits[species_id][Trait.EVERGREEN])
        rootdepth = species_traits[species_id][Trait.ROOTDEPTH]
        leafdiam_a = species_traits[species_id][Trait.LEAFDIAM_A]
        leafarea_c = species_traits[species_id][Trait.LEAFAREA_C]

        # Get current tree structure from states
        diam = states_tensor[agent_index][TreeS.DIAM]
        height = states_tensor[agent_index][TreeS.HEIGHT]
        canopy_ht = states_tensor[agent_index][TreeS.CANOPY_HT]

        # Get internal physiology from params
        age = params_tensor[agent_index][TreeP.AGE]

        # Read intermediates from P2 (same tick)
        env_stress = states_tensor[agent_index][TreeS.ENV_STRESS]
        diam_max = params_tensor[agent_index][TreeP.DIAM_MAX_CALC]

        # ===== READ FROM GAP NEIGHBOR (written at P4, same tick) =====
        n_supply_ratio = 1.0
        fire_intensity = 0.0
        wind_intensity = 0.0

        neighbor_indices = locations[agent_index]
        i = 0
        while i < len(neighbor_indices) and neighbor_indices[i] != -1:
            neighbor_idx = int(neighbor_indices[i])
            neighbor_breed = int(breeds[neighbor_idx])
            if neighbor_breed == Breed.GAP:
                n_supply_ratio = states_tensor[neighbor_idx][GapS.N_SUPPLY_RATIO]
                fire_intensity = states_tensor[neighbor_idx][GapS.FIRE_INTENSITY]
                wind_intensity = states_tensor[neighbor_idx][GapS.WIND_INTENSITY]
            i = i + 1

        # ===== GROWTH COMPUTATION (GAPpy growth() runs before mortality()) =====
        # All living trees grow first; fire/wind death uses post-growth biomass.

        # Read old biomass for N consumed tracking (GAPpy model.py:467,539)
        old_biomC = params_tensor[agent_index][TreeP.BIOMC]
        old_leaf_bm = params_tensor[agent_index][TreeP.LEAF_BM]

        # ===== COMPUTE NUTRIENT RESPONSE (GAPpy poor_soil_rsp quadratic) =====
        lownutr_idx = lownutr_tol
        if lownutr_idx < 1:
            lownutr_idx = 1
        if lownutr_idx > 3:
            lownutr_idx = 3

        # nrc = 4 - lownutr_tol (invert tolerance class, GAPpy Species.f90:235)
        nrc = 4 - lownutr_idx

        # Coefficient lookup by nrc (1-indexed: nrc=1->idx0, nrc=2->idx1, nrc=3->idx2)
        c1 = -0.6274
        c2 = 3.600
        c3 = -1.994
        if nrc == 2:
            c1 = -0.2352
            c2 = 2.771
            c3 = -1.550
        elif nrc == 3:
            c1 = 0.2133
            c2 = 1.789
            c3 = -1.014

        # Clamp supply ratio to [0, 1]
        sf = n_supply_ratio
        if sf < 0.0:
            sf = 0.0
        if sf > 1.0:
            sf = 1.0

        # Quadratic response
        fpoor = c1 + c2 * sf + c3 * sf * sf
        if fpoor < 0.0:
            fpoor = 0.0
        if fpoor > 1.0:
            fpoor = 1.0

        fc_nutrient = fpoor * sf

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
        growth_twigbc = TC_KG * wood_bulk_dens * 0.33667 * growth_dc * growth_dc * growth_crown_depth

        # Root biomass
        growth_root_c = 0.0
        if new_height > 0.01:
            growth_root_c = growth_stembc * rootdepth / new_height + growth_twigbc * 0.5

        growth_biomC = growth_stembc + growth_twigbc + growth_root_c

        # Leaf biomass from growth (GAPpy: dc^2 * leafdiam_a * leafarea_c * 2.0)
        # * 1000.0 converts GAPpy tonnes to GGap kg (TC_KG = TC * 1000)
        growth_leaf_bm = growth_dc * growth_dc * leafdiam_a * leafarea_c * 2.0 * 1000.0

        # ===== CANOPY HEIGHT SELF-PRUNING (GAPpy third tree loop, model.py:564-571) =====
        # GAPpy: check = fc_degday * fc_drought * fc_flood * forska_shade * nutrient
        # forska_shade = light_rsp at canopy BASE (not tree top), computed at P2
        forska_shade = params_tensor[agent_index][TreeP.FORSKA_SHADE]
        fc_degday_val = params_tensor[agent_index][TreeP.FC_DEGDAY]
        fc_drought_val = params_tensor[agent_index][TreeP.FC_DROUGHT]
        fc_flood_val = params_tensor[agent_index][TreeP.FC_FLOOD]
        forska_check = fc_degday_val * fc_drought_val * fc_flood_val * forska_shade * fc_nutrient
        new_canopy_ht = canopy_ht
        if forska_check <= 0.05:
            # GAPpy uses integer height layers for canopy advancement:
            # khc = int(canopy_ht) - 1 (0-based index), khc += 1,
            # guard: khc + 1 < int(forht), new = float(khc + 1) + 0.01
            canht_int = int(canopy_ht)
            forht_int = int(new_height)
            if canht_int + 1 < forht_int:
                new_canopy_ht = float(canht_int + 1) + 0.01

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
        new_twigbc = TC_KG * wood_bulk_dens * 0.33667 * new_dc * new_dc * new_crown_depth

        # Root biomass with new twig
        new_root_c = 0.0
        if new_height > 0.01:
            new_root_c = new_stembc * rootdepth / new_height + new_twigbc * 0.5

        new_biomC = new_stembc + new_twigbc + new_root_c

        # Leaf biomass (GAPpy: dc^2 * leafdiam_a * leafarea_c * 2.0, tonnes->kg)
        new_leaf_bm = new_dc * new_dc * leafdiam_a * leafarea_c * 2.0 * 1000.0

        # ===== COMPUTE N CONSUMED (GAPpy model.py:539-552) =====
        # GAPpy computes N_used in loop 2 (pre-pruning), so use growth_biomC
        # and growth_leaf_bm (OLD canopy_ht), not post-pruning new_biomC/new_leaf_bm.
        # Pruning litter goes to C_into_A0 separately, not counted as N consumed.
        # Fire-killed trees also consume N (GAPpy: growth() runs before mortality()).
        d_bioC = growth_biomC - old_biomC
        n_consumed = d_bioC / STEM_C_N
        # Leaf N: conifer uses CON_LEAF_B multiplier, deciduous uses full leaf mass
        if evergreen > 0:
            prim_prod = CON_LEAF_B * growth_leaf_bm - old_leaf_bm
            n_consumed = n_consumed + prim_prod / CON_LEAF_C_N
        else:
            n_consumed = n_consumed + growth_leaf_bm / DEC_LEAF_C_N

        # ===== DETERMINE DEATH =====
        tree_dies = 0

        if fire_intensity > 0.01 or wind_intensity > 0.01:
            # Fire/wind kills ALL trees (GAPpy mortality() model.py:620-690)
            tree_dies = 1
        else:
            # ===== CHECK MORTALITY (GAPpy: two independent survival checks) =====

            # Read species-specific mortality tolerances from species_traits
            stress_tol = int(species_traits[species_id][Trait.STRESS_TOL])
            age_tol = int(species_traits[species_id][Trait.AGE_TOL])

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
            age_rand = rand_uniform_philox(tick, agent_index, 1)
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
            stress_rand = rand_uniform_philox(tick, agent_index, 2)
            growth_dies = 0
            if mort_marker > 0.5 and stress_rand < stress_check:
                growth_dies = 1

            # --- Combined: tree survives only if BOTH checks pass ---
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
            # Tree dies - all biomass becomes litter (GAPpy: all to A0 layer)
            is_alive = 0.0
            total_c = 0.0
            total_n = 0.0
            if evergreen > 0:
                total_c = new_biomC + new_leaf_bm * CON_LEAF_B
                total_n = new_biomC / STEM_C_N + new_leaf_bm / CON_LEAF_C_N * CON_LEAF_B
            else:
                total_c = new_biomC + new_leaf_bm
                total_n = new_biomC / STEM_C_N + new_leaf_bm / DEC_LEAF_C_N
            litter_c = total_c
            litter_n = total_n
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
        params_tensor[agent_index][TreeP.AGE] = new_age
        params_tensor[agent_index][TreeP.BIOMC] = new_biomC
        params_tensor[agent_index][TreeP.BIOMN] = new_biomC / STEM_C_N
        params_tensor[agent_index][TreeP.LEAF_BM] = new_leaf_bm

        # ===== WRITE TO states (structure) =====
        states_tensor[agent_index][TreeS.IS_ALIVE] = is_alive
        states_tensor[agent_index][TreeS.DIAM] = new_diam
        states_tensor[agent_index][TreeS.HEIGHT] = new_height
        states_tensor[agent_index][TreeS.CANOPY_HT] = new_canopy_ht

    # ===== DORMANT SLOT ACTIVATION =====
    # Free slots (is_alive == 0) check if Gap signals recruitment, select species
    # from gap_seedling_weights breed-local array, and activate as seedlings.
    # By activating at P7, seedlings don't grow until next tick (matching GAPpy).
    # Templates (is_alive == -1) are handled at P5 and skip this step.
    elif is_alive > -0.5:
        # Read recruitment info from Gap neighbor
        recruit_prob = 0.0
        recruit_rand_seed = 0.0
        gap_idx = -1

        neighbor_indices = locations[agent_index]
        i = 0
        while i < len(neighbor_indices) and neighbor_indices[i] != -1:
            neighbor_idx = int(neighbor_indices[i])
            neighbor_breed = int(breeds[neighbor_idx])
            if neighbor_breed == Breed.GAP:
                gap_idx = neighbor_idx
                recruit_prob = states_tensor[neighbor_idx][GapS.NUM_TO_RECRUIT]
                recruit_rand_seed = states_tensor[neighbor_idx][GapS.RECRUIT_RAND_SEED]
            i = i + 1

        # Determine if this free slot should be recruited
        # recruit_prob = nrenew / free_slots (from P6), so
        # expected activations = free_slots * recruit_prob = nrenew
        if recruit_prob > 0.0 and gap_idx >= 0:
            slot_priority = rand_uniform_philox(tick, agent_index, int(recruit_rand_seed) * 97 + 3)

            if slot_priority < recruit_prob:
                # CDF-based species selection from gap breed-local array
                # (matches GAPpy model.py:917-938).
                num_species = len(species_traits)
                sw_row = gap_seedling_weights_idx[gap_idx]

                # Pass 1: sum total_weight across all species
                total_weight = 0.0
                sp = 0
                while sp < num_species:
                    total_weight = total_weight + gap_seedling_weights[sw_row][sp]
                    sp = sp + 1

                # Pass 2: draw random target, cumulative scan to select species
                selected_species = -1
                if total_weight > 1e-10:
                    rand_target = rand_uniform_philox(tick, agent_index, int(recruit_rand_seed) * 97 + 4) * total_weight
                    cum_weight = 0.0
                    last_species = -1
                    sp = 0
                    while sp < num_species:
                        w = gap_seedling_weights[sw_row][sp]
                        if w > 0.0:
                            last_species = sp
                            cum_weight = cum_weight + w
                            if cum_weight > rand_target and selected_species < 0:
                                selected_species = sp
                        sp = sp + 1
                    # Fallback for floating-point overshoot (GAPpy reselection)
                    if selected_species < 0 and last_species >= 0:
                        selected_species = last_species

                if selected_species >= 0:
                    # Set species_id directly (array index = species_id)
                    states_tensor[agent_index][TreeS.SPECIES_ID] = float(selected_species)

                    # Look up species traits from species_traits tensor
                    sel_species_id = selected_species

                    # Seedling diameter: N(1.5, 1) clamped [0.5, 2.5]
                    # Box-Muller normal, z in [-1, 1] -> diam in [0.5, 2.5]
                    seedling_diam = 1.5 + rand_normal_bounded(tick, agent_index, int(recruit_rand_seed) * 97 + 5, -1.0, 1.0)

                    # Calculate initial height from seedling diameter (Forska equation)
                    max_ht = species_traits[sel_species_id][Trait.MAX_HT]
                    arfa_0 = species_traits[sel_species_id][Trait.ARFA_0]
                    delta_ht = max_ht - STD_HT
                    seedling_ht = STD_HT + delta_ht * (1.0 - cp.exp(-arfa_0 * seedling_diam / delta_ht))

                    # Calculate initial biomass from diameter (GAPpy tree.py biomass_c)
                    # GAPpy model.py:951 sets canopy_ht=1.0 for seedlings (not STD_HT)
                    s_wbd = species_traits[sel_species_id][Trait.WOOD_BULK_DENS]
                    s_rdepth = species_traits[sel_species_id][Trait.ROOTDEPTH]
                    s_canopy_ht = 1.0  # GAPpy model.py:951
                    # Basal diameter (canopy_ht=0 internally, GAPpy tree.py:151)
                    s_bd = seedling_diam
                    if seedling_ht > STD_HT:
                        s_bd = seedling_ht / (seedling_ht - STD_HT) * seedling_diam
                    # Canopy diameter (canopy_ht=1.0, GAPpy tree.py:127 stem_shape)
                    s_dc = seedling_diam
                    if seedling_ht > s_canopy_ht and seedling_ht > STD_HT:
                        s_dc = (seedling_ht - s_canopy_ht) / (seedling_ht - STD_HT) * seedling_diam
                    # Stem biomass
                    s_stembc = TC_KG * s_wbd * 0.3 * s_bd * s_bd * seedling_ht
                    # Crown depth and twig biomass (crown from canopy_ht=1.0)
                    s_crown = seedling_ht - s_canopy_ht
                    if s_crown < 0.0:
                        s_crown = 0.0
                    s_twigbc = TC_KG * s_wbd * 0.33667 * s_dc * s_dc * s_crown
                    # Root biomass
                    s_rootc = 0.0
                    if seedling_ht > 0.01:
                        s_rootc = s_stembc * s_rdepth / seedling_ht + s_twigbc * 0.5
                    seedling_biomC = s_stembc + s_twigbc + s_rootc

                    # Seedling N consumed (GAPpy model.py:962-977)
                    # Note: conifer uses leaf_bm/C_N (NOT * CON_LEAF_B, unlike growth)
                    seedling_lda = species_traits[sel_species_id][Trait.LEAFDIAM_A]
                    seedling_lca = species_traits[sel_species_id][Trait.LEAFAREA_C]
                    seedling_leaf_bm = s_dc * s_dc * seedling_lda * seedling_lca * 2.0 * 1000.0
                    seedling_evergreen = int(species_traits[sel_species_id][Trait.EVERGREEN])
                    if seedling_evergreen > 0:
                        n_consumed = seedling_leaf_bm / CON_LEAF_C_N + seedling_biomC / STEM_C_N
                        # Seedling litter (GAPpy model.py:970-971): leaf_bm * (leaf_b - 1.0)
                        litter_c = seedling_leaf_bm * (CON_LEAF_B - 1.0)
                        litter_n = litter_c / CON_LEAF_C_N
                    else:
                        n_consumed = seedling_biomC / STEM_C_N + seedling_leaf_bm / DEC_LEAF_C_N
                        # Seedling litter (GAPpy model.py:976-977): full leaf_bm
                        litter_c = seedling_leaf_bm
                        litter_n = seedling_leaf_bm / DEC_LEAF_C_N

                    # Initialize physiology as seedling
                    params_tensor[agent_index][TreeP.AGE] = SEEDLING_AGE
                    params_tensor[agent_index][TreeP.BIOMC] = seedling_biomC
                    params_tensor[agent_index][TreeP.BIOMN] = seedling_biomC / STEM_C_N
                    # Seedling leaf biomass: set to 0.0 so first-tick P2 conifer N-demand
                    # computes (1.3*new_leaf - 0.0)/60, matching GAPpy bleaf[it]=0.0 init
                    params_tensor[agent_index][TreeP.LEAF_BM] = 0.0
                    params_tensor[agent_index][TreeP.LIGHT_AVAIL] = 1.0
                    params_tensor[agent_index][TreeP.FC_DEGDAY] = 1.0
                    params_tensor[agent_index][TreeP.FC_DROUGHT] = 1.0
                    params_tensor[agent_index][TreeP.FC_FLOOD] = 1.0

                    # Set structure for seedling (canopy_ht=1.0, GAPpy model.py:951)
                    states_tensor[agent_index][TreeS.DIAM] = seedling_diam
                    states_tensor[agent_index][TreeS.HEIGHT] = seedling_ht
                    states_tensor[agent_index][TreeS.CANOPY_HT] = s_canopy_ht

                    # Activate the seedling
                    states_tensor[agent_index][TreeS.IS_ALIVE] = 1.0

    # ===== WRITE LITTER + N CONSUMED TO states (Gap aggregates at P0 next tick) =====
    states_tensor[agent_index][TreeS.LITTER_C] = litter_c          # Above-ground -> A0 layer
    states_tensor[agent_index][TreeS.LITTER_N] = litter_n
    states_tensor[agent_index][TreeS.N_CONSUMED] = n_consumed      # P8 aggregates -> P9 N balance
