# Auto-generated fused GPU kernel with grid barriers
# All priorities and ticks in a single kernel launch

from sagesim.jit_extensions import install_jit_extensions
install_jit_extensions()
from gap_litter_aggregate_step import *
from soil_step import *
from gap_climate_relay_step import *
from tree_potential_growth_step import *
from gap_demand_aggregate_step import *
from tree_template_renewal_step import *
from gap_recruit_aggregate_step import *
from tree_actual_growth_step import *
from gap_nconsumed_aggregate_step import *
from site_final_step import *

# Modified step functions with double buffering
@jit.rawkernel(device='cuda')
def gap_litter_aggregate_step_double_buffer(tick, agent_index, _seed, species_traits, site_configs, rangelists, site_distances, agent_ids, logical_ids, breeds, neighbor_offsets, neighbor_values, params_tensor, states_tensor, gap_lai, gap_lai_idx, gap_avail_spec, gap_avail_spec_idx, gap_imported_seeds, gap_imported_seeds_idx, site_avail_spec, write_site_avail_spec, site_avail_spec_idx, site_imported_seeds, site_imported_seeds_idx):
    """
    Gap litter aggregate step (priority 0).

    Aggregates litter from tree neighbors (written at P7 of previous tick).
    Bins LAI by height layer and computes top-down cumulative sums.
    Sets avail_spec flags for species with mature trees.
    Species traits (EVERGREEN, LEAFDIAM_A, MAX_DIAM) read from globals.
    """
    total_litter_c = 0.0
    total_litter_n = 0.0
    total_seedling_weight = 0.0
    total_lai = 0.0
    lai_row = gap_lai_idx[agent_index]
    sp_row = gap_avail_spec_idx[agent_index]
    for k in range(MAX_HEIGHT_BINS):
        gap_lai[lai_row][k][0] = 0.0
        gap_lai[lai_row][k][1] = 0.0
    sp = 0
    while sp < len(species_traits):
        gap_avail_spec[sp_row][sp] = 0.0
        sp = sp + 1
    i = 0
    while i < neighbor_offsets[agent_index + 1] - neighbor_offsets[agent_index]:
        neighbor_idx = int(neighbor_values[neighbor_offsets[agent_index] + i])
        neighbor_breed = int(breeds[neighbor_idx])
        if neighbor_breed == Breed.TREE:
            tree_alive = states_tensor[neighbor_idx][TreeS.IS_ALIVE]
            if tree_alive > -0.5:
                tree_litter_c = states_tensor[neighbor_idx][TreeS.LITTER_C]
                tree_litter_n = states_tensor[neighbor_idx][TreeS.LITTER_N]
                total_litter_c = total_litter_c + tree_litter_c
                total_litter_n = total_litter_n + tree_litter_n
            if tree_alive > 0.5:
                n_diam = states_tensor[neighbor_idx][TreeS.DIAM]
                n_height = states_tensor[neighbor_idx][TreeS.HEIGHT]
                n_canopy_ht = states_tensor[neighbor_idx][TreeS.CANOPY_HT]
                n_species_id = int(states_tensor[neighbor_idx][TreeS.SPECIES_ID])
                n_leafdiam_a = species_traits[int(n_species_id)][Trait.LEAFDIAM_A]
                n_evergreen = int(species_traits[int(n_species_id)][Trait.EVERGREEN])
                n_max_diam = species_traits[int(n_species_id)][Trait.MAX_DIAM]
                n_dc = n_diam
                if n_height > n_canopy_ht and n_height > STD_HT:
                    n_dc = (n_height - n_canopy_ht) / (n_height - STD_HT) * n_diam
                n_lai = n_dc * n_dc * n_leafdiam_a
                if n_height > STD_HT + 0.1:
                    total_lai = total_lai + n_lai
                canht_int = int(n_canopy_ht) - 1
                if canht_int < 0:
                    canht_int = 0
                forht_int = int(n_height) - 1
                if forht_int < 0:
                    forht_int = 0
                if forht_int > 49:
                    forht_int = 49
                n_canopy_layers = forht_int - canht_int + 1
                if n_canopy_layers < 1:
                    n_canopy_layers = 1
                lai_per_layer = n_lai / float(n_canopy_layers)
                for layer in range(n_canopy_layers):
                    bin_idx = canht_int + layer
                    if bin_idx >= 0 and bin_idx < MAX_HEIGHT_BINS:
                        if n_evergreen > 0:
                            gap_lai[lai_row][bin_idx][0] = gap_lai[lai_row][bin_idx][0] + lai_per_layer
                            gap_lai[lai_row][bin_idx][1] = gap_lai[lai_row][bin_idx][1] + lai_per_layer
                        else:
                            gap_lai[lai_row][bin_idx][0] = gap_lai[lai_row][bin_idx][0] + lai_per_layer
                            gap_lai[lai_row][bin_idx][1] = gap_lai[lai_row][bin_idx][1] + lai_per_layer * 0.8
                if n_species_id >= 0 and n_species_id < len(species_traits):
                    if n_diam > n_max_diam * 0.05:
                        gap_avail_spec[sp_row][n_species_id] = 1.0
            elif tree_alive < -0.5:
                template_weight = states_tensor[neighbor_idx][TreeS.SEEDLING_WEIGHT]
                total_seedling_weight = total_seedling_weight + template_weight
        i = i + 1
    for k in range(49):
        layer = 48 - k
        gap_lai[lai_row][layer][0] = gap_lai[lai_row][layer][0] + gap_lai[lai_row][layer + 1][0]
        gap_lai[lai_row][layer][1] = gap_lai[lai_row][layer][1] + gap_lai[lai_row][layer + 1][1]
    states_tensor[agent_index][GapS.LITTER_ACCUM_C] = total_litter_c
    states_tensor[agent_index][GapS.LITTER_ACCUM_N] = total_litter_n
    total_lai_scaled = total_lai / PLOTSIZE
    states_tensor[agent_index][GapS.TOTAL_LAI] = total_lai_scaled
    states_tensor[agent_index][GapS.TOTAL_SEEDLING_WEIGHT] = total_seedling_weight

@jit.rawkernel(device='cuda')
def site_soil_step_double_buffer(tick, agent_index, _seed, species_traits, site_configs, rangelists, site_distances, agent_ids, logical_ids, breeds, neighbor_offsets, neighbor_values, params_tensor, states_tensor, gap_lai, gap_lai_idx, gap_avail_spec, gap_avail_spec_idx, gap_imported_seeds, gap_imported_seeds_idx, site_avail_spec, write_site_avail_spec, site_avail_spec_idx, site_imported_seeds, site_imported_seeds_idx):
    """
    Soil biogeochemistry step function (priority 1).

    Implements UVAFME's complete soil model:
    1. Reads litter from Gap neighbors
    2. Daily loop (365 days):
       - Interpolate daily climate from monthly data
       - Calculate potential evapotranspiration (Hamon method)
       - Soil water balance
       - Soil decomposition with moisture adjustment
       - Accumulate N mineralization and atmospheric N deposition
    3. Update soil pools and available N

    Reads from Gap neighbors:
    - states: litter_accum_c, litter_accum_n

    Reads site config from site_configs:
    - Monthly climate (tmin, tmax, prcp), std devs
    - Soil properties (field_cap, perm_wp, slope, sigma, lai, latitude)
    - Fire/wind probabilities, base_h

    Writes to own:
    - params: soil pools (A0/A/BL carbon, nitrogen, water), output-only fields (ANNUAL_RAIN, GROW_DAYS, etc.)
    - states: DEG_DAYS, DRY_DAYS, AVAIL_N, FLOOD_DAYS, FIRE_INTENSITY, DRY_DAYS_BASE, WIND_INTENSITY
    """
    total_litter_c = 0.0
    total_litter_n = 0.0
    total_gap_lai = 0.0
    gap_count = 0.0
    i = 0
    while i < neighbor_offsets[agent_index + 1] - neighbor_offsets[agent_index]:
        neighbor_idx = int(neighbor_values[neighbor_offsets[agent_index] + i])
        neighbor_breed = int(breeds[neighbor_idx])
        if neighbor_breed == Breed.GAP:
            gap_litter_c = states_tensor[neighbor_idx][GapS.LITTER_ACCUM_C]
            gap_litter_n = states_tensor[neighbor_idx][GapS.LITTER_ACCUM_N]
            total_litter_c = total_litter_c + gap_litter_c
            total_litter_n = total_litter_n + gap_litter_n
            neighbor_gap_lai = states_tensor[neighbor_idx][GapS.TOTAL_LAI]
            total_gap_lai = total_gap_lai + neighbor_gap_lai
            gap_count = gap_count + 1.0
        i = i + 1
    if gap_count > 0.5:
        uconv = UNIT_CONV / gap_count
        total_litter_c = total_litter_c * uconv
        total_litter_n = total_litter_n * uconv
    ao_c0 = params_tensor[agent_index][SiteP.A0_C]
    ao_n0 = params_tensor[agent_index][SiteP.A0_N]
    sa_c0 = params_tensor[agent_index][SiteP.A_C]
    sa_n0 = params_tensor[agent_index][SiteP.A_N]
    sb_c0 = params_tensor[agent_index][SiteP.BL_C]
    sb_n0 = params_tensor[agent_index][SiteP.BL_N]
    ao_w0 = params_tensor[agent_index][SiteP.A0_W]
    sa_w0 = params_tensor[agent_index][SiteP.A_W]
    sb_w0 = params_tensor[agent_index][SiteP.BL_W]
    lai_w0 = params_tensor[agent_index][SiteP.LAI_W0]
    site_id = int(params_tensor[agent_index][SiteP.SITE_ID])
    sa_fc = site_configs[int(site_id)][Cfg.FIELD_CAP]
    sa_pwp = site_configs[int(site_id)][Cfg.PERM_WP]
    slope = site_configs[int(site_id)][Cfg.SLOPE]
    sigma = site_configs[int(site_id)][Cfg.SIGMA]
    latitude = site_configs[int(site_id)][Cfg.LATITUDE]
    lai = site_configs[int(site_id)][Cfg.LAI]
    if gap_count > 0.5:
        dynamic_lai = total_gap_lai / gap_count
        if dynamic_lai > 0.01:
            lai = dynamic_lai
    ao_c0 = ao_c0 + total_litter_c
    ao_n0 = ao_n0 + total_litter_n
    if lai < 1.0:
        lai = 1.0
    tmin_std_0 = site_configs[int(site_id)][Cfg.TMIN_STD_BASE + 0]
    tmin_std_1 = site_configs[int(site_id)][Cfg.TMIN_STD_BASE + 1]
    tmin_std_2 = site_configs[int(site_id)][Cfg.TMIN_STD_BASE + 2]
    tmin_std_3 = site_configs[int(site_id)][Cfg.TMIN_STD_BASE + 3]
    tmin_std_4 = site_configs[int(site_id)][Cfg.TMIN_STD_BASE + 4]
    tmin_std_5 = site_configs[int(site_id)][Cfg.TMIN_STD_BASE + 5]
    tmin_std_6 = site_configs[int(site_id)][Cfg.TMIN_STD_BASE + 6]
    tmin_std_7 = site_configs[int(site_id)][Cfg.TMIN_STD_BASE + 7]
    tmin_std_8 = site_configs[int(site_id)][Cfg.TMIN_STD_BASE + 8]
    tmin_std_9 = site_configs[int(site_id)][Cfg.TMIN_STD_BASE + 9]
    tmin_std_10 = site_configs[int(site_id)][Cfg.TMIN_STD_BASE + 10]
    tmin_std_11 = site_configs[int(site_id)][Cfg.TMIN_STD_BASE + 11]
    tmax_std_0 = site_configs[int(site_id)][Cfg.TMAX_STD_BASE + 0]
    tmax_std_1 = site_configs[int(site_id)][Cfg.TMAX_STD_BASE + 1]
    tmax_std_2 = site_configs[int(site_id)][Cfg.TMAX_STD_BASE + 2]
    tmax_std_3 = site_configs[int(site_id)][Cfg.TMAX_STD_BASE + 3]
    tmax_std_4 = site_configs[int(site_id)][Cfg.TMAX_STD_BASE + 4]
    tmax_std_5 = site_configs[int(site_id)][Cfg.TMAX_STD_BASE + 5]
    tmax_std_6 = site_configs[int(site_id)][Cfg.TMAX_STD_BASE + 6]
    tmax_std_7 = site_configs[int(site_id)][Cfg.TMAX_STD_BASE + 7]
    tmax_std_8 = site_configs[int(site_id)][Cfg.TMAX_STD_BASE + 8]
    tmax_std_9 = site_configs[int(site_id)][Cfg.TMAX_STD_BASE + 9]
    tmax_std_10 = site_configs[int(site_id)][Cfg.TMAX_STD_BASE + 10]
    tmax_std_11 = site_configs[int(site_id)][Cfg.TMAX_STD_BASE + 11]
    prcp_std_0 = site_configs[int(site_id)][Cfg.PRCP_STD_BASE + 0]
    prcp_std_1 = site_configs[int(site_id)][Cfg.PRCP_STD_BASE + 1]
    prcp_std_2 = site_configs[int(site_id)][Cfg.PRCP_STD_BASE + 2]
    prcp_std_3 = site_configs[int(site_id)][Cfg.PRCP_STD_BASE + 3]
    prcp_std_4 = site_configs[int(site_id)][Cfg.PRCP_STD_BASE + 4]
    prcp_std_5 = site_configs[int(site_id)][Cfg.PRCP_STD_BASE + 5]
    prcp_std_6 = site_configs[int(site_id)][Cfg.PRCP_STD_BASE + 6]
    prcp_std_7 = site_configs[int(site_id)][Cfg.PRCP_STD_BASE + 7]
    prcp_std_8 = site_configs[int(site_id)][Cfg.PRCP_STD_BASE + 8]
    prcp_std_9 = site_configs[int(site_id)][Cfg.PRCP_STD_BASE + 9]
    prcp_std_10 = site_configs[int(site_id)][Cfg.PRCP_STD_BASE + 10]
    prcp_std_11 = site_configs[int(site_id)][Cfg.PRCP_STD_BASE + 11]
    tmin_0 = site_configs[int(site_id)][Cfg.TMIN_BASE + 0]
    tmin_1 = site_configs[int(site_id)][Cfg.TMIN_BASE + 1]
    tmin_2 = site_configs[int(site_id)][Cfg.TMIN_BASE + 2]
    tmin_3 = site_configs[int(site_id)][Cfg.TMIN_BASE + 3]
    tmin_4 = site_configs[int(site_id)][Cfg.TMIN_BASE + 4]
    tmin_5 = site_configs[int(site_id)][Cfg.TMIN_BASE + 5]
    tmin_6 = site_configs[int(site_id)][Cfg.TMIN_BASE + 6]
    tmin_7 = site_configs[int(site_id)][Cfg.TMIN_BASE + 7]
    tmin_8 = site_configs[int(site_id)][Cfg.TMIN_BASE + 8]
    tmin_9 = site_configs[int(site_id)][Cfg.TMIN_BASE + 9]
    tmin_10 = site_configs[int(site_id)][Cfg.TMIN_BASE + 10]
    tmin_11 = site_configs[int(site_id)][Cfg.TMIN_BASE + 11]
    tmax_0 = site_configs[int(site_id)][Cfg.TMAX_BASE + 0]
    tmax_1 = site_configs[int(site_id)][Cfg.TMAX_BASE + 1]
    tmax_2 = site_configs[int(site_id)][Cfg.TMAX_BASE + 2]
    tmax_3 = site_configs[int(site_id)][Cfg.TMAX_BASE + 3]
    tmax_4 = site_configs[int(site_id)][Cfg.TMAX_BASE + 4]
    tmax_5 = site_configs[int(site_id)][Cfg.TMAX_BASE + 5]
    tmax_6 = site_configs[int(site_id)][Cfg.TMAX_BASE + 6]
    tmax_7 = site_configs[int(site_id)][Cfg.TMAX_BASE + 7]
    tmax_8 = site_configs[int(site_id)][Cfg.TMAX_BASE + 8]
    tmax_9 = site_configs[int(site_id)][Cfg.TMAX_BASE + 9]
    tmax_10 = site_configs[int(site_id)][Cfg.TMAX_BASE + 10]
    tmax_11 = site_configs[int(site_id)][Cfg.TMAX_BASE + 11]
    prcp_0 = site_configs[int(site_id)][Cfg.PRCP_BASE + 0]
    prcp_1 = site_configs[int(site_id)][Cfg.PRCP_BASE + 1]
    prcp_2 = site_configs[int(site_id)][Cfg.PRCP_BASE + 2]
    prcp_3 = site_configs[int(site_id)][Cfg.PRCP_BASE + 3]
    prcp_4 = site_configs[int(site_id)][Cfg.PRCP_BASE + 4]
    prcp_5 = site_configs[int(site_id)][Cfg.PRCP_BASE + 5]
    prcp_6 = site_configs[int(site_id)][Cfg.PRCP_BASE + 6]
    prcp_7 = site_configs[int(site_id)][Cfg.PRCP_BASE + 7]
    prcp_8 = site_configs[int(site_id)][Cfg.PRCP_BASE + 8]
    prcp_9 = site_configs[int(site_id)][Cfg.PRCP_BASE + 9]
    prcp_10 = site_configs[int(site_id)][Cfg.PRCP_BASE + 10]
    prcp_11 = site_configs[int(site_id)][Cfg.PRCP_BASE + 11]
    tp = 0.0
    pp = 0.0
    tp = rand_normal_bounded(_seed, tick, logical_ids[agent_index], 100, -1.0, 1.0)
    pp = rand_normal_bounded(_seed, tick, logical_ids[agent_index], 101, -0.5, 0.5)
    tmin_0 = tmin_0 + tp * tmin_std_0
    tmax_0 = tmax_0 + tp * tmax_std_0
    if tmax_0 < tmin_0:
        tmax_0 = tmin_0 + 0.1
    prcp_0 = prcp_0 + pp * prcp_std_0
    if prcp_0 < 0.0:
        prcp_0 = 0.0
    tp = rand_normal_bounded(_seed, tick, logical_ids[agent_index], 102, -1.0, 1.0)
    pp = rand_normal_bounded(_seed, tick, logical_ids[agent_index], 103, -0.5, 0.5)
    tmin_1 = tmin_1 + tp * tmin_std_1
    tmax_1 = tmax_1 + tp * tmax_std_1
    if tmax_1 < tmin_1:
        tmax_1 = tmin_1 + 0.1
    prcp_1 = prcp_1 + pp * prcp_std_1
    if prcp_1 < 0.0:
        prcp_1 = 0.0
    tp = rand_normal_bounded(_seed, tick, logical_ids[agent_index], 104, -1.0, 1.0)
    pp = rand_normal_bounded(_seed, tick, logical_ids[agent_index], 105, -0.5, 0.5)
    tmin_2 = tmin_2 + tp * tmin_std_2
    tmax_2 = tmax_2 + tp * tmax_std_2
    if tmax_2 < tmin_2:
        tmax_2 = tmin_2 + 0.1
    prcp_2 = prcp_2 + pp * prcp_std_2
    if prcp_2 < 0.0:
        prcp_2 = 0.0
    tp = rand_normal_bounded(_seed, tick, logical_ids[agent_index], 106, -1.0, 1.0)
    pp = rand_normal_bounded(_seed, tick, logical_ids[agent_index], 107, -0.5, 0.5)
    tmin_3 = tmin_3 + tp * tmin_std_3
    tmax_3 = tmax_3 + tp * tmax_std_3
    if tmax_3 < tmin_3:
        tmax_3 = tmin_3 + 0.1
    prcp_3 = prcp_3 + pp * prcp_std_3
    if prcp_3 < 0.0:
        prcp_3 = 0.0
    tp = rand_normal_bounded(_seed, tick, logical_ids[agent_index], 108, -1.0, 1.0)
    pp = rand_normal_bounded(_seed, tick, logical_ids[agent_index], 109, -0.5, 0.5)
    tmin_4 = tmin_4 + tp * tmin_std_4
    tmax_4 = tmax_4 + tp * tmax_std_4
    if tmax_4 < tmin_4:
        tmax_4 = tmin_4 + 0.1
    prcp_4 = prcp_4 + pp * prcp_std_4
    if prcp_4 < 0.0:
        prcp_4 = 0.0
    tp = rand_normal_bounded(_seed, tick, logical_ids[agent_index], 110, -1.0, 1.0)
    pp = rand_normal_bounded(_seed, tick, logical_ids[agent_index], 111, -0.5, 0.5)
    tmin_5 = tmin_5 + tp * tmin_std_5
    tmax_5 = tmax_5 + tp * tmax_std_5
    if tmax_5 < tmin_5:
        tmax_5 = tmin_5 + 0.1
    prcp_5 = prcp_5 + pp * prcp_std_5
    if prcp_5 < 0.0:
        prcp_5 = 0.0
    tp = rand_normal_bounded(_seed, tick, logical_ids[agent_index], 112, -1.0, 1.0)
    pp = rand_normal_bounded(_seed, tick, logical_ids[agent_index], 113, -0.5, 0.5)
    tmin_6 = tmin_6 + tp * tmin_std_6
    tmax_6 = tmax_6 + tp * tmax_std_6
    if tmax_6 < tmin_6:
        tmax_6 = tmin_6 + 0.1
    prcp_6 = prcp_6 + pp * prcp_std_6
    if prcp_6 < 0.0:
        prcp_6 = 0.0
    tp = rand_normal_bounded(_seed, tick, logical_ids[agent_index], 114, -1.0, 1.0)
    pp = rand_normal_bounded(_seed, tick, logical_ids[agent_index], 115, -0.5, 0.5)
    tmin_7 = tmin_7 + tp * tmin_std_7
    tmax_7 = tmax_7 + tp * tmax_std_7
    if tmax_7 < tmin_7:
        tmax_7 = tmin_7 + 0.1
    prcp_7 = prcp_7 + pp * prcp_std_7
    if prcp_7 < 0.0:
        prcp_7 = 0.0
    tp = rand_normal_bounded(_seed, tick, logical_ids[agent_index], 116, -1.0, 1.0)
    pp = rand_normal_bounded(_seed, tick, logical_ids[agent_index], 117, -0.5, 0.5)
    tmin_8 = tmin_8 + tp * tmin_std_8
    tmax_8 = tmax_8 + tp * tmax_std_8
    if tmax_8 < tmin_8:
        tmax_8 = tmin_8 + 0.1
    prcp_8 = prcp_8 + pp * prcp_std_8
    if prcp_8 < 0.0:
        prcp_8 = 0.0
    tp = rand_normal_bounded(_seed, tick, logical_ids[agent_index], 118, -1.0, 1.0)
    pp = rand_normal_bounded(_seed, tick, logical_ids[agent_index], 119, -0.5, 0.5)
    tmin_9 = tmin_9 + tp * tmin_std_9
    tmax_9 = tmax_9 + tp * tmax_std_9
    if tmax_9 < tmin_9:
        tmax_9 = tmin_9 + 0.1
    prcp_9 = prcp_9 + pp * prcp_std_9
    if prcp_9 < 0.0:
        prcp_9 = 0.0
    tp = rand_normal_bounded(_seed, tick, logical_ids[agent_index], 120, -1.0, 1.0)
    pp = rand_normal_bounded(_seed, tick, logical_ids[agent_index], 121, -0.5, 0.5)
    tmin_10 = tmin_10 + tp * tmin_std_10
    tmax_10 = tmax_10 + tp * tmax_std_10
    if tmax_10 < tmin_10:
        tmax_10 = tmin_10 + 0.1
    prcp_10 = prcp_10 + pp * prcp_std_10
    if prcp_10 < 0.0:
        prcp_10 = 0.0
    tp = rand_normal_bounded(_seed, tick, logical_ids[agent_index], 122, -1.0, 1.0)
    pp = rand_normal_bounded(_seed, tick, logical_ids[agent_index], 123, -0.5, 0.5)
    tmin_11 = tmin_11 + tp * tmin_std_11
    tmax_11 = tmax_11 + tp * tmax_std_11
    if tmax_11 < tmin_11:
        tmax_11 = tmin_11 + 0.1
    prcp_11 = prcp_11 + pp * prcp_std_11
    if prcp_11 < 0.0:
        prcp_11 = 0.0
    annual_prcp_cm = prcp_0 + prcp_1 + prcp_2 + prcp_3 + prcp_4 + prcp_5 + prcp_6 + prcp_7 + prcp_8 + prcp_9 + prcp_10 + prcp_11
    sbh = site_configs[int(site_id)][Cfg.BASE_H]
    if sbh < 1.0:
        sbh = 70.0
    laiw_min = lai * LAI_MIN
    laiw_max = lai * LAI_MAX
    aow_min = ao_c0 * AO_MIN
    aow_max = ao_c0 * AO_MAX
    sbw_min = sbh * BASE_MIN
    sbw_max = sbh * BASE_MAX
    lat_rad = latitude * PI / 180.0
    total_avail_n = 0.0
    total_resp = 0.0
    rain_n = 0.0
    freeze_days = 0.0
    flood_days = 0.0
    drydays_upper = 0.0
    drydays_base = 0.0
    total_pet = 0.0
    total_aet = 0.0
    deg_days = 0.0
    grow_days_5 = 0.0
    annual_runoff = 0.0
    m0_raindays = prcp_0 / 4.0 + 1.0
    if m0_raindays > 25.0:
        m0_raindays = 25.0
    m0_ik = int(m0_raindays)
    if m0_ik < 1:
        m0_ik = 1
    m0_rr = prcp_0 / float(m0_ik)
    m0_ss = m0_raindays / 31.0
    m0_inum = float(m0_ik)
    m1_raindays = prcp_1 / 4.0 + 1.0
    if m1_raindays > 25.0:
        m1_raindays = 25.0
    m1_ik = int(m1_raindays)
    if m1_ik < 1:
        m1_ik = 1
    m1_rr = prcp_1 / float(m1_ik)
    m1_ss = m1_raindays / 28.0
    m1_inum = float(m1_ik)
    m2_raindays = prcp_2 / 4.0 + 1.0
    if m2_raindays > 25.0:
        m2_raindays = 25.0
    m2_ik = int(m2_raindays)
    if m2_ik < 1:
        m2_ik = 1
    m2_rr = prcp_2 / float(m2_ik)
    m2_ss = m2_raindays / 31.0
    m2_inum = float(m2_ik)
    m3_raindays = prcp_3 / 4.0 + 1.0
    if m3_raindays > 25.0:
        m3_raindays = 25.0
    m3_ik = int(m3_raindays)
    if m3_ik < 1:
        m3_ik = 1
    m3_rr = prcp_3 / float(m3_ik)
    m3_ss = m3_raindays / 30.0
    m3_inum = float(m3_ik)
    m4_raindays = prcp_4 / 4.0 + 1.0
    if m4_raindays > 25.0:
        m4_raindays = 25.0
    m4_ik = int(m4_raindays)
    if m4_ik < 1:
        m4_ik = 1
    m4_rr = prcp_4 / float(m4_ik)
    m4_ss = m4_raindays / 31.0
    m4_inum = float(m4_ik)
    m5_raindays = prcp_5 / 4.0 + 1.0
    if m5_raindays > 25.0:
        m5_raindays = 25.0
    m5_ik = int(m5_raindays)
    if m5_ik < 1:
        m5_ik = 1
    m5_rr = prcp_5 / float(m5_ik)
    m5_ss = m5_raindays / 30.0
    m5_inum = float(m5_ik)
    m6_raindays = prcp_6 / 4.0 + 1.0
    if m6_raindays > 25.0:
        m6_raindays = 25.0
    m6_ik = int(m6_raindays)
    if m6_ik < 1:
        m6_ik = 1
    m6_rr = prcp_6 / float(m6_ik)
    m6_ss = m6_raindays / 31.0
    m6_inum = float(m6_ik)
    m7_raindays = prcp_7 / 4.0 + 1.0
    if m7_raindays > 25.0:
        m7_raindays = 25.0
    m7_ik = int(m7_raindays)
    if m7_ik < 1:
        m7_ik = 1
    m7_rr = prcp_7 / float(m7_ik)
    m7_ss = m7_raindays / 31.0
    m7_inum = float(m7_ik)
    m8_raindays = prcp_8 / 4.0 + 1.0
    if m8_raindays > 25.0:
        m8_raindays = 25.0
    m8_ik = int(m8_raindays)
    if m8_ik < 1:
        m8_ik = 1
    m8_rr = prcp_8 / float(m8_ik)
    m8_ss = m8_raindays / 30.0
    m8_inum = float(m8_ik)
    m9_raindays = prcp_9 / 4.0 + 1.0
    if m9_raindays > 25.0:
        m9_raindays = 25.0
    m9_ik = int(m9_raindays)
    if m9_ik < 1:
        m9_ik = 1
    m9_rr = prcp_9 / float(m9_ik)
    m9_ss = m9_raindays / 31.0
    m9_inum = float(m9_ik)
    m10_raindays = prcp_10 / 4.0 + 1.0
    if m10_raindays > 25.0:
        m10_raindays = 25.0
    m10_ik = int(m10_raindays)
    if m10_ik < 1:
        m10_ik = 1
    m10_rr = prcp_10 / float(m10_ik)
    m10_ss = m10_raindays / 30.0
    m10_inum = float(m10_ik)
    m11_raindays = prcp_11 / 4.0 + 1.0
    if m11_raindays > 25.0:
        m11_raindays = 25.0
    m11_ik = int(m11_raindays)
    if m11_ik < 1:
        m11_ik = 1
    m11_rr = prcp_11 / float(m11_ik)
    m11_ss = m11_raindays / 31.0
    m11_inum = float(m11_ik)
    day = 0
    while day < 365:
        d_s = day
        if day < 15:
            d_s = day + 365
        day_tmin = 0.0
        day_tmax = 0.0
        frac = 0.0
        if d_s < 44:
            frac = float(d_s - 15) / 29.0
            day_tmin = tmin_0 + frac * (tmin_1 - tmin_0)
            day_tmax = tmax_0 + frac * (tmax_1 - tmax_0)
        elif d_s < 74:
            frac = float(d_s - 44) / 30.0
            day_tmin = tmin_1 + frac * (tmin_2 - tmin_1)
            day_tmax = tmax_1 + frac * (tmax_2 - tmax_1)
        elif d_s < 104:
            frac = float(d_s - 74) / 30.0
            day_tmin = tmin_2 + frac * (tmin_3 - tmin_2)
            day_tmax = tmax_2 + frac * (tmax_3 - tmax_2)
        elif d_s < 135:
            frac = float(d_s - 104) / 31.0
            day_tmin = tmin_3 + frac * (tmin_4 - tmin_3)
            day_tmax = tmax_3 + frac * (tmax_4 - tmax_3)
        elif d_s < 165:
            frac = float(d_s - 135) / 30.0
            day_tmin = tmin_4 + frac * (tmin_5 - tmin_4)
            day_tmax = tmax_4 + frac * (tmax_5 - tmax_4)
        elif d_s < 195:
            frac = float(d_s - 165) / 30.0
            day_tmin = tmin_5 + frac * (tmin_6 - tmin_5)
            day_tmax = tmax_5 + frac * (tmax_6 - tmax_5)
        elif d_s < 226:
            frac = float(d_s - 195) / 31.0
            day_tmin = tmin_6 + frac * (tmin_7 - tmin_6)
            day_tmax = tmax_6 + frac * (tmax_7 - tmax_6)
        elif d_s < 257:
            frac = float(d_s - 226) / 31.0
            day_tmin = tmin_7 + frac * (tmin_8 - tmin_7)
            day_tmax = tmax_7 + frac * (tmax_8 - tmax_7)
        elif d_s < 287:
            frac = float(d_s - 257) / 30.0
            day_tmin = tmin_8 + frac * (tmin_9 - tmin_8)
            day_tmax = tmax_8 + frac * (tmax_9 - tmax_8)
        elif d_s < 318:
            frac = float(d_s - 287) / 31.0
            day_tmin = tmin_9 + frac * (tmin_10 - tmin_9)
            day_tmax = tmax_9 + frac * (tmax_10 - tmax_9)
        elif d_s < 348:
            frac = float(d_s - 318) / 30.0
            day_tmin = tmin_10 + frac * (tmin_11 - tmin_10)
            day_tmax = tmax_10 + frac * (tmax_11 - tmax_10)
        else:
            frac = float(d_s - 348) / 32.0
            day_tmin = tmin_11 + frac * (tmin_0 - tmin_11)
            day_tmax = tmax_11 + frac * (tmax_0 - tmax_11)
        day_prcp = 0.0
        if day < 31:
            if m0_inum > 0.5:
                rain_hash = rand_uniform_xorshift(_seed, tick * 365 + day, logical_ids[agent_index], 200)
                if rain_hash <= m0_ss:
                    day_prcp = m0_rr
                    m0_inum = m0_inum - 1.0
            if day == 30 and m0_inum > 0.5:
                day_prcp = day_prcp + m0_inum * m0_rr
                m0_inum = 0.0
        elif day < 59:
            if m1_inum > 0.5:
                rain_hash = rand_uniform_xorshift(_seed, tick * 365 + day, logical_ids[agent_index], 200)
                if rain_hash <= m1_ss:
                    day_prcp = m1_rr
                    m1_inum = m1_inum - 1.0
            if day == 58 and m1_inum > 0.5:
                day_prcp = day_prcp + m1_inum * m1_rr
                m1_inum = 0.0
        elif day < 90:
            if m2_inum > 0.5:
                rain_hash = rand_uniform_xorshift(_seed, tick * 365 + day, logical_ids[agent_index], 200)
                if rain_hash <= m2_ss:
                    day_prcp = m2_rr
                    m2_inum = m2_inum - 1.0
            if day == 89 and m2_inum > 0.5:
                day_prcp = day_prcp + m2_inum * m2_rr
                m2_inum = 0.0
        elif day < 120:
            if m3_inum > 0.5:
                rain_hash = rand_uniform_xorshift(_seed, tick * 365 + day, logical_ids[agent_index], 200)
                if rain_hash <= m3_ss:
                    day_prcp = m3_rr
                    m3_inum = m3_inum - 1.0
            if day == 119 and m3_inum > 0.5:
                day_prcp = day_prcp + m3_inum * m3_rr
                m3_inum = 0.0
        elif day < 151:
            if m4_inum > 0.5:
                rain_hash = rand_uniform_xorshift(_seed, tick * 365 + day, logical_ids[agent_index], 200)
                if rain_hash <= m4_ss:
                    day_prcp = m4_rr
                    m4_inum = m4_inum - 1.0
            if day == 150 and m4_inum > 0.5:
                day_prcp = day_prcp + m4_inum * m4_rr
                m4_inum = 0.0
        elif day < 181:
            if m5_inum > 0.5:
                rain_hash = rand_uniform_xorshift(_seed, tick * 365 + day, logical_ids[agent_index], 200)
                if rain_hash <= m5_ss:
                    day_prcp = m5_rr
                    m5_inum = m5_inum - 1.0
            if day == 180 and m5_inum > 0.5:
                day_prcp = day_prcp + m5_inum * m5_rr
                m5_inum = 0.0
        elif day < 212:
            if m6_inum > 0.5:
                rain_hash = rand_uniform_xorshift(_seed, tick * 365 + day, logical_ids[agent_index], 200)
                if rain_hash <= m6_ss:
                    day_prcp = m6_rr
                    m6_inum = m6_inum - 1.0
            if day == 211 and m6_inum > 0.5:
                day_prcp = day_prcp + m6_inum * m6_rr
                m6_inum = 0.0
        elif day < 243:
            if m7_inum > 0.5:
                rain_hash = rand_uniform_xorshift(_seed, tick * 365 + day, logical_ids[agent_index], 200)
                if rain_hash <= m7_ss:
                    day_prcp = m7_rr
                    m7_inum = m7_inum - 1.0
            if day == 242 and m7_inum > 0.5:
                day_prcp = day_prcp + m7_inum * m7_rr
                m7_inum = 0.0
        elif day < 273:
            if m8_inum > 0.5:
                rain_hash = rand_uniform_xorshift(_seed, tick * 365 + day, logical_ids[agent_index], 200)
                if rain_hash <= m8_ss:
                    day_prcp = m8_rr
                    m8_inum = m8_inum - 1.0
            if day == 272 and m8_inum > 0.5:
                day_prcp = day_prcp + m8_inum * m8_rr
                m8_inum = 0.0
        elif day < 304:
            if m9_inum > 0.5:
                rain_hash = rand_uniform_xorshift(_seed, tick * 365 + day, logical_ids[agent_index], 200)
                if rain_hash <= m9_ss:
                    day_prcp = m9_rr
                    m9_inum = m9_inum - 1.0
            if day == 303 and m9_inum > 0.5:
                day_prcp = day_prcp + m9_inum * m9_rr
                m9_inum = 0.0
        elif day < 334:
            if m10_inum > 0.5:
                rain_hash = rand_uniform_xorshift(_seed, tick * 365 + day, logical_ids[agent_index], 200)
                if rain_hash <= m10_ss:
                    day_prcp = m10_rr
                    m10_inum = m10_inum - 1.0
            if day == 333 and m10_inum > 0.5:
                day_prcp = day_prcp + m10_inum * m10_rr
                m10_inum = 0.0
        else:
            if m11_inum > 0.5:
                rain_hash = rand_uniform_xorshift(_seed, tick * 365 + day, logical_ids[agent_index], 200)
                if rain_hash <= m11_ss:
                    day_prcp = m11_rr
                    m11_inum = m11_inum - 1.0
            if day == 364 and m11_inum > 0.5:
                day_prcp = day_prcp + m11_inum * m11_rr
                m11_inum = 0.0
        day_temp = (day_tmin + day_tmax) / 2.0
        if day_temp < 0.0:
            freeze_days = freeze_days + 1.0
        if day_temp >= 5.0:
            deg_days = deg_days + (day_temp - 5.0)
            grow_days_5 = grow_days_5 + 1.0
        rain_n = rain_n + day_prcp * PRCP_N
        julia = day + 1
        dairta = H_AS * cp.sin(H_B * float(julia) + H_PHASE)
        yxd_pet = -cp.tan(lat_rad) * cp.tan(dairta)
        if yxd_pet >= 1.0:
            yxd_pet = 1.0
        if yxd_pet <= -1.0:
            yxd_pet = -1.0
        omega = 0.0
        if yxd_pet >= 1.0:
            omega = 0.0
        elif yxd_pet <= -1.0:
            omega = PI
        else:
            omega = cp.arccos(yxd_pet)
        erad = H_AMP * cp.cos(lat_rad) * cp.cos(dairta) * (cp.sin(omega) - omega * cp.cos(omega))
        if erad < 0.0:
            erad = 0.0
        pot_ev_day = 0.0
        if day_temp > 0.0:
            tdiff = day_tmax - day_tmin
            if tdiff < 0.0:
                tdiff = 0.0
            pot_ev_day = H_COEFF * tdiff ** 0.5 * (day_temp + H_ADDON) * erad
        act_ev_day = pot_ev_day
        total_pet = total_pet + pot_ev_day
        total_aet = total_aet + act_ev_day
        total_resp = total_resp + 0.0
        day = day + 1
    if day > 99999:
        freeze = 0.0
        aow_min = ao_c0 * AO_MIN
        aow_max = ao_c0 * AO_MAX
        if aow_max < 0.01:
            aow_max = 0.01
        if aow_min < 0.001:
            aow_min = 0.001
        if day_prcp > 0.01:
            table_water = day_prcp * sigma * freeze
            sb_w0 = sb_w0 + table_water
            if sb_w0 > sbw_max:
                sb_w0 = sbw_max
        act_ev_day = 0.0
        runoff = 0.0
        if pot_ev_day <= 0.0:
            laiw = lai_w0 + day_prcp
            if laiw > laiw_max:
                laiw = laiw_max
            yxd1 = day_prcp - (laiw - lai_w0)
            if yxd1 < 0.0:
                yxd1 = 0.0
            aow = ao_w0 + yxd1
            if aow > aow_max:
                aow = aow_max
            yxd2 = yxd1 - (aow - ao_w0)
            if yxd2 < 0.0:
                yxd2 = 0.0
            sbw = sb_w0 + yxd2
            if sbw > sbw_max:
                sbw = sbw_max
            runoff = yxd2 - (sbw - sb_w0)
            if runoff < 0.0:
                runoff = 0.0
            lai_w0 = laiw
            ao_w0 = aow
            sb_w0 = sbw
        else:
            lai_loss = laiw_max - lai_w0
            if lai_loss > day_prcp:
                lai_loss = day_prcp
            if lai_loss < 0.0:
                lai_loss = 0.0
            yxd1 = day_prcp - lai_loss
            if yxd1 < 0.0:
                yxd1 = 0.0
            laiw = lai_w0 + lai_loss
            yxd = slope / 90.0 * (slope / 90.0)
            lossslp = yxd * yxd1
            yxd2 = yxd1 - lossslp - pot_ev_day
            if yxd2 > 0.0:
                saw_add = sa_fc - sa_w0
                if saw_add > yxd2:
                    saw_add = yxd2
                if saw_add < 0.0:
                    saw_add = 0.0
                saw = sa_w0 + saw_add
                yxd3 = yxd2 - saw_add
                if yxd3 < 0.0:
                    yxd3 = 0.0
                aow_add = aow_max - ao_w0
                if aow_add > yxd3:
                    aow_add = yxd3
                if aow_add < 0.0:
                    aow_add = 0.0
                aow = ao_w0 + aow_add
                yxd4 = yxd3 - aow_add
                if yxd4 < 0.0:
                    yxd4 = 0.0
                sbw_add = sbw_max - sb_w0
                if sbw_add > yxd4:
                    sbw_add = yxd4
                if sbw_add < 0.0:
                    sbw_add = 0.0
                sbw = sb_w0 + sbw_add
                act_ev_day = pot_ev_day
                runoff = yxd2 + lossslp
                sa_w0 = saw
                sb_w0 = sbw
                ao_w0 = aow
            else:
                lai_w1 = -yxd2
                if lai_w1 > lai_w0 - laiw_min:
                    lai_w1 = lai_w0 - laiw_min
                if lai_w1 < 0.0:
                    lai_w1 = 0.0
                lai_w0 = lai_w0 - lai_w1
                act_ev_day = act_ev_day + lai_w1
                yxd3 = yxd2 + lai_w1
                if yxd3 > 0.0:
                    yxd3 = 0.0
                ao_w1 = -yxd3
                if ao_w1 > ao_w0 - aow_min:
                    ao_w1 = ao_w0 - aow_min
                if ao_w1 < 0.0:
                    ao_w1 = 0.0
                act_ev_day = act_ev_day + ao_w1
                ao_w0 = ao_w0 - ao_w1
                yxd4 = yxd3 + ao_w1
                if yxd4 > 0.0:
                    yxd4 = 0.0
                sa_w1 = -yxd4
                if sa_w1 > sa_w0 - sa_pwp:
                    sa_w1 = sa_w0 - sa_pwp
                if sa_w1 < 0.0:
                    sa_w1 = 0.0
                act_ev_day = act_ev_day + sa_w1
                sa_w0 = sa_w0 - sa_w1
                yxd5 = yxd4 + sa_w1
                if yxd5 > 0.0:
                    yxd5 = 0.0
                sb_w1 = -yxd5
                if sb_w1 > sb_w0 - sbw_min:
                    sb_w1 = sb_w0 - sbw_min
                if sb_w1 < 0.0:
                    sb_w1 = 0.0
                sb_w0 = sb_w0 - sb_w1
                runoff = lossslp
        aow0_scaled = ao_w0 / aow_max
        saw0_scaled = 0.5
        if sa_fc > 0.0:
            saw0_scaled = sa_w0 / sa_fc
        if sa_fc > 0.0 and sa_w0 >= sa_fc * 0.95:
            flood_days = flood_days + 1.0
        sbw0_scaled_by_min = 1.0
        if sbw_min > 0.001:
            sbw0_scaled_by_min = sb_w0 / sbw_min
        sbw0_scaled_by_max = 1.0
        if sbw_max > 0.001:
            sbw0_scaled_by_max = sb_w0 / sbw_max
        saw0_scaled_by_wp = 1.0
        if sa_pwp > 0.001:
            saw0_scaled_by_wp = sa_w0 / sa_pwp
        if saw0_scaled < 1.0001 and sbw0_scaled_by_min < 1.0001 and (sbw0_scaled_by_max < 1.0001):
            drydays_upper = drydays_upper + 1.0
        if saw0_scaled_by_wp < 1.0001:
            drydays_base = drydays_base + 1.0
        total_pet = total_pet + pot_ev_day
        total_aet = total_aet + act_ev_day
        annual_runoff = annual_runoff + runoff
        ao_cn = AO_CN_0
        if ao_n0 > 0.0001:
            ao_cn = ao_c0 / ao_n0
        if aow0_scaled > 0.5:
            aow0_scaled = 0.5
        aofunc = 1.0 - (1.0 - aow0_scaled / 0.3) * (1.0 - aow0_scaled / 0.3)
        if aofunc < 0.2:
            aofunc = 0.2
        tadjst = 0.0
        tadjst1 = 0.0
        if day_temp >= -5.0:
            tadjst = cp.power(3.0, 0.1 * (day_temp - 1.0))
            tadjst1 = cp.power(2.5, 0.1 * (day_temp - 1.0))
        resp1 = tadjst * aofunc * AO_RESP * ao_c0
        yxdn = 0.0
        yxdc = 0.0
        if ao_cn > 0.001:
            yxdn = resp1 / ao_cn
            yxdc = yxdn * AO_CN_0
        ao_c0 = ao_c0 - yxdc - resp1
        ao_n0 = ao_n0 - yxdn
        if ao_c0 < 0.0:
            ao_c0 = 0.0
        if ao_n0 < 0.0:
            ao_n0 = 0.0
        sa_c0 = sa_c0 + yxdc
        sa_n0 = sa_n0 + yxdn
        sa_cn = SA_CN_0
        if sa_n0 > 0.0001:
            sa_cn = sa_c0 / sa_n0
        safunc = 1.0 - (1.0 - saw0_scaled / 0.8) * (1.0 - saw0_scaled / 0.8)
        if safunc < 0.2:
            safunc = 0.2
        resp2 = tadjst1 * safunc * SA_RESP * sa_c0
        tosb = resp2 / SB_CN_0
        n_efficiency = 0.5
        avail_n_day = 0.0
        if sa_cn > 0.001:
            n_efficiency = (sa_cn - SA_CN_0) / sa_cn
            if n_efficiency < 0.5:
                n_efficiency = 0.5
            avail_n_day = resp2 / sa_cn * n_efficiency
            if avail_n_day < 0.0:
                avail_n_day = 0.0
        sa_c0 = sa_c0 - resp2 - tosb
        sa_n0 = sa_n0 - avail_n_day
        if sa_c0 < 0.0:
            sa_c0 = 0.0
        if sa_n0 < 0.0:
            sa_n0 = 0.0
        sb_c0 = sb_c0 + tosb
        resp3 = sb_c0 * SB_RESP * tadjst1
        sb_c0 = sb_c0 - resp3
        if sb_c0 < 0.0:
            sb_c0 = 0.0
        total_avail_n = total_avail_n + avail_n_day
        total_resp = total_resp + resp1 + resp2 + resp3
        day = day + 1
    growdays = grow_days_5
    if growdays < 1.0:
        growdays = 1.0
    dry_days_frac = drydays_upper / growdays
    dry_days_base_frac = drydays_base / growdays
    if total_pet > 0.001:
        rain_ratio = annual_prcp_cm / total_pet
        if rain_ratio > 1.0:
            rain_ratio = 1.0
        aet_ratio = total_aet / total_pet
        if aet_ratio > 1.0:
            aet_ratio = 1.0
        tmp_cap = rain_ratio
        if aet_ratio > rain_ratio:
            tmp_cap = aet_ratio
        cap = 1.0 - tmp_cap
        if dry_days_frac > cap:
            dry_days_frac = cap
    if dry_days_frac < 0.0:
        dry_days_frac = 0.0
    if dry_days_frac > 1.0:
        dry_days_frac = 1.0
    if dry_days_base_frac < 0.0:
        dry_days_base_frac = 0.0
    if dry_days_base_frac > 1.0:
        dry_days_base_frac = 1.0
    dry_days = dry_days_frac
    flood_days = flood_days / growdays
    params_tensor[agent_index][SiteP.A0_C] = ao_c0
    params_tensor[agent_index][SiteP.A0_N] = ao_n0
    params_tensor[agent_index][SiteP.A_C] = sa_c0
    params_tensor[agent_index][SiteP.A_N] = sa_n0
    params_tensor[agent_index][SiteP.BL_C] = sb_c0
    params_tensor[agent_index][SiteP.BL_N] = sb_n0
    params_tensor[agent_index][SiteP.A0_W] = ao_w0
    params_tensor[agent_index][SiteP.A_W] = sa_w0
    params_tensor[agent_index][SiteP.BL_W] = sb_w0
    params_tensor[agent_index][SiteP.LAI_W0] = lai_w0
    params_tensor[agent_index][SiteP.ANNUAL_RUNOFF] = annual_runoff
    states_tensor[agent_index][SiteS.DEG_DAYS] = deg_days
    states_tensor[agent_index][SiteS.AVAIL_N] = total_avail_n + rain_n
    states_tensor[agent_index][SiteS.FLOOD_DAYS] = flood_days
    states_tensor[agent_index][SiteS.DRY_DAYS] = dry_days
    states_tensor[agent_index][SiteS.DRY_DAYS_BASE] = dry_days_base_frac
    fire_prob = site_configs[int(site_id)][Cfg.FIRE_PROB]
    estimated_dry_days = 0.0
    if saw0_scaled < 0.3:
        estimated_dry_days = 30.0
    if saw0_scaled < 0.2:
        estimated_dry_days = 60.0
    if saw0_scaled < 0.1:
        estimated_dry_days = 120.0
    if estimated_dry_days > 60.0:
        dry_fire_adj = 0.03 + (estimated_dry_days - 60.0) * 0.001
        if dry_fire_adj > fire_prob:
            fire_prob = dry_fire_adj
    if fire_prob > 0.15:
        fire_prob = 0.15
    fire_rand = rand_uniform_philox(_seed, tick, logical_ids[agent_index], 10)
    fire_intensity = 0.0
    if fire_rand < fire_prob:
        fire_intensity = 0.3 + fire_rand * 2.0
        if fire_intensity > 1.0:
            fire_intensity = 1.0
    states_tensor[agent_index][SiteS.FIRE_INTENSITY] = fire_intensity
    wind_prob = site_configs[int(site_id)][Cfg.WIND_PROB]
    wind_rand = rand_uniform_philox(_seed, tick, logical_ids[agent_index], 11)
    wind_intensity = 0.0
    if wind_rand < wind_prob and fire_intensity < 0.01:
        wind_intensity = 0.5 + wind_rand * 1.0
        if wind_intensity > 1.0:
            wind_intensity = 1.0
    states_tensor[agent_index][SiteS.WIND_INTENSITY] = wind_intensity
    params_tensor[agent_index][SiteP.ANNUAL_RAIN] = annual_prcp_cm
    params_tensor[agent_index][SiteP.GROW_DAYS] = grow_days_5
    params_tensor[agent_index][SiteP.POT_EVAP] = total_pet
    params_tensor[agent_index][SiteP.ACT_EVAP] = total_aet
    params_tensor[agent_index][SiteP.SOIL_RESP] = total_resp
    params_tensor[agent_index][SiteP.C_INTO_A0] = total_litter_c
    params_tensor[agent_index][SiteP.N_INTO_A0] = total_litter_n

@jit.rawkernel(device='cuda')
def gap_climate_relay_step_double_buffer(tick, agent_index, _seed, species_traits, site_configs, rangelists, site_distances, agent_ids, logical_ids, breeds, neighbor_offsets, neighbor_values, params_tensor, states_tensor, gap_lai, gap_lai_idx, gap_avail_spec, gap_avail_spec_idx, gap_imported_seeds, gap_imported_seeds_idx, site_avail_spec, write_site_avail_spec, site_avail_spec_idx, site_imported_seeds, site_imported_seeds_idx):
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
    site_idx = -1
    i = 0
    while i < neighbor_offsets[agent_index + 1] - neighbor_offsets[agent_index]:
        neighbor_idx = int(neighbor_values[neighbor_offsets[agent_index] + i])
        neighbor_breed = int(breeds[neighbor_idx])
        if neighbor_breed == Breed.SITE:
            site_idx = neighbor_idx
            site_deg_days = states_tensor[neighbor_idx][SiteS.DEG_DAYS]
            site_dry_days = states_tensor[neighbor_idx][SiteS.DRY_DAYS]
            site_avail_n = states_tensor[neighbor_idx][SiteS.AVAIL_N]
            site_flood_days = states_tensor[neighbor_idx][SiteS.FLOOD_DAYS]
            site_fire_intensity = states_tensor[neighbor_idx][SiteS.FIRE_INTENSITY]
            site_wind_intensity = states_tensor[neighbor_idx][SiteS.WIND_INTENSITY]
            site_dry_days_base = states_tensor[neighbor_idx][SiteS.DRY_DAYS_BASE]
        i = i + 1
    states_tensor[agent_index][GapS.DEG_DAYS] = site_deg_days
    states_tensor[agent_index][GapS.DRY_DAYS] = site_dry_days
    states_tensor[agent_index][GapS.AVAIL_N] = site_avail_n
    states_tensor[agent_index][GapS.FLOOD_DAYS] = site_flood_days
    states_tensor[agent_index][GapS.FIRE_INTENSITY] = site_fire_intensity
    states_tensor[agent_index][GapS.WIND_INTENSITY] = site_wind_intensity
    states_tensor[agent_index][GapS.DRY_DAYS_BASE] = site_dry_days_base
    if site_fire_intensity > 0.01:
        states_tensor[agent_index][GapS.RECOVERY_YEARS] = 5.0
    elif site_wind_intensity > 0.01:
        states_tensor[agent_index][GapS.RECOVERY_YEARS] = 3.0
    num_species = len(species_traits)
    if site_idx >= 0:
        sp = 0
        while sp < num_species:
            gap_imported_seeds[gap_imported_seeds_idx[agent_index]][sp] = site_imported_seeds[site_imported_seeds_idx[site_idx]][sp]
            sp = sp + 1

@jit.rawkernel(device='cuda')
def tree_potential_growth_step_double_buffer(tick, agent_index, _seed, species_traits, site_configs, rangelists, site_distances, agent_ids, logical_ids, breeds, neighbor_offsets, neighbor_values, params_tensor, states_tensor, gap_lai, gap_lai_idx, gap_avail_spec, gap_avail_spec_idx, gap_imported_seeds, gap_imported_seeds_idx, site_avail_spec, write_site_avail_spec, site_avail_spec_idx, site_imported_seeds, site_imported_seeds_idx):
    """
    Phase A (P3): Environmental stress + potential growth + N demand.

    Computes env_stress = fc_degday * fc_drought * fc_light * fc_flood (no nutrient).
    Stores env_stress in states and diam_max in params for P5 to consume same-tick.
    Species traits are read from species_traits tensor instead of params_tensor.
    """
    is_alive = states_tensor[agent_index][TreeS.IS_ALIVE]
    n_demand = 0.0
    if is_alive > 0.5:
        species_id = states_tensor[agent_index][TreeS.SPECIES_ID]
        max_diam = species_traits[int(species_id)][Trait.MAX_DIAM]
        max_ht = species_traits[int(species_id)][Trait.MAX_HT]
        arfa_0 = species_traits[int(species_id)][Trait.ARFA_0]
        g = species_traits[int(species_id)][Trait.G]
        shade_tol = int(species_traits[int(species_id)][Trait.SHADE_TOL])
        deg_day_min = species_traits[int(species_id)][Trait.DEG_DAY_MIN]
        deg_day_opt = species_traits[int(species_id)][Trait.DEG_DAY_OPT]
        deg_day_max = species_traits[int(species_id)][Trait.DEG_DAY_MAX]
        wood_bulk_dens = species_traits[int(species_id)][Trait.WOOD_BULK_DENS]
        flood_tol = int(species_traits[int(species_id)][Trait.FLOOD_TOL])
        drought_tol = int(species_traits[int(species_id)][Trait.DROUGHT_TOL])
        evergreen = int(species_traits[int(species_id)][Trait.EVERGREEN])
        rootdepth = species_traits[int(species_id)][Trait.ROOTDEPTH]
        leafdiam_a = species_traits[int(species_id)][Trait.LEAFDIAM_A]
        leafarea_c = species_traits[int(species_id)][Trait.LEAFAREA_C]
        diam = states_tensor[agent_index][TreeS.DIAM]
        height = states_tensor[agent_index][TreeS.HEIGHT]
        canopy_ht = states_tensor[agent_index][TreeS.CANOPY_HT]
        biomC = params_tensor[agent_index][TreeP.BIOMC]
        leaf_bm = params_tensor[agent_index][TreeP.LEAF_BM]
        deg_days = 2500.0
        dry_days = 0.0
        dry_days_base = 0.0
        flood_days = 0.0
        cum_dec_lai = 0.0
        cum_con_lai = 0.0
        cum_dec_lai_base = 0.0
        cum_con_lai_base = 0.0
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
        i = 0
        while i < neighbor_offsets[agent_index + 1] - neighbor_offsets[agent_index]:
            neighbor_idx = int(neighbor_values[neighbor_offsets[agent_index] + i])
            neighbor_breed = int(breeds[neighbor_idx])
            if neighbor_breed == Breed.GAP:
                deg_days = states_tensor[neighbor_idx][GapS.DEG_DAYS]
                dry_days = states_tensor[neighbor_idx][GapS.DRY_DAYS]
                dry_days_base = states_tensor[neighbor_idx][GapS.DRY_DAYS_BASE]
                flood_days = states_tensor[neighbor_idx][GapS.FLOOD_DAYS]
                grow = gap_lai_idx[neighbor_idx]
                cum_dec_lai = gap_lai[grow][tree_height_layer][0]
                cum_con_lai = gap_lai[grow][tree_height_layer][1]
                cum_dec_lai_base = gap_lai[grow][tree_base_layer][0]
                cum_con_lai_base = gap_lai[grow][tree_base_layer][1]
            i = i + 1
        light_avail = 1.0
        if evergreen > 0:
            light_avail = cp.exp(XT * cum_con_lai / PLOTSIZE)
        else:
            light_avail = cp.exp(XT * cum_dec_lai / PLOTSIZE)
        if light_avail < 0.01:
            light_avail = 0.01
        if light_avail > 1.0:
            light_avail = 1.0
        fc_degday = 0.0
        if deg_days > deg_day_min and deg_days < deg_day_max:
            a = (deg_day_opt - deg_day_min) / (deg_day_max - deg_day_min)
            b = (deg_day_max - deg_day_opt) / (deg_day_max - deg_day_min)
            tmp1 = (deg_days - deg_day_min) / (deg_day_opt - deg_day_min)
            tmp2 = (deg_day_max - deg_days) / (deg_day_max - deg_day_opt)
            if tmp1 > 0.0 and tmp2 > 0.0:
                fc_degday = tmp1 ** a * tmp2 ** b
        fc_drought = 1.0
        drought_idx = drought_tol - 1
        if drought_idx < 0:
            drought_idx = 0
        if drought_idx > 5:
            drought_idx = 5
        gamma = 0.35
        if drought_idx == 0:
            gamma = 0.5
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
        if dry_days < gamma:
            tmp_d = (gamma - dry_days) / gamma
            fc_drought = tmp_d ** 0.5
        else:
            fc_drought = 0.0
        if drought_tol == 1:
            fcdry_base = 0.0
            if dry_days_base < 0.5:
                tmp_b = (0.5 - dry_days_base) / 0.5
                fcdry_base = tmp_b ** 0.5
            if evergreen > 0:
                fcdry_base = fcdry_base * 0.33
            else:
                fcdry_base = fcdry_base * 0.2
            if fcdry_base > fc_drought:
                fc_drought = fcdry_base
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
        fc_flood = 1.0
        env_stress = fc_degday * fc_drought * fc_light * fc_flood
        if env_stress < 0.0:
            env_stress = 0.0
        if env_stress > 1.0:
            env_stress = 1.0
        diam_max = 0.0
        if diam < max_diam and height < max_ht:
            delta_ht = max_ht - STD_HT
            exp_term = cp.exp(-arfa_0 * diam / delta_ht)
            denom = 2.0 * height + arfa_0 * exp_term * diam
            if denom > 0.001:
                diam_max = g * diam * (1.0 - diam * height / (max_diam * max_ht)) / denom
        pot_diam_increment = diam_max * env_stress
        if pot_diam_increment < 0.0:
            pot_diam_increment = 0.0
        pot_new_diam = diam + pot_diam_increment
        if pot_new_diam > max_diam:
            pot_new_diam = max_diam
        delta_ht = max_ht - STD_HT
        pot_new_height = STD_HT + delta_ht * (1.0 - cp.exp(-arfa_0 * pot_new_diam / delta_ht))
        pot_bd = pot_new_diam
        if pot_new_height > STD_HT:
            pot_bd = pot_new_height / (pot_new_height - STD_HT) * pot_new_diam
        pot_dc = pot_new_diam
        if pot_new_height > canopy_ht and pot_new_height > STD_HT:
            pot_dc = (pot_new_height - canopy_ht) / (pot_new_height - STD_HT) * pot_new_diam
        pot_stembc = TC_KG * wood_bulk_dens * 0.3 * pot_bd * pot_bd * pot_new_height
        pot_crown_depth = pot_new_height - canopy_ht
        if pot_crown_depth < 0.0:
            pot_crown_depth = 0.0
        pot_twigbc = TC_KG * wood_bulk_dens * 0.33667 * pot_dc * pot_dc * pot_crown_depth
        pot_root_c = 0.0
        if pot_new_height > 0.01:
            pot_root_c = pot_stembc * rootdepth / pot_new_height + pot_twigbc * 0.5
        pot_new_biomC = pot_stembc + pot_twigbc + pot_root_c
        pot_new_leaf_bm = pot_dc * pot_dc * leafdiam_a * leafarea_c * 2.0 * 1000.0
        biomC_increment = pot_new_biomC - biomC
        if biomC_increment < 0.0:
            biomC_increment = 0.0
        stem_n_demand = biomC_increment / STEM_C_N
        leaf_n_demand = 0.0
        if evergreen > 0:
            leaf_n_demand = (CON_LEAF_B * pot_new_leaf_bm - leaf_bm) / CON_LEAF_C_N
        else:
            leaf_n_demand = pot_new_leaf_bm / DEC_LEAF_C_N
        if leaf_n_demand < 0.0:
            leaf_n_demand = 0.0
        n_demand = stem_n_demand + leaf_n_demand
        params_tensor[agent_index][TreeP.LIGHT_AVAIL] = light_avail
        params_tensor[agent_index][TreeP.FC_DEGDAY] = fc_degday
        params_tensor[agent_index][TreeP.FC_DROUGHT] = fc_drought
        params_tensor[agent_index][TreeP.FC_FLOOD] = fc_flood
        states_tensor[agent_index][TreeS.ENV_STRESS] = env_stress
        params_tensor[agent_index][TreeP.DIAM_MAX_CALC] = diam_max
        params_tensor[agent_index][TreeP.FORSKA_SHADE] = forska_shade
    states_tensor[agent_index][TreeS.N_DEMAND] = n_demand

@jit.rawkernel(device='cuda')
def gap_demand_aggregate_step_double_buffer(tick, agent_index, _seed, species_traits, site_configs, rangelists, site_distances, agent_ids, logical_ids, breeds, neighbor_offsets, neighbor_values, params_tensor, states_tensor, gap_lai, gap_lai_idx, gap_avail_spec, gap_avail_spec_idx, gap_imported_seeds, gap_imported_seeds_idx, site_avail_spec, write_site_avail_spec, site_avail_spec_idx, site_imported_seeds, site_imported_seeds_idx):
    """
    Gap N demand aggregate + sync step (priority 4).

    Reads n_demand from living tree neighbors (written at P3, same tick).
    Computes per-gap N supply ratio using avail_n (from P2 climate relay)
    and own N demand. Clears accumulators consumed by P0/P1.
    """
    total_n_dem = 0.0
    i = 0
    while i < neighbor_offsets[agent_index + 1] - neighbor_offsets[agent_index]:
        neighbor_idx = int(neighbor_values[neighbor_offsets[agent_index] + i])
        neighbor_breed = int(breeds[neighbor_idx])
        if neighbor_breed == Breed.TREE:
            tree_alive = states_tensor[neighbor_idx][TreeS.IS_ALIVE]
            if tree_alive > 0.5:
                tree_n_demand = states_tensor[neighbor_idx][TreeS.N_DEMAND]
                total_n_dem = total_n_dem + tree_n_demand
        i = i + 1
    params_tensor[agent_index][GapP.TOTAL_N_DEMAND] = total_n_dem
    avail_n = states_tensor[agent_index][GapS.AVAIL_N]
    gap_n_demand_scaled = total_n_dem * UNIT_CONV
    gap_n_supply_ratio = 1.0
    if gap_n_demand_scaled > 1e-05:
        gap_n_supply_ratio = avail_n / gap_n_demand_scaled
        if gap_n_supply_ratio > 1.0:
            gap_n_supply_ratio = 1.0
    states_tensor[agent_index][GapS.N_SUPPLY_RATIO] = gap_n_supply_ratio
    states_tensor[agent_index][GapS.LITTER_ACCUM_C] = 0.0
    states_tensor[agent_index][GapS.LITTER_ACCUM_N] = 0.0
    states_tensor[agent_index][GapS.TOTAL_LAI] = 0.0
    states_tensor[agent_index][GapS.N_CONSUMED] = 0.0

@jit.rawkernel(device='cuda')
def tree_template_renewal_step_double_buffer(tick, agent_index, _seed, species_traits, site_configs, rangelists, site_distances, agent_ids, logical_ids, breeds, neighbor_offsets, neighbor_values, params_tensor, states_tensor, gap_lai, gap_lai_idx, gap_avail_spec, gap_avail_spec_idx, gap_imported_seeds, gap_imported_seeds_idx, site_avail_spec, write_site_avail_spec, site_avail_spec_idx, site_imported_seeds, site_imported_seeds_idx):
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
        i = 0
        while i < neighbor_offsets[agent_index + 1] - neighbor_offsets[agent_index]:
            neighbor_idx = int(neighbor_values[neighbor_offsets[agent_index] + i])
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
                num_to_recruit = states_tensor[neighbor_idx][GapS.NUM_TO_RECRUIT]
                total_seedling_weight = states_tensor[neighbor_idx][GapS.TOTAL_SEEDLING_WEIGHT]
                cum_dec_lai = gap_lai[gap_lai_idx[neighbor_idx]][1][0]
                cum_con_lai = gap_lai[gap_lai_idx[neighbor_idx]][1][1]
            i = i + 1
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
        seedbank = params_tensor[agent_index][TreeP.SEEDBANK]
        seedling = params_tensor[agent_index][TreeP.SEEDLING]
        fc_degday = 0.0
        if deg_days > deg_day_min and deg_days < deg_day_max:
            a = (deg_day_opt - deg_day_min) / (deg_day_max - deg_day_min)
            b = (deg_day_max - deg_day_opt) / (deg_day_max - deg_day_min)
            tmp1 = (deg_days - deg_day_min) / (deg_day_opt - deg_day_min)
            tmp2 = (deg_day_max - deg_days) / (deg_day_max - deg_day_opt)
            if tmp1 > 0.0 and tmp2 > 0.0:
                fc_degday = tmp1 ** a * tmp2 ** b
        fc_drought = 1.0
        drought_idx = drought_tol - 1
        if drought_idx < 0:
            drought_idx = 0
        if drought_idx > 5:
            drought_idx = 5
        gamma_d = 0.35
        if drought_idx == 0:
            gamma_d = 0.5
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
            if dry_days_base < 0.5:
                tmp_b = (0.5 - dry_days_base) / 0.5
                fcdry_base = tmp_b ** 0.5
            if evergreen > 0:
                fcdry_base = fcdry_base * 0.33
            else:
                fcdry_base = fcdry_base * 0.2
            if fcdry_base > fc_drought:
                fc_drought = fcdry_base
        fc_flood = 1.0
        lownutr_idx = lownutr_tol
        if lownutr_idx < 1:
            lownutr_idx = 1
        if lownutr_idx > 3:
            lownutr_idx = 3
        nrc = 4 - lownutr_idx
        nutr_c1 = -0.6274
        nutr_c2 = 3.6
        nutr_c3 = -1.994
        if nrc == 2:
            nutr_c1 = -0.2352
            nutr_c2 = 2.771
            nutr_c3 = -1.55
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
        light_avail = 1.0
        if evergreen > 0:
            light_avail = cp.exp(XT * cum_con_lai / PLOTSIZE)
        else:
            light_avail = cp.exp(XT * cum_dec_lai / PLOTSIZE)
        if light_avail < 0.01:
            light_avail = 0.01
        if light_avail > 1.0:
            light_avail = 1.0
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
        raw_regrowth = 0.0
        regrowth = 0.0
        if avail_n > 0.0:
            raw_regrowth = fc_degday * fc_drought * fc_flood * fc_nutrient * fc_light
            regrowth = raw_regrowth
            if regrowth <= 0.05:
                regrowth = 0.0
        num_species = len(species_traits)
        avail_spec = 0.0
        imported_seeds = 0.0
        species_idx = int(species_id)
        if gap_idx >= 0 and species_idx >= 0 and (species_idx < num_species):
            avail_spec = gap_avail_spec[gap_avail_spec_idx[gap_idx]][species_idx]
            imported_seeds = gap_imported_seeds[gap_imported_seeds_idx[gap_idx]][species_idx]
        weight = 0.0
        in_pipeline = 0
        if fire_intensity > 0.01:
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
        elif wind_intensity > 0.01:
            seedling = invader_val + seedling + sprout_val * avail_spec
        elif recovery_years > 1.5:
            regrowth = 0.0
        elif recovery_years > 0.5:
            regrowth = 0.0
            in_pipeline = 1
        else:
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
        if in_pipeline > 0:
            seedling = seedling * PLOTSIZE
            if fire_intensity < 0.01 and wind_intensity < 0.01:
                old_weight = states_tensor[agent_index][TreeS.SEEDLING_WEIGHT]
                if total_seedling_weight > 0.01 and num_to_recruit > 0.5:
                    my_share = num_to_recruit * old_weight / total_seedling_weight
                    seedling = seedling - my_share
                    if seedling < 0.0:
                        seedling = 0.0
        seedling = seedling * seedling_lg / PLOTSIZE
        params_tensor[agent_index][TreeP.SEEDBANK] = seedbank
        params_tensor[agent_index][TreeP.SEEDLING] = seedling
        states_tensor[agent_index][TreeS.ENV_STRESS] = regrowth
        states_tensor[agent_index][TreeS.SEEDLING_WEIGHT] = weight

@jit.rawkernel(device='cuda')
def gap_recruit_aggregate_step_double_buffer(tick, agent_index, _seed, species_traits, site_configs, rangelists, site_distances, agent_ids, logical_ids, breeds, neighbor_offsets, neighbor_values, params_tensor, states_tensor, gap_lai, gap_lai_idx, gap_avail_spec, gap_avail_spec_idx, gap_imported_seeds, gap_imported_seeds_idx, site_avail_spec, write_site_avail_spec, site_avail_spec_idx, site_imported_seeds, site_imported_seeds_idx):
    """
    Gap recruitment aggregate step (priority 6).

    Reads regrowth from template neighbors (written at P5, same tick).
    Computes num_to_recruit from growmax (GAPpy model.py:833-837).
    Writes recruitment info to Gap states for P7 free slots to read.
    """
    living_tree_count = 0.0
    free_slot_tree_count = 0.0
    growmax = 0.0
    i = 0
    while i < neighbor_offsets[agent_index + 1] - neighbor_offsets[agent_index]:
        neighbor_idx = int(neighbor_values[neighbor_offsets[agent_index] + i])
        neighbor_breed = int(breeds[neighbor_idx])
        if neighbor_breed == Breed.TREE:
            tree_alive = states_tensor[neighbor_idx][TreeS.IS_ALIVE]
            if tree_alive > 0.5:
                living_tree_count = living_tree_count + 1.0
            elif tree_alive > -0.5:
                free_slot_tree_count = free_slot_tree_count + 1.0
            else:
                template_regrowth = states_tensor[neighbor_idx][TreeS.ENV_STRESS]
                if template_regrowth > growmax:
                    growmax = template_regrowth
        i = i + 1
    num_to_recruit = 0.0
    if free_slot_tree_count > 0.5:
        max_renew = float(int(PLOTSIZE * growmax)) - living_tree_count
        half_cap = float(int(PLOTSIZE * 0.5))
        if max_renew > half_cap:
            max_renew = half_cap
        nrenew = max_renew
        if nrenew < 3.0:
            nrenew = 3.0
        cap = float(int(PLOTSIZE)) - living_tree_count
        if nrenew > cap:
            nrenew = cap
        if nrenew > free_slot_tree_count:
            nrenew = free_slot_tree_count
        if nrenew < 0.0:
            nrenew = 0.0
        num_to_recruit = nrenew
    recruit_prob = 0.0
    if num_to_recruit > 0.5 and free_slot_tree_count > 0.5:
        recruit_prob = num_to_recruit / free_slot_tree_count
    recruit_rand_seed = rand_uniform_philox(_seed, tick, logical_ids[agent_index], 9) * 10000.0
    recovery_years = states_tensor[agent_index][GapS.RECOVERY_YEARS]
    if recovery_years > 0.5:
        recruit_prob = 0.0
        states_tensor[agent_index][GapS.RECOVERY_YEARS] = recovery_years - 1.0
    states_tensor[agent_index][GapS.NUM_TO_RECRUIT] = recruit_prob
    states_tensor[agent_index][GapS.RECRUIT_RAND_SEED] = recruit_rand_seed

@jit.rawkernel(device='cuda')
def tree_actual_growth_step_double_buffer(tick, agent_index, _seed, species_traits, site_configs, rangelists, site_distances, agent_ids, logical_ids, breeds, neighbor_offsets, neighbor_values, params_tensor, states_tensor, gap_lai, gap_lai_idx, gap_avail_spec, gap_avail_spec_idx, gap_imported_seeds, gap_imported_seeds_idx, site_avail_spec, write_site_avail_spec, site_avail_spec_idx, site_imported_seeds, site_imported_seeds_idx):
    """
    Phase C (P7): Nutrient response + final growth + mortality + litter + free slot activation.

    Living trees: reads env_stress/diam_max from P2, n_supply_ratio from P4.
    Free slots: reads num_to_recruit from P6, seedling_weight from P5 templates.
    Templates: skipped.
    """
    is_alive = states_tensor[agent_index][TreeS.IS_ALIVE]
    litter_c = 0.0
    litter_n = 0.0
    n_consumed = 0.0
    if is_alive > 0.5:
        species_id = int(states_tensor[agent_index][TreeS.SPECIES_ID])
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
        diam = states_tensor[agent_index][TreeS.DIAM]
        height = states_tensor[agent_index][TreeS.HEIGHT]
        canopy_ht = states_tensor[agent_index][TreeS.CANOPY_HT]
        age = params_tensor[agent_index][TreeP.AGE]
        env_stress = states_tensor[agent_index][TreeS.ENV_STRESS]
        diam_max = params_tensor[agent_index][TreeP.DIAM_MAX_CALC]
        n_supply_ratio = 1.0
        fire_intensity = 0.0
        wind_intensity = 0.0
        i = 0
        while i < neighbor_offsets[agent_index + 1] - neighbor_offsets[agent_index]:
            neighbor_idx = int(neighbor_values[neighbor_offsets[agent_index] + i])
            neighbor_breed = int(breeds[neighbor_idx])
            if neighbor_breed == Breed.GAP:
                n_supply_ratio = states_tensor[neighbor_idx][GapS.N_SUPPLY_RATIO]
                fire_intensity = states_tensor[neighbor_idx][GapS.FIRE_INTENSITY]
                wind_intensity = states_tensor[neighbor_idx][GapS.WIND_INTENSITY]
            i = i + 1
        old_biomC = params_tensor[agent_index][TreeP.BIOMC]
        old_leaf_bm = params_tensor[agent_index][TreeP.LEAF_BM]
        lownutr_idx = lownutr_tol
        if lownutr_idx < 1:
            lownutr_idx = 1
        if lownutr_idx > 3:
            lownutr_idx = 3
        nrc = 4 - lownutr_idx
        c1 = -0.6274
        c2 = 3.6
        c3 = -1.994
        if nrc == 2:
            c1 = -0.2352
            c2 = 2.771
            c3 = -1.55
        elif nrc == 3:
            c1 = 0.2133
            c2 = 1.789
            c3 = -1.014
        sf = n_supply_ratio
        if sf < 0.0:
            sf = 0.0
        if sf > 1.0:
            sf = 1.0
        fpoor = c1 + c2 * sf + c3 * sf * sf
        if fpoor < 0.0:
            fpoor = 0.0
        if fpoor > 1.0:
            fpoor = 1.0
        fc_nutrient = fpoor * sf
        growth_factor = env_stress * fc_nutrient
        if growth_factor < 0.0:
            growth_factor = 0.0
        if growth_factor > 1.0:
            growth_factor = 1.0
        diam_increment = diam_max * growth_factor
        if diam_increment < 0.0:
            diam_increment = 0.0
        pp = max_diam / max_age * 0.1
        if pp > 0.05:
            pp = 0.05
        mort_marker = 0.0
        if diam_increment <= pp or growth_factor <= 0.05:
            mort_marker = 1.0
        new_diam = diam + diam_increment
        if new_diam > max_diam:
            new_diam = max_diam
        delta_ht = max_ht - STD_HT
        new_height = STD_HT + delta_ht * (1.0 - cp.exp(-arfa_0 * new_diam / delta_ht))
        growth_bd = new_diam
        if new_height > STD_HT:
            growth_bd = new_height / (new_height - STD_HT) * new_diam
        growth_dc = new_diam
        if new_height > canopy_ht and new_height > STD_HT:
            growth_dc = (new_height - canopy_ht) / (new_height - STD_HT) * new_diam
        growth_stembc = TC_KG * wood_bulk_dens * 0.3 * growth_bd * growth_bd * new_height
        growth_crown_depth = new_height - canopy_ht
        if growth_crown_depth < 0.0:
            growth_crown_depth = 0.0
        growth_twigbc = TC_KG * wood_bulk_dens * 0.33667 * growth_dc * growth_dc * growth_crown_depth
        growth_root_c = 0.0
        if new_height > 0.01:
            growth_root_c = growth_stembc * rootdepth / new_height + growth_twigbc * 0.5
        growth_biomC = growth_stembc + growth_twigbc + growth_root_c
        growth_leaf_bm = growth_dc * growth_dc * leafdiam_a * leafarea_c * 2.0 * 1000.0
        forska_shade = params_tensor[agent_index][TreeP.FORSKA_SHADE]
        fc_degday_val = params_tensor[agent_index][TreeP.FC_DEGDAY]
        fc_drought_val = params_tensor[agent_index][TreeP.FC_DROUGHT]
        fc_flood_val = params_tensor[agent_index][TreeP.FC_FLOOD]
        forska_check = fc_degday_val * fc_drought_val * fc_flood_val * forska_shade * fc_nutrient
        new_canopy_ht = canopy_ht
        if forska_check <= 0.05:
            canht_int = int(canopy_ht)
            forht_int = int(new_height)
            if canht_int + 1 < forht_int:
                new_canopy_ht = float(canht_int + 1) + 0.01
        new_stembc = growth_stembc
        new_dc = new_diam
        if new_height > new_canopy_ht and new_height > STD_HT:
            new_dc = (new_height - new_canopy_ht) / (new_height - STD_HT) * new_diam
        new_crown_depth = new_height - new_canopy_ht
        if new_crown_depth < 0.0:
            new_crown_depth = 0.0
        new_twigbc = TC_KG * wood_bulk_dens * 0.33667 * new_dc * new_dc * new_crown_depth
        new_root_c = 0.0
        if new_height > 0.01:
            new_root_c = new_stembc * rootdepth / new_height + new_twigbc * 0.5
        new_biomC = new_stembc + new_twigbc + new_root_c
        new_leaf_bm = new_dc * new_dc * leafdiam_a * leafarea_c * 2.0 * 1000.0
        d_bioC = growth_biomC - old_biomC
        n_consumed = d_bioC / STEM_C_N
        if evergreen > 0:
            prim_prod = CON_LEAF_B * growth_leaf_bm - old_leaf_bm
            n_consumed = n_consumed + prim_prod / CON_LEAF_C_N
        else:
            n_consumed = n_consumed + growth_leaf_bm / DEC_LEAF_C_N
        tree_dies = 0
        if fire_intensity > 0.01 or wind_intensity > 0.01:
            tree_dies = 1
        else:
            stress_tol = int(species_traits[species_id][Trait.STRESS_TOL])
            age_tol = int(species_traits[species_id][Trait.AGE_TOL])
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
            age_rand = rand_uniform_philox(_seed, tick, logical_ids[agent_index], 1)
            age_dies = 0
            if age_rand < age_mort_prob:
                age_dies = 1
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
                stress_check = 0.4
            if stress_k == 4:
                stress_check = 0.43
            stress_rand = rand_uniform_philox(_seed, tick, logical_ids[agent_index], 2)
            growth_dies = 0
            if mort_marker > 0.5 and stress_rand < stress_check:
                growth_dies = 1
            if age_dies > 0 or growth_dies > 0:
                tree_dies = 1
        canopy_litter_c = 0.0
        canopy_litter_n = 0.0
        d_bc = growth_biomC - new_biomC
        if d_bc > 0.0:
            canopy_litter_c = canopy_litter_c + d_bc
            canopy_litter_n = canopy_litter_n + d_bc / STEM_C_N
        d_leafb = growth_leaf_bm - new_leaf_bm
        if d_leafb > 0.0:
            if evergreen > 0:
                canopy_litter_c = canopy_litter_c + d_leafb * CON_LEAF_B
                canopy_litter_n = canopy_litter_n + d_leafb / CON_LEAF_C_N * CON_LEAF_B
            else:
                canopy_litter_c = canopy_litter_c + d_leafb
                canopy_litter_n = canopy_litter_n + d_leafb / DEC_LEAF_C_N
        if tree_dies > 0:
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
            is_alive = 1.0
            if evergreen > 0:
                litter_c = new_leaf_bm * (CON_LEAF_B - 1.0)
                litter_n = litter_c / CON_LEAF_C_N
            else:
                litter_c = new_leaf_bm
                litter_n = litter_c / DEC_LEAF_C_N
        litter_c = litter_c + canopy_litter_c
        litter_n = litter_n + canopy_litter_n
        new_age = age + 1.0
        params_tensor[agent_index][TreeP.AGE] = new_age
        params_tensor[agent_index][TreeP.BIOMC] = new_biomC
        params_tensor[agent_index][TreeP.BIOMN] = new_biomC / STEM_C_N
        params_tensor[agent_index][TreeP.LEAF_BM] = new_leaf_bm
        states_tensor[agent_index][TreeS.IS_ALIVE] = is_alive
        states_tensor[agent_index][TreeS.DIAM] = new_diam
        states_tensor[agent_index][TreeS.HEIGHT] = new_height
        states_tensor[agent_index][TreeS.CANOPY_HT] = new_canopy_ht
    elif is_alive > -0.5:
        recruit_prob = 0.0
        recruit_rand_seed = 0.0
        i = 0
        while i < neighbor_offsets[agent_index + 1] - neighbor_offsets[agent_index]:
            neighbor_idx = int(neighbor_values[neighbor_offsets[agent_index] + i])
            neighbor_breed = int(breeds[neighbor_idx])
            if neighbor_breed == Breed.GAP:
                recruit_prob = states_tensor[neighbor_idx][GapS.NUM_TO_RECRUIT]
                recruit_rand_seed = states_tensor[neighbor_idx][GapS.RECRUIT_RAND_SEED]
            i = i + 1
        if recruit_prob > 0.0:
            slot_priority = rand_uniform_philox(_seed, tick, logical_ids[agent_index], int(recruit_rand_seed) * 97 + 3)
            if slot_priority < recruit_prob:
                total_weight = 0.0
                i = 0
                while i < neighbor_offsets[agent_index + 1] - neighbor_offsets[agent_index]:
                    neighbor_idx = int(neighbor_values[neighbor_offsets[agent_index] + i])
                    if int(breeds[neighbor_idx]) == Breed.TREE:
                        if states_tensor[neighbor_idx][TreeS.IS_ALIVE] < -0.5:
                            total_weight = total_weight + states_tensor[neighbor_idx][TreeS.SEEDLING_WEIGHT]
                    i = i + 1
                selected_neighbor_idx = -1
                last_template_idx = -1
                if total_weight > 1e-10:
                    q0 = rand_uniform_philox(_seed, tick, logical_ids[agent_index], int(recruit_rand_seed) * 97 + 4)
                    cum_prob = 0.0
                    sp = 0
                    while sp < len(species_traits):
                        i = 0
                        while i < neighbor_offsets[agent_index + 1] - neighbor_offsets[agent_index]:
                            neighbor_idx = int(neighbor_values[neighbor_offsets[agent_index] + i])
                            if int(breeds[neighbor_idx]) == Breed.TREE:
                                if states_tensor[neighbor_idx][TreeS.IS_ALIVE] < -0.5:
                                    last_template_idx = neighbor_idx
                                    if int(states_tensor[neighbor_idx][TreeS.SPECIES_ID]) == sp:
                                        weight = states_tensor[neighbor_idx][TreeS.SEEDLING_WEIGHT]
                                        cum_prob = cum_prob + weight / total_weight
                                        if cum_prob > q0 and selected_neighbor_idx < 0:
                                            selected_neighbor_idx = neighbor_idx
                            i = i + 1
                        sp = sp + 1
                    if selected_neighbor_idx < 0 and last_template_idx >= 0:
                        selected_neighbor_idx = last_template_idx
                if selected_neighbor_idx >= 0:
                    states_tensor[agent_index][TreeS.SPECIES_ID] = states_tensor[selected_neighbor_idx][TreeS.SPECIES_ID]
                    sel_species_id = int(states_tensor[agent_index][TreeS.SPECIES_ID])
                    seedling_diam = 1.5 + rand_normal_bounded(_seed, tick, logical_ids[agent_index], int(recruit_rand_seed) * 97 + 5, -1.0, 1.0)
                    max_ht = species_traits[sel_species_id][Trait.MAX_HT]
                    arfa_0 = species_traits[sel_species_id][Trait.ARFA_0]
                    delta_ht = max_ht - STD_HT
                    seedling_ht = STD_HT + delta_ht * (1.0 - cp.exp(-arfa_0 * seedling_diam / delta_ht))
                    s_wbd = species_traits[sel_species_id][Trait.WOOD_BULK_DENS]
                    s_rdepth = species_traits[sel_species_id][Trait.ROOTDEPTH]
                    s_canopy_ht = 1.0
                    s_bd = seedling_diam
                    if seedling_ht > STD_HT:
                        s_bd = seedling_ht / (seedling_ht - STD_HT) * seedling_diam
                    s_dc = seedling_diam
                    if seedling_ht > s_canopy_ht and seedling_ht > STD_HT:
                        s_dc = (seedling_ht - s_canopy_ht) / (seedling_ht - STD_HT) * seedling_diam
                    s_stembc = TC_KG * s_wbd * 0.3 * s_bd * s_bd * seedling_ht
                    s_crown = seedling_ht - s_canopy_ht
                    if s_crown < 0.0:
                        s_crown = 0.0
                    s_twigbc = TC_KG * s_wbd * 0.33667 * s_dc * s_dc * s_crown
                    s_rootc = 0.0
                    if seedling_ht > 0.01:
                        s_rootc = s_stembc * s_rdepth / seedling_ht + s_twigbc * 0.5
                    seedling_biomC = s_stembc + s_twigbc + s_rootc
                    seedling_lda = species_traits[sel_species_id][Trait.LEAFDIAM_A]
                    seedling_lca = species_traits[sel_species_id][Trait.LEAFAREA_C]
                    seedling_leaf_bm = s_dc * s_dc * seedling_lda * seedling_lca * 2.0 * 1000.0
                    seedling_evergreen = int(species_traits[sel_species_id][Trait.EVERGREEN])
                    if seedling_evergreen > 0:
                        n_consumed = seedling_leaf_bm / CON_LEAF_C_N + seedling_biomC / STEM_C_N
                        litter_c = seedling_leaf_bm * (CON_LEAF_B - 1.0)
                        litter_n = litter_c / CON_LEAF_C_N
                    else:
                        n_consumed = seedling_biomC / STEM_C_N + seedling_leaf_bm / DEC_LEAF_C_N
                        litter_c = seedling_leaf_bm
                        litter_n = seedling_leaf_bm / DEC_LEAF_C_N
                    params_tensor[agent_index][TreeP.AGE] = SEEDLING_AGE
                    params_tensor[agent_index][TreeP.BIOMC] = seedling_biomC
                    params_tensor[agent_index][TreeP.BIOMN] = seedling_biomC / STEM_C_N
                    params_tensor[agent_index][TreeP.LEAF_BM] = 0.0
                    params_tensor[agent_index][TreeP.LIGHT_AVAIL] = 1.0
                    params_tensor[agent_index][TreeP.FC_DEGDAY] = 1.0
                    params_tensor[agent_index][TreeP.FC_DROUGHT] = 1.0
                    params_tensor[agent_index][TreeP.FC_FLOOD] = 1.0
                    states_tensor[agent_index][TreeS.DIAM] = seedling_diam
                    states_tensor[agent_index][TreeS.HEIGHT] = seedling_ht
                    states_tensor[agent_index][TreeS.CANOPY_HT] = s_canopy_ht
                    states_tensor[agent_index][TreeS.IS_ALIVE] = 1.0
    states_tensor[agent_index][TreeS.LITTER_C] = litter_c
    states_tensor[agent_index][TreeS.LITTER_N] = litter_n
    states_tensor[agent_index][TreeS.N_CONSUMED] = n_consumed

@jit.rawkernel(device='cuda')
def gap_nconsumed_aggregate_step_double_buffer(tick, agent_index, _seed, species_traits, site_configs, rangelists, site_distances, agent_ids, logical_ids, breeds, neighbor_offsets, neighbor_values, params_tensor, states_tensor, gap_lai, gap_lai_idx, gap_avail_spec, gap_avail_spec_idx, gap_imported_seeds, gap_imported_seeds_idx, site_avail_spec, write_site_avail_spec, site_avail_spec_idx, site_imported_seeds, site_imported_seeds_idx):
    """
    Gap N consumed aggregate step (priority 8).

    Reads n_consumed from tree neighbors (written at P7, same tick).
    Writes total to Gap states (Site reads at P9 for same-tick N balance).
    """
    total_n_consumed = 0.0
    i = 0
    while i < neighbor_offsets[agent_index + 1] - neighbor_offsets[agent_index]:
        neighbor_idx = int(neighbor_values[neighbor_offsets[agent_index] + i])
        neighbor_breed = int(breeds[neighbor_idx])
        if neighbor_breed == Breed.TREE:
            tree_alive = states_tensor[neighbor_idx][TreeS.IS_ALIVE]
            if tree_alive > -0.5:
                tree_n_consumed = states_tensor[neighbor_idx][TreeS.N_CONSUMED]
                total_n_consumed = total_n_consumed + tree_n_consumed
        i = i + 1
    states_tensor[agent_index][GapS.N_CONSUMED] = total_n_consumed

@jit.rawkernel(device='cuda')
def site_final_step_double_buffer(tick, agent_index, _seed, species_traits, site_configs, rangelists, site_distances, agent_ids, logical_ids, breeds, neighbor_offsets, neighbor_values, params_tensor, states_tensor, gap_lai, gap_lai_idx, gap_avail_spec, gap_avail_spec_idx, gap_imported_seeds, gap_imported_seeds_idx, site_avail_spec, write_site_avail_spec, site_avail_spec_idx, site_imported_seeds, site_imported_seeds_idx):
    """
    Site final step (priority 9).

    Part A - N Balance:
      Reads avail_N (P1 same tick), N consumed (P8 same tick), annual_runoff (P1 same tick).
      Applies surplus/deficit to A layer, leaching to base layer.
      Matches GAPpy model.py:993-1005.

    Part B - Seed Dispersal:
      Aggregates avail_spec from gap neighbors (for ghost export).
      Reads neighbor site ghost avail_spec (previous tick) and computes
      dispersal using negative exponential kernel.
      Writes imported_seeds to site_species for P2 gap relay next tick.
    """
    num_species = len(species_traits)
    own_site_id = int(params_tensor[agent_index][SiteP.SITE_ID])
    total_n_consumed = 0.0
    gap_count = 0.0
    site_neighbor_0 = -1
    site_neighbor_1 = -1
    site_neighbor_2 = -1
    site_neighbor_3 = -1
    num_site_neighbors = 0
    srow_avail = site_avail_spec_idx[agent_index]
    sp = 0
    while sp < num_species:
        write_site_avail_spec[srow_avail][sp] = 0.0
        sp = sp + 1
    i = 0
    while i < neighbor_offsets[agent_index + 1] - neighbor_offsets[agent_index]:
        neighbor_idx = int(neighbor_values[neighbor_offsets[agent_index] + i])
        neighbor_breed = int(breeds[neighbor_idx])
        if neighbor_breed == Breed.GAP:
            gap_count = gap_count + 1.0
            gap_n_consumed = states_tensor[neighbor_idx][GapS.N_CONSUMED]
            total_n_consumed = total_n_consumed + gap_n_consumed
            sp = 0
            while sp < num_species:
                gap_avail = gap_avail_spec[gap_avail_spec_idx[neighbor_idx]][sp]
                write_site_avail_spec[srow_avail][sp] = site_avail_spec[srow_avail][sp] + gap_avail
                sp = sp + 1
        elif neighbor_breed == Breed.SITE:
            if num_site_neighbors == 0:
                site_neighbor_0 = neighbor_idx
            elif num_site_neighbors == 1:
                site_neighbor_1 = neighbor_idx
            elif num_site_neighbors == 2:
                site_neighbor_2 = neighbor_idx
            elif num_site_neighbors == 3:
                site_neighbor_3 = neighbor_idx
            num_site_neighbors = num_site_neighbors + 1
        i = i + 1
    if gap_count > 0.5:
        total_n_consumed = total_n_consumed * UNIT_CONV / gap_count
    avail_n = states_tensor[agent_index][SiteS.AVAIL_N]
    annual_runoff = params_tensor[agent_index][SiteP.ANNUAL_RUNOFF]
    sa_n0 = params_tensor[agent_index][SiteP.A_N]
    sa_c0 = params_tensor[agent_index][SiteP.A_C]
    sb_c0 = params_tensor[agent_index][SiteP.BL_C]
    sb_n0 = params_tensor[agent_index][SiteP.BL_N]
    surplus = avail_n - total_n_consumed
    net_n_leach = 0.0
    if surplus > 0.0:
        leach_frac = annual_runoff / 1000.0
        if leach_frac > 0.1:
            leach_frac = 0.1
        net_n_leach = surplus * leach_frac
        sa_n0 = sa_n0 + surplus - net_n_leach
    else:
        sa_n0 = sa_n0 + surplus
        net_n_leach = 0.0
    sa_n0 = sa_n0 - 2e-05 * annual_runoff
    sa_c0 = sa_c0 - net_n_leach * 20.0
    sb_c0 = sb_c0 + net_n_leach * 20.0
    sb_n0 = sb_n0 + net_n_leach
    params_tensor[agent_index][SiteP.A_N] = sa_n0
    params_tensor[agent_index][SiteP.A_C] = sa_c0
    params_tensor[agent_index][SiteP.BL_C] = sb_c0
    params_tensor[agent_index][SiteP.BL_N] = sb_n0
    params_tensor[agent_index][SiteP.NET_N_INTO_A0] = net_n_leach
    if gap_count > 0.5:
        sp = 0
        while sp < num_species:
            write_site_avail_spec[srow_avail][sp] = site_avail_spec[srow_avail][sp] / gap_count
            sp = sp + 1
    srow_imported = site_imported_seeds_idx[agent_index]
    sp = 0
    while sp < num_species:
        site_imported_seeds[srow_imported][sp] = 0.0
        sp = sp + 1
    ns = 0
    while ns < num_site_neighbors:
        neighbor_idx = site_neighbor_0
        if ns == 1:
            neighbor_idx = site_neighbor_1
        elif ns == 2:
            neighbor_idx = site_neighbor_2
        elif ns == 3:
            neighbor_idx = site_neighbor_3
        if neighbor_idx >= 0:
            neighbor_site_id = int(states_tensor[neighbor_idx][SiteS.SITE_ID])
            distance = site_distances[own_site_id][neighbor_site_id]
            sp = 0
            while sp < num_species:
                neighbor_avail = site_avail_spec[site_avail_spec_idx[neighbor_idx]][sp]
                if neighbor_avail > 0.0:
                    in_range = rangelists[own_site_id][sp]
                    if in_range > 0.5:
                        max_disp = species_traits[sp][Trait.MAX_DISPERSAL_DIST]
                        if max_disp > 0.0:
                            weight = cp.exp(-distance / max_disp)
                            seed_num = species_traits[sp][Trait.SEED]
                            seed_import = seed_num * neighbor_avail * weight
                            site_imported_seeds[srow_imported][sp] = site_imported_seeds[srow_imported][sp] + seed_import
                sp = sp + 1
        ns = ns + 1
    if gap_count > 0.5:
        sp = 0
        while sp < num_species:
            site_imported_seeds[srow_imported][sp] = site_imported_seeds[srow_imported][sp] / gap_count
            sp = sp + 1

@jit.rawkernel(device='cuda')
def stepfunc(
global_tick,
_seed,
g0,g1,g2,g3,
a0,neighbor_offsets,neighbor_values,a2,a3,
sync_workers_every_n_ticks,
num_rank_local_agents,
priority_0_start,priority_0_count,priority_1_start,priority_1_count,priority_2_start,priority_2_count,priority_3_start,priority_3_count,priority_4_start,priority_4_count,priority_5_start,priority_5_count,priority_6_start,priority_6_count,priority_7_start,priority_7_count,priority_8_start,priority_8_count,priority_9_start,priority_9_count,
agent_ids,
logical_ids,
barrier_counter,
num_blocks_param,
gap_lai,
gap_lai_idx,
gap_avail_spec,
gap_avail_spec_idx,
gap_imported_seeds,
gap_imported_seeds_idx,
site_avail_spec,
write_site_avail_spec,
site_avail_spec_nrows,
site_avail_spec_idx,
site_imported_seeds,
site_imported_seeds_idx,
):
	thread_id = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
	total_threads = jit.gridDim.x * jit.blockDim.x
	barrier_id = 0

	for tick in range(sync_workers_every_n_ticks):
		thread_local_tick = int(global_tick) + tick

		agent_index = thread_id
		while agent_index < priority_0_count:
			_real_idx = int(agent_index) + int(priority_0_start)
			breed_id = a0[_real_idx]
			if breed_id == 1:
				gap_litter_aggregate_step_double_buffer(
					thread_local_tick,
					_real_idx,
					_seed,
					g0,g1,g2,g3,
					agent_ids,
					logical_ids,
					a0,neighbor_offsets,neighbor_values,a2,a3,
					gap_lai,gap_lai_idx,gap_avail_spec,gap_avail_spec_idx,gap_imported_seeds,gap_imported_seeds_idx,site_avail_spec,write_site_avail_spec,site_avail_spec_idx,site_imported_seeds,site_imported_seeds_idx,
				)
			agent_index = agent_index + total_threads

		jit.syncthreads()
		if jit.threadIdx.x == 0:
			jit.threadfence()
			jit.atomic_add(barrier_counter, 0, 1)
			_barrier_target = (barrier_id + 1) * num_blocks_param
			while jit.atomic_add(barrier_counter, 0, 0) < _barrier_target:
				pass
			jit.threadfence()
		jit.syncthreads()
		barrier_id = barrier_id + 1
		agent_index = thread_id
		while agent_index < priority_1_count:
			_real_idx = int(agent_index) + int(priority_1_start)
			breed_id = a0[_real_idx]
			if breed_id == 2:
				site_soil_step_double_buffer(
					thread_local_tick,
					_real_idx,
					_seed,
					g0,g1,g2,g3,
					agent_ids,
					logical_ids,
					a0,neighbor_offsets,neighbor_values,a2,a3,
					gap_lai,gap_lai_idx,gap_avail_spec,gap_avail_spec_idx,gap_imported_seeds,gap_imported_seeds_idx,site_avail_spec,write_site_avail_spec,site_avail_spec_idx,site_imported_seeds,site_imported_seeds_idx,
				)
			agent_index = agent_index + total_threads

		jit.syncthreads()
		if jit.threadIdx.x == 0:
			jit.threadfence()
			jit.atomic_add(barrier_counter, 0, 1)
			_barrier_target = (barrier_id + 1) * num_blocks_param
			while jit.atomic_add(barrier_counter, 0, 0) < _barrier_target:
				pass
			jit.threadfence()
		jit.syncthreads()
		barrier_id = barrier_id + 1
		agent_index = thread_id
		while agent_index < priority_2_count:
			_real_idx = int(agent_index) + int(priority_2_start)
			breed_id = a0[_real_idx]
			if breed_id == 1:
				gap_climate_relay_step_double_buffer(
					thread_local_tick,
					_real_idx,
					_seed,
					g0,g1,g2,g3,
					agent_ids,
					logical_ids,
					a0,neighbor_offsets,neighbor_values,a2,a3,
					gap_lai,gap_lai_idx,gap_avail_spec,gap_avail_spec_idx,gap_imported_seeds,gap_imported_seeds_idx,site_avail_spec,write_site_avail_spec,site_avail_spec_idx,site_imported_seeds,site_imported_seeds_idx,
				)
			agent_index = agent_index + total_threads

		jit.syncthreads()
		if jit.threadIdx.x == 0:
			jit.threadfence()
			jit.atomic_add(barrier_counter, 0, 1)
			_barrier_target = (barrier_id + 1) * num_blocks_param
			while jit.atomic_add(barrier_counter, 0, 0) < _barrier_target:
				pass
			jit.threadfence()
		jit.syncthreads()
		barrier_id = barrier_id + 1
		agent_index = thread_id
		while agent_index < priority_3_count:
			_real_idx = int(agent_index) + int(priority_3_start)
			breed_id = a0[_real_idx]
			if breed_id == 0:
				tree_potential_growth_step_double_buffer(
					thread_local_tick,
					_real_idx,
					_seed,
					g0,g1,g2,g3,
					agent_ids,
					logical_ids,
					a0,neighbor_offsets,neighbor_values,a2,a3,
					gap_lai,gap_lai_idx,gap_avail_spec,gap_avail_spec_idx,gap_imported_seeds,gap_imported_seeds_idx,site_avail_spec,write_site_avail_spec,site_avail_spec_idx,site_imported_seeds,site_imported_seeds_idx,
				)
			agent_index = agent_index + total_threads

		jit.syncthreads()
		if jit.threadIdx.x == 0:
			jit.threadfence()
			jit.atomic_add(barrier_counter, 0, 1)
			_barrier_target = (barrier_id + 1) * num_blocks_param
			while jit.atomic_add(barrier_counter, 0, 0) < _barrier_target:
				pass
			jit.threadfence()
		jit.syncthreads()
		barrier_id = barrier_id + 1
		agent_index = thread_id
		while agent_index < priority_4_count:
			_real_idx = int(agent_index) + int(priority_4_start)
			breed_id = a0[_real_idx]
			if breed_id == 1:
				gap_demand_aggregate_step_double_buffer(
					thread_local_tick,
					_real_idx,
					_seed,
					g0,g1,g2,g3,
					agent_ids,
					logical_ids,
					a0,neighbor_offsets,neighbor_values,a2,a3,
					gap_lai,gap_lai_idx,gap_avail_spec,gap_avail_spec_idx,gap_imported_seeds,gap_imported_seeds_idx,site_avail_spec,write_site_avail_spec,site_avail_spec_idx,site_imported_seeds,site_imported_seeds_idx,
				)
			agent_index = agent_index + total_threads

		jit.syncthreads()
		if jit.threadIdx.x == 0:
			jit.threadfence()
			jit.atomic_add(barrier_counter, 0, 1)
			_barrier_target = (barrier_id + 1) * num_blocks_param
			while jit.atomic_add(barrier_counter, 0, 0) < _barrier_target:
				pass
			jit.threadfence()
		jit.syncthreads()
		barrier_id = barrier_id + 1
		agent_index = thread_id
		while agent_index < priority_5_count:
			_real_idx = int(agent_index) + int(priority_5_start)
			breed_id = a0[_real_idx]
			if breed_id == 0:
				tree_template_renewal_step_double_buffer(
					thread_local_tick,
					_real_idx,
					_seed,
					g0,g1,g2,g3,
					agent_ids,
					logical_ids,
					a0,neighbor_offsets,neighbor_values,a2,a3,
					gap_lai,gap_lai_idx,gap_avail_spec,gap_avail_spec_idx,gap_imported_seeds,gap_imported_seeds_idx,site_avail_spec,write_site_avail_spec,site_avail_spec_idx,site_imported_seeds,site_imported_seeds_idx,
				)
			agent_index = agent_index + total_threads

		jit.syncthreads()
		if jit.threadIdx.x == 0:
			jit.threadfence()
			jit.atomic_add(barrier_counter, 0, 1)
			_barrier_target = (barrier_id + 1) * num_blocks_param
			while jit.atomic_add(barrier_counter, 0, 0) < _barrier_target:
				pass
			jit.threadfence()
		jit.syncthreads()
		barrier_id = barrier_id + 1
		agent_index = thread_id
		while agent_index < priority_6_count:
			_real_idx = int(agent_index) + int(priority_6_start)
			breed_id = a0[_real_idx]
			if breed_id == 1:
				gap_recruit_aggregate_step_double_buffer(
					thread_local_tick,
					_real_idx,
					_seed,
					g0,g1,g2,g3,
					agent_ids,
					logical_ids,
					a0,neighbor_offsets,neighbor_values,a2,a3,
					gap_lai,gap_lai_idx,gap_avail_spec,gap_avail_spec_idx,gap_imported_seeds,gap_imported_seeds_idx,site_avail_spec,write_site_avail_spec,site_avail_spec_idx,site_imported_seeds,site_imported_seeds_idx,
				)
			agent_index = agent_index + total_threads

		jit.syncthreads()
		if jit.threadIdx.x == 0:
			jit.threadfence()
			jit.atomic_add(barrier_counter, 0, 1)
			_barrier_target = (barrier_id + 1) * num_blocks_param
			while jit.atomic_add(barrier_counter, 0, 0) < _barrier_target:
				pass
			jit.threadfence()
		jit.syncthreads()
		barrier_id = barrier_id + 1
		agent_index = thread_id
		while agent_index < priority_7_count:
			_real_idx = int(agent_index) + int(priority_7_start)
			breed_id = a0[_real_idx]
			if breed_id == 0:
				tree_actual_growth_step_double_buffer(
					thread_local_tick,
					_real_idx,
					_seed,
					g0,g1,g2,g3,
					agent_ids,
					logical_ids,
					a0,neighbor_offsets,neighbor_values,a2,a3,
					gap_lai,gap_lai_idx,gap_avail_spec,gap_avail_spec_idx,gap_imported_seeds,gap_imported_seeds_idx,site_avail_spec,write_site_avail_spec,site_avail_spec_idx,site_imported_seeds,site_imported_seeds_idx,
				)
			agent_index = agent_index + total_threads

		jit.syncthreads()
		if jit.threadIdx.x == 0:
			jit.threadfence()
			jit.atomic_add(barrier_counter, 0, 1)
			_barrier_target = (barrier_id + 1) * num_blocks_param
			while jit.atomic_add(barrier_counter, 0, 0) < _barrier_target:
				pass
			jit.threadfence()
		jit.syncthreads()
		barrier_id = barrier_id + 1
		agent_index = thread_id
		while agent_index < priority_8_count:
			_real_idx = int(agent_index) + int(priority_8_start)
			breed_id = a0[_real_idx]
			if breed_id == 1:
				gap_nconsumed_aggregate_step_double_buffer(
					thread_local_tick,
					_real_idx,
					_seed,
					g0,g1,g2,g3,
					agent_ids,
					logical_ids,
					a0,neighbor_offsets,neighbor_values,a2,a3,
					gap_lai,gap_lai_idx,gap_avail_spec,gap_avail_spec_idx,gap_imported_seeds,gap_imported_seeds_idx,site_avail_spec,write_site_avail_spec,site_avail_spec_idx,site_imported_seeds,site_imported_seeds_idx,
				)
			agent_index = agent_index + total_threads

		jit.syncthreads()
		if jit.threadIdx.x == 0:
			jit.threadfence()
			jit.atomic_add(barrier_counter, 0, 1)
			_barrier_target = (barrier_id + 1) * num_blocks_param
			while jit.atomic_add(barrier_counter, 0, 0) < _barrier_target:
				pass
			jit.threadfence()
		jit.syncthreads()
		barrier_id = barrier_id + 1
		agent_index = thread_id
		while agent_index < priority_9_count:
			_real_idx = int(agent_index) + int(priority_9_start)
			breed_id = a0[_real_idx]
			if breed_id == 2:
				site_final_step_double_buffer(
					thread_local_tick,
					_real_idx,
					_seed,
					g0,g1,g2,g3,
					agent_ids,
					logical_ids,
					a0,neighbor_offsets,neighbor_values,a2,a3,
					gap_lai,gap_lai_idx,gap_avail_spec,gap_avail_spec_idx,gap_imported_seeds,gap_imported_seeds_idx,site_avail_spec,write_site_avail_spec,site_avail_spec_idx,site_imported_seeds,site_imported_seeds_idx,
				)
			agent_index = agent_index + total_threads

		jit.syncthreads()
		if jit.threadIdx.x == 0:
			jit.threadfence()
			jit.atomic_add(barrier_counter, 0, 1)
			_barrier_target = (barrier_id + 1) * num_blocks_param
			while jit.atomic_add(barrier_counter, 0, 0) < _barrier_target:
				pass
			jit.threadfence()
		jit.syncthreads()
		barrier_id = barrier_id + 1

		_bla_row = thread_id
		while _bla_row < site_avail_spec_nrows:
			for _bla_j in range(235):
				site_avail_spec[_bla_row][_bla_j] = write_site_avail_spec[_bla_row][_bla_j]
			_bla_row = _bla_row + total_threads

		jit.syncthreads()
		if jit.threadIdx.x == 0:
			jit.threadfence()
			jit.atomic_add(barrier_counter, 0, 1)
			_barrier_target = (barrier_id + 1) * num_blocks_param
			while jit.atomic_add(barrier_counter, 0, 0) < _barrier_target:
				pass
			jit.threadfence()
		jit.syncthreads()
		barrier_id = barrier_id + 1