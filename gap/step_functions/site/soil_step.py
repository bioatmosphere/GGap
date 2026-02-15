"""
Site step function for GGap model (Priority 1).
GPU kernel implementing UVAFME soil biogeochemistry.

Matches UVAFME's bio_geo_climate() and process_soil_biogeochemistry():
- Daily climate interpolation from monthly data
- Soil water balance (soil_water function)
- Three-layer decomposition (A0 -> A -> Base)
- Atmospheric N deposition from precipitation
- Fire probability calculation

Execution Flow:
    1. READ LITTER FROM GAPS
       - Sum litter_accum_c/n from all Gap neighbors
       - Convert to daily inputs (/ 365)

    2. DAILY LOOP (365 iterations)
       For each day:

       a. Climate Interpolation
          - Determine month from day of year
          - Get daily tmin, tmax, precip from monthly values
          - Track freeze days
          - Accumulate atmospheric N from rain

       b. Potential Evapotranspiration (Hamon method)
          - Calculate solar declination
          - Calculate day length from latitude
          - Calculate PET from temperature and day length

       c. Soil Water Balance
          - Route precipitation: canopy -> A0 -> A -> Base
          - Apply slope runoff
          - Evapotranspiration draws from pools
          - Track flood days (A layer saturated)

       d. Soil Decomposition
          - A0 respiration -> transfer to A layer
          - A layer respiration -> N mineralization (avail_n)
          - A layer transfer to Base layer
          - Base layer respiration
          - Temperature and moisture adjustments

    3. FIRE PROBABILITY
       - Base 1% annual probability
       - Increases with dry soil moisture
       - Cap at 15%
       - Stochastic fire occurrence
       - Fire intensity 0.3-1.0

Soil Layers:
    A0 (Litter):  Fresh organic matter, fast decomposition
    A  (Humus):   Decomposed organic matter, N mineralization source
    Base:         Stable organic matter, slow turnover

Property scheme (3 properties):
- params[116]: soil pools + monthly climate + site properties + fire/wind/soil + climate_std - private
- states[6]: climate + avail_n + flood_days + fire_intensity + n_supply_ratio - public
- states_db[1]: placeholder (public, double buffered but unused)
"""

import cupy as cp
from cupyx import jit

# === Breed IDs ===
BREED_TREE = 0
BREED_GAP = 1
BREED_SITE = 2

# === Site params[116] (private) ===
# Soil pools [0-8]:
SITE_P_A0_C = 0
SITE_P_A0_N = 1
SITE_P_A_C = 2
SITE_P_A_N = 3
SITE_P_BL_C = 4
SITE_P_BL_N = 5
SITE_P_A0_W = 6
SITE_P_A_W = 7
SITE_P_BL_W = 8
# Monthly climate [9-44]:
SITE_P_TMIN_BASE = 9   # tmin[0..11] at indices 9-20
SITE_P_TMAX_BASE = 21  # tmax[0..11] at indices 21-32
SITE_P_PRCP_BASE = 33  # prcp[0..11] at indices 33-44
# Additional soil/site params [45-52]:
SITE_P_FIELD_CAP = 45
SITE_P_PERM_WP = 46
SITE_P_SLOPE = 47
SITE_P_SIGMA = 48
SITE_P_LAI = 49
SITE_P_LAI_W0 = 50
SITE_P_LATITUDE = 51
SITE_P_RAIN_N = 52
# Fire/wind/soil [53-55]:
SITE_P_FIRE_PROB = 53
SITE_P_WIND_PROB = 54
SITE_P_BASE_H = 55
# Climate standard deviations [56-91]:
SITE_P_TMIN_STD_BASE = 56   # tmin_std[0..11] at 56-67
SITE_P_TMAX_STD_BASE = 68   # tmax_std[0..11] at 68-79
SITE_P_PRCP_STD_BASE = 80   # prcp_std[0..11] at 80-91

# === Site states[6] (public) ===
SITE_S_DEG_DAYS = 0
SITE_S_DRY_DAYS = 1
SITE_S_AVAIL_N = 2
SITE_S_FLOOD_DAYS = 3  # Days when A layer is saturated
SITE_S_FIRE_INTENSITY = 4  # Fire intensity this year (0-1)

# === Gap states[14] (for reading from Gap neighbors) ===
GAP_S_LITTER_ACCUM_C = 4       # Above-ground litter -> A0 layer
GAP_S_LITTER_ACCUM_N = 5
GAP_S_LITTER_ACCUM_C_BG = 12   # Below-ground litter -> A layer (roots)
GAP_S_LITTER_ACCUM_N_BG = 13

# === UVAFME Constants (from soil.py) ===
AO_CN_0 = 30.0
SA_CN_0 = 4.0
SB_CN_0 = 20.0
AO_RESP = 5.24e-4
SA_RESP = 1.24e-5
SB_RESP = 2.74e-7

BASE_MAX = 0.6
BASE_MIN = 0.1
AO_MIN = 0.025
AO_MAX = 0.25
LAI_MIN = 0.01
LAI_MAX = 0.15

# Atmospheric N in precipitation (tn N per cm precip)
PRCP_N = 0.00002

# Days per month for interpolation
DAYS_PER_MONTH_0 = 31
DAYS_PER_MONTH_1 = 28
DAYS_PER_MONTH_2 = 31
DAYS_PER_MONTH_3 = 30
DAYS_PER_MONTH_4 = 31
DAYS_PER_MONTH_5 = 30
DAYS_PER_MONTH_6 = 31
DAYS_PER_MONTH_7 = 31
DAYS_PER_MONTH_8 = 30
DAYS_PER_MONTH_9 = 31
DAYS_PER_MONTH_10 = 30
DAYS_PER_MONTH_11 = 31

PI = 3.14159265359


@jit.rawkernel(device="cuda")
def site_soil_step(
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

    Writes to own:
    - params: soil pools (A0/A/BL carbon, nitrogen, water)
    - states: avail_n (for Gap to read)
    """
    # ========== READ LITTER FROM GAP NEIGHBORS ==========
    total_litter_c = 0.0       # Above-ground -> A0 layer
    total_litter_n = 0.0
    total_litter_c_bg = 0.0    # Below-ground -> A layer (roots)
    total_litter_n_bg = 0.0

    neighbor_indices = locations[agent_index]
    i = 0
    while i < len(neighbor_indices) and neighbor_indices[i] != -1:
        neighbor_idx = int(neighbor_indices[i])
        neighbor_breed = int(breeds[neighbor_idx])

        if neighbor_breed == BREED_GAP:
            gap_litter_c = states_tensor[neighbor_idx][GAP_S_LITTER_ACCUM_C]
            gap_litter_n = states_tensor[neighbor_idx][GAP_S_LITTER_ACCUM_N]
            total_litter_c = total_litter_c + gap_litter_c
            total_litter_n = total_litter_n + gap_litter_n

            gap_litter_c_bg = states_tensor[neighbor_idx][GAP_S_LITTER_ACCUM_C_BG]
            gap_litter_n_bg = states_tensor[neighbor_idx][GAP_S_LITTER_ACCUM_N_BG]
            total_litter_c_bg = total_litter_c_bg + gap_litter_c_bg
            total_litter_n_bg = total_litter_n_bg + gap_litter_n_bg

        i = i + 1

    # ========== READ CURRENT SOIL STATE ==========
    ao_c0 = params_tensor[agent_index][SITE_P_A0_C]
    ao_n0 = params_tensor[agent_index][SITE_P_A0_N]
    sa_c0 = params_tensor[agent_index][SITE_P_A_C]
    sa_n0 = params_tensor[agent_index][SITE_P_A_N]
    sb_c0 = params_tensor[agent_index][SITE_P_BL_C]
    sb_n0 = params_tensor[agent_index][SITE_P_BL_N]
    ao_w0 = params_tensor[agent_index][SITE_P_A0_W]
    sa_w0 = params_tensor[agent_index][SITE_P_A_W]
    sb_w0 = params_tensor[agent_index][SITE_P_BL_W]

    # Read site properties
    sa_fc = params_tensor[agent_index][SITE_P_FIELD_CAP]
    sa_pwp = params_tensor[agent_index][SITE_P_PERM_WP]
    slope = params_tensor[agent_index][SITE_P_SLOPE]
    sigma = params_tensor[agent_index][SITE_P_SIGMA]
    lai = params_tensor[agent_index][SITE_P_LAI]
    lai_w0 = params_tensor[agent_index][SITE_P_LAI_W0]
    latitude = params_tensor[agent_index][SITE_P_LATITUDE]

    # ========== ADD LITTER AS ANNUAL PULSE (matches GAPpy) ==========
    ao_c0 = ao_c0 + total_litter_c       # Above-ground litter -> A0 layer
    ao_n0 = ao_n0 + total_litter_n
    sa_c0 = sa_c0 + total_litter_c_bg    # Below-ground litter -> A layer (roots)
    sa_n0 = sa_n0 + total_litter_n_bg

    # Ensure minimum LAI
    if lai < 1.0:
        lai = 1.0

    # ========== READ MONTHLY CLIMATE STD DEVS ==========
    tmin_std_0 = params_tensor[agent_index][SITE_P_TMIN_STD_BASE + 0]
    tmin_std_1 = params_tensor[agent_index][SITE_P_TMIN_STD_BASE + 1]
    tmin_std_2 = params_tensor[agent_index][SITE_P_TMIN_STD_BASE + 2]
    tmin_std_3 = params_tensor[agent_index][SITE_P_TMIN_STD_BASE + 3]
    tmin_std_4 = params_tensor[agent_index][SITE_P_TMIN_STD_BASE + 4]
    tmin_std_5 = params_tensor[agent_index][SITE_P_TMIN_STD_BASE + 5]
    tmin_std_6 = params_tensor[agent_index][SITE_P_TMIN_STD_BASE + 6]
    tmin_std_7 = params_tensor[agent_index][SITE_P_TMIN_STD_BASE + 7]
    tmin_std_8 = params_tensor[agent_index][SITE_P_TMIN_STD_BASE + 8]
    tmin_std_9 = params_tensor[agent_index][SITE_P_TMIN_STD_BASE + 9]
    tmin_std_10 = params_tensor[agent_index][SITE_P_TMIN_STD_BASE + 10]
    tmin_std_11 = params_tensor[agent_index][SITE_P_TMIN_STD_BASE + 11]

    tmax_std_0 = params_tensor[agent_index][SITE_P_TMAX_STD_BASE + 0]
    tmax_std_1 = params_tensor[agent_index][SITE_P_TMAX_STD_BASE + 1]
    tmax_std_2 = params_tensor[agent_index][SITE_P_TMAX_STD_BASE + 2]
    tmax_std_3 = params_tensor[agent_index][SITE_P_TMAX_STD_BASE + 3]
    tmax_std_4 = params_tensor[agent_index][SITE_P_TMAX_STD_BASE + 4]
    tmax_std_5 = params_tensor[agent_index][SITE_P_TMAX_STD_BASE + 5]
    tmax_std_6 = params_tensor[agent_index][SITE_P_TMAX_STD_BASE + 6]
    tmax_std_7 = params_tensor[agent_index][SITE_P_TMAX_STD_BASE + 7]
    tmax_std_8 = params_tensor[agent_index][SITE_P_TMAX_STD_BASE + 8]
    tmax_std_9 = params_tensor[agent_index][SITE_P_TMAX_STD_BASE + 9]
    tmax_std_10 = params_tensor[agent_index][SITE_P_TMAX_STD_BASE + 10]
    tmax_std_11 = params_tensor[agent_index][SITE_P_TMAX_STD_BASE + 11]

    prcp_std_0 = params_tensor[agent_index][SITE_P_PRCP_STD_BASE + 0]
    prcp_std_1 = params_tensor[agent_index][SITE_P_PRCP_STD_BASE + 1]
    prcp_std_2 = params_tensor[agent_index][SITE_P_PRCP_STD_BASE + 2]
    prcp_std_3 = params_tensor[agent_index][SITE_P_PRCP_STD_BASE + 3]
    prcp_std_4 = params_tensor[agent_index][SITE_P_PRCP_STD_BASE + 4]
    prcp_std_5 = params_tensor[agent_index][SITE_P_PRCP_STD_BASE + 5]
    prcp_std_6 = params_tensor[agent_index][SITE_P_PRCP_STD_BASE + 6]
    prcp_std_7 = params_tensor[agent_index][SITE_P_PRCP_STD_BASE + 7]
    prcp_std_8 = params_tensor[agent_index][SITE_P_PRCP_STD_BASE + 8]
    prcp_std_9 = params_tensor[agent_index][SITE_P_PRCP_STD_BASE + 9]
    prcp_std_10 = params_tensor[agent_index][SITE_P_PRCP_STD_BASE + 10]
    prcp_std_11 = params_tensor[agent_index][SITE_P_PRCP_STD_BASE + 11]

    # ========== READ MONTHLY CLIMATE ==========
    tmin_0 = params_tensor[agent_index][SITE_P_TMIN_BASE + 0]
    tmin_1 = params_tensor[agent_index][SITE_P_TMIN_BASE + 1]
    tmin_2 = params_tensor[agent_index][SITE_P_TMIN_BASE + 2]
    tmin_3 = params_tensor[agent_index][SITE_P_TMIN_BASE + 3]
    tmin_4 = params_tensor[agent_index][SITE_P_TMIN_BASE + 4]
    tmin_5 = params_tensor[agent_index][SITE_P_TMIN_BASE + 5]
    tmin_6 = params_tensor[agent_index][SITE_P_TMIN_BASE + 6]
    tmin_7 = params_tensor[agent_index][SITE_P_TMIN_BASE + 7]
    tmin_8 = params_tensor[agent_index][SITE_P_TMIN_BASE + 8]
    tmin_9 = params_tensor[agent_index][SITE_P_TMIN_BASE + 9]
    tmin_10 = params_tensor[agent_index][SITE_P_TMIN_BASE + 10]
    tmin_11 = params_tensor[agent_index][SITE_P_TMIN_BASE + 11]

    tmax_0 = params_tensor[agent_index][SITE_P_TMAX_BASE + 0]
    tmax_1 = params_tensor[agent_index][SITE_P_TMAX_BASE + 1]
    tmax_2 = params_tensor[agent_index][SITE_P_TMAX_BASE + 2]
    tmax_3 = params_tensor[agent_index][SITE_P_TMAX_BASE + 3]
    tmax_4 = params_tensor[agent_index][SITE_P_TMAX_BASE + 4]
    tmax_5 = params_tensor[agent_index][SITE_P_TMAX_BASE + 5]
    tmax_6 = params_tensor[agent_index][SITE_P_TMAX_BASE + 6]
    tmax_7 = params_tensor[agent_index][SITE_P_TMAX_BASE + 7]
    tmax_8 = params_tensor[agent_index][SITE_P_TMAX_BASE + 8]
    tmax_9 = params_tensor[agent_index][SITE_P_TMAX_BASE + 9]
    tmax_10 = params_tensor[agent_index][SITE_P_TMAX_BASE + 10]
    tmax_11 = params_tensor[agent_index][SITE_P_TMAX_BASE + 11]

    prcp_0 = params_tensor[agent_index][SITE_P_PRCP_BASE + 0]
    prcp_1 = params_tensor[agent_index][SITE_P_PRCP_BASE + 1]
    prcp_2 = params_tensor[agent_index][SITE_P_PRCP_BASE + 2]
    prcp_3 = params_tensor[agent_index][SITE_P_PRCP_BASE + 3]
    prcp_4 = params_tensor[agent_index][SITE_P_PRCP_BASE + 4]
    prcp_5 = params_tensor[agent_index][SITE_P_PRCP_BASE + 5]
    prcp_6 = params_tensor[agent_index][SITE_P_PRCP_BASE + 6]
    prcp_7 = params_tensor[agent_index][SITE_P_PRCP_BASE + 7]
    prcp_8 = params_tensor[agent_index][SITE_P_PRCP_BASE + 8]
    prcp_9 = params_tensor[agent_index][SITE_P_PRCP_BASE + 9]
    prcp_10 = params_tensor[agent_index][SITE_P_PRCP_BASE + 10]
    prcp_11 = params_tensor[agent_index][SITE_P_PRCP_BASE + 11]

    # ========== MONTHLY CLIMATE PERTURBATION (matches GAPpy) ==========
    # Generate one pseudo-random perturbation per month and apply to monthly
    # climate variables. Temp perturbation clamped to [-1,1], precip to [-0.5,0.5].
    u1 = 0.0
    u2 = 0.0
    u3 = 0.0
    u4 = 0.0
    tp = 0.0
    pp = 0.0

    # Month 0 (January)
    u1 = ((tick * 7919 + 0 * 6271 + agent_index * 1013) % 10000) / 10000.0
    u2 = ((tick * 5381 + 0 * 3571 + agent_index * 2017) % 10000) / 10000.0
    tp = u1 + u2 - 1.0
    if tp < -1.0:
        tp = -1.0
    if tp > 1.0:
        tp = 1.0
    u3 = ((tick * 4219 + 0 * 8461 + agent_index * 3037) % 10000) / 10000.0
    u4 = ((tick * 3079 + 0 * 7517 + agent_index * 5039) % 10000) / 10000.0
    pp = (u3 + u4 - 1.0) * 0.5
    if pp < -0.5:
        pp = -0.5
    if pp > 0.5:
        pp = 0.5
    tmin_0 = tmin_0 + tp * tmin_std_0
    tmax_0 = tmax_0 + tp * tmax_std_0
    if tmax_0 < tmin_0:
        tmax_0 = tmin_0 + 0.1
    prcp_0 = prcp_0 + pp * prcp_std_0
    if prcp_0 < 0.0:
        prcp_0 = 0.0

    # Month 1 (February)
    u1 = ((tick * 7919 + 1 * 6271 + agent_index * 1013) % 10000) / 10000.0
    u2 = ((tick * 5381 + 1 * 3571 + agent_index * 2017) % 10000) / 10000.0
    tp = u1 + u2 - 1.0
    if tp < -1.0:
        tp = -1.0
    if tp > 1.0:
        tp = 1.0
    u3 = ((tick * 4219 + 1 * 8461 + agent_index * 3037) % 10000) / 10000.0
    u4 = ((tick * 3079 + 1 * 7517 + agent_index * 5039) % 10000) / 10000.0
    pp = (u3 + u4 - 1.0) * 0.5
    if pp < -0.5:
        pp = -0.5
    if pp > 0.5:
        pp = 0.5
    tmin_1 = tmin_1 + tp * tmin_std_1
    tmax_1 = tmax_1 + tp * tmax_std_1
    if tmax_1 < tmin_1:
        tmax_1 = tmin_1 + 0.1
    prcp_1 = prcp_1 + pp * prcp_std_1
    if prcp_1 < 0.0:
        prcp_1 = 0.0

    # Month 2 (March)
    u1 = ((tick * 7919 + 2 * 6271 + agent_index * 1013) % 10000) / 10000.0
    u2 = ((tick * 5381 + 2 * 3571 + agent_index * 2017) % 10000) / 10000.0
    tp = u1 + u2 - 1.0
    if tp < -1.0:
        tp = -1.0
    if tp > 1.0:
        tp = 1.0
    u3 = ((tick * 4219 + 2 * 8461 + agent_index * 3037) % 10000) / 10000.0
    u4 = ((tick * 3079 + 2 * 7517 + agent_index * 5039) % 10000) / 10000.0
    pp = (u3 + u4 - 1.0) * 0.5
    if pp < -0.5:
        pp = -0.5
    if pp > 0.5:
        pp = 0.5
    tmin_2 = tmin_2 + tp * tmin_std_2
    tmax_2 = tmax_2 + tp * tmax_std_2
    if tmax_2 < tmin_2:
        tmax_2 = tmin_2 + 0.1
    prcp_2 = prcp_2 + pp * prcp_std_2
    if prcp_2 < 0.0:
        prcp_2 = 0.0

    # Month 3 (April)
    u1 = ((tick * 7919 + 3 * 6271 + agent_index * 1013) % 10000) / 10000.0
    u2 = ((tick * 5381 + 3 * 3571 + agent_index * 2017) % 10000) / 10000.0
    tp = u1 + u2 - 1.0
    if tp < -1.0:
        tp = -1.0
    if tp > 1.0:
        tp = 1.0
    u3 = ((tick * 4219 + 3 * 8461 + agent_index * 3037) % 10000) / 10000.0
    u4 = ((tick * 3079 + 3 * 7517 + agent_index * 5039) % 10000) / 10000.0
    pp = (u3 + u4 - 1.0) * 0.5
    if pp < -0.5:
        pp = -0.5
    if pp > 0.5:
        pp = 0.5
    tmin_3 = tmin_3 + tp * tmin_std_3
    tmax_3 = tmax_3 + tp * tmax_std_3
    if tmax_3 < tmin_3:
        tmax_3 = tmin_3 + 0.1
    prcp_3 = prcp_3 + pp * prcp_std_3
    if prcp_3 < 0.0:
        prcp_3 = 0.0

    # Month 4 (May)
    u1 = ((tick * 7919 + 4 * 6271 + agent_index * 1013) % 10000) / 10000.0
    u2 = ((tick * 5381 + 4 * 3571 + agent_index * 2017) % 10000) / 10000.0
    tp = u1 + u2 - 1.0
    if tp < -1.0:
        tp = -1.0
    if tp > 1.0:
        tp = 1.0
    u3 = ((tick * 4219 + 4 * 8461 + agent_index * 3037) % 10000) / 10000.0
    u4 = ((tick * 3079 + 4 * 7517 + agent_index * 5039) % 10000) / 10000.0
    pp = (u3 + u4 - 1.0) * 0.5
    if pp < -0.5:
        pp = -0.5
    if pp > 0.5:
        pp = 0.5
    tmin_4 = tmin_4 + tp * tmin_std_4
    tmax_4 = tmax_4 + tp * tmax_std_4
    if tmax_4 < tmin_4:
        tmax_4 = tmin_4 + 0.1
    prcp_4 = prcp_4 + pp * prcp_std_4
    if prcp_4 < 0.0:
        prcp_4 = 0.0

    # Month 5 (June)
    u1 = ((tick * 7919 + 5 * 6271 + agent_index * 1013) % 10000) / 10000.0
    u2 = ((tick * 5381 + 5 * 3571 + agent_index * 2017) % 10000) / 10000.0
    tp = u1 + u2 - 1.0
    if tp < -1.0:
        tp = -1.0
    if tp > 1.0:
        tp = 1.0
    u3 = ((tick * 4219 + 5 * 8461 + agent_index * 3037) % 10000) / 10000.0
    u4 = ((tick * 3079 + 5 * 7517 + agent_index * 5039) % 10000) / 10000.0
    pp = (u3 + u4 - 1.0) * 0.5
    if pp < -0.5:
        pp = -0.5
    if pp > 0.5:
        pp = 0.5
    tmin_5 = tmin_5 + tp * tmin_std_5
    tmax_5 = tmax_5 + tp * tmax_std_5
    if tmax_5 < tmin_5:
        tmax_5 = tmin_5 + 0.1
    prcp_5 = prcp_5 + pp * prcp_std_5
    if prcp_5 < 0.0:
        prcp_5 = 0.0

    # Month 6 (July)
    u1 = ((tick * 7919 + 6 * 6271 + agent_index * 1013) % 10000) / 10000.0
    u2 = ((tick * 5381 + 6 * 3571 + agent_index * 2017) % 10000) / 10000.0
    tp = u1 + u2 - 1.0
    if tp < -1.0:
        tp = -1.0
    if tp > 1.0:
        tp = 1.0
    u3 = ((tick * 4219 + 6 * 8461 + agent_index * 3037) % 10000) / 10000.0
    u4 = ((tick * 3079 + 6 * 7517 + agent_index * 5039) % 10000) / 10000.0
    pp = (u3 + u4 - 1.0) * 0.5
    if pp < -0.5:
        pp = -0.5
    if pp > 0.5:
        pp = 0.5
    tmin_6 = tmin_6 + tp * tmin_std_6
    tmax_6 = tmax_6 + tp * tmax_std_6
    if tmax_6 < tmin_6:
        tmax_6 = tmin_6 + 0.1
    prcp_6 = prcp_6 + pp * prcp_std_6
    if prcp_6 < 0.0:
        prcp_6 = 0.0

    # Month 7 (August)
    u1 = ((tick * 7919 + 7 * 6271 + agent_index * 1013) % 10000) / 10000.0
    u2 = ((tick * 5381 + 7 * 3571 + agent_index * 2017) % 10000) / 10000.0
    tp = u1 + u2 - 1.0
    if tp < -1.0:
        tp = -1.0
    if tp > 1.0:
        tp = 1.0
    u3 = ((tick * 4219 + 7 * 8461 + agent_index * 3037) % 10000) / 10000.0
    u4 = ((tick * 3079 + 7 * 7517 + agent_index * 5039) % 10000) / 10000.0
    pp = (u3 + u4 - 1.0) * 0.5
    if pp < -0.5:
        pp = -0.5
    if pp > 0.5:
        pp = 0.5
    tmin_7 = tmin_7 + tp * tmin_std_7
    tmax_7 = tmax_7 + tp * tmax_std_7
    if tmax_7 < tmin_7:
        tmax_7 = tmin_7 + 0.1
    prcp_7 = prcp_7 + pp * prcp_std_7
    if prcp_7 < 0.0:
        prcp_7 = 0.0

    # Month 8 (September)
    u1 = ((tick * 7919 + 8 * 6271 + agent_index * 1013) % 10000) / 10000.0
    u2 = ((tick * 5381 + 8 * 3571 + agent_index * 2017) % 10000) / 10000.0
    tp = u1 + u2 - 1.0
    if tp < -1.0:
        tp = -1.0
    if tp > 1.0:
        tp = 1.0
    u3 = ((tick * 4219 + 8 * 8461 + agent_index * 3037) % 10000) / 10000.0
    u4 = ((tick * 3079 + 8 * 7517 + agent_index * 5039) % 10000) / 10000.0
    pp = (u3 + u4 - 1.0) * 0.5
    if pp < -0.5:
        pp = -0.5
    if pp > 0.5:
        pp = 0.5
    tmin_8 = tmin_8 + tp * tmin_std_8
    tmax_8 = tmax_8 + tp * tmax_std_8
    if tmax_8 < tmin_8:
        tmax_8 = tmin_8 + 0.1
    prcp_8 = prcp_8 + pp * prcp_std_8
    if prcp_8 < 0.0:
        prcp_8 = 0.0

    # Month 9 (October)
    u1 = ((tick * 7919 + 9 * 6271 + agent_index * 1013) % 10000) / 10000.0
    u2 = ((tick * 5381 + 9 * 3571 + agent_index * 2017) % 10000) / 10000.0
    tp = u1 + u2 - 1.0
    if tp < -1.0:
        tp = -1.0
    if tp > 1.0:
        tp = 1.0
    u3 = ((tick * 4219 + 9 * 8461 + agent_index * 3037) % 10000) / 10000.0
    u4 = ((tick * 3079 + 9 * 7517 + agent_index * 5039) % 10000) / 10000.0
    pp = (u3 + u4 - 1.0) * 0.5
    if pp < -0.5:
        pp = -0.5
    if pp > 0.5:
        pp = 0.5
    tmin_9 = tmin_9 + tp * tmin_std_9
    tmax_9 = tmax_9 + tp * tmax_std_9
    if tmax_9 < tmin_9:
        tmax_9 = tmin_9 + 0.1
    prcp_9 = prcp_9 + pp * prcp_std_9
    if prcp_9 < 0.0:
        prcp_9 = 0.0

    # Month 10 (November)
    u1 = ((tick * 7919 + 10 * 6271 + agent_index * 1013) % 10000) / 10000.0
    u2 = ((tick * 5381 + 10 * 3571 + agent_index * 2017) % 10000) / 10000.0
    tp = u1 + u2 - 1.0
    if tp < -1.0:
        tp = -1.0
    if tp > 1.0:
        tp = 1.0
    u3 = ((tick * 4219 + 10 * 8461 + agent_index * 3037) % 10000) / 10000.0
    u4 = ((tick * 3079 + 10 * 7517 + agent_index * 5039) % 10000) / 10000.0
    pp = (u3 + u4 - 1.0) * 0.5
    if pp < -0.5:
        pp = -0.5
    if pp > 0.5:
        pp = 0.5
    tmin_10 = tmin_10 + tp * tmin_std_10
    tmax_10 = tmax_10 + tp * tmax_std_10
    if tmax_10 < tmin_10:
        tmax_10 = tmin_10 + 0.1
    prcp_10 = prcp_10 + pp * prcp_std_10
    if prcp_10 < 0.0:
        prcp_10 = 0.0

    # Month 11 (December)
    u1 = ((tick * 7919 + 11 * 6271 + agent_index * 1013) % 10000) / 10000.0
    u2 = ((tick * 5381 + 11 * 3571 + agent_index * 2017) % 10000) / 10000.0
    tp = u1 + u2 - 1.0
    if tp < -1.0:
        tp = -1.0
    if tp > 1.0:
        tp = 1.0
    u3 = ((tick * 4219 + 11 * 8461 + agent_index * 3037) % 10000) / 10000.0
    u4 = ((tick * 3079 + 11 * 7517 + agent_index * 5039) % 10000) / 10000.0
    pp = (u3 + u4 - 1.0) * 0.5
    if pp < -0.5:
        pp = -0.5
    if pp > 0.5:
        pp = 0.5
    tmin_11 = tmin_11 + tp * tmin_std_11
    tmax_11 = tmax_11 + tp * tmax_std_11
    if tmax_11 < tmin_11:
        tmax_11 = tmin_11 + 0.1
    prcp_11 = prcp_11 + pp * prcp_std_11
    if prcp_11 < 0.0:
        prcp_11 = 0.0

    # ========== COMPUTE ANNUAL PRECIP AND DRY_DAYS (matches GAPpy) ==========
    total_prcp_mm = prcp_0 + prcp_1 + prcp_2 + prcp_3 + prcp_4 + prcp_5 + prcp_6 + prcp_7 + prcp_8 + prcp_9 + prcp_10 + prcp_11
    annual_prcp_cm = total_prcp_mm / 10.0
    dry_days = 100.0 - annual_prcp_cm
    if dry_days < 0.0:
        dry_days = 0.0

    # ========== WATER BALANCE LIMITS ==========
    sbh = params_tensor[agent_index][SITE_P_BASE_H]
    if sbh < 1.0:
        sbh = 70.0  # Fallback if not set
    laiw_min = lai * LAI_MIN
    laiw_max = lai * LAI_MAX
    aow_min = ao_c0 * AO_MIN
    aow_max = ao_c0 * AO_MAX
    sbw_min = sbh * BASE_MIN
    sbw_max = sbh * BASE_MAX

    # Latitude in radians for PET calculation
    lat_rad = latitude * PI / 180.0

    # ========== DAILY LOOP (365 DAYS) ==========
    total_avail_n = 0.0
    total_resp = 0.0
    rain_n = 0.0
    freeze_days = 0.0
    flood_days = 0.0  # Days when A layer is saturated

    day = 0
    while day < 365:
        # === Determine month and interpolate daily climate ===
        # Using simple month assignment based on day of year
        month = 0
        day_in_month = day

        if day < 31:
            month = 0
            day_in_month = day
        elif day < 59:
            month = 1
            day_in_month = day - 31
        elif day < 90:
            month = 2
            day_in_month = day - 59
        elif day < 120:
            month = 3
            day_in_month = day - 90
        elif day < 151:
            month = 4
            day_in_month = day - 120
        elif day < 181:
            month = 5
            day_in_month = day - 151
        elif day < 212:
            month = 6
            day_in_month = day - 181
        elif day < 243:
            month = 7
            day_in_month = day - 212
        elif day < 273:
            month = 8
            day_in_month = day - 243
        elif day < 304:
            month = 9
            day_in_month = day - 273
        elif day < 334:
            month = 10
            day_in_month = day - 304
        else:
            month = 11
            day_in_month = day - 334

        # Get monthly values (unrolled due to CuPy JIT limitations)
        day_tmin = 0.0
        day_tmax = 0.0
        day_prcp = 0.0

        if month == 0:
            day_tmin = tmin_0
            day_tmax = tmax_0
            day_prcp = prcp_0 / 31.0
        elif month == 1:
            day_tmin = tmin_1
            day_tmax = tmax_1
            day_prcp = prcp_1 / 28.0
        elif month == 2:
            day_tmin = tmin_2
            day_tmax = tmax_2
            day_prcp = prcp_2 / 31.0
        elif month == 3:
            day_tmin = tmin_3
            day_tmax = tmax_3
            day_prcp = prcp_3 / 30.0
        elif month == 4:
            day_tmin = tmin_4
            day_tmax = tmax_4
            day_prcp = prcp_4 / 31.0
        elif month == 5:
            day_tmin = tmin_5
            day_tmax = tmax_5
            day_prcp = prcp_5 / 30.0
        elif month == 6:
            day_tmin = tmin_6
            day_tmax = tmax_6
            day_prcp = prcp_6 / 31.0
        elif month == 7:
            day_tmin = tmin_7
            day_tmax = tmax_7
            day_prcp = prcp_7 / 31.0
        elif month == 8:
            day_tmin = tmin_8
            day_tmax = tmax_8
            day_prcp = prcp_8 / 30.0
        elif month == 9:
            day_tmin = tmin_9
            day_tmax = tmax_9
            day_prcp = prcp_9 / 31.0
        elif month == 10:
            day_tmin = tmin_10
            day_tmax = tmax_10
            day_prcp = prcp_10 / 30.0
        else:
            day_tmin = tmin_11
            day_tmax = tmax_11
            day_prcp = prcp_11 / 31.0

        day_temp = (day_tmin + day_tmax) / 2.0

        # Track freeze days
        if day_temp < 0.0:
            freeze_days = freeze_days + 1.0

        # Accumulate atmospheric N from precipitation (mm units)
        rain_n = rain_n + day_prcp * PRCP_N

        # Convert mm -> cm for water balance
        day_prcp = day_prcp / 10.0

        # === Calculate Potential Evapotranspiration (Hamon method) ===
        day_of_year = day + 1

        # Solar declination
        decl = 0.409 * cp.sin(2.0 * PI * day_of_year / 365.0 - 1.39)

        # Day length calculation
        # Note: cp.acos not available in CuPy JIT, use polynomial approximation
        # acos(x) ≈ PI/2 - x*(1 + x^2*(1/6 + x^2*3/40)) for |x| < 1
        cos_arg = -cp.tan(lat_rad) * cp.tan(decl)
        if cos_arg < -1.0:
            cos_arg = -1.0
        if cos_arg > 1.0:
            cos_arg = 1.0

        # Polynomial approximation for acos (accurate to ~0.01 radians)
        # acos(x) = PI/2 - asin(x), and asin(x) ≈ x + x^3/6 + 3x^5/40 + ...
        x2 = cos_arg * cos_arg
        asin_approx = cos_arg * (1.0 + x2 * (0.16666667 + x2 * 0.075))
        ws = 1.5707963 - asin_approx  # PI/2 - asin(cos_arg) = acos(cos_arg)
        daylength_hours = 24.0 * ws / PI

        # Potential evapotranspiration
        pot_ev_day = 0.0
        if day_temp > 0.0:
            sat_vap_density = 0.622 * 6.108 * cp.exp(17.27 * day_temp / (237.3 + day_temp))
            pot_ev_day = 0.1651 * daylength_hours * sat_vap_density / (day_temp + 273.3)

        # === SOIL WATER BALANCE (from UVAFME soil.py:soil_water) ===
        freeze = freeze_days / (day + 1.0)  # Running freeze fraction

        # Update water limits based on current A0 carbon
        aow_min = ao_c0 * AO_MIN
        aow_max = ao_c0 * AO_MAX
        if aow_max < 0.01:
            aow_max = 0.01
        if aow_min < 0.001:
            aow_min = 0.001

        # Water table recharge from rain
        if day_prcp > 0.01:
            table_water = day_prcp * sigma * freeze
            sb_w0 = sb_w0 + table_water
            if sb_w0 > sbw_max:
                sb_w0 = sbw_max

        # Water balance calculations
        act_ev_day = 0.0
        runoff = 0.0

        if pot_ev_day <= 0.0:
            # No evaporation - water accumulates
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
            # Evaporation occurs - complex water routing
            lai_loss = laiw_max - lai_w0
            if lai_loss > day_prcp:
                lai_loss = day_prcp
            if lai_loss < 0.0:
                lai_loss = 0.0

            yxd1 = day_prcp - lai_loss
            if yxd1 < 0.0:
                yxd1 = 0.0
            laiw = lai_w0 + lai_loss

            # Slope runoff
            yxd = (slope / 90.0) * (slope / 90.0)
            lossslp = yxd * yxd1
            yxd2 = yxd1 - lossslp - pot_ev_day

            if yxd2 > 0.0:
                # Excess water - infiltration
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
                # Water deficit - draw from pools
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

            lai_w0 = laiw

        # Calculate moisture scaling factors
        aow0_scaled = ao_w0 / aow_max
        saw0_scaled = 0.5
        if sa_fc > 0.0:
            saw0_scaled = sa_w0 / sa_fc

        # Track flood days: when A layer is near field capacity (saturated)
        # UVAFME considers flooding when soil water is at or above field capacity
        if sa_fc > 0.0 and sa_w0 >= sa_fc * 0.95:
            flood_days = flood_days + 1.0

        # === SOIL DECOMPOSITION (from UVAFME soil.py:soil_decomp) ===
        # A0 layer C/N ratio
        ao_cn = AO_CN_0
        if ao_n0 > 0.0001:
            ao_cn = ao_c0 / ao_n0

        # Moisture function for A0 (clamp to valid range)
        if aow0_scaled > 0.5:
            aow0_scaled = 0.5
        aofunc = 1.0 - (1.0 - aow0_scaled / 0.3) * (1.0 - aow0_scaled / 0.3)
        if aofunc < 0.2:
            aofunc = 0.2

        # Temperature adjustment
        tadjst = 0.0
        tadjst1 = 0.0
        if day_temp >= -5.0:
            tadjst = cp.power(3.0, 0.1 * (day_temp - 1.0))
            tadjst1 = cp.power(2.5, 0.1 * (day_temp - 1.0))

        # A0 layer respiration
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

        # Soil A layer (receives A0 decomposition products)
        sa_c0 = sa_c0 + yxdc
        sa_n0 = sa_n0 + yxdn
        sa_cn = SA_CN_0
        if sa_n0 > 0.0001:
            sa_cn = sa_c0 / sa_n0

        # A layer moisture function
        safunc = 1.0 - (1.0 - saw0_scaled / 0.8) * (1.0 - saw0_scaled / 0.8)
        if safunc < 0.2:
            safunc = 0.2

        resp2 = tadjst1 * safunc * SA_RESP * sa_c0

        # N mineralization from A layer
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

        # Base layer
        sb_c0 = sb_c0 + tosb
        resp3 = sb_c0 * SB_RESP * tadjst1
        sb_c0 = sb_c0 - resp3
        if sb_c0 < 0.0:
            sb_c0 = 0.0

        # Accumulate totals
        total_avail_n = total_avail_n + avail_n_day
        total_resp = total_resp + resp1 + resp2 + resp3

        day = day + 1

    # ========== WRITE FINAL RESULTS ==========
    # Soil pools (params - private)
    params_tensor[agent_index][SITE_P_A0_C] = ao_c0
    params_tensor[agent_index][SITE_P_A0_N] = ao_n0
    params_tensor[agent_index][SITE_P_A_C] = sa_c0
    params_tensor[agent_index][SITE_P_A_N] = sa_n0
    params_tensor[agent_index][SITE_P_BL_C] = sb_c0
    params_tensor[agent_index][SITE_P_BL_N] = sb_n0
    params_tensor[agent_index][SITE_P_A0_W] = ao_w0
    params_tensor[agent_index][SITE_P_A_W] = sa_w0
    params_tensor[agent_index][SITE_P_BL_W] = sb_w0
    params_tensor[agent_index][SITE_P_LAI_W0] = lai_w0

    # Available N = mineralization + atmospheric deposition
    states_tensor[agent_index][SITE_S_AVAIL_N] = total_avail_n + rain_n

    # Flood days for this year
    states_tensor[agent_index][SITE_S_FLOOD_DAYS] = flood_days

    # Dry days recomputed from perturbed annual precipitation
    states_tensor[agent_index][SITE_S_DRY_DAYS] = dry_days

    # ========== FIRE PROBABILITY ==========
    # Fire probability from CSV (per 1000 years, already converted to annual at load)
    fire_prob = params_tensor[agent_index][SITE_P_FIRE_PROB]

    # Dry conditions increase fire risk beyond base probability
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
        fire_prob = 0.15  # Cap at 15% annual probability

    # Stochastic fire check (deterministic based on tick)
    fire_rand = ((tick * 1021 + agent_index * 1019) % 10000) / 10000.0
    fire_intensity = 0.0
    if fire_rand < fire_prob:
        # Fire occurs - intensity based on dry conditions
        fire_intensity = 0.3 + fire_rand * 2.0  # 0.3 to ~0.7 intensity
        if fire_intensity > 1.0:
            fire_intensity = 1.0

    states_tensor[agent_index][SITE_S_FIRE_INTENSITY] = fire_intensity
