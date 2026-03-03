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
- states[16]: climate + avail_n + flood_days + fire/wind + stochastic climate + soil outputs - public
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
# N balance state [92]:
SITE_P_ANNUAL_RUNOFF = 92   # Annual accumulated runoff (for N balance at P10 same tick)

# === Site states[16] (public) ===
SITE_S_DEG_DAYS = 0
SITE_S_DRY_DAYS = 1
SITE_S_AVAIL_N = 2
SITE_S_FLOOD_DAYS = 3  # Days when A layer is saturated
SITE_S_FIRE_INTENSITY = 4  # Fire intensity this year (0-1)
SITE_S_DRY_DAYS_BASE = 6   # Base layer drought fraction (for intolerant species)
SITE_S_WIND_INTENSITY = 7  # Wind intensity this year (0-1, 0=no wind)
SITE_S_ANNUAL_RAIN = 8     # Perturbed annual rainfall (cm)
SITE_S_GROW_DAYS = 9       # Growing season days (tavg >= 5°C)
SITE_S_POT_EVAP = 10       # Annual potential evapotranspiration (cm)
SITE_S_ACT_EVAP = 11       # Annual actual evapotranspiration (cm)
SITE_S_SOIL_RESP = 12      # Annual soil respiration (tn C/ha)
SITE_S_C_INTO_A0 = 13      # Annual C litter input to A0 layer (tn C/ha)
SITE_S_N_INTO_A0 = 14      # Annual N litter input to A0 layer (tn N/ha)
SITE_S_NET_N_INTO_A0 = 15  # Net N leached to base layer (tn N/ha)

# === Gap states[16] (for reading from Gap neighbors) ===
GAP_S_LITTER_ACCUM_C = 4       # Above-ground litter -> A0 layer
GAP_S_LITTER_ACCUM_N = 5
GAP_S_TOTAL_N_DEMAND = 11      # N demand from trees (written at P4, used by P5 for N ratio)
GAP_S_TOTAL_LAI = 12           # Per-gap normalized LAI (from P0, GAPpy canopy())
GAP_S_N_CONSUMED = 13          # N consumed by trees (now aggregated at P9, read at P10)

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

# Unit conversion: kg (tree-level) → tn/ha (soil pools)
# = HEC_TO_M2 / plotsize / 1000 = 10000 / 500 / 1000
# Divide by gap_count at runtime (= /numplots equivalent)
UNIT_CONV = 0.02

# Days per month (for precip distribution)
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
DEG2RAD = 0.017453

# Hargreaves PET constants (GAPpy constants.py)
H_B = 0.017214        # ≈ 2*PI/365
H_AS = 0.409           # Max solar declination (radians)
H_AC = 0.033           # Eccentricity correction amplitude
H_PHASE = -1.39        # Phase offset for declination
H_AMP = 37.58603       # Radiation amplitude
H_COEFF = 0.000093876  # Hargreaves coefficient
H_ADDON = 17.8         # Hargreaves temperature offset


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
    # ========== READ LITTER + LAI FROM GAP NEIGHBORS ==========
    total_litter_c = 0.0       # Above-ground -> A0 layer
    total_litter_n = 0.0
    total_gap_lai = 0.0        # Sum of per-gap normalized LAI
    gap_count = 0.0

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

            # Read per-gap normalized LAI (GAPpy canopy():362)
            gap_lai = states_tensor[neighbor_idx][GAP_S_TOTAL_LAI]
            total_gap_lai = total_gap_lai + gap_lai

            gap_count = gap_count + 1.0

        i = i + 1

    # ========== CONVERT TREE KG → SOIL TN/HA (GAPpy uconvert) ==========
    if gap_count > 0.5:
        uconv = UNIT_CONV / gap_count
        total_litter_c = total_litter_c * uconv
        total_litter_n = total_litter_n * uconv

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
    lai_w0 = params_tensor[agent_index][SITE_P_LAI_W0]
    latitude = params_tensor[agent_index][SITE_P_LATITUDE]

    # N balance moved to site_nbalance_step (P10) — runs AFTER P8 growth,
    # so it uses same-tick avail_N and n_consumed (no 1-tick delay).

    # ========== DYNAMIC LAI FROM TREE CANOPY (GAPpy canopy():282-362) ==========
    # Average per-gap normalized LAI across gaps (= /numplots equivalent)
    lai = params_tensor[agent_index][SITE_P_LAI]  # Fallback: initial CSV value
    if gap_count > 0.5:
        dynamic_lai = total_gap_lai / gap_count
        if dynamic_lai > 0.01:
            lai = dynamic_lai

    # ========== ADD LITTER AS ANNUAL PULSE (matches GAPpy) ==========
    ao_c0 = ao_c0 + total_litter_c       # Above-ground litter -> A0 layer
    ao_n0 = ao_n0 + total_litter_n

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

    # ========== MONTHLY CLIMATE PERTURBATION ==========
    # Beasley-Springer-Moro (BSM) inverse normal CDF approximation applied to
    # hash-based uniform RNG. Matches GAPpy's Box-Muller normal + clamp approach.
    # Hash uniform mapped to [0.08, 0.92] (BSM central region), then rational
    # approximation gives ~N(0,1). Clamped to [-1,1] for temp, [-0.5,0.5] for precip.
    u_t = 0.0
    u_p = 0.0
    q_t = 0.0
    q_p = 0.0
    r_t = 0.0
    r_p = 0.0
    numer = 0.0
    denom = 0.0
    tp = 0.0
    pp = 0.0

    # Month 0 (January)
    u_t = ((tick * 7919 + 0 * 6271 + agent_index * 1013) % 1000000) / 1000000.0
    u_t = 0.08 + u_t * 0.84
    q_t = u_t - 0.5
    r_t = q_t * q_t
    numer = q_t * (2.50662824 + r_t * (-18.6150006 + r_t * (41.3911977 + r_t * (-25.4410605))))
    denom = 1.0 + r_t * (-8.47351093 + r_t * (23.0833674 + r_t * (-21.0622410 + r_t * 3.13082910)))
    tp = numer / denom
    if tp < -1.0:
        tp = -1.0
    if tp > 1.0:
        tp = 1.0
    u_p = ((tick * 4219 + 0 * 8461 + agent_index * 3037) % 1000000) / 1000000.0
    u_p = 0.08 + u_p * 0.84
    q_p = u_p - 0.5
    r_p = q_p * q_p
    numer = q_p * (2.50662824 + r_p * (-18.6150006 + r_p * (41.3911977 + r_p * (-25.4410605))))
    denom = 1.0 + r_p * (-8.47351093 + r_p * (23.0833674 + r_p * (-21.0622410 + r_p * 3.13082910)))
    pp = numer / denom
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
    u_t = ((tick * 7919 + 1 * 6271 + agent_index * 1013) % 1000000) / 1000000.0
    u_t = 0.08 + u_t * 0.84
    q_t = u_t - 0.5
    r_t = q_t * q_t
    numer = q_t * (2.50662824 + r_t * (-18.6150006 + r_t * (41.3911977 + r_t * (-25.4410605))))
    denom = 1.0 + r_t * (-8.47351093 + r_t * (23.0833674 + r_t * (-21.0622410 + r_t * 3.13082910)))
    tp = numer / denom
    if tp < -1.0:
        tp = -1.0
    if tp > 1.0:
        tp = 1.0
    u_p = ((tick * 4219 + 1 * 8461 + agent_index * 3037) % 1000000) / 1000000.0
    u_p = 0.08 + u_p * 0.84
    q_p = u_p - 0.5
    r_p = q_p * q_p
    numer = q_p * (2.50662824 + r_p * (-18.6150006 + r_p * (41.3911977 + r_p * (-25.4410605))))
    denom = 1.0 + r_p * (-8.47351093 + r_p * (23.0833674 + r_p * (-21.0622410 + r_p * 3.13082910)))
    pp = numer / denom
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
    u_t = ((tick * 7919 + 2 * 6271 + agent_index * 1013) % 1000000) / 1000000.0
    u_t = 0.08 + u_t * 0.84
    q_t = u_t - 0.5
    r_t = q_t * q_t
    numer = q_t * (2.50662824 + r_t * (-18.6150006 + r_t * (41.3911977 + r_t * (-25.4410605))))
    denom = 1.0 + r_t * (-8.47351093 + r_t * (23.0833674 + r_t * (-21.0622410 + r_t * 3.13082910)))
    tp = numer / denom
    if tp < -1.0:
        tp = -1.0
    if tp > 1.0:
        tp = 1.0
    u_p = ((tick * 4219 + 2 * 8461 + agent_index * 3037) % 1000000) / 1000000.0
    u_p = 0.08 + u_p * 0.84
    q_p = u_p - 0.5
    r_p = q_p * q_p
    numer = q_p * (2.50662824 + r_p * (-18.6150006 + r_p * (41.3911977 + r_p * (-25.4410605))))
    denom = 1.0 + r_p * (-8.47351093 + r_p * (23.0833674 + r_p * (-21.0622410 + r_p * 3.13082910)))
    pp = numer / denom
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
    u_t = ((tick * 7919 + 3 * 6271 + agent_index * 1013) % 1000000) / 1000000.0
    u_t = 0.08 + u_t * 0.84
    q_t = u_t - 0.5
    r_t = q_t * q_t
    numer = q_t * (2.50662824 + r_t * (-18.6150006 + r_t * (41.3911977 + r_t * (-25.4410605))))
    denom = 1.0 + r_t * (-8.47351093 + r_t * (23.0833674 + r_t * (-21.0622410 + r_t * 3.13082910)))
    tp = numer / denom
    if tp < -1.0:
        tp = -1.0
    if tp > 1.0:
        tp = 1.0
    u_p = ((tick * 4219 + 3 * 8461 + agent_index * 3037) % 1000000) / 1000000.0
    u_p = 0.08 + u_p * 0.84
    q_p = u_p - 0.5
    r_p = q_p * q_p
    numer = q_p * (2.50662824 + r_p * (-18.6150006 + r_p * (41.3911977 + r_p * (-25.4410605))))
    denom = 1.0 + r_p * (-8.47351093 + r_p * (23.0833674 + r_p * (-21.0622410 + r_p * 3.13082910)))
    pp = numer / denom
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
    u_t = ((tick * 7919 + 4 * 6271 + agent_index * 1013) % 1000000) / 1000000.0
    u_t = 0.08 + u_t * 0.84
    q_t = u_t - 0.5
    r_t = q_t * q_t
    numer = q_t * (2.50662824 + r_t * (-18.6150006 + r_t * (41.3911977 + r_t * (-25.4410605))))
    denom = 1.0 + r_t * (-8.47351093 + r_t * (23.0833674 + r_t * (-21.0622410 + r_t * 3.13082910)))
    tp = numer / denom
    if tp < -1.0:
        tp = -1.0
    if tp > 1.0:
        tp = 1.0
    u_p = ((tick * 4219 + 4 * 8461 + agent_index * 3037) % 1000000) / 1000000.0
    u_p = 0.08 + u_p * 0.84
    q_p = u_p - 0.5
    r_p = q_p * q_p
    numer = q_p * (2.50662824 + r_p * (-18.6150006 + r_p * (41.3911977 + r_p * (-25.4410605))))
    denom = 1.0 + r_p * (-8.47351093 + r_p * (23.0833674 + r_p * (-21.0622410 + r_p * 3.13082910)))
    pp = numer / denom
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
    u_t = ((tick * 7919 + 5 * 6271 + agent_index * 1013) % 1000000) / 1000000.0
    u_t = 0.08 + u_t * 0.84
    q_t = u_t - 0.5
    r_t = q_t * q_t
    numer = q_t * (2.50662824 + r_t * (-18.6150006 + r_t * (41.3911977 + r_t * (-25.4410605))))
    denom = 1.0 + r_t * (-8.47351093 + r_t * (23.0833674 + r_t * (-21.0622410 + r_t * 3.13082910)))
    tp = numer / denom
    if tp < -1.0:
        tp = -1.0
    if tp > 1.0:
        tp = 1.0
    u_p = ((tick * 4219 + 5 * 8461 + agent_index * 3037) % 1000000) / 1000000.0
    u_p = 0.08 + u_p * 0.84
    q_p = u_p - 0.5
    r_p = q_p * q_p
    numer = q_p * (2.50662824 + r_p * (-18.6150006 + r_p * (41.3911977 + r_p * (-25.4410605))))
    denom = 1.0 + r_p * (-8.47351093 + r_p * (23.0833674 + r_p * (-21.0622410 + r_p * 3.13082910)))
    pp = numer / denom
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
    u_t = ((tick * 7919 + 6 * 6271 + agent_index * 1013) % 1000000) / 1000000.0
    u_t = 0.08 + u_t * 0.84
    q_t = u_t - 0.5
    r_t = q_t * q_t
    numer = q_t * (2.50662824 + r_t * (-18.6150006 + r_t * (41.3911977 + r_t * (-25.4410605))))
    denom = 1.0 + r_t * (-8.47351093 + r_t * (23.0833674 + r_t * (-21.0622410 + r_t * 3.13082910)))
    tp = numer / denom
    if tp < -1.0:
        tp = -1.0
    if tp > 1.0:
        tp = 1.0
    u_p = ((tick * 4219 + 6 * 8461 + agent_index * 3037) % 1000000) / 1000000.0
    u_p = 0.08 + u_p * 0.84
    q_p = u_p - 0.5
    r_p = q_p * q_p
    numer = q_p * (2.50662824 + r_p * (-18.6150006 + r_p * (41.3911977 + r_p * (-25.4410605))))
    denom = 1.0 + r_p * (-8.47351093 + r_p * (23.0833674 + r_p * (-21.0622410 + r_p * 3.13082910)))
    pp = numer / denom
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
    u_t = ((tick * 7919 + 7 * 6271 + agent_index * 1013) % 1000000) / 1000000.0
    u_t = 0.08 + u_t * 0.84
    q_t = u_t - 0.5
    r_t = q_t * q_t
    numer = q_t * (2.50662824 + r_t * (-18.6150006 + r_t * (41.3911977 + r_t * (-25.4410605))))
    denom = 1.0 + r_t * (-8.47351093 + r_t * (23.0833674 + r_t * (-21.0622410 + r_t * 3.13082910)))
    tp = numer / denom
    if tp < -1.0:
        tp = -1.0
    if tp > 1.0:
        tp = 1.0
    u_p = ((tick * 4219 + 7 * 8461 + agent_index * 3037) % 1000000) / 1000000.0
    u_p = 0.08 + u_p * 0.84
    q_p = u_p - 0.5
    r_p = q_p * q_p
    numer = q_p * (2.50662824 + r_p * (-18.6150006 + r_p * (41.3911977 + r_p * (-25.4410605))))
    denom = 1.0 + r_p * (-8.47351093 + r_p * (23.0833674 + r_p * (-21.0622410 + r_p * 3.13082910)))
    pp = numer / denom
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
    u_t = ((tick * 7919 + 8 * 6271 + agent_index * 1013) % 1000000) / 1000000.0
    u_t = 0.08 + u_t * 0.84
    q_t = u_t - 0.5
    r_t = q_t * q_t
    numer = q_t * (2.50662824 + r_t * (-18.6150006 + r_t * (41.3911977 + r_t * (-25.4410605))))
    denom = 1.0 + r_t * (-8.47351093 + r_t * (23.0833674 + r_t * (-21.0622410 + r_t * 3.13082910)))
    tp = numer / denom
    if tp < -1.0:
        tp = -1.0
    if tp > 1.0:
        tp = 1.0
    u_p = ((tick * 4219 + 8 * 8461 + agent_index * 3037) % 1000000) / 1000000.0
    u_p = 0.08 + u_p * 0.84
    q_p = u_p - 0.5
    r_p = q_p * q_p
    numer = q_p * (2.50662824 + r_p * (-18.6150006 + r_p * (41.3911977 + r_p * (-25.4410605))))
    denom = 1.0 + r_p * (-8.47351093 + r_p * (23.0833674 + r_p * (-21.0622410 + r_p * 3.13082910)))
    pp = numer / denom
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
    u_t = ((tick * 7919 + 9 * 6271 + agent_index * 1013) % 1000000) / 1000000.0
    u_t = 0.08 + u_t * 0.84
    q_t = u_t - 0.5
    r_t = q_t * q_t
    numer = q_t * (2.50662824 + r_t * (-18.6150006 + r_t * (41.3911977 + r_t * (-25.4410605))))
    denom = 1.0 + r_t * (-8.47351093 + r_t * (23.0833674 + r_t * (-21.0622410 + r_t * 3.13082910)))
    tp = numer / denom
    if tp < -1.0:
        tp = -1.0
    if tp > 1.0:
        tp = 1.0
    u_p = ((tick * 4219 + 9 * 8461 + agent_index * 3037) % 1000000) / 1000000.0
    u_p = 0.08 + u_p * 0.84
    q_p = u_p - 0.5
    r_p = q_p * q_p
    numer = q_p * (2.50662824 + r_p * (-18.6150006 + r_p * (41.3911977 + r_p * (-25.4410605))))
    denom = 1.0 + r_p * (-8.47351093 + r_p * (23.0833674 + r_p * (-21.0622410 + r_p * 3.13082910)))
    pp = numer / denom
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
    u_t = ((tick * 7919 + 10 * 6271 + agent_index * 1013) % 1000000) / 1000000.0
    u_t = 0.08 + u_t * 0.84
    q_t = u_t - 0.5
    r_t = q_t * q_t
    numer = q_t * (2.50662824 + r_t * (-18.6150006 + r_t * (41.3911977 + r_t * (-25.4410605))))
    denom = 1.0 + r_t * (-8.47351093 + r_t * (23.0833674 + r_t * (-21.0622410 + r_t * 3.13082910)))
    tp = numer / denom
    if tp < -1.0:
        tp = -1.0
    if tp > 1.0:
        tp = 1.0
    u_p = ((tick * 4219 + 10 * 8461 + agent_index * 3037) % 1000000) / 1000000.0
    u_p = 0.08 + u_p * 0.84
    q_p = u_p - 0.5
    r_p = q_p * q_p
    numer = q_p * (2.50662824 + r_p * (-18.6150006 + r_p * (41.3911977 + r_p * (-25.4410605))))
    denom = 1.0 + r_p * (-8.47351093 + r_p * (23.0833674 + r_p * (-21.0622410 + r_p * 3.13082910)))
    pp = numer / denom
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
    u_t = ((tick * 7919 + 11 * 6271 + agent_index * 1013) % 1000000) / 1000000.0
    u_t = 0.08 + u_t * 0.84
    q_t = u_t - 0.5
    r_t = q_t * q_t
    numer = q_t * (2.50662824 + r_t * (-18.6150006 + r_t * (41.3911977 + r_t * (-25.4410605))))
    denom = 1.0 + r_t * (-8.47351093 + r_t * (23.0833674 + r_t * (-21.0622410 + r_t * 3.13082910)))
    tp = numer / denom
    if tp < -1.0:
        tp = -1.0
    if tp > 1.0:
        tp = 1.0
    u_p = ((tick * 4219 + 11 * 8461 + agent_index * 3037) % 1000000) / 1000000.0
    u_p = 0.08 + u_p * 0.84
    q_p = u_p - 0.5
    r_p = q_p * q_p
    numer = q_p * (2.50662824 + r_p * (-18.6150006 + r_p * (41.3911977 + r_p * (-25.4410605))))
    denom = 1.0 + r_p * (-8.47351093 + r_p * (23.0833674 + r_p * (-21.0622410 + r_p * 3.13082910)))
    pp = numer / denom
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

    # ========== COMPUTE ANNUAL PRECIP (already in cm from gap_model.py) ==========
    annual_prcp_cm = prcp_0 + prcp_1 + prcp_2 + prcp_3 + prcp_4 + prcp_5 + prcp_6 + prcp_7 + prcp_8 + prcp_9 + prcp_10 + prcp_11

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
    drydays_upper = 0.0   # Count of upper-layer dry days
    drydays_base = 0.0    # Count of base-layer dry days
    total_pet = 0.0       # Accumulated potential ET
    total_aet = 0.0       # Accumulated actual ET
    deg_days = 0.0        # Growing degree days (base 5°C, GAPpy model.py:233-236)
    grow_days_5 = 0.0     # Days with tavg >= 5°C (GAPpy growing season definition)
    annual_runoff = 0.0   # Accumulated runoff for N balance (GAPpy model.py:227)

    # ========== BERNOULLI RAIN-DAY PARAMETERS (GAPpy cov365a) ==========
    # Per month: raindays = min(25, prcp/4+1), ik = int(raindays),
    # rr = prcp/ik (amount per rain day), ss = raindays/days_in_month (rain probability),
    # inum = float(ik) (countdown of remaining rain events)
    # Month 0 (Jan, 31 days)
    m0_raindays = prcp_0 / 4.0 + 1.0
    if m0_raindays > 25.0:
        m0_raindays = 25.0
    m0_ik = int(m0_raindays)
    if m0_ik < 1:
        m0_ik = 1
    m0_rr = prcp_0 / float(m0_ik)
    m0_ss = m0_raindays / 31.0
    m0_inum = float(m0_ik)
    # Month 1 (Feb, 28 days)
    m1_raindays = prcp_1 / 4.0 + 1.0
    if m1_raindays > 25.0:
        m1_raindays = 25.0
    m1_ik = int(m1_raindays)
    if m1_ik < 1:
        m1_ik = 1
    m1_rr = prcp_1 / float(m1_ik)
    m1_ss = m1_raindays / 28.0
    m1_inum = float(m1_ik)
    # Month 2 (Mar, 31 days)
    m2_raindays = prcp_2 / 4.0 + 1.0
    if m2_raindays > 25.0:
        m2_raindays = 25.0
    m2_ik = int(m2_raindays)
    if m2_ik < 1:
        m2_ik = 1
    m2_rr = prcp_2 / float(m2_ik)
    m2_ss = m2_raindays / 31.0
    m2_inum = float(m2_ik)
    # Month 3 (Apr, 30 days)
    m3_raindays = prcp_3 / 4.0 + 1.0
    if m3_raindays > 25.0:
        m3_raindays = 25.0
    m3_ik = int(m3_raindays)
    if m3_ik < 1:
        m3_ik = 1
    m3_rr = prcp_3 / float(m3_ik)
    m3_ss = m3_raindays / 30.0
    m3_inum = float(m3_ik)
    # Month 4 (May, 31 days)
    m4_raindays = prcp_4 / 4.0 + 1.0
    if m4_raindays > 25.0:
        m4_raindays = 25.0
    m4_ik = int(m4_raindays)
    if m4_ik < 1:
        m4_ik = 1
    m4_rr = prcp_4 / float(m4_ik)
    m4_ss = m4_raindays / 31.0
    m4_inum = float(m4_ik)
    # Month 5 (Jun, 30 days)
    m5_raindays = prcp_5 / 4.0 + 1.0
    if m5_raindays > 25.0:
        m5_raindays = 25.0
    m5_ik = int(m5_raindays)
    if m5_ik < 1:
        m5_ik = 1
    m5_rr = prcp_5 / float(m5_ik)
    m5_ss = m5_raindays / 30.0
    m5_inum = float(m5_ik)
    # Month 6 (Jul, 31 days)
    m6_raindays = prcp_6 / 4.0 + 1.0
    if m6_raindays > 25.0:
        m6_raindays = 25.0
    m6_ik = int(m6_raindays)
    if m6_ik < 1:
        m6_ik = 1
    m6_rr = prcp_6 / float(m6_ik)
    m6_ss = m6_raindays / 31.0
    m6_inum = float(m6_ik)
    # Month 7 (Aug, 31 days)
    m7_raindays = prcp_7 / 4.0 + 1.0
    if m7_raindays > 25.0:
        m7_raindays = 25.0
    m7_ik = int(m7_raindays)
    if m7_ik < 1:
        m7_ik = 1
    m7_rr = prcp_7 / float(m7_ik)
    m7_ss = m7_raindays / 31.0
    m7_inum = float(m7_ik)
    # Month 8 (Sep, 30 days)
    m8_raindays = prcp_8 / 4.0 + 1.0
    if m8_raindays > 25.0:
        m8_raindays = 25.0
    m8_ik = int(m8_raindays)
    if m8_ik < 1:
        m8_ik = 1
    m8_rr = prcp_8 / float(m8_ik)
    m8_ss = m8_raindays / 30.0
    m8_inum = float(m8_ik)
    # Month 9 (Oct, 31 days)
    m9_raindays = prcp_9 / 4.0 + 1.0
    if m9_raindays > 25.0:
        m9_raindays = 25.0
    m9_ik = int(m9_raindays)
    if m9_ik < 1:
        m9_ik = 1
    m9_rr = prcp_9 / float(m9_ik)
    m9_ss = m9_raindays / 31.0
    m9_inum = float(m9_ik)
    # Month 10 (Nov, 30 days)
    m10_raindays = prcp_10 / 4.0 + 1.0
    if m10_raindays > 25.0:
        m10_raindays = 25.0
    m10_ik = int(m10_raindays)
    if m10_ik < 1:
        m10_ik = 1
    m10_rr = prcp_10 / float(m10_ik)
    m10_ss = m10_raindays / 30.0
    m10_inum = float(m10_ik)
    # Month 11 (Dec, 31 days)
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
        # === TEMPERATURE: Linear interpolation between monthly midpoints (GAPpy cov365) ===
        # Anchor days (0-indexed): [15,44,74,104,135,165,195,226,257,287,318,348]
        # Days 0-14 wrap around from December
        d_s = day
        if day < 15:
            d_s = day + 365

        day_tmin = 0.0
        day_tmax = 0.0
        frac = 0.0

        if d_s < 44:
            # Jan mid (15) → Feb mid (44), width 29
            frac = float(d_s - 15) / 29.0
            day_tmin = tmin_0 + frac * (tmin_1 - tmin_0)
            day_tmax = tmax_0 + frac * (tmax_1 - tmax_0)
        elif d_s < 74:
            # Feb mid (44) → Mar mid (74), width 30
            frac = float(d_s - 44) / 30.0
            day_tmin = tmin_1 + frac * (tmin_2 - tmin_1)
            day_tmax = tmax_1 + frac * (tmax_2 - tmax_1)
        elif d_s < 104:
            # Mar mid (74) → Apr mid (104), width 30
            frac = float(d_s - 74) / 30.0
            day_tmin = tmin_2 + frac * (tmin_3 - tmin_2)
            day_tmax = tmax_2 + frac * (tmax_3 - tmax_2)
        elif d_s < 135:
            # Apr mid (104) → May mid (135), width 31
            frac = float(d_s - 104) / 31.0
            day_tmin = tmin_3 + frac * (tmin_4 - tmin_3)
            day_tmax = tmax_3 + frac * (tmax_4 - tmax_3)
        elif d_s < 165:
            # May mid (135) → Jun mid (165), width 30
            frac = float(d_s - 135) / 30.0
            day_tmin = tmin_4 + frac * (tmin_5 - tmin_4)
            day_tmax = tmax_4 + frac * (tmax_5 - tmax_4)
        elif d_s < 195:
            # Jun mid (165) → Jul mid (195), width 30
            frac = float(d_s - 165) / 30.0
            day_tmin = tmin_5 + frac * (tmin_6 - tmin_5)
            day_tmax = tmax_5 + frac * (tmax_6 - tmax_5)
        elif d_s < 226:
            # Jul mid (195) → Aug mid (226), width 31
            frac = float(d_s - 195) / 31.0
            day_tmin = tmin_6 + frac * (tmin_7 - tmin_6)
            day_tmax = tmax_6 + frac * (tmax_7 - tmax_6)
        elif d_s < 257:
            # Aug mid (226) → Sep mid (257), width 31
            frac = float(d_s - 226) / 31.0
            day_tmin = tmin_7 + frac * (tmin_8 - tmin_7)
            day_tmax = tmax_7 + frac * (tmax_8 - tmax_7)
        elif d_s < 287:
            # Sep mid (257) → Oct mid (287), width 30
            frac = float(d_s - 257) / 30.0
            day_tmin = tmin_8 + frac * (tmin_9 - tmin_8)
            day_tmax = tmax_8 + frac * (tmax_9 - tmax_8)
        elif d_s < 318:
            # Oct mid (287) → Nov mid (318), width 31
            frac = float(d_s - 287) / 31.0
            day_tmin = tmin_9 + frac * (tmin_10 - tmin_9)
            day_tmax = tmax_9 + frac * (tmax_10 - tmax_9)
        elif d_s < 348:
            # Nov mid (318) → Dec mid (348), width 30
            frac = float(d_s - 318) / 30.0
            day_tmin = tmin_10 + frac * (tmin_11 - tmin_10)
            day_tmax = tmax_10 + frac * (tmax_11 - tmax_10)
        else:
            # Dec mid (348) → Jan mid next year (380), width 32
            frac = float(d_s - 348) / 32.0
            day_tmin = tmin_11 + frac * (tmin_0 - tmin_11)
            day_tmax = tmax_11 + frac * (tmax_0 - tmax_11)

        # === PRECIPITATION: Bernoulli rain-day allocation (GAPpy cov365a) ===
        # Each month has ik rain events of size rr, with daily probability ss.
        # Remaining events dumped on last day of month to preserve monthly totals.
        day_prcp = 0.0
        if day < 31:
            if m0_inum > 0.5:
                rain_hash = ((tick * 5003 + day * 7013 + agent_index * 3019) % 1000000) / 1000000.0
                if rain_hash <= m0_ss:
                    day_prcp = m0_rr
                    m0_inum = m0_inum - 1.0
            if day == 30 and m0_inum > 0.5:
                day_prcp = day_prcp + m0_inum * m0_rr
                m0_inum = 0.0
        elif day < 59:
            if m1_inum > 0.5:
                rain_hash = ((tick * 5003 + day * 7013 + agent_index * 3019) % 1000000) / 1000000.0
                if rain_hash <= m1_ss:
                    day_prcp = m1_rr
                    m1_inum = m1_inum - 1.0
            if day == 58 and m1_inum > 0.5:
                day_prcp = day_prcp + m1_inum * m1_rr
                m1_inum = 0.0
        elif day < 90:
            if m2_inum > 0.5:
                rain_hash = ((tick * 5003 + day * 7013 + agent_index * 3019) % 1000000) / 1000000.0
                if rain_hash <= m2_ss:
                    day_prcp = m2_rr
                    m2_inum = m2_inum - 1.0
            if day == 89 and m2_inum > 0.5:
                day_prcp = day_prcp + m2_inum * m2_rr
                m2_inum = 0.0
        elif day < 120:
            if m3_inum > 0.5:
                rain_hash = ((tick * 5003 + day * 7013 + agent_index * 3019) % 1000000) / 1000000.0
                if rain_hash <= m3_ss:
                    day_prcp = m3_rr
                    m3_inum = m3_inum - 1.0
            if day == 119 and m3_inum > 0.5:
                day_prcp = day_prcp + m3_inum * m3_rr
                m3_inum = 0.0
        elif day < 151:
            if m4_inum > 0.5:
                rain_hash = ((tick * 5003 + day * 7013 + agent_index * 3019) % 1000000) / 1000000.0
                if rain_hash <= m4_ss:
                    day_prcp = m4_rr
                    m4_inum = m4_inum - 1.0
            if day == 150 and m4_inum > 0.5:
                day_prcp = day_prcp + m4_inum * m4_rr
                m4_inum = 0.0
        elif day < 181:
            if m5_inum > 0.5:
                rain_hash = ((tick * 5003 + day * 7013 + agent_index * 3019) % 1000000) / 1000000.0
                if rain_hash <= m5_ss:
                    day_prcp = m5_rr
                    m5_inum = m5_inum - 1.0
            if day == 180 and m5_inum > 0.5:
                day_prcp = day_prcp + m5_inum * m5_rr
                m5_inum = 0.0
        elif day < 212:
            if m6_inum > 0.5:
                rain_hash = ((tick * 5003 + day * 7013 + agent_index * 3019) % 1000000) / 1000000.0
                if rain_hash <= m6_ss:
                    day_prcp = m6_rr
                    m6_inum = m6_inum - 1.0
            if day == 211 and m6_inum > 0.5:
                day_prcp = day_prcp + m6_inum * m6_rr
                m6_inum = 0.0
        elif day < 243:
            if m7_inum > 0.5:
                rain_hash = ((tick * 5003 + day * 7013 + agent_index * 3019) % 1000000) / 1000000.0
                if rain_hash <= m7_ss:
                    day_prcp = m7_rr
                    m7_inum = m7_inum - 1.0
            if day == 242 and m7_inum > 0.5:
                day_prcp = day_prcp + m7_inum * m7_rr
                m7_inum = 0.0
        elif day < 273:
            if m8_inum > 0.5:
                rain_hash = ((tick * 5003 + day * 7013 + agent_index * 3019) % 1000000) / 1000000.0
                if rain_hash <= m8_ss:
                    day_prcp = m8_rr
                    m8_inum = m8_inum - 1.0
            if day == 272 and m8_inum > 0.5:
                day_prcp = day_prcp + m8_inum * m8_rr
                m8_inum = 0.0
        elif day < 304:
            if m9_inum > 0.5:
                rain_hash = ((tick * 5003 + day * 7013 + agent_index * 3019) % 1000000) / 1000000.0
                if rain_hash <= m9_ss:
                    day_prcp = m9_rr
                    m9_inum = m9_inum - 1.0
            if day == 303 and m9_inum > 0.5:
                day_prcp = day_prcp + m9_inum * m9_rr
                m9_inum = 0.0
        elif day < 334:
            if m10_inum > 0.5:
                rain_hash = ((tick * 5003 + day * 7013 + agent_index * 3019) % 1000000) / 1000000.0
                if rain_hash <= m10_ss:
                    day_prcp = m10_rr
                    m10_inum = m10_inum - 1.0
            if day == 333 and m10_inum > 0.5:
                day_prcp = day_prcp + m10_inum * m10_rr
                m10_inum = 0.0
        else:
            if m11_inum > 0.5:
                rain_hash = ((tick * 5003 + day * 7013 + agent_index * 3019) % 1000000) / 1000000.0
                if rain_hash <= m11_ss:
                    day_prcp = m11_rr
                    m11_inum = m11_inum - 1.0
            if day == 364 and m11_inum > 0.5:
                day_prcp = day_prcp + m11_inum * m11_rr
                m11_inum = 0.0

        day_temp = (day_tmin + day_tmax) / 2.0

        # Track freeze days
        if day_temp < 0.0:
            freeze_days = freeze_days + 1.0

        # Accumulate growing degree days (base 5°C, GAPpy model.py:233-236)
        if day_temp >= 5.0:
            deg_days = deg_days + (day_temp - 5.0)
            grow_days_5 = grow_days_5 + 1.0

        # Accumulate atmospheric N from precipitation (already in cm from gap_model.py)
        rain_n = rain_n + day_prcp * PRCP_N

        # === Calculate PET: Hargreaves method (GAPpy climate.py:85-112) ===
        julia = day + 1  # 1-based Julian day

        # Extraterrestrial radiation (GAPpy ex_rad function)
        # Note: dr (distance ratio) omitted; not used in erad formula (GAPpy climate.py:100)
        dairta = H_AS * cp.sin(H_B * float(julia) + H_PHASE)
        yxd_pet = -cp.tan(lat_rad) * cp.tan(dairta)

        # Clamp for acos
        if yxd_pet >= 1.0:
            yxd_pet = 1.0
        if yxd_pet <= -1.0:
            yxd_pet = -1.0

        # Polynomial approximation for acos (accurate to ~0.01 radians)
        # acos(x) = PI/2 - asin(x), asin(x) ≈ x + x^3/6 + 3x^5/40
        omega = 0.0
        if yxd_pet >= 1.0:
            omega = 0.0
        elif yxd_pet <= -1.0:
            omega = PI
        else:
            x2_pet = yxd_pet * yxd_pet
            asin_pet = yxd_pet * (1.0 + x2_pet * (0.16666667 + x2_pet * 0.075))
            omega = 1.5707963 - asin_pet  # PI/2 - asin = acos

        erad = H_AMP * cp.cos(lat_rad) * cp.cos(dairta) * (cp.sin(omega) - omega * cp.cos(omega))
        if erad < 0.0:
            erad = 0.0

        # Hargreaves PET (GAPpy climate.py:107-112)
        pot_ev_day = 0.0
        if day_temp > 0.0:
            tdiff = day_tmax - day_tmin
            if tdiff < 0.0:
                tdiff = 0.0
            pot_ev_day = H_COEFF * (tdiff ** 0.5) * (day_temp + H_ADDON) * erad

        # === SOIL WATER BALANCE (from UVAFME soil.py:soil_water) ===
        # GAPpy: freeze is always 0.0 (local variable never updated, dead code)
        freeze = 0.0

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

            # GAPpy: lai_w0 is NOT set to laiw in the evaporation path.
            # In excess path: lai_w0 stays unchanged (original value).
            # In deficit path: lai_w0 was already decremented by lai_w1 above.
            # Only the no-evaporation path (pot_ev <= 0) sets lai_w0 = laiw.

        # Calculate moisture scaling factors
        aow0_scaled = ao_w0 / aow_max
        saw0_scaled = 0.5
        if sa_fc > 0.0:
            saw0_scaled = sa_w0 / sa_fc

        # Track flood days: when A layer is near field capacity (saturated)
        # UVAFME considers flooding when soil water is at or above field capacity
        if sa_fc > 0.0 and sa_w0 >= sa_fc * 0.95:
            flood_days = flood_days + 1.0

        # Track dry days from soil water state (GAPpy model.py:238-245)
        sbw0_scaled_by_min = 1.0
        if sbw_min > 0.001:
            sbw0_scaled_by_min = sb_w0 / sbw_min
        sbw0_scaled_by_max = 1.0
        if sbw_max > 0.001:
            sbw0_scaled_by_max = sb_w0 / sbw_max
        saw0_scaled_by_wp = 1.0
        if sa_pwp > 0.001:
            saw0_scaled_by_wp = sa_w0 / sa_pwp

        # Upper layer dry: all three conditions below capacity
        if saw0_scaled < 1.0001 and sbw0_scaled_by_min < 1.0001 and sbw0_scaled_by_max < 1.0001:
            drydays_upper = drydays_upper + 1.0

        # Base layer dry: A layer water below wilting point
        if saw0_scaled_by_wp < 1.0001:
            drydays_base = drydays_base + 1.0

        # Accumulate PET, AET, and runoff
        total_pet = total_pet + pot_ev_day
        total_aet = total_aet + act_ev_day
        annual_runoff = annual_runoff + runoff  # For N balance (GAPpy model.py:227)

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

    # ========== NORMALIZE DRY DAYS TO FRACTIONS (GAPpy model.py:261-262) ==========
    # GAPpy: growdays = count of days with tavg >= 5°C (not 365 - freeze_days)
    growdays = grow_days_5
    if growdays < 1.0:
        growdays = 1.0

    dry_days_frac = drydays_upper / growdays
    dry_days_base_frac = drydays_base / growdays

    # Cap upper-layer drought by rain/PET ratio (GAPpy)
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

    # Normalize flood days by growing season (GAPpy model.py:263)
    flood_days = flood_days / growdays

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
    params_tensor[agent_index][SITE_P_ANNUAL_RUNOFF] = annual_runoff  # For N balance at P10

    # Growing degree days (accumulated daily, base 5°C)
    states_tensor[agent_index][SITE_S_DEG_DAYS] = deg_days

    # Available N = mineralization + atmospheric deposition
    states_tensor[agent_index][SITE_S_AVAIL_N] = total_avail_n + rain_n

    # Flood days for this year
    states_tensor[agent_index][SITE_S_FLOOD_DAYS] = flood_days

    # Dry days from soil water balance (fraction 0-1)
    states_tensor[agent_index][SITE_S_DRY_DAYS] = dry_days

    # Base layer drought fraction (for intolerant species dual-metric)
    states_tensor[agent_index][SITE_S_DRY_DAYS_BASE] = dry_days_base_frac

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
    fire_rand = ((tick * 1021 + agent_index * 1019) % 1000000) / 1000000.0
    fire_intensity = 0.0
    if fire_rand < fire_prob:
        # Fire occurs - intensity based on dry conditions
        fire_intensity = 0.3 + fire_rand * 2.0  # 0.3 to ~0.7 intensity
        if fire_intensity > 1.0:
            fire_intensity = 1.0

    states_tensor[agent_index][SITE_S_FIRE_INTENSITY] = fire_intensity

    # ========== WIND PROBABILITY (GAPpy model.py:623-655) ==========
    wind_prob = params_tensor[agent_index][SITE_P_WIND_PROB]

    # Stochastic wind check (separate hash from fire)
    wind_rand = ((tick * 2039 + agent_index * 2027) % 1000000) / 1000000.0
    wind_intensity = 0.0
    if wind_rand < wind_prob and fire_intensity < 0.01:
        # Wind occurs (only if no fire this year - fire takes precedence, GAPpy model.py:630)
        wind_intensity = 0.5 + wind_rand * 1.0  # 0.5 to ~1.0 intensity
        if wind_intensity > 1.0:
            wind_intensity = 1.0

    states_tensor[agent_index][SITE_S_WIND_INTENSITY] = wind_intensity

    # Stochastic climate outputs (for CSV export)
    states_tensor[agent_index][SITE_S_ANNUAL_RAIN] = annual_prcp_cm
    states_tensor[agent_index][SITE_S_GROW_DAYS] = grow_days_5
    states_tensor[agent_index][SITE_S_POT_EVAP] = total_pet
    states_tensor[agent_index][SITE_S_ACT_EVAP] = total_aet

    # Soil outputs (for CSV export)
    states_tensor[agent_index][SITE_S_SOIL_RESP] = total_resp
    states_tensor[agent_index][SITE_S_C_INTO_A0] = total_litter_c
    states_tensor[agent_index][SITE_S_N_INTO_A0] = total_litter_n
