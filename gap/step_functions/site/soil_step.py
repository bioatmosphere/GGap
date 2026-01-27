"""
Site step function for GGap model.
GPU kernel implementing UVAFME soil biogeochemistry.

Matches UVAFME's bio_geo_climate() and process_soil_biogeochemistry():
- Daily climate interpolation from monthly data
- Soil water balance (soil_water function)
- Three-layer decomposition (A0 -> A -> Base)
- Atmospheric N deposition from precipitation

Priority 2: Reads litter_accum from Gap neighbors, does decomposition.

Property scheme (3 properties):
- params[53]: soil pools + monthly climate + site properties - private
- states[4]: climate + available (deg_days, dry_days, base_mortality, avail_n) - public
- states_db[1]: placeholder (public, double buffered but unused)
"""

import cupy as cp
from cupyx import jit

# === Breed IDs ===
BREED_TREE = 0
BREED_GAP = 1
BREED_SITE = 2

# === Site params[53] (private) ===
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

# === Site states[6] (public) ===
SITE_S_DEG_DAYS = 0
SITE_S_DRY_DAYS = 1
SITE_S_BASE_MORTALITY = 2
SITE_S_AVAIL_N = 3
SITE_S_FLOOD_DAYS = 4  # Days when A layer is saturated
SITE_S_FIRE_INTENSITY = 5  # Fire intensity this year (0-1)

# === Gap states[7] (for reading from Gap neighbors) ===
GAP_S_LITTER_ACCUM_C = 5
GAP_S_LITTER_ACCUM_N = 6

# === UVAFME Constants (from soil.py) ===
AO_CN_0 = 30.0
SA_CN_0 = 4.0
SB_CN_0 = 20.0
AO_RESP = 5.24e-4
SA_RESP = 1.24e-5
SB_RESP = 2.74e-7

SOIL_BASE_DEPTH = 70.0
BASE_MAX = 0.6
BASE_MIN = 0.1
AO_MIN = 0.025
AO_MAX = 0.25
LAI_MIN = 0.01
LAI_MAX = 0.15

# Atmospheric N in precipitation (tn N per cm precip)
PRCP_N = 0.00001

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
    Soil biogeochemistry step function (priority 2).

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
    total_litter_c = 0.0
    total_litter_n = 0.0

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

        i = i + 1

    # Convert annual litter to daily inputs
    daily_litter_c = total_litter_c / 365.0
    daily_litter_n = total_litter_n / 365.0

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

    # Ensure minimum LAI
    if lai < 1.0:
        lai = 1.0

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

    # ========== WATER BALANCE LIMITS ==========
    sbh = SOIL_BASE_DEPTH
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

        # Accumulate atmospheric N from precipitation
        rain_n = rain_n + day_prcp * PRCP_N

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
            table_water = day_prcp * sigma * (1.0 - freeze)
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
        # Add daily litter to A0 layer
        ao_c0 = ao_c0 + daily_litter_c
        ao_n0 = ao_n0 + daily_litter_n

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
        yxdn = resp1 / ao_cn
        yxdc = yxdn * AO_CN_0
        ao_c0 = ao_c0 - yxdc - resp1
        ao_n0 = ao_n0 - yxdn
        if ao_c0 < 0.0:
            ao_c0 = 0.0
        if ao_n0 < 0.0:
            ao_n0 = 0.0

        # Soil A layer
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

    # ========== FIRE PROBABILITY ==========
    # Fire probability increases with dry days and decreases with precipitation
    # Base fire probability: 0.01 (1% per year baseline)
    # Dry conditions increase fire risk
    fire_prob = 0.01  # Base probability
    dry_day_fraction = freeze_days / 365.0  # Reuse freeze_days counter for dry conditions

    # More dry days = higher fire probability
    # At 120+ dry days, fire probability increases significantly
    if freeze_days < 30.0:  # freeze_days was actually calculated but let's use a simple dry proxy
        # Use dry_days from soil moisture tracking (estimated from water balance)
        estimated_dry_days = 0.0
        if saw0_scaled < 0.3:
            estimated_dry_days = 30.0
        if saw0_scaled < 0.2:
            estimated_dry_days = 60.0
        if saw0_scaled < 0.1:
            estimated_dry_days = 120.0

        if estimated_dry_days > 60.0:
            fire_prob = 0.03 + (estimated_dry_days - 60.0) * 0.001

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
