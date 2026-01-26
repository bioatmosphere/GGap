"""
Site step function for GGap model.
GPU kernel implementing UVAFME soil biogeochemistry.

Soil layers: A0 (litter) -> A (humus) -> Base (stable)
Daily decomposition with temperature and moisture adjustment.

Priority 2: Reads litter_accum from Gap neighbors, does decomposition.

All step functions share same signature with 5 property tensors:
- params_tensor: static parameters
- state_db_tensor: state needing double buffer
- state_tensor: state NOT needing double buffer
- output_tensor: outputs
- soil_tensor: soil state
"""

import sys
import os

_current_dir = os.path.dirname(os.path.abspath(__file__))
_sagesim_path = os.path.join(os.path.dirname(_current_dir), "SAGESim")
if _sagesim_path not in sys.path:
    sys.path.insert(0, _sagesim_path)

import cupy as cp
from cupyx import jit

# === Index constants (shared across all breeds) ===

# params indices - site_params at indices 10-12
SITE_DEG_DAYS = 10

# state indices (20 floats)
S_AVAIL_N = 12

# output indices (8 floats)
O_LITTER_ACCUM_C = 3
O_LITTER_ACCUM_N = 4
O_SOIL_RESP = 5

# soil indices (10 floats)
SOIL_A0_C = 0
SOIL_A0_N = 1
SOIL_A_C = 2
SOIL_A_N = 3
SOIL_BL_C = 4
SOIL_BL_N = 5
SOIL_A0_W = 6
SOIL_A_W = 7
SOIL_BL_W = 8

# Decomposition rate constants (per day)
A0_RESP = 5.24e-4   # Fast litter decomposition
A_RESP = 1.24e-5    # Medium humus decomposition
BL_RESP = 2.74e-7   # Slow stable carbon

# C/N ratio targets
A0_CN_TARGET = 30.0  # Litter layer target C/N
A_CN_TARGET = 4.0    # Humus layer target C/N
BL_CN_TARGET = 20.0  # Base layer target C/N

# Transfer efficiencies
A0_TO_A_EFF = 0.3    # Fraction of A0 decomposition going to A layer
A_TO_BL_EFF = 0.1    # Fraction of A decomposition going to Base layer

# Breed IDs
BREED_TREE = 0
BREED_GAP = 1
BREED_SITE = 2


@jit.rawkernel(device="cuda")
def site_soil_step(
    tick,
    agent_index,
    globals,
    agent_ids,
    breeds,
    locations,
    # 5 consolidated property tensors
    params_tensor,
    state_db_tensor,
    state_tensor,
    output_tensor,
    soil_tensor,
):
    """
    Soil biogeochemistry step function (priority 2).

    Reads from Gap neighbors:
    - litter_accum [C, N] aggregated from trees (output[3:5])

    Writes to own state:
    - soil: updated C/N in each layer (soil[0:6])
    - avail_n: available nitrogen for Gap to read (state[12])
    - soil_resp: total respiration (output[5])

    Processes daily decomposition for one simulated year (365 days).
    """
    # Note: Breed check is done by SAGESim wrapper (stepfunc), no need to check here

    # Read litter from Gap neighbors
    total_litter_c = 0.0
    total_litter_n = 0.0

    neighbor_indices = locations[agent_index]
    i = 0
    while i < len(neighbor_indices) and neighbor_indices[i] != -1:
        neighbor_idx = int(neighbor_indices[i])
        neighbor_breed = int(breeds[neighbor_idx])

        if neighbor_breed == BREED_GAP:
            gap_litter_c = output_tensor[neighbor_idx][O_LITTER_ACCUM_C]
            gap_litter_n = output_tensor[neighbor_idx][O_LITTER_ACCUM_N]
            total_litter_c = total_litter_c + gap_litter_c
            total_litter_n = total_litter_n + gap_litter_n

        i = i + 1

    # Read current soil pools
    A0_c = soil_tensor[agent_index][SOIL_A0_C]
    A0_n = soil_tensor[agent_index][SOIL_A0_N]
    A_c = soil_tensor[agent_index][SOIL_A_C]
    A_n = soil_tensor[agent_index][SOIL_A_N]
    BL_c = soil_tensor[agent_index][SOIL_BL_C]
    BL_n = soil_tensor[agent_index][SOIL_BL_N]

    # Read water content
    A0_w = soil_tensor[agent_index][SOIL_A0_W]
    A_w = soil_tensor[agent_index][SOIL_A_W]

    # Add litter to A0 layer
    A0_c = A0_c + total_litter_c
    A0_n = A0_n + total_litter_n

    # Get temperature factor from params (deg_days at index 10)
    deg_days = params_tensor[agent_index][SITE_DEG_DAYS]
    # Approximate mean annual temp from degree days (rough: deg_days/365 - 5)
    mean_temp = deg_days / 365.0 - 5.0
    if mean_temp < -5.0:
        mean_temp = -5.0

    # Temperature adjustment (Q10 response)
    tadjst = 1.0
    if mean_temp >= -5.0:
        tadjst = cp.exp(0.1 * (mean_temp - 1.0) * cp.log(3.0))

    # Process 365 days of decomposition
    total_avail_n = 0.0
    total_resp = 0.0

    day = 0
    while day < 365:
        # Moisture adjustment for A0 layer
        a0_moist_factor = 0.5 + 0.5 * A0_w
        if a0_moist_factor < 0.1:
            a0_moist_factor = 0.1
        if a0_moist_factor > 1.0:
            a0_moist_factor = 1.0

        # Moisture adjustment for A layer
        a_moist_factor = 0.5 + 0.5 * A_w
        if a_moist_factor < 0.1:
            a_moist_factor = 0.1
        if a_moist_factor > 1.0:
            a_moist_factor = 1.0

        # ===== A0 LAYER DECOMPOSITION =====
        a0_resp_rate = A0_RESP * tadjst * a0_moist_factor
        a0_c_resp = A0_c * a0_resp_rate

        # Calculate C/N ratio of A0
        a0_cn = 30.0
        if A0_n > 0.0001:
            a0_cn = A0_c / A0_n

        # N mineralization from A0
        n_efficiency = 0.0
        if a0_cn > A0_CN_TARGET:
            n_efficiency = 0.5 * (a0_cn - A0_CN_TARGET) / a0_cn

        a0_n_release = a0_c_resp / a0_cn * n_efficiency
        if a0_n_release < 0.0:
            a0_n_release = 0.0
        if a0_n_release > A0_n * 0.01:
            a0_n_release = A0_n * 0.01

        # Transfer to A layer
        a0_to_a_c = a0_c_resp * A0_TO_A_EFF
        a0_to_a_n = a0_n_release * A0_TO_A_EFF

        # Update A0 pools
        A0_c = A0_c - a0_c_resp
        A0_n = A0_n - a0_n_release
        if A0_c < 0.0:
            A0_c = 0.0
        if A0_n < 0.0:
            A0_n = 0.0

        # ===== A LAYER DECOMPOSITION =====
        A_c = A_c + a0_to_a_c
        A_n = A_n + a0_to_a_n

        a_resp_rate = A_RESP * tadjst * a_moist_factor
        a_c_resp = A_c * a_resp_rate

        # Calculate C/N ratio of A
        a_cn = 4.0
        if A_n > 0.0001:
            a_cn = A_c / A_n

        # N mineralization from A layer (main N source)
        a_n_efficiency = 0.0
        if a_cn > A_CN_TARGET:
            a_n_efficiency = 0.8 * (a_cn - A_CN_TARGET) / a_cn

        a_n_release = a_c_resp / a_cn * a_n_efficiency
        if a_n_release < 0.0:
            a_n_release = 0.0
        if a_n_release > A_n * 0.01:
            a_n_release = A_n * 0.01

        # Transfer to Base layer
        a_to_bl_c = a_c_resp * A_TO_BL_EFF
        a_to_bl_n = a_n_release * A_TO_BL_EFF

        # Update A pools
        A_c = A_c - a_c_resp
        A_n = A_n - a_n_release
        if A_c < 0.0:
            A_c = 0.0
        if A_n < 0.0:
            A_n = 0.0

        # ===== BASE LAYER DECOMPOSITION =====
        BL_c = BL_c + a_to_bl_c
        BL_n = BL_n + a_to_bl_n

        bl_resp_rate = BL_RESP * tadjst
        bl_c_resp = BL_c * bl_resp_rate

        bl_n_release = bl_c_resp / BL_CN_TARGET * 0.1
        if bl_n_release > BL_n * 0.001:
            bl_n_release = BL_n * 0.001

        # Update Base pools
        BL_c = BL_c - bl_c_resp
        BL_n = BL_n - bl_n_release
        if BL_c < 0.0:
            BL_c = 0.0
        if BL_n < 0.0:
            BL_n = 0.0

        # Accumulate available N and respiration
        daily_avail_n = a0_n_release + a_n_release + bl_n_release
        daily_resp = a0_c_resp + a_c_resp + bl_c_resp

        total_avail_n = total_avail_n + daily_avail_n
        total_resp = total_resp + daily_resp

        day = day + 1

    # ===== WRITE FINAL RESULTS =====
    # soil: C/N pools
    soil_tensor[agent_index][SOIL_A0_C] = A0_c
    soil_tensor[agent_index][SOIL_A0_N] = A0_n
    soil_tensor[agent_index][SOIL_A_C] = A_c
    soil_tensor[agent_index][SOIL_A_N] = A_n
    soil_tensor[agent_index][SOIL_BL_C] = BL_c
    soil_tensor[agent_index][SOIL_BL_N] = BL_n

    # state: avail_n for Gap to read
    state_tensor[agent_index][S_AVAIL_N] = total_avail_n

    # output: soil_resp
    output_tensor[agent_index][O_SOIL_RESP] = total_resp
