"""
Site N balance step function for GGap model (Priority 10).
Closes the nitrogen loop in the SAME tick (no 1-tick delay).

Matches GAPpy model.py:993-1005 where N balance is applied at end of renewal(),
after all growth and recruitment N consumption is known.

Execution Flow:
    1. Read avail_N from own states (computed at P1, same tick)
    2. Read annual_runoff from own params (computed at P1, same tick)
    3. Read total N consumed from all Gap neighbors (aggregated at P9, same tick)
    4. Apply unit conversion (kg -> tn/ha)
    5. Compute surplus = avail_N - total_N_consumed
    6. If surplus > 0: return to A layer minus leach fraction
    7. If surplus <= 0: debit A layer
    8. Apply runoff leaching (always)
    9. Transfer leached N*20 C to base layer
"""

import cupy as cp  # noqa: F401
from cupyx import jit

# === Breed IDs ===
BREED_TREE = 0
BREED_GAP = 1
BREED_SITE = 2

# === Site params (private, soil pools adjusted in-place) ===
SITE_P_A_C = 2
SITE_P_A_N = 3
SITE_P_BL_C = 4
SITE_P_BL_N = 5
SITE_P_ANNUAL_RUNOFF = 92

# === Site states[8] (public) ===
SITE_S_AVAIL_N = 2

# === Gap states[16] (for reading N consumed from Gap neighbors) ===
GAP_S_N_CONSUMED = 13

# Unit conversion: kg (tree-level) -> tn/ha (soil pools)
# = HEC_TO_M2 / plotsize / 1000 = 10000 / 500 / 1000
UNIT_CONV = 0.02


@jit.rawkernel(device="cuda")
def site_nbalance_step(
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
    Site N balance step (priority 10).

    Reads avail_N (P1 same tick), N consumed (P9 same tick), annual_runoff (P1 same tick).
    Applies surplus/deficit to A layer, leaching to base layer.
    Matches GAPpy model.py:993-1005.
    """
    # Read avail_N computed by P1 this tick
    avail_n = states_tensor[agent_index][SITE_S_AVAIL_N]

    # Read annual runoff computed by P1 this tick
    annual_runoff = params_tensor[agent_index][SITE_P_ANNUAL_RUNOFF]

    # Sum N consumed from all Gap neighbors (aggregated at P9, same tick)
    total_n_consumed = 0.0
    gap_count = 0.0

    neighbor_indices = locations[agent_index]
    i = 0
    while i < len(neighbor_indices) and neighbor_indices[i] != -1:
        neighbor_idx = int(neighbor_indices[i])
        neighbor_breed = int(breeds[neighbor_idx])

        if neighbor_breed == BREED_GAP:
            gap_n_consumed = states_tensor[neighbor_idx][GAP_S_N_CONSUMED]
            total_n_consumed = total_n_consumed + gap_n_consumed
            gap_count = gap_count + 1.0

        i = i + 1

    # Convert N consumed from kg to tn/ha (GAPpy uconvert)
    if gap_count > 0.5:
        total_n_consumed = total_n_consumed * UNIT_CONV / gap_count

    # Read current soil pools (written by P1 same tick, params has no double buffer)
    sa_n0 = params_tensor[agent_index][SITE_P_A_N]
    sa_c0 = params_tensor[agent_index][SITE_P_A_C]
    sb_c0 = params_tensor[agent_index][SITE_P_BL_C]
    sb_n0 = params_tensor[agent_index][SITE_P_BL_N]

    # Surplus = available N - consumed N (GAPpy model.py:993)
    surplus = avail_n - total_n_consumed
    net_n_leach = 0.0

    if surplus > 0.0:
        # Fraction leached via runoff (capped at 10%, GAPpy model.py:994)
        leach_frac = annual_runoff / 1000.0
        if leach_frac > 0.1:
            leach_frac = 0.1
        net_n_leach = surplus * leach_frac
        # Return remainder to A layer (GAPpy model.py:995)
        sa_n0 = sa_n0 + surplus - net_n_leach
    else:
        # Deficit: debit A layer (GAPpy model.py:997)
        sa_n0 = sa_n0 + surplus  # surplus is negative
        net_n_leach = 0.0

    # Runoff leaching from A layer (always applied, GAPpy model.py:1000)
    sa_n0 = sa_n0 - 0.00002 * annual_runoff

    # Transfer leached N and C to base layer (GAPpy model.py:1002-1005)
    sa_c0 = sa_c0 - net_n_leach * 20.0
    sb_c0 = sb_c0 + net_n_leach * 20.0
    sb_n0 = sb_n0 + net_n_leach

    # Write adjusted soil pools back
    params_tensor[agent_index][SITE_P_A_N] = sa_n0
    params_tensor[agent_index][SITE_P_A_C] = sa_c0
    params_tensor[agent_index][SITE_P_BL_C] = sb_c0
    params_tensor[agent_index][SITE_P_BL_N] = sb_n0
