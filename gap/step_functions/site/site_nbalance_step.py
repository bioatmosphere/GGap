"""
Site N balance step function for GGap model (Priority 9).
Closes the nitrogen loop in the SAME tick (no 1-tick delay).

Matches GAPpy model.py:993-1005 where N balance is applied at end of renewal(),
after all growth and recruitment N consumption is known.

Execution Flow:
    1. Read avail_N from own states (computed at P1, same tick)
    2. Read annual_runoff from own params (computed at P1, same tick)
    3. Read total N consumed from all Gap neighbors (aggregated at P8, same tick)
    4. Apply unit conversion (kg -> tn/ha)
    5. Compute surplus = avail_N - total_N_consumed
    6. If surplus > 0: return to A layer minus leach fraction
    7. If surplus <= 0: debit A layer
    8. Apply runoff leaching (always)
    9. Transfer leached N*20 C to base layer
"""

import cupy as cp  # noqa: F401
from cupyx import jit

from gap.constants import (
    Breed, SiteP, SiteS, GapS,
    UNIT_CONV,
)


@jit.rawkernel(device="cuda")
def site_nbalance_step(
    tick,
    agent_index,
    species_traits, site_configs, rangelists, site_distances,
    agent_ids,
    breeds,
    locations,
    params_tensor,
    states_tensor,
    gap_lai, gap_species, site_species,
    gap_lai_idx, gap_species_idx, site_species_idx,
):
    """
    Site N balance step (priority 9).

    Reads avail_N (P1 same tick), N consumed (P8 same tick), annual_runoff (P1 same tick).
    Applies surplus/deficit to A layer, leaching to base layer.
    Matches GAPpy model.py:993-1005.
    """
    # Read avail_N computed by P1 this tick
    avail_n = states_tensor[agent_index][SiteS.AVAIL_N]

    # Read annual runoff computed by P1 this tick
    annual_runoff = params_tensor[agent_index][SiteP.ANNUAL_RUNOFF]

    # Sum N consumed from all Gap neighbors (aggregated at P8, same tick)
    total_n_consumed = 0.0
    gap_count = 0.0

    neighbor_indices = locations[agent_index]
    i = 0
    while i < len(neighbor_indices) and neighbor_indices[i] != -1:
        neighbor_idx = int(neighbor_indices[i])
        neighbor_breed = int(breeds[neighbor_idx])

        if neighbor_breed == Breed.GAP:
            gap_n_consumed = states_tensor[neighbor_idx][GapS.N_CONSUMED]
            total_n_consumed = total_n_consumed + gap_n_consumed
            gap_count = gap_count + 1.0

        i = i + 1

    # Convert N consumed from kg to tn/ha (GAPpy uconvert)
    if gap_count > 0.5:
        total_n_consumed = total_n_consumed * UNIT_CONV / gap_count

    # Read current soil pools (written by P1 same tick, params has no double buffer)
    sa_n0 = params_tensor[agent_index][SiteP.A_N]
    sa_c0 = params_tensor[agent_index][SiteP.A_C]
    sb_c0 = params_tensor[agent_index][SiteP.BL_C]
    sb_n0 = params_tensor[agent_index][SiteP.BL_N]

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
    params_tensor[agent_index][SiteP.A_N] = sa_n0
    params_tensor[agent_index][SiteP.A_C] = sa_c0
    params_tensor[agent_index][SiteP.BL_C] = sb_c0
    params_tensor[agent_index][SiteP.BL_N] = sb_n0

    # Export net N leached for CSV output (output-only in params)
    params_tensor[agent_index][SiteP.NET_N_INTO_A0] = net_n_leach
