"""
Gap sync step function for GGap model (Priority 5).
Computes per-gap N supply ratio and clears accumulators.

Climate relay is now handled by gap_climate_relay_step (P2), which runs
before tree_potential_growth_step (P3) so trees read current-tick climate.

This step computes the per-gap N ratio (needs P4 demand) and clears
accumulators consumed by P1.

GAPpy computes N_supply_demand per-plot (model.py:475-488):
    N_req = max(N_req * HEC_TO_M2 / plotsize, 0.00001)
    N_supply_demand = site.soil.avail_N / N_req
Each plot gets its own ratio from its own trees' demand.
This step matches that by computing the ratio per-gap (not per-site).
"""

import cupy as cp  # noqa: F401
from cupyx import jit

# === Breed IDs ===
BREED_TREE = 0
BREED_GAP = 1
BREED_SITE = 2

# === Gap states[16] (public, no buffer) ===
GAP_S_AVAIL_N = 2             # avail_n (copied from Site at P2 climate relay)
GAP_S_N_SUPPLY_RATIO = 3
GAP_S_LITTER_ACCUM_C = 4
GAP_S_LITTER_ACCUM_N = 5
GAP_S_TOTAL_N_DEMAND = 11     # Own N demand from P4 (same tick)
GAP_S_TOTAL_LAI = 12          # Per-gap normalized LAI (from P0)
GAP_S_N_CONSUMED = 13         # N consumed by trees (written at P9, read at P10)

# Unit conversion: kg (tree-level) → tn/ha (soil pools)
# = HEC_TO_M2 / plotsize / 1000 = 10000 / 500 / 1000
UNIT_CONV = 0.02


@jit.rawkernel(device="cuda")
def gap_sync_step(
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
    Gap sync step (priority 5).

    Computes per-gap N supply ratio using avail_n (from P2 climate relay)
    and own N demand (from P4 demand aggregate).
    Clears accumulators consumed by P1.
    """
    # Read avail_n already copied to own states by P2 (gap_climate_relay_step)
    avail_n = states_tensor[agent_index][GAP_S_AVAIL_N]

    # Compute per-gap N supply ratio (GAPpy model.py:475-488)
    # Each gap uses its own N demand (from P4) with the site-wide avail_N
    gap_n_demand = states_tensor[agent_index][GAP_S_TOTAL_N_DEMAND]
    gap_n_demand_scaled = gap_n_demand * UNIT_CONV
    gap_n_supply_ratio = 1.0
    if gap_n_demand_scaled > 0.00001:
        gap_n_supply_ratio = avail_n / gap_n_demand_scaled
        if gap_n_supply_ratio > 1.0:
            gap_n_supply_ratio = 1.0
    states_tensor[agent_index][GAP_S_N_SUPPLY_RATIO] = gap_n_supply_ratio

    # Clear accumulators (consumed by P1, P0 rewrites next tick)
    states_tensor[agent_index][GAP_S_LITTER_ACCUM_C] = 0.0
    states_tensor[agent_index][GAP_S_LITTER_ACCUM_N] = 0.0
    states_tensor[agent_index][GAP_S_TOTAL_LAI] = 0.0
    states_tensor[agent_index][GAP_S_N_CONSUMED] = 0.0
    # Note: NUM_TO_RECRUIT and RECRUIT_RAND_SEED are NOT cleared here.
    # Trees read them at P8 (free slot activation). P7 overwrites them each tick.
