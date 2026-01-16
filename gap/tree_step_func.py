"""
Tree step functions for GGap model.
GPU kernels implementing UVAFME-based forest gap dynamics.
"""

import sys
import os

# Add SAGESim to path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_sagesim_path = os.path.join(os.path.dirname(_current_dir), "SAGESim")
if _sagesim_path not in sys.path:
    sys.path.insert(0, _sagesim_path)

import cupy as cp
from cupyx import jit
from sagesim.utils import (
    get_this_agent_data_from_tensor,
    set_this_agent_data_from_tensor,
)
# Note: get_neighbor_data_from_tensor is DEPRECATED in SAGESim v0.3.0+
# Use direct indexing instead: property_tensor[neighbor_index]
# The 'locations' property now contains pre-converted indices, not agent IDs


@jit.rawkernel(device="cuda")
def light_step(
    tick,
    agent_index,
    globals,
    agent_ids,
    breeds,
    locations,
    species_id_tensor,
    is_alive_tensor,
    age_tensor,
    diam_bht_tensor,
    forska_ht_tensor,
    canopy_ht_tensor,
    biomC_tensor,
    biomN_tensor,
    leaf_bm_tensor,
    x_tensor,
    y_tensor,
    light_avail_tensor,
    fc_degday_tensor,
    fc_drought_tensor,
    fc_flood_tensor,
    growth_factor_tensor
):
    """
    Calculate light availability for each tree based on shading from neighbors.

    Taller neighboring trees cast shade, reducing light availability.
    This implements a simplified version of UVAFME's canopy light model.

    Global parameters:
    [0] neighborhood_radius: Distance threshold for light competition (meters)
    [1] deg_days: Annual degree days for temperature response
    [2] dry_days: Annual drought days
    """
    # Get global parameters
    neighborhood_radius = globals[0]

    # Get tree properties
    is_alive = int(get_this_agent_data_from_tensor(agent_index, is_alive_tensor))

    # Only process living trees
    if is_alive == 1:
        height = get_this_agent_data_from_tensor(agent_index, forska_ht_tensor)
        x = get_this_agent_data_from_tensor(agent_index, x_tensor)
        y = get_this_agent_data_from_tensor(agent_index, y_tensor)
        species_id = int(get_this_agent_data_from_tensor(agent_index, species_id_tensor))

        # Start with full light
        light_available = 1.0

        # For initial implementation, use simple light availability
        # Full model would calculate shading from neighbors
        light_available = 1.0

        # Write light availability (SAGESim handles write buffering)
        set_this_agent_data_from_tensor(
            agent_index, light_avail_tensor, light_available
        )


@jit.rawkernel(device="cuda")
def growth_step(
    tick,
    agent_index,
    globals,
    agent_ids,
    breeds,
    locations,
    species_id_tensor,
    is_alive_tensor,
    age_tensor,
    diam_bht_tensor,
    forska_ht_tensor,
    canopy_ht_tensor,
    biomC_tensor,
    biomN_tensor,
    leaf_bm_tensor,
    x_tensor,
    y_tensor,
    light_avail_tensor,
    fc_degday_tensor,
    fc_drought_tensor,
    fc_flood_tensor,
    growth_factor_tensor
):
    """
    Calculate tree growth based on UVAFME equations.

    Growth is modulated by:
    - Species-specific growth rates
    - Environmental factors (temperature, drought, flood)
    - Light availability
    - Asymptotic approach to maximum size

    Global parameters:
    [0] neighborhood_radius
    [1] deg_days: Annual degree days
    [2] dry_days: Annual drought days
    """
    # Get global environmental parameters
    deg_days = globals[1]
    dry_days = globals[2]

    # Get tree properties
    is_alive = int(get_this_agent_data_from_tensor(agent_index, is_alive_tensor))

    if is_alive == 1:
        species_id = int(get_this_agent_data_from_tensor(agent_index, species_id_tensor))
        age = get_this_agent_data_from_tensor(agent_index, age_tensor)
        diam = get_this_agent_data_from_tensor(agent_index, diam_bht_tensor)
        height = get_this_agent_data_from_tensor(agent_index, forska_ht_tensor)
        light_avail = get_this_agent_data_from_tensor(agent_index, light_avail_tensor)
        fc_degday = get_this_agent_data_from_tensor(agent_index, fc_degday_tensor)
        fc_drought = get_this_agent_data_from_tensor(agent_index, fc_drought_tensor)

        # Species-specific parameters
        # Format: max_age, max_diam, max_ht, arfa_0, g, shade_tol
        max_age = 150.0
        max_diam = 90.0
        max_ht = 25.0
        arfa_0 = 0.35
        g = 100.0
        shade_tol = 3
        deg_day_min = 600.0
        deg_day_opt = 2500.0
        deg_day_max = 4500.0

        if species_id == 1:  # Red Maple
            max_age = 150.0
            max_diam = 90.0
            max_ht = 25.0
            arfa_0 = 0.35
            g = 100.0
            shade_tol = 3
            deg_day_min = 600.0
            deg_day_opt = 2500.0
            deg_day_max = 4500.0
        elif species_id == 2:  # Loblolly Pine
            max_age = 200.0
            max_diam = 120.0
            max_ht = 35.0
            arfa_0 = 0.40
            g = 120.0
            shade_tol = 2
            deg_day_min = 1200.0
            deg_day_opt = 3000.0
            deg_day_max = 5000.0
        elif species_id == 3:  # White Oak
            max_age = 300.0
            max_diam = 150.0
            max_ht = 30.0
            arfa_0 = 0.30
            g = 80.0
            shade_tol = 3
            deg_day_min = 800.0
            deg_day_opt = 2800.0
            deg_day_max = 4800.0
        elif species_id == 4:  # Sweetgum
            max_age = 200.0
            max_diam = 120.0
            max_ht = 28.0
            arfa_0 = 0.38
            g = 95.0
            shade_tol = 2
            deg_day_min = 1000.0
            deg_day_opt = 2900.0
            deg_day_max = 5200.0
        elif species_id == 5:  # Eastern Hemlock
            max_age = 400.0
            max_diam = 140.0
            max_ht = 32.0
            arfa_0 = 0.28
            g = 70.0
            shade_tol = 5
            deg_day_min = 300.0
            deg_day_opt = 2000.0
            deg_day_max = 3500.0
        elif species_id == 6:  # Tulip Poplar
            max_age = 200.0
            max_diam = 150.0
            max_ht = 35.0
            arfa_0 = 0.42
            g = 130.0
            shade_tol = 1
            deg_day_min = 900.0
            deg_day_opt = 2700.0
            deg_day_max = 4800.0

        # Calculate temperature response factor (parabolic response)
        fc_temp = 0.0
        if deg_days >= deg_day_max or deg_days <= deg_day_min:
            fc_temp = 0.0
        else:
            a = (deg_day_opt - deg_day_min) / (deg_day_max - deg_day_min)
            b = (deg_day_max - deg_day_opt) / (deg_day_max - deg_day_min)
            tmp1 = (deg_days - deg_day_min) / (deg_day_opt - deg_day_min)
            tmp2 = (deg_day_max - deg_days) / (deg_day_max - deg_day_opt)
            fc_temp = (tmp1 ** a) * (tmp2 ** b)

        # Calculate light response factor (exponential saturation)
        # Based on UVAFME light_rsp function
        light_c1 = 1.11
        light_c2 = 2.52
        light_c3 = 0.07

        if shade_tol == 1:
            light_c1 = 1.01
            light_c2 = 4.62
            light_c3 = 0.05
        elif shade_tol == 2:
            light_c1 = 1.04
            light_c2 = 3.44
            light_c3 = 0.06
        elif shade_tol == 3:
            light_c1 = 1.11
            light_c2 = 2.52
            light_c3 = 0.07
        elif shade_tol == 4:
            light_c1 = 1.24
            light_c2 = 1.78
            light_c3 = 0.08
        elif shade_tol == 5:
            light_c1 = 1.49
            light_c2 = 1.23
            light_c3 = 0.09

        fc_light = light_c1 * (1.0 - cp.exp(-light_c2 * (light_avail - light_c3)))
        if fc_light < 0.0:
            fc_light = 0.0
        if fc_light > 1.0:
            fc_light = 1.0

        # Combined growth factor
        growth_factor = fc_temp * fc_drought * fc_light
        if growth_factor < 0.0:
            growth_factor = 0.0
        if growth_factor > 1.0:
            growth_factor = 1.0

        # Diameter growth (UVAFME equation)
        # Growth rate scaled by environmental factors and approach to max size
        diam_increment = 0.0
        if diam < max_diam:
            # Basic growth rate
            base_growth = g * growth_factor / 100.0  # cm per year

            # Asymptotic approach to maximum diameter
            size_factor = 1.0 - (diam / max_diam)

            diam_increment = base_growth * size_factor

        new_diam = diam + diam_increment
        if new_diam > max_diam:
            new_diam = max_diam

        # Height calculation using Forska equation
        # h = STD_HT + (h_max - STD_HT) * (1 - exp(-arfa_0 * d / (h_max - STD_HT)))
        STD_HT = 1.3  # Standard height (breast height)
        delta_ht = max_ht - STD_HT

        new_height = STD_HT + delta_ht * (1.0 - cp.exp(-(arfa_0 * new_diam / delta_ht)))

        # Biomass calculation (simplified)
        # In full UVAFME, this involves stem shape calculations
        # Here we use allometric relationships
        # biomC ~ wood_bulk_dens * volume
        # Assuming cylindrical approximation: V = pi * (d/200)^2 * h
        PI = 3.14159265359
        wood_bulk_dens = 0.54  # g/cm3, varies by species

        radius_m = new_diam / 200.0  # Convert cm to meters, then diameter to radius
        volume_m3 = PI * radius_m * radius_m * new_height

        # Convert to biomass (rough approximation)
        # wood_bulk_dens is in g/cm3, need kg/m3
        biomass_kg = volume_m3 * wood_bulk_dens * 1000.0

        # Carbon content is approximately 50% of dry biomass
        new_biomC = biomass_kg * 0.5

        # Age increment
        new_age = age + 1.0

        # Write updates (SAGESim handles write buffering)
        set_this_agent_data_from_tensor(agent_index, age_tensor, new_age)
        set_this_agent_data_from_tensor(agent_index, diam_bht_tensor, new_diam)
        set_this_agent_data_from_tensor(agent_index, forska_ht_tensor, new_height)
        set_this_agent_data_from_tensor(agent_index, biomC_tensor, new_biomC)
        set_this_agent_data_from_tensor(agent_index, growth_factor_tensor, growth_factor)


@jit.rawkernel(device="cuda")
def mortality_step(
    tick,
    agent_index,
    globals,
    agent_ids,
    breeds,
    locations,
    species_id_tensor,
    is_alive_tensor,
    age_tensor,
    diam_bht_tensor,
    forska_ht_tensor,
    canopy_ht_tensor,
    biomC_tensor,
    biomN_tensor,
    leaf_bm_tensor,
    x_tensor,
    y_tensor,
    light_avail_tensor,
    fc_degday_tensor,
    fc_drought_tensor,
    fc_flood_tensor,
    growth_factor_tensor
):
    """
    Calculate tree mortality based on age and environmental stress.

    Mortality increases with:
    - Age (approaching max age)
    - Environmental stress (poor growing conditions)
    - Suppression (low growth factor)

    Global parameters:
    [0] neighborhood_radius
    [1] deg_days
    [2] dry_days
    [3] base_mortality_rate: Base annual mortality probability
    """
    # Get parameters
    base_mortality = globals[3]

    # Get tree properties
    is_alive = int(get_this_agent_data_from_tensor(agent_index, is_alive_tensor))

    if is_alive == 1:
        species_id = int(get_this_agent_data_from_tensor(agent_index, species_id_tensor))
        age = get_this_agent_data_from_tensor(agent_index, age_tensor)
        growth_factor = get_this_agent_data_from_tensor(agent_index, growth_factor_tensor)

        # Get species max age
        max_age = 150.0
        if species_id == 1:
            max_age = 150.0
        elif species_id == 2:
            max_age = 200.0
        elif species_id == 3:
            max_age = 300.0
        elif species_id == 4:
            max_age = 200.0
        elif species_id == 5:
            max_age = 400.0
        elif species_id == 6:
            max_age = 200.0

        # Age-related mortality (increases as tree approaches max age)
        age_factor = age / max_age
        age_mortality = base_mortality * age_factor

        # Stress-related mortality (poor growth conditions)
        stress_factor = 1.0 - growth_factor
        stress_mortality = base_mortality * stress_factor * 2.0  # Stress has double impact

        # Total mortality probability
        total_mortality = age_mortality + stress_mortality
        if total_mortality > 1.0:
            total_mortality = 1.0

        # Random mortality check
        # Note: random() in GPU context needs special handling
        # For now, use a deterministic threshold based on tick and agent_index
        # In production, would use proper random number generation
        rand_val = cp.float32((tick * 997 + agent_index * 991) % 1000) / 1000.0

        if rand_val < total_mortality:
            set_this_agent_data_from_tensor(agent_index, is_alive_tensor, 0.0)
        else:
            set_this_agent_data_from_tensor(agent_index, is_alive_tensor, 1.0)
