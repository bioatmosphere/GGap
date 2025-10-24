# Auto-generated GPU kernel with cross-breed synchronization
# Contains all necessary imports and modified step functions

import os
import sys
module_path = os.path.abspath('/global/cfs/cdirs/m2467/wangb/GGap/gap/gap')
if module_path not in sys.path:
	sys.path.append(module_path)
from tree_step_func import *

# Modified step functions with double buffering
@jit.rawkernel(device='cuda')
def light_step_double_buffer(tick, agent_index, globals, agent_ids, breeds, locations, species_id_tensor, is_alive_tensor, age_tensor, diam_bht_tensor, forska_ht_tensor, canopy_ht_tensor, biomC_tensor, biomN_tensor, leaf_bm_tensor, x_tensor, y_tensor, light_avail_tensor, fc_degday_tensor, fc_drought_tensor, fc_flood_tensor, growth_factor_tensor, write_is_alive_tensor, write_age_tensor, write_diam_bht_tensor, write_forska_ht_tensor, write_biomC_tensor, write_light_avail_tensor, write_growth_factor_tensor):
    """
    Calculate light availability for each tree based on shading from neighbors.

    Taller neighboring trees cast shade, reducing light availability.
    This implements a simplified version of UVAFME's canopy light model.

    Global parameters:
    [0] neighborhood_radius: Distance threshold for light competition (meters)
    [1] deg_days: Annual degree days for temperature response
    [2] dry_days: Annual drought days
    """
    neighborhood_radius = globals[0]
    is_alive = int(get_this_agent_data_from_tensor(agent_index, is_alive_tensor))
    if is_alive == 1:
        height = get_this_agent_data_from_tensor(agent_index, forska_ht_tensor)
        x = get_this_agent_data_from_tensor(agent_index, x_tensor)
        y = get_this_agent_data_from_tensor(agent_index, y_tensor)
        species_id = int(get_this_agent_data_from_tensor(agent_index, species_id_tensor))
        light_available = 1.0
        light_available = 1.0
        set_this_agent_data_from_tensor(agent_index, write_light_avail_tensor, light_available)

@jit.rawkernel(device='cuda')
def growth_step_double_buffer(tick, agent_index, globals, agent_ids, breeds, locations, species_id_tensor, is_alive_tensor, age_tensor, diam_bht_tensor, forska_ht_tensor, canopy_ht_tensor, biomC_tensor, biomN_tensor, leaf_bm_tensor, x_tensor, y_tensor, light_avail_tensor, fc_degday_tensor, fc_drought_tensor, fc_flood_tensor, growth_factor_tensor, write_is_alive_tensor, write_age_tensor, write_diam_bht_tensor, write_forska_ht_tensor, write_biomC_tensor, write_light_avail_tensor, write_growth_factor_tensor):
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
    deg_days = globals[1]
    dry_days = globals[2]
    is_alive = int(get_this_agent_data_from_tensor(agent_index, is_alive_tensor))
    if is_alive == 1:
        species_id = int(get_this_agent_data_from_tensor(agent_index, species_id_tensor))
        age = get_this_agent_data_from_tensor(agent_index, age_tensor)
        diam = get_this_agent_data_from_tensor(agent_index, diam_bht_tensor)
        height = get_this_agent_data_from_tensor(agent_index, forska_ht_tensor)
        light_avail = get_this_agent_data_from_tensor(agent_index, light_avail_tensor)
        fc_degday = get_this_agent_data_from_tensor(agent_index, fc_degday_tensor)
        fc_drought = get_this_agent_data_from_tensor(agent_index, fc_drought_tensor)
        max_age = 150.0
        max_diam = 90.0
        max_ht = 25.0
        arfa_0 = 0.35
        g = 100.0
        shade_tol = 3
        deg_day_min = 600.0
        deg_day_opt = 2500.0
        deg_day_max = 4500.0
        if species_id == 1:
            max_age = 150.0
            max_diam = 90.0
            max_ht = 25.0
            arfa_0 = 0.35
            g = 100.0
            shade_tol = 3
            deg_day_min = 600.0
            deg_day_opt = 2500.0
            deg_day_max = 4500.0
        elif species_id == 2:
            max_age = 200.0
            max_diam = 120.0
            max_ht = 35.0
            arfa_0 = 0.4
            g = 120.0
            shade_tol = 2
            deg_day_min = 1200.0
            deg_day_opt = 3000.0
            deg_day_max = 5000.0
        elif species_id == 3:
            max_age = 300.0
            max_diam = 150.0
            max_ht = 30.0
            arfa_0 = 0.3
            g = 80.0
            shade_tol = 3
            deg_day_min = 800.0
            deg_day_opt = 2800.0
            deg_day_max = 4800.0
        elif species_id == 4:
            max_age = 200.0
            max_diam = 120.0
            max_ht = 28.0
            arfa_0 = 0.38
            g = 95.0
            shade_tol = 2
            deg_day_min = 1000.0
            deg_day_opt = 2900.0
            deg_day_max = 5200.0
        elif species_id == 5:
            max_age = 400.0
            max_diam = 140.0
            max_ht = 32.0
            arfa_0 = 0.28
            g = 70.0
            shade_tol = 5
            deg_day_min = 300.0
            deg_day_opt = 2000.0
            deg_day_max = 3500.0
        elif species_id == 6:
            max_age = 200.0
            max_diam = 150.0
            max_ht = 35.0
            arfa_0 = 0.42
            g = 130.0
            shade_tol = 1
            deg_day_min = 900.0
            deg_day_opt = 2700.0
            deg_day_max = 4800.0
        fc_temp = 0.0
        if deg_days >= deg_day_max or deg_days <= deg_day_min:
            fc_temp = 0.0
        else:
            a = (deg_day_opt - deg_day_min) / (deg_day_max - deg_day_min)
            b = (deg_day_max - deg_day_opt) / (deg_day_max - deg_day_min)
            tmp1 = (deg_days - deg_day_min) / (deg_day_opt - deg_day_min)
            tmp2 = (deg_day_max - deg_days) / (deg_day_max - deg_day_opt)
            fc_temp = tmp1 ** a * tmp2 ** b
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
        growth_factor = fc_temp * fc_drought * fc_light
        if growth_factor < 0.0:
            growth_factor = 0.0
        if growth_factor > 1.0:
            growth_factor = 1.0
        diam_increment = 0.0
        if diam < max_diam:
            base_growth = g * growth_factor / 100.0
            size_factor = 1.0 - diam / max_diam
            diam_increment = base_growth * size_factor
        new_diam = diam + diam_increment
        if new_diam > max_diam:
            new_diam = max_diam
        STD_HT = 1.3
        delta_ht = max_ht - STD_HT
        new_height = STD_HT + delta_ht * (1.0 - cp.exp(-(arfa_0 * new_diam / delta_ht)))
        PI = 3.14159265359
        wood_bulk_dens = 0.54
        radius_m = new_diam / 200.0
        volume_m3 = PI * radius_m * radius_m * new_height
        biomass_kg = volume_m3 * wood_bulk_dens * 1000.0
        new_biomC = biomass_kg * 0.5
        new_age = age + 1.0
        set_this_agent_data_from_tensor(agent_index, write_age_tensor, new_age)
        set_this_agent_data_from_tensor(agent_index, write_diam_bht_tensor, new_diam)
        set_this_agent_data_from_tensor(agent_index, write_forska_ht_tensor, new_height)
        set_this_agent_data_from_tensor(agent_index, write_biomC_tensor, new_biomC)
        set_this_agent_data_from_tensor(agent_index, write_growth_factor_tensor, growth_factor)

@jit.rawkernel(device='cuda')
def mortality_step_double_buffer(tick, agent_index, globals, agent_ids, breeds, locations, species_id_tensor, is_alive_tensor, age_tensor, diam_bht_tensor, forska_ht_tensor, canopy_ht_tensor, biomC_tensor, biomN_tensor, leaf_bm_tensor, x_tensor, y_tensor, light_avail_tensor, fc_degday_tensor, fc_drought_tensor, fc_flood_tensor, growth_factor_tensor, write_is_alive_tensor, write_age_tensor, write_diam_bht_tensor, write_forska_ht_tensor, write_biomC_tensor, write_light_avail_tensor, write_growth_factor_tensor):
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
    base_mortality = globals[3]
    is_alive = int(get_this_agent_data_from_tensor(agent_index, is_alive_tensor))
    if is_alive == 1:
        species_id = int(get_this_agent_data_from_tensor(agent_index, species_id_tensor))
        age = get_this_agent_data_from_tensor(agent_index, age_tensor)
        growth_factor = get_this_agent_data_from_tensor(agent_index, growth_factor_tensor)
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
        age_factor = age / max_age
        age_mortality = base_mortality * age_factor
        stress_factor = 1.0 - growth_factor
        stress_mortality = base_mortality * stress_factor * 2.0
        total_mortality = age_mortality + stress_mortality
        if total_mortality > 1.0:
            total_mortality = 1.0
        rand_val = cp.float32((tick * 997 + agent_index * 991) % 1000) / 1000.0
        if rand_val < total_mortality:
            set_this_agent_data_from_tensor(agent_index, write_is_alive_tensor, 0.0)
        else:
            set_this_agent_data_from_tensor(agent_index, write_is_alive_tensor, 1.0)

@jit.rawkernel(device='cuda')
def stepfunc(
global_tick,
device_global_data_vector,
a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,write_a3,write_a4,write_a5,write_a6,write_a8,write_a13,write_a17,
sync_workers_every_n_ticks,
num_rank_local_agents,
agent_ids,
current_priority_index,
):
	thread_id = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
	agent_index = thread_id
	if agent_index < num_rank_local_agents:
		breed_id = a0[agent_index]
		for tick in range(sync_workers_every_n_ticks):
			thread_local_tick = int(global_tick) + tick

			if current_priority_index == 0:
				if breed_id == 0:
					light_step_double_buffer(
						thread_local_tick,
						agent_index,
						device_global_data_vector,
						agent_ids,
						a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,write_a3,write_a4,write_a5,write_a6,write_a8,write_a13,write_a17,
					)
			if current_priority_index == 1:
				if breed_id == 0:
					growth_step_double_buffer(
						thread_local_tick,
						agent_index,
						device_global_data_vector,
						agent_ids,
						a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,write_a3,write_a4,write_a5,write_a6,write_a8,write_a13,write_a17,
					)
			if current_priority_index == 2:
				if breed_id == 0:
					mortality_step_double_buffer(
						thread_local_tick,
						agent_index,
						device_global_data_vector,
						agent_ids,
						a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,write_a3,write_a4,write_a5,write_a6,write_a8,write_a13,write_a17,
					)