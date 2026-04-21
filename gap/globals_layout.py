"""
Globals layout constants for GGap model.

Defines the memory layout for SAGESim global properties that store
constant species traits and site configuration data. All step functions
copy these as integer constants (CuPy JIT can't import).

Layout:
    globals_data[0]                          = num_species
    globals_data[1]                          = num_sites
    globals_data[2 + sp*26 + trait]          = species trait value
    globals_data[1302 + site_id*107 + cfg]   = site config value
    globals_data[RANGELIST_BASE + site_id*50 + sp] = 1.0 or 0.0
"""

# === Header ===
GLOB_NUM_SPECIES = 0
GLOB_NUM_SITES = 1

# === Species traits block ===
NUM_SPECIES_TRAITS = 26
SPECIES_BASE = 2
MAX_SPECIES = 50

# Trait offsets within a species block (globals_data[SPECIES_BASE + sp*26 + offset])
TRAIT_MAX_AGE = 0
TRAIT_MAX_DIAM = 1
TRAIT_MAX_HT = 2
TRAIT_ARFA_0 = 3
TRAIT_G = 4
TRAIT_SHADE_TOL = 5
TRAIT_DEG_DAY_MIN = 6
TRAIT_DEG_DAY_OPT = 7
TRAIT_DEG_DAY_MAX = 8
TRAIT_INVADER = 9
TRAIT_SEED = 10
TRAIT_SPROUT = 11
TRAIT_WOOD_BULK_DENS = 12
TRAIT_LOWNUTR_TOL = 13
TRAIT_FLOOD_TOL = 14
TRAIT_DROUGHT_TOL = 15
TRAIT_EVERGREEN = 16
TRAIT_FIRE_TOL = 17
TRAIT_ROOTDEPTH = 18
TRAIT_STRESS_TOL = 19
TRAIT_AGE_TOL = 20
TRAIT_SEED_SURV = 21
TRAIT_SEEDLING_LG = 22
TRAIT_LEAFDIAM_A = 23
TRAIT_LEAFAREA_C = 24
TRAIT_MAX_DISPERSAL_DIST = 25

# === Site config block ===
NUM_SITE_CONFIGS = 107
SITE_CONFIG_BASE = SPECIES_BASE + MAX_SPECIES * NUM_SPECIES_TRAITS  # 1302
MAX_SITES = 20

# Config offsets within a site block (globals_data[SITE_CONFIG_BASE + site_id*107 + offset])
CFG_TMIN_BASE = 0       # tmin[0..11] at offsets 0-11
CFG_TMAX_BASE = 12      # tmax[0..11] at offsets 12-23
CFG_PRCP_BASE = 24      # prcp[0..11] at offsets 24-35
CFG_FIELD_CAP = 36
CFG_PERM_WP = 37
CFG_SLOPE = 38
CFG_SIGMA = 39
CFG_LAI = 40
CFG_LATITUDE = 41
CFG_LONGITUDE = 42
CFG_RAIN_N = 43
CFG_FIRE_PROB = 44
CFG_WIND_PROB = 45
CFG_BASE_H = 46
CFG_TMIN_STD_BASE = 47  # tmin_std[0..11] at offsets 47-58
CFG_TMAX_STD_BASE = 59  # tmax_std[0..11] at offsets 59-70
CFG_PRCP_STD_BASE = 71  # prcp_std[0..11] at offsets 71-82
CFG_TMP_LAPSE_BASE = 83  # temp_lapse[0..11] at offsets 83-94
CFG_PRCP_LAPSE_BASE = 95  # prcp_lapse[0..11] at offsets 95-106

# === Rangelist block ===
RANGELIST_BASE = SITE_CONFIG_BASE + MAX_SITES * NUM_SITE_CONFIGS  # 3442
# globals_data[RANGELIST_BASE + site_id * MAX_SPECIES + sp] = 1.0 or 0.0

# Total globals size for N sites:
# RANGELIST_BASE + MAX_SITES * MAX_SPECIES = 3442 + 20*50 = 4442
GLOBALS_SIZE = RANGELIST_BASE + MAX_SITES * MAX_SPECIES  # 4442
