"""
GGap model constants — single source of truth.

All index constants, physics values, and CSV mappings for the GGap model.
Step functions import only the constants they need.
"""
from enum import IntEnum


# ============================================================
# Breed IDs
# ============================================================
class Breed(IntEnum):
    TREE = 0
    GAP = 1
    SITE = 2


# ============================================================
# Species trait column indices (species_traits tensor)
# ============================================================
class Trait(IntEnum):
    MAX_AGE = 0
    MAX_DIAM = 1          # CSV: DBHmax
    MAX_HT = 2            # CSV: Hmax
    ARFA_0 = 3            # CSV: s
    G = 4                 # CSV: g (adjusted by shade_tol)
    SHADE_TOL = 5         # CSV: l
    DEG_DAY_MIN = 6       # CSV: DEGDmin
    DEG_DAY_OPT = 7       # CSV: DEGDoptimum
    DEG_DAY_MAX = 8       # CSV: DEGDmax
    INVADER = 9
    SEED = 10
    SPROUT = 11
    WOOD_BULK_DENS = 12   # CSV: bulk
    LOWNUTR_TOL = 13      # CSV: n
    FLOOD_TOL = 14        # CSV: f
    DROUGHT_TOL = 15      # CSV: d
    EVERGREEN = 16
    FIRE_TOL = 17         # CSV: fire
    ROOTDEPTH = 18
    STRESS_TOL = 19       # CSV: stress
    AGE_TOL = 20          # CSV: old
    SEED_SURV = 21        # CSV: NDE
    SEEDLING_LG = 22      # CSV: NDS
    LEAFDIAM_A = 23       # CSV: D_L (adjusted)
    LEAFAREA_C = 24       # CSV: L_C (adjusted)
    MAX_DISPERSAL_DIST = 25

NUM_SPECIES_TRAITS = 26

# CSV column name → Trait index mapping
TRAIT_CSV_MAPPING = {
    'AGEmax': Trait.MAX_AGE,
    'DBHmax': Trait.MAX_DIAM,
    'Hmax': Trait.MAX_HT,
    's': Trait.ARFA_0,
    'g': Trait.G,
    'l': Trait.SHADE_TOL,
    'DEGDmin': Trait.DEG_DAY_MIN,
    'DEGDoptimum': Trait.DEG_DAY_OPT,
    'DEGDmax': Trait.DEG_DAY_MAX,
    'invader': Trait.INVADER,
    'seed': Trait.SEED,
    'sprout': Trait.SPROUT,
    'bulk': Trait.WOOD_BULK_DENS,
    'n': Trait.LOWNUTR_TOL,
    'f': Trait.FLOOD_TOL,
    'd': Trait.DROUGHT_TOL,
    'evergreen': Trait.EVERGREEN,
    'fire': Trait.FIRE_TOL,
    'rootdepth': Trait.ROOTDEPTH,
    'stress': Trait.STRESS_TOL,
    'old': Trait.AGE_TOL,
    'NDE': Trait.SEED_SURV,
    'NDS': Trait.SEEDLING_LG,
    'D_L': Trait.LEAFDIAM_A,
    'L_C': Trait.LEAFAREA_C,
}


# ============================================================
# Site config column indices (site_configs tensor)
# ============================================================
class Cfg(IntEnum):
    TMIN_BASE = 0         # tmin[0..11] at offsets 0-11
    TMAX_BASE = 12        # tmax[0..11] at offsets 12-23
    PRCP_BASE = 24        # prcp[0..11] at offsets 24-35
    FIELD_CAP = 36
    PERM_WP = 37
    SLOPE = 38
    SIGMA = 39
    LAI = 40
    LATITUDE = 41
    LONGITUDE = 42
    RAIN_N = 43
    FIRE_PROB = 44
    WIND_PROB = 45
    BASE_H = 46
    TMIN_STD_BASE = 47    # tmin_std[0..11] at offsets 47-58
    TMAX_STD_BASE = 59    # tmax_std[0..11] at offsets 59-70
    PRCP_STD_BASE = 71    # prcp_std[0..11] at offsets 71-82
    TMP_LAPSE_BASE = 83   # temp_lapse[0..11] at offsets 83-94
    PRCP_LAPSE_BASE = 95  # prcp_lapse[0..11] at offsets 95-106

NUM_SITE_CONFIGS = 107


# ============================================================
# Tree params indices (params_tensor columns, private)
# ============================================================
class TreeP(IntEnum):
    SPECIES_ID = 0
    AGE = 1
    BIOMC = 2
    BIOMN = 3
    LEAF_BM = 4
    X = 5
    Y = 6
    LIGHT_AVAIL = 7
    FC_DEGDAY = 8
    FC_DROUGHT = 9
    FC_FLOOD = 10
    ENV_STRESS = 11
    DIAM_MAX_CALC = 12
    FORSKA_SHADE = 13
    SEEDBANK = 14
    SEEDLING = 15
    SEEDLING_WEIGHT = 16


# ============================================================
# Tree states indices (states_tensor columns, public)
# ============================================================
class TreeS(IntEnum):
    LITTER_C = 0          # Above-ground litter carbon
    LITTER_N = 1          # Above-ground litter nitrogen
    N_DEMAND = 2
    N_CONSUMED = 3        # Actual N consumed this tick
    LITTER_N_BG = 4       # Below-ground litter nitrogen (roots)


# ============================================================
# Tree states_db indices (states_db_tensor columns, double buffered)
# ============================================================
class TreeDB(IntEnum):
    IS_ALIVE = 0
    DIAM = 1
    HEIGHT = 2
    CANOPY_HT = 3
    SEEDLING_WEIGHT = 4


# ============================================================
# Gap params indices
# ============================================================
class GapP(IntEnum):
    SITE_ID = 0
    TOTAL_N_DEMAND = 1


# ============================================================
# Gap states indices (states_tensor columns, public)
# ============================================================
class GapS(IntEnum):
    DEG_DAYS = 0
    DRY_DAYS = 1
    AVAIL_N = 2
    N_SUPPLY_RATIO = 3
    LITTER_ACCUM_C = 4
    LITTER_ACCUM_N = 5
    NUM_TO_RECRUIT = 6
    RECRUIT_RAND_SEED = 7
    FLOOD_DAYS = 8
    TOTAL_SEEDLING_WEIGHT = 9
    FIRE_INTENSITY = 10
    TOTAL_N_DEMAND = 11
    TOTAL_LAI = 12
    N_CONSUMED = 13
    DRY_DAYS_BASE = 14
    WIND_INTENSITY = 15
    RECOVERY_YEARS = 16
    # Total: 17 columns (was 167+num_species)
    # CUM_DEC_LAI, CUM_CON_LAI, AVAIL_SPEC moved to breed-local arrays


# ============================================================
# Site params indices (params_tensor columns, private)
# ============================================================
class SiteP(IntEnum):
    A0_C = 0
    A0_N = 1
    A_C = 2
    A_N = 3
    BL_C = 4
    BL_N = 5
    A0_W = 6
    A_W = 7
    BL_W = 8
    LAI_W0 = 9
    ANNUAL_RUNOFF = 10
    SITE_ID = 11


# ============================================================
# Site states indices (states_tensor columns, public)
# ============================================================
class SiteS(IntEnum):
    DEG_DAYS = 0
    DRY_DAYS = 1
    AVAIL_N = 2
    FLOOD_DAYS = 3
    FIRE_INTENSITY = 4
    N_SUPPLY_RATIO = 5
    DRY_DAYS_BASE = 6
    WIND_INTENSITY = 7
    ANNUAL_RAIN = 8
    GROW_DAYS = 9
    POT_EVAP = 10
    ACT_EVAP = 11
    SOIL_RESP = 12
    C_INTO_A0 = 13
    N_INTO_A0 = 14
    NET_N_INTO_A0 = 15
    # Total: 16 columns (was 16+2*num_species)
    # SITE_AVAIL_SPEC and IMPORTED_SEEDS moved to breed-local arrays


# ============================================================
# Physics / model constants
# ============================================================
PI = 3.14159265359
STD_HT = 1.3              # Standard height for DBH measurement (m)
TC_KG = 0.039269908        # PI / 80 — stem volume constant (cm, m, g/cm³)
XT = -0.40                 # Universal light extinction coefficient
PLOTSIZE = 500.0           # Plot area (m²), also max trees per plot
SEEDLING_DIAM = 1.0
SEEDLING_AGE = 1.0

# Biomass C:N ratios
STEM_C_N = 450.0
CON_LEAF_C_N = 60.0
DEC_LEAF_C_N = 40.0
CON_LEAF_B = 1.3           # 1.0 + CON_LEAF_RATIO (0.3)

# Unit conversion: kg (tree-level) → tn/ha (soil pools)
# = HEC_TO_M2 / plotsize / 1000 = 10000 / 500 / 1000
UNIT_CONV = 0.02

# Max height bins for LAI profiles
MAX_HEIGHT_BINS = 50

# Dispersal constants
DISPERSAL_CUTOFF_FACTOR = 5.0  # sites within this × max_dispersal_dist are connected
EARTH_RADIUS_KM = 6371.0       # for haversine distance (CPU-side only)


# ============================================================
# Soil / decomposition constants
# ============================================================
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

# Hargreaves PET coefficients
DEG2RAD = 0.017453
H_B = 0.017214
H_AS = 0.409
H_AC = 0.033
H_PHASE = -1.39
H_AMP = 37.58603
H_COEFF = 0.000093876
H_ADDON = 17.8

# Days per month
DAYS_PER_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
# Individual constants for CuPy JIT (can't index lists)
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
