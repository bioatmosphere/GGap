"""
Tree species data for GGap model.
Based on UVAFME species parameters.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class SpeciesParameters:
    """Parameters for a tree species."""

    # Identification
    species_id: int
    genus_name: str
    common_name: str

    # Maximum limits
    max_age: float  # years
    max_diam: float  # cm
    max_ht: float  # meters

    # Growth parameters
    arfa_0: float  # Height-diameter relationship parameter
    g: float  # Diameter growth rate

    # Tolerance classes (1-5, 1=intolerant, 5=very tolerant)
    shade_tol: int
    drought_tol: int
    flood_tol: int

    # Temperature response (degree days)
    deg_day_min: float
    deg_day_opt: float
    deg_day_max: float

    # Other parameters
    wood_bulk_dens: float  # g/cm3
    rootdepth: float  # meters
    leafarea_c: float  # m2/m2
    conifer: bool


# Define common Eastern US species
SPECIES_DATA: Dict[int, SpeciesParameters] = {
    # Species 1: Red Maple (Acer rubrum) - Early/Mid successional
    1: SpeciesParameters(
        species_id=1,
        genus_name="Acer",
        common_name="Red Maple",
        max_age=150.0,
        max_diam=90.0,
        max_ht=25.0,
        arfa_0=0.35,
        g=100.0,
        shade_tol=3,  # Moderate shade tolerance
        drought_tol=3,
        flood_tol=4,  # Flood tolerant
        deg_day_min=600.0,
        deg_day_opt=2500.0,
        deg_day_max=4500.0,
        wood_bulk_dens=0.54,
        rootdepth=3.0,
        leafarea_c=4.5,
        conifer=False
    ),

    # Species 2: Loblolly Pine (Pinus taeda) - Early successional
    2: SpeciesParameters(
        species_id=2,
        genus_name="Pinus",
        common_name="Loblolly Pine",
        max_age=200.0,
        max_diam=120.0,
        max_ht=35.0,
        arfa_0=0.40,
        g=120.0,
        shade_tol=2,  # Shade intolerant
        drought_tol=3,
        flood_tol=3,
        deg_day_min=1200.0,
        deg_day_opt=3000.0,
        deg_day_max=5000.0,
        wood_bulk_dens=0.51,
        rootdepth=4.5,
        leafarea_c=6.0,
        conifer=True
    ),

    # Species 3: White Oak (Quercus alba) - Mid/Late successional
    3: SpeciesParameters(
        species_id=3,
        genus_name="Quercus",
        common_name="White Oak",
        max_age=300.0,
        max_diam=150.0,
        max_ht=30.0,
        arfa_0=0.30,
        g=80.0,
        shade_tol=3,
        drought_tol=2,  # Drought sensitive
        flood_tol=2,  # Flood sensitive
        deg_day_min=800.0,
        deg_day_opt=2800.0,
        deg_day_max=4800.0,
        wood_bulk_dens=0.68,
        rootdepth=4.0,
        leafarea_c=5.0,
        conifer=False
    ),

    # Species 4: Sweetgum (Liquidambar styraciflua) - Mid successional
    4: SpeciesParameters(
        species_id=4,
        genus_name="Liquidambar",
        common_name="Sweetgum",
        max_age=200.0,
        max_diam=120.0,
        max_ht=28.0,
        arfa_0=0.38,
        g=95.0,
        shade_tol=2,
        drought_tol=3,
        flood_tol=4,  # Flood tolerant
        deg_day_min=1000.0,
        deg_day_opt=2900.0,
        deg_day_max=5200.0,
        wood_bulk_dens=0.52,
        rootdepth=3.5,
        leafarea_c=4.8,
        conifer=False
    ),

    # Species 5: Eastern Hemlock (Tsuga canadensis) - Late successional
    5: SpeciesParameters(
        species_id=5,
        genus_name="Tsuga",
        common_name="Eastern Hemlock",
        max_age=400.0,
        max_diam=140.0,
        max_ht=32.0,
        arfa_0=0.28,
        g=70.0,
        shade_tol=5,  # Very shade tolerant
        drought_tol=2,
        flood_tol=2,
        deg_day_min=300.0,
        deg_day_opt=2000.0,
        deg_day_max=3500.0,
        wood_bulk_dens=0.40,
        rootdepth=3.0,
        leafarea_c=7.0,
        conifer=True
    ),

    # Species 6: Tulip Poplar (Liriodendron tulipifera) - Early successional
    6: SpeciesParameters(
        species_id=6,
        genus_name="Liriodendron",
        common_name="Tulip Poplar",
        max_age=200.0,
        max_diam=150.0,
        max_ht=35.0,
        arfa_0=0.42,
        g=130.0,
        shade_tol=1,  # Very shade intolerant
        drought_tol=2,
        flood_tol=3,
        deg_day_min=900.0,
        deg_day_opt=2700.0,
        deg_day_max=4800.0,
        wood_bulk_dens=0.43,
        rootdepth=3.5,
        leafarea_c=4.2,
        conifer=False
    )
}


def get_species_params(species_id: int) -> SpeciesParameters:
    """Get species parameters by ID."""
    if species_id not in SPECIES_DATA:
        raise ValueError(f"Unknown species ID: {species_id}")
    return SPECIES_DATA[species_id]


def get_all_species_ids():
    """Get list of all available species IDs."""
    return list(SPECIES_DATA.keys())
