"""
Create synthetic CSV files for weak scaling tests.

Generates minimal UVAFME-compatible CSV files with:
- 10-20 synthetic species
- Up to 100 synthetic sites on a grid
- Uniform climate
- All species present at all sites

Usage:
    python create_synthetic_csv.py --num-sites 100 --num-species 20 --output-dir ../synthetic_data
"""

import argparse
import csv
import os
import numpy as np


def create_specieslist(output_dir, num_species=20):
    """Create synthetic species list CSV with UVAFME-compatible field names."""
    filepath = os.path.join(output_dir, "SYNTHETIC_specieslist.csv")

    # UVAFME/CONUS field names (matches input_data/*_specieslist.csv)
    fieldnames = [
        'Group', 'Genus', 'Individual', 'Scientific name', 'Common name',
        'AGEmax', 'DBHmax', 'Hmax', 's', 'g', 'bulk', 'D_L', 'L_C',
        'DEGDmin', 'DEGDoptimum', 'DEGDmax', 'l', 'd', 'f', 'n',
        'fire', 'stress', 'old', 'evergreen', 'invader', 'seed', 'sprout',
        'NDE', 'NDS', 'Species_code', 'max_dispersal_dist'
    ]

    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(num_species):
            # Vary traits across species
            frac = i / max(num_species - 1, 1)
            shade_tol = 1 + int(frac * 4)  # 1-5

            row = {
                'Group': i % 10 + 1,
                'Genus': f'Genus{i % 10}',
                'Individual': i + 1,
                'Scientific name': f'Genus{i % 10}_species{i}',
                'Common name': f'Synthetic_species_{i}',
                'AGEmax': 100 + frac * 400,     # 100-500 years
                'DBHmax': 20 + frac * 100,      # 20-120 cm
                'Hmax': 15 + frac * 30,         # 15-45 m
                's': 0.7,                        # ARFA_0 (slope parameter)
                'g': 1.0 + frac * 2.0,          # Growth rate 1.0-3.0
                'bulk': 0.45,                    # Wood bulk density g/cm³
                'D_L': 0.15,                     # Leaf diameter adjustment
                'L_C': 3160.0,                   # Leaf area constant (m²/ha)
                'DEGDmin': 800 + frac * 2700,   # 800-3500
                'DEGDoptimum': 1300 + frac * 2700,  # 1300-4000
                'DEGDmax': 2300 + frac * 2700,  # 2300-5000
                'l': shade_tol,                  # Shade tolerance 1-5
                'd': 1 + int((1-frac) * 4),     # Drought tolerance 1-5 (inverse)
                'f': 3,                          # Flood tolerance 1-5
                'n': 3,                          # Low nutrient tolerance
                'fire': 1,                       # Fire tolerance
                'stress': 3,                     # Stress tolerance
                'old': 3,                        # Age tolerance
                'evergreen': 0,                  # Deciduous
                'invader': 1,                    # Can invade
                'seed': 10,                      # Seed production
                'sprout': 1,                     # Can sprout
                'NDE': 0.31,                     # Seed survival
                'NDS': 0.78,                     # Seedling survival
                'Species_code': f'SYN{i:03d}',
                'max_dispersal_dist': 5.0,       # km
            }
            writer.writerow(row)

    print(f"Created: {filepath} ({num_species} species)")


def create_site_csv(output_dir, num_sites=100):
    """Create synthetic site CSV with all UVAFME fields."""
    filepath = os.path.join(output_dir, "SYNTHETIC_site.csv")

    # Create grid (approximately square)
    grid_size = int(np.ceil(np.sqrt(num_sites)))

    # All fields from UVAFME site CSV
    fieldnames = [
        'site', 'latitude', 'longitude', 'wmo', 'name', 'region', 'elevation', 'slope',
        'soilA_field_cap', 'soilA_perm_wp', 'lai', 'soil_base_h', 'lai_w0',
        'soilAO_w0', 'soilA_w0', 'sbase_w0', 'fire_prob', 'wind_prob',
        'soilAO_c0', 'soilAO_n0', 'soilA_c0', 'soilA_n0', 'sbase_c0', 'sbase_n0', 'sigma'
    ]
    # Add temperature lapse rates
    for month in ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']:
        fieldnames.append(f'tmp_lapse_{month}')
    # Add precipitation lapse rates
    for month in ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']:
        fieldnames.append(f'prcp_lapse_{month}')

    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for site_id in range(num_sites):
            i = site_id // grid_size
            j = site_id % grid_size

            # Spread across mid-latitudes
            lat = 40.0 + i * 0.5  # Roughly 0.5° spacing
            lon = -90.0 + j * 0.5

            row = {
                'site': site_id,
                'latitude': f'{lat:.6f}',
                'longitude': f'{lon:.6f}',
                'wmo': site_id,
                'name': f'Site_{site_id}',
                'region': 'SYNTHETIC',
                'elevation': 300,  # meters
                'slope': 0,  # flat
                'soilA_field_cap': 25,  # cm
                'soilA_perm_wp': 12.5,  # cm (50% of field capacity)
                'lai': 3,  # leaf area index
                'soil_base_h': 35,  # cm
                'lai_w0': 0.5,  # initial LAI water
                'soilAO_w0': 0.6,  # initial A0 water
                'soilA_w0': 15.9,  # initial A water
                'sbase_w0': 10.9,  # initial base water
                'fire_prob': 0,  # no fire
                'wind_prob': 5,  # low wind
                'soilAO_c0': 5,  # initial A0 carbon kg/m²
                'soilAO_n0': 0.1,  # initial A0 nitrogen kg/m²
                'soilA_c0': 33.7,  # initial A carbon kg/m²
                'soilA_n0': 2.6,  # initial A nitrogen kg/m²
                'sbase_c0': 3,  # initial base carbon kg/m²
                'sbase_n0': 0,  # initial base nitrogen kg/m²
                'sigma': 0,  # no precipitation lapse adjustment
            }

            # Temperature lapse rates (°C/100m)
            for month in ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']:
                row[f'tmp_lapse_{month}'] = 0.6

            # Precipitation lapse rates (mm/100m)
            for month in ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']:
                row[f'prcp_lapse_{month}'] = 1.0

            writer.writerow(row)

    print(f"Created: {filepath} ({num_sites} sites)")


def create_climate_csv(output_dir, num_sites=100):
    """Create synthetic climate CSV with seasonal cycle (UVAFME format)."""
    filepath = os.path.join(output_dir, "SYNTHETIC_climate.csv")

    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
              'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

    # Seasonal temperature cycle (temperate, °C)
    # Tmin/tmax with ~10°C diurnal range
    monthly_tmin = [-2.0, 0.0, 5.0, 9.0, 14.0, 18.0,
                    20.0, 19.0, 15.0, 9.0, 3.0, -1.0]
    monthly_tmax = [8.0, 10.0, 15.0, 19.0, 24.0, 28.0,
                    30.0, 29.0, 25.0, 19.0, 13.0, 9.0]

    # Uniform precipitation (mm)
    monthly_precip = [80.0] * 12

    # Build fieldnames (UVAFME format)
    fieldnames = ['site', 'LATITUDE', 'LONGITUDE']
    for month in months:
        fieldnames.append(f'tmin_{month}')
    for month in months:
        fieldnames.append(f'tmax_{month}')
    for month in months:
        fieldnames.append(f'prcp_{month}')

    # Create grid for lat/lon
    grid_size = int(np.ceil(np.sqrt(num_sites)))

    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for site_id in range(num_sites):
            i = site_id // grid_size
            j = site_id % grid_size
            lat = 40.0 + i * 0.5
            lon = -90.0 + j * 0.5

            row = {
                'site': site_id,
                'LATITUDE': f'{lat:.6f}',
                'LONGITUDE': f'{lon:.6f}',
            }

            # Add tmin
            for month, tmin in zip(months, monthly_tmin):
                row[f'tmin_{month}'] = f'{tmin:.2f}'

            # Add tmax
            for month, tmax in zip(months, monthly_tmax):
                row[f'tmax_{month}'] = f'{tmax:.2f}'

            # Add precipitation
            for month, precip in zip(months, monthly_precip):
                row[f'prcp_{month}'] = f'{precip:.2f}'

            writer.writerow(row)

    print(f"Created: {filepath} ({num_sites} sites)")


def create_climate_stddev_csv(output_dir, num_sites=100):
    """Create synthetic climate stddev CSV (UVAFME format)."""
    filepath = os.path.join(output_dir, "SYNTHETIC_climate_stddev.csv")

    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
              'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

    # Small stddev for temperature and precip
    tmin_stddev = 2.0  # °C
    tmax_stddev = 2.0  # °C
    precip_stddev = 10.0  # mm

    fieldnames = ['site', 'LATITUDE', 'LONGITUDE']
    for month in months:
        fieldnames.append(f'tmn_std_{month}')
    for month in months:
        fieldnames.append(f'tmx_std_{month}')
    for month in months:
        fieldnames.append(f'prcp_std_{month}')

    # Create grid for lat/lon
    grid_size = int(np.ceil(np.sqrt(num_sites)))

    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for site_id in range(num_sites):
            i = site_id // grid_size
            j = site_id % grid_size
            lat = 40.0 + i * 0.5
            lon = -90.0 + j * 0.5

            row = {
                'site': site_id,
                'LATITUDE': f'{lat:.6f}',
                'LONGITUDE': f'{lon:.6f}',
            }

            for month in months:
                row[f'tmn_std_{month}'] = f'{tmin_stddev:.2f}'
                row[f'tmx_std_{month}'] = f'{tmax_stddev:.2f}'
                row[f'prcp_std_{month}'] = f'{precip_stddev:.2f}'

            writer.writerow(row)

    print(f"Created: {filepath} ({num_sites} sites)")


def create_rangelist_csv(output_dir, num_sites=100, num_species=20):
    """Create synthetic rangelist CSV (all species at all sites)."""
    filepath = os.path.join(output_dir, "SYNTHETIC_rangelist.csv")

    # Generate species codes
    species_codes = [f'SYN{i:03d}' for i in range(num_species)]

    fieldnames = ['site', 'latitude', 'longitude'] + species_codes

    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for site_id in range(num_sites):
            # Get coordinates from site CSV logic
            grid_size = int(np.ceil(np.sqrt(num_sites)))
            i = site_id // grid_size
            j = site_id % grid_size
            lat = 40.0 + i * 0.5
            lon = -90.0 + j * 0.5

            row = {
                'site': site_id,
                'latitude': f'{lat:.6f}',
                'longitude': f'{lon:.6f}',
            }

            # All species present at all sites (for simplicity)
            for sp_code in species_codes:
                row[sp_code] = 1

            writer.writerow(row)

    print(f"Created: {filepath} ({num_sites} sites × {num_species} species)")


def create_altitudes_csv(output_dir, num_sites=100):
    """Create synthetic altitudes CSV."""
    filepath = os.path.join(output_dir, "SYNTHETIC_altitudes.csv")

    fieldnames = ['site', 'altitude']

    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for site_id in range(num_sites):
            row = {
                'site': site_id,
                'altitude': 300,  # meters, uniform
            }
            writer.writerow(row)

    print(f"Created: {filepath} ({num_sites} sites)")


def main():
    parser = argparse.ArgumentParser(description="Create synthetic CSV files for GGap testing")
    parser.add_argument('--num-sites', type=int, default=25000,
                       help='Number of sites to generate (default: 25000, covers up to 2048-GPU weak scaling)')
    parser.add_argument('--num-species', type=int, default=100,
                       help='Number of species to generate (default: 100)')
    parser.add_argument('--output-dir', type=str, default='../input_data',
                       help='Output directory (default: ../input_data)')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Creating Synthetic CSV Files")
    print("=" * 60)
    print(f"Sites: {args.num_sites}")
    print(f"Species: {args.num_species}")
    print(f"Output: {args.output_dir}")
    print()

    # Create all CSV files
    create_specieslist(args.output_dir, args.num_species)
    create_site_csv(args.output_dir, args.num_sites)
    create_climate_csv(args.output_dir, args.num_sites)
    create_climate_stddev_csv(args.output_dir, args.num_sites)
    create_rangelist_csv(args.output_dir, args.num_sites, args.num_species)
    create_altitudes_csv(args.output_dir, args.num_sites)

    print()
    print("=" * 60)
    print("All files created successfully!")
    print("=" * 60)
    print()
    print("To use these files:")
    print(f"  model.load_globals(data_dir='{args.output_dir}', prefix='SYNTHETIC')")


if __name__ == "__main__":
    main()
