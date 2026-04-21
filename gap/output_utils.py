"""Output utilities for GGap - produces GAPpy-compatible CSV files.

Matches GAPpy's output_module.py format for:
  - genus_data.csv: per-genus biomass, basal area, diameter classes
  - species_data.csv: per-species biomass, basal area, diameter classes
  - site_data.csv: climate and environmental data
  - soil_data.csv: soil C/N pools and tree biomass totals
  - tree_data.csv: per-tree individual data

Scaling conventions (from GAPpy):
  plotsize  = 500.0 m²  (area per gap/plot)
  plotscale = HEC_TO_M2 / plotsize           (= 20.0)
  plotadj   = plotscale / num_gaps           (biomC scaling: kg -> kg/ha)
  plotrenorm= 1.0 / plotsize / num_gaps      (basal area, LAI scaling)
"""

import os
import csv
import math
import numpy as np

# Constants matching GAPpy
HEC_TO_M2 = 10000.0
NHC = 7  # Number of diameter categories
RNVALID = -999.0
CON_LEAF_C_N = 60.0
DEC_LEAF_C_N = 40.0
PI = math.pi


def get_diam_class(diam):
    """Return diameter class index (0-6) matching GAPpy categories.

    Classes: <0 (dead), <=8, <=28, <=48, <=68, <=88, >88 cm
    """
    if diam < 0.0:
        return 0
    elif diam <= 8.0:
        return 1
    elif diam <= 28.0:
        return 2
    elif diam <= 48.0:
        return 3
    elif diam <= 68.0:
        return 4
    elif diam <= 88.0:
        return 5
    else:
        return 6


def stddev(values):
    """Compute mean and sample standard deviation of a list."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    return mean, math.sqrt(max(0.0, variance))


class OutputWriter:
    """Write GAPpy-compatible CSV output files."""

    def __init__(self, output_dir, site_id=0, plotsize=500.0):
        self.output_dir = output_dir
        self.site_id = site_id
        self.plotsize = plotsize
        self.files = {}
        self.writers = {}
        self.species_info = {}
        self.species_list = []  # [(genus, species_code), ...]
        self.genus_list = []    # [genus_name, ...]
        self.num_gaps = 0
        self.plotscale = 0.0
        self.plotadj = 0.0
        self.plotrenorm = 0.0

        # Lookup tables (populated in open())
        self._species_code_to_idx = {}
        self._genus_to_idx = {}
        self._species_id_to_code = {}
        self._species_id_to_genus = {}
        self._species_id_to_leafarea_c = {}

    def open(self, species_by_id, num_gaps):
        """Open output files and write headers.

        Args:
            species_by_id: dict mapping global_id -> species info dict
            num_gaps: number of gaps (= GAPpy plots)
        """
        os.makedirs(self.output_dir, exist_ok=True)
        self.species_info = species_by_id
        self.num_gaps = num_gaps

        # Build sorted species and genus lists
        genera = set()
        spec_pairs = []
        for gid in sorted(species_by_id.keys()):
            sp = species_by_id[gid]
            genera.add(sp['genus'])
            spec_pairs.append((sp['genus'], sp['species_code']))
        self.species_list = spec_pairs
        self.genus_list = sorted(genera)

        # Compute scaling factors
        self.plotscale = HEC_TO_M2 / self.plotsize
        self.plotadj = self.plotscale / num_gaps if num_gaps > 0 else self.plotscale
        self.plotrenorm = 1.0 / self.plotsize / num_gaps if num_gaps > 0 else 1.0 / self.plotsize

        # Build lookup tables
        for idx, (g, c) in enumerate(self.species_list):
            self._species_code_to_idx[c] = idx
        self._genus_to_idx = {g: i for i, g in enumerate(self.genus_list)}
        for gid, sp in species_by_id.items():
            self._species_id_to_code[gid] = sp['species_code']
            self._species_id_to_genus[gid] = sp['genus']
            self._species_id_to_leafarea_c[gid] = sp['leafarea_c']

        # Open files
        file_defs = {
            'genus': 'genus_data.csv',
            'species': 'species_data.csv',
            'site': 'site_data.csv',
            'soil': 'soil_data.csv',
            'tree': 'tree_data.csv',
        }
        for key, filename in file_defs.items():
            path = os.path.join(self.output_dir, filename)
            f = open(path, 'w', newline='')
            self.files[key] = f
            self.writers[key] = csv.writer(f)
        self._write_headers()

    def _write_headers(self):
        bio_cols = [
            '<0', '0-8', '8-28', '-48', '-68', '-88', '>88',
            'max_diam', 'max_hgt', 'leaf_area_ind',
            'basal_area', 'total_biomC', 'pl_biomC_std',
            'total_biomN', 'pl_biomN_std',
        ]
        self.writers['genus'].writerow(['siteID', 'year', 'genus'] + bio_cols)
        self.writers['species'].writerow(
            ['siteID', 'year', 'genus', 'species'] + bio_cols
        )
        self.writers['site'].writerow([
            'siteID', 'year',
            'rain', 'pet', 'aet', 'grow',
            'degd', 'dryd_upper', 'dryd_base', 'flood_d',
        ])
        self.writers['soil'].writerow([
            'siteID', 'year',
            'a0c0', 'ac0', 'a0n0', 'an0', 'bc0', 'bn0',
            'soilresp', 'biomassC', 'C_into_A0', 'net_C_into_A0', 'net_prim_prodC',
            'biomassN', 'N_into_A0', 'net_N_into_A0', 'net_prim_prodN', 'avail_n',
        ])
        self.writers['tree'].writerow([
            'siteID', 'year', 'plot', 'tree',
            'genus', 'species', 'diam bh', 'forska_height',
            'leaf biomass', 'stem biomC', 'stem biomN',
        ])

    def write_site_data(self, year, rain, pot_evap_day, act_evap_day,
                        grow_days, deg_days, dry_days_upper, dry_days_base,
                        flood_days):
        """Write one row of site data (matches GAPpy output_module.py)."""
        self.writers['site'].writerow([
            self.site_id, year,
            rain, pot_evap_day, act_evap_day, grow_days,
            deg_days, dry_days_upper, dry_days_base, flood_days,
        ])

    def write_soil_data(self, year, a0_c, a_c, a0_n, a_n, bl_c, bl_n,
                        avail_n, soilresp=0.0, c_into_a0=0.0,
                        n_into_a0=0.0, net_n_into_a0=0.0):
        """Write one row of soil data."""
        self.writers['soil'].writerow([
            self.site_id, year,
            a0_c, a_c, a0_n, a_n, bl_c, bl_n,
            soilresp,
            0.0,  # biomassC (always 0.0, matches GAPpy)
            c_into_a0, 0.0, 0.0,  # C_into_A0, net_C_into_A0, net_prim_prodC
            0.0,  # biomassN (always 0.0, matches GAPpy)
            n_into_a0, net_n_into_a0, 0.0,  # N_into_A0, net_N_into_A0, net_prim_prodN
            avail_n,
        ])

    def write_tree_data(self, year, tree_data, gap_agents):
        """Write per-tree data rows (matches GAPpy output_module.py).

        tree_data is dict of numpy arrays.
        """
        if tree_data['count'] == 0:
            return
        gap_idx = {gid: i + 1 for i, gid in enumerate(gap_agents)}
        # Convert numpy columns to Python lists once for fast iteration
        gap_ids = tree_data['gap_agent_id'].tolist()
        sp_ids = tree_data['species_id'].tolist()
        diams = tree_data['diam'].tolist()
        heights = tree_data['height'].tolist()
        biomCs = tree_data['biomC'].tolist()
        biomNs = tree_data['biomN'].tolist()
        leaf_bms = tree_data['leaf_bm'].tolist()
        tree_num = {}
        for i in range(tree_data['count']):
            plot = gap_idx.get(gap_ids[i], 0)
            tree_num[plot] = tree_num.get(plot, 0) + 1
            sp_id = sp_ids[i]
            genus = self._species_id_to_genus.get(sp_id, '')
            species_code = self._species_id_to_code.get(sp_id, '')
            self.writers['tree'].writerow([
                self.site_id, year, plot, tree_num[plot],
                genus, species_code, diams[i], heights[i],
                leaf_bms[i], biomCs[i], biomNs[i],
            ])

    def _write_bio_data(self, writer_key, year, tree_data, gap_agents, group_by):
        """Write species or genus level biomass data.

        Matches GAPpy output_module.py scaling and aggregation.
        Per-gap accumulation -> cross-gap totals -> scaling.
        tree_data is dict of numpy arrays.
        """
        gap_idx = {gid: i for i, gid in enumerate(gap_agents)}
        num_gaps = len(gap_agents)

        if group_by == 'genus':
            groups = self.genus_list
        else:
            groups = self.species_list

        num_groups = len(groups)

        # Per-gap, per-group accumulators
        biomC = [[0.0] * num_groups for _ in range(num_gaps)]
        biomN = [[0.0] * num_groups for _ in range(num_gaps)]
        basal = [[0.0] * num_groups for _ in range(num_gaps)]
        leaf = [[0.0] * num_groups for _ in range(num_gaps)]
        max_ht = [[0.0] * num_groups for _ in range(num_gaps)]
        max_diam = [[0.0] * num_groups for _ in range(num_gaps)]
        diam_cats = [[[0] * NHC for _ in range(num_groups)] for _ in range(num_gaps)]

        # Convert numpy columns to Python lists once for fast iteration
        n = tree_data['count']
        if n == 0:
            # Still need to write zero rows for each group
            pass
        else:
            t_gap_ids = tree_data['gap_agent_id'].tolist()
            t_sp_ids = tree_data['species_id'].tolist()
            t_diams = tree_data['diam'].tolist()
            t_heights = tree_data['height'].tolist()
            t_biomCs = tree_data['biomC'].tolist()
            t_biomNs = tree_data['biomN'].tolist()
            t_leaf_bms = tree_data['leaf_bm'].tolist()
            t_evergreens = tree_data['evergreen'].tolist()

            # Accumulate per-gap, per-group
            for i in range(n):
                gi = gap_idx.get(t_gap_ids[i])
                if gi is None:
                    continue
                sp_id = t_sp_ids[i]
                code = self._species_id_to_code.get(sp_id)
                if code is None:
                    continue

                if group_by == 'genus':
                    genus = self._species_id_to_genus.get(sp_id)
                    grp_i = self._genus_to_idx.get(genus)
                else:
                    grp_i = self._species_code_to_idx.get(code)

                if grp_i is None:
                    continue

                diam = t_diams[i]
                lf = t_leaf_bms[i]
                l_cn = CON_LEAF_C_N if t_evergreens[i] else DEC_LEAF_C_N
                tot_bc = t_biomCs[i] + lf
                tot_bn = t_biomNs[i] + lf / l_cn

                biomC[gi][grp_i] += tot_bc
                biomN[gi][grp_i] += tot_bn
                basal[gi][grp_i] += 0.25 * PI * diam ** 2
                leaf[gi][grp_i] += lf
                max_ht[gi][grp_i] = max(max_ht[gi][grp_i], t_heights[i])
                max_diam[gi][grp_i] = max(max_diam[gi][grp_i], diam)
                diam_cats[gi][grp_i][get_diam_class(diam)] += 1

        # Compute totals and write rows
        for grp_i in range(num_groups):
            t_biomC = sum(biomC[gi][grp_i] for gi in range(num_gaps))
            t_biomN = sum(biomN[gi][grp_i] for gi in range(num_gaps))
            t_basal = sum(basal[gi][grp_i] for gi in range(num_gaps))
            t_max_ht = max((max_ht[gi][grp_i] for gi in range(num_gaps)), default=0.0)
            t_max_diam = max(
                (max_diam[gi][grp_i] for gi in range(num_gaps)), default=0.0
            )
            total_dc = [
                sum(diam_cats[gi][grp_i][c] for gi in range(num_gaps))
                for c in range(NHC)
            ]

            # LAI from leaf biomass (GAPpy: leaf_bm / (leafarea_c * 2.0))
            t_leaf = sum(leaf[gi][grp_i] for gi in range(num_gaps))
            leafarea_c = self._find_leafarea_c(groups[grp_i], group_by)
            t_lai = t_leaf / (leafarea_c * 2.0) if leafarea_c > 0 else 0.0

            # Std dev of per-gap biomC/N (matches GAPpy stddev across plots)
            gap_biomC = [biomC[gi][grp_i] for gi in range(num_gaps)]
            gap_biomN = [biomN[gi][grp_i] for gi in range(num_gaps)]
            _, std_biomC = stddev(gap_biomC)
            _, std_biomN = stddev(gap_biomN)

            # Apply scaling (matches GAPpy output_module.py)
            t_biomC *= self.plotadj
            t_biomN *= self.plotrenorm * 10.0
            t_basal *= self.plotrenorm
            t_lai *= self.plotrenorm
            std_biomC *= self.plotscale
            std_biomN *= self.plotscale
            scaled_dc = [int(c * self.plotadj) for c in total_dc]

            # Build row
            if group_by == 'genus':
                row = [self.site_id, year, groups[grp_i]]
            else:
                genus, code = groups[grp_i]
                row = [self.site_id, year, genus, code]
            row.extend(scaled_dc)
            row.extend([
                t_max_diam, t_max_ht, t_lai,
                t_basal, t_biomC, std_biomC,
                t_biomN, std_biomN,
            ])
            self.writers[writer_key].writerow(row)

    def _find_leafarea_c(self, group_key, group_by):
        """Find leafarea_c for a genus or species group."""
        if group_by == 'genus':
            for sp_id, g in self._species_id_to_genus.items():
                if g == group_key:
                    return self._species_id_to_leafarea_c[sp_id]
        else:
            _, code = group_key
            for sp_id, c in self._species_id_to_code.items():
                if c == code:
                    return self._species_id_to_leafarea_c[sp_id]
        return 0.0

    def write_genus_data(self, year, tree_data, gap_agents):
        """Write genus-level biomass data."""
        self._write_bio_data('genus', year, tree_data, gap_agents, 'genus')

    def write_species_data(self, year, tree_data, gap_agents):
        """Write species-level biomass data."""
        self._write_bio_data('species', year, tree_data, gap_agents, 'species')

    def close(self):
        """Close all output files."""
        for f in self.files.values():
            f.close()
        self.files.clear()
        self.writers.clear()
