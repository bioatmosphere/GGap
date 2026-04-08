"""
CONUS forest fraction map with 10 sites, surrounded by paired plots:
  - Size distribution bar chart (proportion %)
  - Species composition stacked area (biomass C % over time)

Layout: paired charts around the perimeter, map in the center.
  Left:   [species_comp | size_dist] ← map
  Right:  map → [size_dist | species_comp]
  Bottom: [size_dist | species_comp] below map
"""

import csv
from collections import defaultdict
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch
from shapely.geometry import Point
from shapely.ops import unary_union
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader

SURFDATA = "grids/surfdata_0.125nldas2_hist_2000_78pfts_c240908.nc"
SITES_DIR = "sites"
OUTPUT_MAP = "conus_10sites_with_bars.png"
TARGET_YEAR = 1000
TOP_N = 6  # top species for composition plot

SITES = [
    ("Pacific NW",          "site_0083",  83, 47.25, -122.25),
    ("N. Rockies",          "site_0124", 124, 46.75, -114.75),
    ("Sierra Nevada",       "site_0690", 690, 39.25, -120.75),
    ("S. Rockies",          "site_0913", 913, 36.25, -106.75),
    ("Upper Midwest",       "site_0236", 236, 45.75,  -88.75),
    ("Central Appalachia",  "site_0784", 784, 38.25,  -80.25),
    ("S. Appalachia",       "site_0978", 978, 35.75,  -82.75),
    ("Northeast",           "site_0349", 349, 44.75,  -72.75),
    ("SE Coastal",          "site_1274",1274, 31.75,  -81.75),
    ("Gulf Coast",          "site_1313",1313, 30.75,  -92.75),
]

SIZE_CLASSES = ["0–8", "8–28", "28–48", "48–68", "68–88", ">88"]
SIZE_COL_INDICES = [5, 6, 7, 8, 9, 10]
BAR_COLORS = ["#90EE90", "#66BB6A", "#228B22", "#006400", "#8B4513", "#4E2A04"]

COMP_PALETTE = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]
OTHER_COLOR = "#cccccc"

# ── Read size distributions ──────────────────────────────────────────
def read_size_dist(site_num):
    path = f"{SITES_DIR}/site_{site_num}/species_data.csv"
    counts = np.zeros(6)
    with open(path, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if int(row[1]) == TARGET_YEAR:
                for i, ci in enumerate(SIZE_COL_INDICES):
                    counts[i] += int(row[ci])
    total = counts.sum()
    return counts / total * 100.0 if total > 0 else counts

# ── Read species composition over time ───────────────────────────────
def read_species_biomass(site_num):
    path = f"{SITES_DIR}/site_{site_num}/species_data.csv"
    data = defaultdict(lambda: defaultdict(float))
    with open(path, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            year = int(row[1])
            species = row[3]
            biomC = float(row[15])
            if biomC > 0:
                data[year][species] += biomC
    return data

def get_top_species(data, top_n):
    totals = defaultdict(float)
    for year_data in data.values():
        for sp, bc in year_data.items():
            totals[sp] += bc
    ranked = sorted(totals.items(), key=lambda x: -x[1])
    return [sp for sp, _ in ranked[:top_n]]

def build_stacked_data(data, top_species):
    years = sorted(data.keys())
    n_sp = len(top_species) + 1
    pcts = np.zeros((n_sp, len(years)))
    for j, yr in enumerate(years):
        year_data = data[yr]
        total = sum(year_data.values())
        if total <= 0:
            continue
        for i, sp in enumerate(top_species):
            pcts[i, j] = year_data.get(sp, 0) / total * 100.0
        other = total - sum(year_data.get(sp, 0) for sp in top_species)
        pcts[-1, j] = other / total * 100.0
    return np.array(years), top_species + ["Other"], pcts

# ── Load all data ────────────────────────────────────────────────────
print("Reading site data...")
site_size_props = {}
site_comp_data = {}
for label, site_id, site_num, slat, slon in SITES:
    site_size_props[site_num] = read_size_dist(site_num)
    raw = read_species_biomass(site_num)
    top_sp = get_top_species(raw, TOP_N)
    site_comp_data[site_num] = build_stacked_data(raw, top_sp)

# ── Load US boundary ──────────────────────────────────────────────────
print("Loading US boundary...")
shpname = shpreader.natural_earth(resolution="10m", category="cultural",
                                  name="admin_1_states_provinces")
reader = shpreader.Reader(shpname)
exclude = {"Alaska", "Hawaii", "United States Virgin Islands",
           "Puerto Rico", "American Samoa", "Guam",
           "Northern Mariana Islands"}
us_states = []
for record in reader.records():
    if (record.attributes["admin"] == "United States of America"
            and record.attributes["name"] not in exclude):
        us_states.append(record.geometry)
us_geom = unary_union(us_states)

# ── Load forest data ──────────────────────────────────────────────────
ds = nc.Dataset(SURFDATA, "r")
lat = ds.variables["LATIXY"][:].data.copy()
lon = ds.variables["LONGXY"][:].data.copy()
lon = np.where(lon > 180.0, lon - 360.0, lon)
landfrac = ds.variables["LANDFRAC_PFT"][:].data.copy()
pct_nat_pft = ds.variables["PCT_NAT_PFT"][:].data.copy()
pct_natveg = ds.variables["PCT_NATVEG"][:].data.copy()
ds.close()

forest_pct_of_cell = pct_nat_pft[1:9].sum(axis=0) * pct_natveg / 100.0
forest_pct_of_cell[landfrac <= 0.0] = np.nan
forest_pct_of_cell[forest_pct_of_cell <= 0.0] = np.nan

print("Filtering cells to US boundary...")
nr, nc_ = lat.shape
for r in range(nr):
    for c in range(nc_):
        if np.isnan(forest_pct_of_cell[r, c]):
            continue
        if not us_geom.contains(Point(lon[r, c], lat[r, c])):
            forest_pct_of_cell[r, c] = np.nan

# ── Figure layout ────────────────────────────────────────────────────
# 5 rows x 8 cols:
#   Row 0 (top):    cols 2-3 = site pair, cols 4-5 = site pair
#   Rows 1-3 (mid): cols 0-1 = left 3 sites, cols 2-5 = map, cols 6-7 = right 3 sites
#   Row 4 (bottom): cols 2-3 = site pair, cols 4-5 = site pair
#
# Geographic assignment:
#   Top 2:    Pacific NW (0), Northeast (7)
#   Left 3:   N. Rockies (1), Sierra Nevada (2), S. Rockies (3)
#   Right 3:  Upper Midwest (4), Central Appalachia (5), S. Appalachia (6)
#   Bottom 2: Gulf Coast (9), SE Coastal (8)

proj = ccrs.AlbersEqualArea(central_longitude=-96, central_latitude=37.5,
                            standard_parallels=(29.5, 45.5))

fig = plt.figure(figsize=(7.16, 3.8))
fig.patch.set_facecolor("white")

gs = gridspec.GridSpec(5, 8, figure=fig,
                       width_ratios=[1, 1, 1, 1, 1, 1, 1, 1],
                       height_ratios=[1, 1, 1, 1, 1],
                       hspace=0.08, wspace=0.45)
fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.02)

# ── Map (center, rows 1-3, cols 2-5) ────────────────────────────────
ax_map = fig.add_subplot(gs[1:4, 2:6], projection=proj)
ax_map.set_extent([-123, -68, 25, 50], crs=ccrs.PlateCarree())
ax_map.spines['geo'].set_visible(False) if 'geo' in ax_map.spines else None
ax_map.set_facecolor("white")

ax_map.add_geometries(us_states, ccrs.PlateCarree(),
                      facecolor="#E8E8E8", edgecolor="none", zorder=0)

forest_cmap = LinearSegmentedColormap.from_list(
    "forest_frac", ["#F0FFF0", "#ADFF2F", "#228B22", "#006400", "#002200"], N=256)

half = 0.0625
lat_edges = np.linspace(lat.min() - half, lat.max() + half, lat.shape[0] + 1)
lon_edges = np.linspace(lon.min() - half, lon.max() + half, lon.shape[1] + 1)
mesh_lon, mesh_lat = np.meshgrid(lon_edges, lat_edges)

im = ax_map.pcolormesh(mesh_lon, mesh_lat, forest_pct_of_cell,
                       cmap=forest_cmap, vmin=0, vmax=100,
                       transform=ccrs.PlateCarree(), zorder=1)

ax_map.add_geometries(us_states, ccrs.PlateCarree(),
                      facecolor="none", edgecolor="gray", linewidth=0.3, zorder=3)

cax = ax_map.inset_axes([0.12, 0.12, 0.25, 0.02])
cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
cbar.set_label("Forest fraction (%)", fontsize=7, labelpad=2)
cbar.ax.tick_params(labelsize=6)

for idx, (label, site_id, site_num, slat, slon) in enumerate(SITES):
    ax_map.plot(slon, slat, "o", markersize=7, color="red",
                markeredgecolor="black", markeredgewidth=0.7,
                transform=ccrs.PlateCarree(), zorder=8)
    ax_map.text(slon, slat, str(idx + 1), fontsize=5, fontweight="bold",
                color="white", ha="center", va="center",
                transform=ccrs.PlateCarree(), zorder=9)

# ── Annotations (matplotlib, appears in all formats) ─────────────────
# Place at top of figure, aligned with top panel row
# Left: size distribution label + short arrow pointing right
# Right: species composition label + short arrow pointing left

# ── Helper: make size distribution bar chart ─────────────────────────
def make_bar(ax, site_idx, show_title=True):
    label, site_id, site_num, slat, slon = SITES[site_idx]
    props = site_size_props[site_num]
    x = np.arange(len(SIZE_CLASSES))
    ax.bar(x, props, color=BAR_COLORS, edgecolor="black", linewidth=0.3, width=0.7)
    if show_title:
        ax.text(0.97, 0.97, f"{site_idx+1}.\n{label}", fontsize=5.5, fontweight="bold",
                ha="right", va="top", transform=ax.transAxes, zorder=10,
                linespacing=1.1,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="none", alpha=0.8))
    ax.set_xticks(x)
    ax.set_xticklabels(SIZE_CLASSES, fontsize=6, rotation=45, ha="right")
    ax.tick_params(axis="y", labelsize=6, pad=1)
    ax.tick_params(axis="x", pad=1)
    ax.set_ylim(0, 60)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.3)
    ax.spines["bottom"].set_linewidth(0.3)

# ── Helper: make species composition stacked area ────────────────────
def make_comp(ax, site_idx, show_title=True):
    label, site_id, site_num, slat, slon = SITES[site_idx]
    years, names, pcts = site_comp_data[site_num]
    colors = COMP_PALETTE[:len(names)-1] + [OTHER_COLOR]
    ax.stackplot(years, pcts, labels=names, colors=colors, alpha=0.85, linewidth=0)
    if show_title:
        parts = label.rsplit(" (", 1)
        if len(parts) == 2:
            wrapped = f"{site_idx+1}. {parts[0]}\n({parts[1]}"
        else:
            wrapped = f"{site_idx+1}. {label}"
        ax.text(0.97, 0.95, wrapped, fontsize=6, fontweight="bold",
                ha="right", va="top", transform=ax.transAxes, zorder=10,
                linespacing=1.1)
    ax.set_xlim(years[0], years[-1])
    ax.set_ylim(0, 100)
    ax.tick_params(axis="both", labelsize=6, pad=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.3)
    ax.spines["bottom"].set_linewidth(0.3)
    ax.legend(fontsize=5, loc="upper left", frameon=True, framealpha=0.4,
              edgecolor="none", handlelength=0.6, handletextpad=0.2,
              borderpad=0.2, labelspacing=0.1)

# ── Place paired charts ──────────────────────────────────────────────
# Top row (row 0): 2 northern sites
#   Pacific NW (0): cols 2-3, Northeast (7): cols 4-5
top_sites = [(0, 2, 3), (7, 4, 5)]
for si, bc, cc in top_sites:
    ax_bar = fig.add_subplot(gs[0, bc])
    make_bar(ax_bar, si)
    ax_comp = fig.add_subplot(gs[0, cc])
    make_comp(ax_comp, si, show_title=False)

# Left (rows 1-3, cols 0-1): 3 western sites
#   col 0 = species comp (no title), col 1 = size dist (with title)
#   Only bottom row (site 4, S. Rockies) shows x-axis tick labels
left_sites = [1, 2, 3]  # N. Rockies, Sierra Nevada, S. Rockies
for i, si in enumerate(left_sites):
    ax_comp = fig.add_subplot(gs[1 + i, 0])
    make_comp(ax_comp, si, show_title=False)
    if i < 2:
        ax_comp.set_xticklabels([])
    ax_bar = fig.add_subplot(gs[1 + i, 1])
    make_bar(ax_bar, si, show_title=True)
    if i < 2:
        ax_bar.set_xticklabels([])

# Right (rows 1-3, cols 6-7): 3 eastern sites
#   col 6 = size dist (with title), col 7 = species comp (no title)
#   Only bottom row (site 7, S. Appalachia) shows x-axis tick labels
right_sites = [4, 5, 6]  # Upper Midwest, Central Appalachia, S. Appalachia
for i, si in enumerate(right_sites):
    ax_bar = fig.add_subplot(gs[1 + i, 6])
    make_bar(ax_bar, si, show_title=True)
    if i < 2:
        ax_bar.set_xticklabels([])
    ax_comp = fig.add_subplot(gs[1 + i, 7])
    make_comp(ax_comp, si, show_title=False)
    if i < 2:
        ax_comp.set_xticklabels([])

# Bottom row (row 4): 2 southern sites
#   Gulf Coast (9): cols 2-3, SE Coastal (8): cols 4-5
bottom_sites = [(9, 2, 3), (8, 4, 5)]
for si, bc, cc in bottom_sites:
    ax_bar = fig.add_subplot(gs[4, bc])
    make_bar(ax_bar, si)
    ax_comp = fig.add_subplot(gs[4, cc])
    make_comp(ax_comp, si, show_title=False)

# ── Draw connector lines ─────────────────────────────────────────────
def get_map_fig_coords(slat, slon):
    map_pos = ax_map.get_position()
    proj_pt = proj.transform_point(slon, slat, ccrs.PlateCarree())
    xlim = ax_map.get_xlim()
    ylim = ax_map.get_ylim()
    x_frac = (proj_pt[0] - xlim[0]) / (xlim[1] - xlim[0])
    y_frac = (proj_pt[1] - ylim[0]) / (ylim[1] - ylim[0])
    return (map_pos.x0 + x_frac * map_pos.width,
            map_pos.y0 + y_frac * map_pos.height)

# Collect all panel axes (skip ax_map at index 0 and colorbar)
# Each site panel = 2 axes (bar + comp). Order: top(2×2), left(3×2), right(3×2), bottom(2×2)
panel_axes = fig.axes[1:]  # skip ax_map

def connect_panel(ax_inner, side, slat, slon):
    """Draw line from inner panel edge to map marker."""
    pos = ax_inner.get_position()
    x_map, y_map = get_map_fig_coords(slat, slon)
    if side == "top":
        x_p = pos.x0 + pos.width / 2
        y_p = pos.y0
    elif side == "bottom":
        x_p = pos.x0 + pos.width / 2
        y_p = pos.y1
    elif side == "left":
        x_p = pos.x1
        y_p = pos.y0 + pos.height / 2
    elif side == "right":
        x_p = pos.x0
        y_p = pos.y0 + pos.height / 2
    fig.add_artist(FancyArrowPatch(
        (x_p, y_p), (x_map, y_map), transform=fig.transFigure,
        arrowstyle="-", color="gray", linewidth=0.5, alpha=0.4, zorder=0))

# Top sites: connect bar (inner) bottom edge → map
# panel_axes[0]=bar(0), [1]=comp(0), [2]=bar(7), [3]=comp(7)
for k, (si, bc, cc) in enumerate(top_sites):
    _, _, site_num, slat, slon = SITES[si]
    connect_panel(panel_axes[2*k], "top", slat, slon)  # bar axes

# Left sites: connect bar (col 1) right edge → map
# panel_axes[4]=comp(1), [5]=bar(1), [6]=comp(2), [7]=bar(2), [8]=comp(3), [9]=bar(3)
for i, si in enumerate(left_sites):
    _, _, site_num, slat, slon = SITES[si]
    connect_panel(panel_axes[4 + 2*i + 1], "left", slat, slon)  # bar axes

# Right sites: connect bar (col 6) left edge → map
# panel_axes[10]=bar(4), [11]=comp(4), [12]=bar(5), [13]=comp(5), [14]=bar(6), [15]=comp(6)
for i, si in enumerate(right_sites):
    _, _, site_num, slat, slon = SITES[si]
    connect_panel(panel_axes[10 + 2*i], "right", slat, slon)  # bar axes

# Bottom sites: connect bar top edge → map
# panel_axes[16]=bar(9), [17]=comp(9), [18]=bar(8), [19]=comp(8)
for k, (si, bc, cc) in enumerate(bottom_sites):
    _, _, site_num, slat, slon = SITES[si]
    connect_panel(panel_axes[16 + 2*k], "bottom", slat, slon)  # bar axes

# ── Annotations at vertical center of top panel row ──────────────────
_top_panel = panel_axes[0].get_position()  # first top bar chart
_ann_y = _top_panel.y0 + _top_panel.height / 2
fig.text(0.01, _ann_y, "Tree size (diameter) distribution\nat yr 1000 (cm %)  \u2192",
         ha="left", va="center", fontsize=6, fontweight="bold", color="black")
fig.text(0.97, _ann_y, "\u2190  Species composition\nover time (biomass C %)",
         ha="right", va="center", fontsize=6, fontweight="bold", color="black")

# ── Save PNG (auto-cropped) and SVG ──────────────────────────────────
from PIL import Image

# SVG and PDF (vector, no cropping needed)
OUTPUT_SVG = OUTPUT_MAP.replace(".png", ".svg")
OUTPUT_PDF = OUTPUT_MAP.replace(".png", ".pdf")
fig.savefig(OUTPUT_SVG, format="svg", bbox_inches="tight", pad_inches=0)
fig.savefig(OUTPUT_PDF, format="pdf", bbox_inches="tight", pad_inches=0)
print(f"Saved to {OUTPUT_SVG}")
print(f"Saved to {OUTPUT_PDF}")

# PNG (raster, auto-crop whitespace)
fig.savefig(OUTPUT_MAP, dpi=300, bbox_inches="tight", pad_inches=0)
img = Image.open(OUTPUT_MAP).convert("RGB")
arr = np.array(img)
non_white = np.any(arr < 252, axis=2)
rows_any = np.any(non_white, axis=1)
cols_any = np.any(non_white, axis=0)
r0, r1 = np.where(rows_any)[0][[0, -1]]
c0, c1 = np.where(cols_any)[0][[0, -1]]
margin = 3
r0 = max(0, r0 - margin)
r1 = min(arr.shape[0], r1 + margin)
c0 = max(0, c0 - margin)
c1 = min(arr.shape[1], c1 + margin)
img = img.crop((c0, r0, c1, r1))
# Ensure width ≤ 7.16" at 300 DPI (2148 px)
max_w = int(7.16 * 300)
if img.size[0] > max_w:
    ratio = max_w / img.size[0]
    new_h = int(img.size[1] * ratio)
    img = img.resize((max_w, new_h), Image.LANCZOS)

img.save(OUTPUT_MAP)
w, h = img.size
print(f"Saved to {OUTPUT_MAP} ({w}x{h}, {w/300:.2f}\"x{h/300:.2f}\" at 300 DPI)")
