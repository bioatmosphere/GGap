"""
Site / agent hierarchy figure for SC2026 — the science figure showing
how an individual GGap site nests gaps which contain trees, with the
~100 billion total trees headline that motivates the paper.

Drawn at EXACT IEEE single-column print size — see _style.py.
matplotlib pt == LaTeX pt; the IEEE template includes figures with
\\centerline{\\includegraphics{...}} (no width override), so the
script saves at exact print dimensions.

Style conventions (same as ggap_architecture.py):
  - 3.487 in wide (COL_SINGLE) at exact print size
  - Font and line width constants imported from `_style.py`
  - Same Site/Gap/Tree palette and Material-style structural colors

Outputs:
  papers/SC2026/figs/site_hierarchy.pdf  (vector, transparent bg, for LaTeX)
  papers/SC2026/figs/site_hierarchy.png  (raster preview at 600 DPI for dev)
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg

from _style import (
    figsize_single, apply_rcparams,
    F_TITLE, F_HEAD, F_LABEL, F_BODY, F_SMALL,
    LW_THICK, LW_MED, LW_THIN, LW_HAIR,
    C_SITE, C_GAP, C_TREE, C_TEXT, C_DISP, C_EXPAND,
)

apply_rcparams()

# Local-only colours (not part of the shared paper palette)
C_ARROW = '#444444'

# Tree row depth line widths (front of isometric grid vs back rows).
LW_TREE_FRONT = LW_THIN
LW_TREE_BACK  = LW_HAIR


def rounded_box(ax, xy, w, h, label, color, fontsize=F_HEAD, text_color='white',
                alpha=1.0, lw=LW_MED, zorder=2, sublabel=None, sublabel_size=None):
    box = FancyBboxPatch(xy, w, h,
                         boxstyle="round,pad=0.015",
                         facecolor=color, edgecolor='#333333',
                         linewidth=lw, alpha=alpha, zorder=zorder)
    ax.add_patch(box)
    cx, cy = xy[0] + w/2, xy[1] + h/2
    if sublabel:
        ss = sublabel_size if sublabel_size else fontsize - 1
        ax.text(cx, cy + 0.018, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=text_color, zorder=zorder+1)
        ax.text(cx, cy - 0.018, sublabel, ha='center', va='center',
                fontsize=ss, color=text_color, alpha=0.85, zorder=zorder+1)
    else:
        ax.text(cx, cy, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=text_color, zorder=zorder+1)
    return box


def arrow(ax, start, end, color=C_ARROW, lw=LW_THICK):
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                mutation_scale=10))


script_dir = os.path.dirname(os.path.abspath(__file__))


# ════════════════════════════════════════════════════════════════════
#  CREATE FIGURE — single panel
# ════════════════════════════════════════════════════════════════════
# Height preserves the original 8/7 aspect ratio at the new width:
#   3.487 * (8/7) = 3.985 inches.
FIG_HEIGHT = 3.985
fig, ax_a = plt.subplots(1, 1, figsize=figsize_single(FIG_HEIGHT))
fig.patch.set_facecolor('white')

ax_a.set_xlim(0, 1)
ax_a.set_ylim(-0.07, 1.15)
ax_a.axis('off')


# ════════════════════════════════════════════════════════════════════
#  Agent Hierarchy
# ════════════════════════════════════════════════════════════════════

MID = 0.50

# ── Globe inset ──
globe_path = os.path.join(script_dir, 'globe.png')
globe_img = mpimg.imread(globe_path)

globe_cx, globe_cy = 0.07, 0.93
imagebox = OffsetImage(globe_img, zoom=0.14)
imagebox.image.axes = ax_a
ab = AnnotationBbox(imagebox, (globe_cx, globe_cy), frameon=False, zorder=5)
ax_a.add_artist(ab)

# ── Dotted bracket: globe (US region) → CONUS map ──
conus_cx = 0.45
us_x = globe_cx + 0.012
us_y = globe_cy + 0.010
ax_a.plot([us_x, conus_cx - 0.08],
          [us_y, globe_cy + 0.065],
          color=C_EXPAND, lw=LW_THICK, ls=':', zorder=1)
ax_a.plot([us_x, conus_cx - 0.08],
          [us_y, globe_cy - 0.035],
          color=C_EXPAND, lw=LW_THICK, ls=':', zorder=1)

# ── CONUS forest fraction map ──
conus_path = os.path.join(script_dir, 'conus_nldas2_forest_fraction_notitle.png')
conus_img = mpimg.imread(conus_path)

conus_box = OffsetImage(conus_img, zoom=0.0325)
conus_box.image.axes = ax_a
conus_ab = AnnotationBbox(conus_box, (conus_cx, globe_cy),
                          frameon=False, zorder=4)
ax_a.add_artist(conus_ab)

# ── Dotted bracket: CONUS red box → site inset ──
redbox_x = conus_cx + 0.048
redbox_y = globe_cy - 0.012
site_inset_cx = 0.86
ax_a.plot([redbox_x, site_inset_cx - 0.06],
          [redbox_y + 0.010, globe_cy + 0.065],
          color=C_EXPAND, lw=LW_THICK, ls=':', zorder=1)
ax_a.plot([redbox_x, site_inset_cx - 0.06],
          [redbox_y - 0.010, globe_cy - 0.045],
          color=C_EXPAND, lw=LW_THICK, ls=':', zorder=1)

# ── Site inset (isometric forest gap illustration, right of CONUS) ──
site_inset_path = os.path.join(script_dir, 'site_inset.png')
site_inset_img = mpimg.imread(site_inset_path)
site_inset_box = OffsetImage(site_inset_img, zoom=0.045)
site_inset_box.image.axes = ax_a
site_inset_ab = AnnotationBbox(site_inset_box, (0.86, globe_cy + 0.02),
                               frameon=False, zorder=4, clip_on=False)
ax_a.add_artist(site_inset_ab)

# ── Single Gap callout on site inset ──
# (Original had 3 callouts, all pointing into the inset, which crowded
#  the small image. One label is enough; the inset itself shows multiple
#  gaps and the reader recognises them by analogy.)
site_ix = 0.86
site_iy = globe_cy + 0.02
ax_a.annotate('Gap', xy=(site_ix - 0.035, site_iy + 0.060),
              xytext=(site_ix - 0.16, site_iy + 0.075),
              fontsize=F_BODY, fontweight='bold', color=C_GAP,
              arrowprops=dict(arrowstyle='->', color=C_GAP, lw=LW_THIN,
                              connectionstyle='arc3,rad=-0.2'),
              zorder=12, clip_on=False)
# Recovering seedling gap (bottom-left of inset)
ax_a.annotate('Gap', xy=(site_ix - 0.09, site_iy - 0.035),
              xytext=(site_ix - 0.13, site_iy - 0.15),
              fontsize=F_BODY, fontweight='bold', color=C_GAP,
              arrowprops=dict(arrowstyle='->', color=C_GAP, lw=LW_THIN,
                              connectionstyle='arc3,rad=0.2'),
              zorder=12, clip_on=False)
# Sparse scattered trees gap (right side of inset)
ax_a.annotate('Gap', xy=(site_ix + 0.03, site_iy - 0.01),
              xytext=(site_ix + 0.07, site_iy - 0.15),
              fontsize=F_BODY, fontweight='bold', color=C_GAP,
              arrowprops=dict(arrowstyle='->', color=C_GAP, lw=LW_THIN,
                              connectionstyle='arc3,rad=-0.2'),
              zorder=12, clip_on=False)

# ── Labels above insets ──
ax_a.text(globe_cx, globe_cy + 0.15, 'Globe',
          ha='center', va='bottom', fontsize=F_HEAD, fontweight='bold',
          color=C_TEXT, zorder=10, clip_on=False)
ax_a.text(conus_cx, globe_cy + 0.15, 'CONUS',
          ha='center', va='bottom', fontsize=F_HEAD, fontweight='bold',
          color=C_TEXT, zorder=10, clip_on=False)
ax_a.text(0.86, globe_cy + 0.15, 'Site',
          ha='center', va='bottom', fontsize=F_HEAD, fontweight='bold',
          color=C_TEXT, zorder=10, clip_on=False)

# ── "~100 billion trees" badge (the paper's scale hook) ──
# Size and transparency match the coworker's original site_hierarchy.py;
# kept the rest (bold upright, slightly thicker pad) from the v2 retune.
ax_a.text(conus_cx, globe_cy + 0.05,
          ' ~100 billion trees ',
          ha='center', va='center', fontsize=F_LABEL, color='white',
          fontweight='bold', zorder=10,
          bbox=dict(boxstyle='round,pad=0.3', facecolor=C_DISP,
                    edgecolor='none', alpha=0.70),
          clip_on=False)

# ── Dotted bracket: CONUS → site network ──
conus_bot_y = globe_cy - 0.06
network_top_y = 0.600 + 0.055 + 0.040 + 0.025  # back_y + sh + margin
network_left = MID - 0.28
network_right = MID + 0.16 + 0.11  # rightmost front box right edge
ax_a.plot([conus_cx, 0.14],
          [conus_bot_y, network_top_y],
          color=C_EXPAND, lw=LW_THIN, ls='--', alpha=0.45, zorder=1)
ax_a.plot([conus_cx, 0.86],
          [conus_bot_y, network_top_y],
          color=C_EXPAND, lw=LW_THIN, ls='--', alpha=0.45, zorder=1)

site_w = 0.15
site_h = 0.050
site_y = 0.600

# ── Site boxes ──
site_gap = 0.10
site_total = 3 * site_w + 2 * site_gap
site_xs = [MID - site_total/2 + i * (site_w + site_gap) for i in range(3)]
site_labels = ['Site 0', 'Site 1', 'Site \u2026']

# 3-D network: front row + back row of sites connected by edges
sw, sh = 0.11, 0.040  # smaller boxes for network layout

# Front row (3 labeled sites)
front_xs = [MID - 0.28, MID - 0.06, MID + 0.16]
front_y  = site_y
front_centers = [(fx + sw/2, front_y + sh/2) for fx in front_xs]

# Back row (2 unlabeled sites, offset up-right for depth)
back_dx, back_dy = 0.06, 0.055
back_xs = [MID - 0.17 + back_dx, MID + 0.05 + back_dx]
back_y  = front_y + back_dy
back_centers = [(bx + sw/2, back_y + sh/2) for bx in back_xs]

all_centers = front_centers + back_centers

# Draw network edges (all-to-all connections) behind boxes
for i in range(len(all_centers)):
    for j in range(i+1, len(all_centers)):
        ax_a.plot([all_centers[i][0], all_centers[j][0]],
                  [all_centers[i][1], all_centers[j][1]],
                  color=C_DISP, lw=LW_THICK, ls='-', alpha=0.40, zorder=3)

# Back row boxes (faded, no labels — suggest more sites in depth)
for bx in back_xs:
    rounded_box(ax_a, (bx, back_y), sw, sh, '\u2026', C_SITE, fontsize=F_BODY,
                text_color='#BBCCDD', alpha=0.55, lw=LW_THIN, zorder=4)

# Front row boxes (full opacity, labeled)
front_labels = ['Site 0', 'Site 1', 'Site \u2026']
for fx, lbl in zip(front_xs, front_labels):
    rounded_box(ax_a, (fx, front_y), sw, sh, lbl, C_SITE, fontsize=F_HEAD,
                alpha=1.0, lw=LW_MED, zorder=5)

# (typo fix: was "(e.g.,seed dispersal)" — added missing space after comma)
ax_a.text(MID, back_y + sh + 0.020,
          'Inter-site connections (e.g., seed dispersal)',
          ha='center', va='bottom', fontsize=F_BODY, color=C_DISP,
          fontweight='bold', zorder=6)

# ── Expansion bracket: Site 1 → hierarchy ──
expand_left  = 0.02
expand_right = 0.98
expand_top   = 0.470
expand_bot   = -0.05

site1_left  = front_xs[1]
site1_right = front_xs[1] + sw
flare_y = front_y - 0.010
ax_a.plot([site1_left, site1_left],   [front_y, flare_y],
          color=C_EXPAND, lw=LW_THIN, ls='--', alpha=0.45, zorder=0)
ax_a.plot([site1_right, site1_right], [front_y, flare_y],
          color=C_EXPAND, lw=LW_THIN, ls='--', alpha=0.45, zorder=0)
pad = 0.012
ax_a.plot([site1_left, expand_left + pad],   [flare_y, expand_top + pad],
          color=C_EXPAND, lw=LW_THIN, ls='--', alpha=0.45, zorder=0)
ax_a.plot([site1_right, expand_right - pad], [flare_y, expand_top + pad],
          color=C_EXPAND, lw=LW_THIN, ls='--', alpha=0.45, zorder=0)

expand_bg = FancyBboxPatch(
    (expand_left, expand_bot), expand_right - expand_left, expand_top - expand_bot,
    boxstyle="round,pad=0.012",
    facecolor='#F5F8F5', edgecolor=C_EXPAND,
    linewidth=LW_THIN, ls='--', alpha=0.45, zorder=0)
ax_a.add_patch(expand_bg)

# ── Ensemble annotation ──
ax_a.text(MID, 0.400 + 0.055 + 0.040,
          'A large ensemble\nof independent gaps',
          ha='center', va='bottom', fontsize=F_BODY, color='black',
          fontweight='bold', zorder=6)

# ── Site properties bar ──
props_x = expand_left + 0.015
props_w = (expand_right - expand_left) - 0.030
props_y = 0.400
props_h = 0.055
rounded_box(ax_a, (props_x, props_y), props_w, props_h,
            'Site properties', C_SITE, fontsize=F_LABEL, alpha=1.0,
            sublabel='Nutrient(N) \u2022 Water balance \u2022 Meteorology \u2022 Disturbance',
            sublabel_size=F_BODY)

# ── Gap agents ──
gap_h = 0.055
gap_y_a = 0.210
gap_margin = 0.04
gap_w = (props_w - 2 * gap_margin) / 3
gap_start = props_x
gap_xs_inner = [gap_start + i * (gap_w + gap_margin) for i in range(3)]

for i, gx in enumerate(gap_xs_inner):
    lbl = f'Gap {i}' if i < 2 else 'Gap \u2026'
    rounded_box(ax_a, (gx, gap_y_a), gap_w, gap_h, lbl, C_GAP, fontsize=F_HEAD,
                sublabel='Leaf area \u2022 Litter', sublabel_size=F_BODY)

# ── Arrows: site ↔ gaps ──
for idx, gx in enumerate(gap_xs_inner):
    gc = gx + gap_w / 2
    top_y = props_y - 0.015
    bot_y = gap_y_a + gap_h + 0.015
    mid_y_sg = (top_y + bot_y) / 2
    arrow(ax_a, (gc - 0.010, top_y), (gc - 0.010, bot_y), color=C_SITE)
    arrow(ax_a, (gc + 0.010, bot_y), (gc + 0.010, top_y), color=C_GAP)
    if idx == 1:
        ax_a.text(gc - 0.018, mid_y_sg, 'climate,\navail N',
                  ha='right', va='center', fontsize=F_BODY, color=C_SITE,
                  linespacing=1.2, zorder=5)
        ax_a.text(gc + 0.018, mid_y_sg, 'litter,\nN consumed',
                  ha='left', va='center', fontsize=F_BODY, color=C_GAP,
                  linespacing=1.2, zorder=5)

# ── Tree agents ──
tree_w = 0.035
tree_h = 0.038
tree_y = 0.000

# Isometric 3-D grid of trees per gap: multiple rows receding into depth
tree_cols = 4
tree_rows = 3   # front to back
iso_dx = 0.006  # rightward shift per depth row
iso_dy = 0.012  # upward shift per depth row

for gx in gap_xs_inner:
    gc = gx + gap_w / 2
    col_span = (tree_cols - 1) * (tree_w + 0.004)
    x0 = gc - col_span / 2 - tree_w / 2

    # Draw back rows first, then front on top
    for row in range(tree_rows - 1, -1, -1):
        ox = row * iso_dx
        oy = row * iso_dy
        depth_alpha = 1.0 - row * 0.22
        depth_lw = LW_TREE_FRONT if row == 0 else LW_TREE_BACK
        zord = 2 + (tree_rows - row)
        for col in range(tree_cols):
            tx = x0 + col * (tree_w + 0.004) + ox
            ty = tree_y + oy
            if row == 0:
                lbl = 'T' if col < tree_cols - 1 else '\u2026'
                tc = C_TEXT if col < tree_cols - 1 else '#777777'
                a = 1.0 if col < tree_cols - 1 else 0.6
            else:
                lbl = ''
                tc = C_TEXT
                a = depth_alpha
            rounded_box(ax_a, (tx, ty), tree_w, tree_h, lbl, C_TREE,
                        fontsize=F_BODY, text_color=tc, alpha=a,
                        lw=depth_lw, zorder=zord)

# ── Arrows: gaps ↔ trees ──
for idx, gx in enumerate(gap_xs_inner):
    gc = gx + gap_w / 2
    top_y = gap_y_a - 0.015
    bot_y = tree_y + (tree_rows - 1) * iso_dy + tree_h + 0.018
    mid_y_gt = (top_y + bot_y) / 2
    arrow(ax_a, (gc - 0.010, top_y), (gc - 0.010, bot_y), color=C_GAP)
    arrow(ax_a, (gc + 0.010, bot_y), (gc + 0.010, top_y), color=C_TREE)
    if idx == 1:
        ax_a.text(gc - 0.018, mid_y_gt, 'light,\nN supply',
                  ha='right', va='center', fontsize=F_BODY, color=C_GAP,
                  linespacing=1.2, zorder=5)
        ax_a.text(gc + 0.018, mid_y_gt, 'growth,\nN demand',
                  ha='left', va='center', fontsize=F_BODY, color=C_TREE,
                  linespacing=1.2, zorder=5)

# ── Tree state annotation ──
ax_a.text(MID, tree_y - 0.022,
          'Tree life cycle: Growth \u2192 Reproduction \u2192 Mortality \u2192 Recruitment',
          ha='center', va='top', fontsize=F_BODY, color='black')


# ════════════════════════════════════════════════════════════════════
#  SAVE
# ════════════════════════════════════════════════════════════════════
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figs')
os.makedirs(out_dir, exist_ok=True)
out_base = os.path.join(out_dir, 'site_hierarchy')
# PNG for dev preview: white background for readability against any
# IDE / browser background.
fig.savefig(f'{out_base}.png', format='png', dpi=600,
            facecolor='white', edgecolor='none')
# PDF for LaTeX: transparent background so the figure blends with the
# page color regardless of document-level styling.
fig.savefig(f'{out_base}.pdf', format='pdf',
            transparent=True, edgecolor='none')
print(f"Saved: {out_base}.{{png,pdf}}")
