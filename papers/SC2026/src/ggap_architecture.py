"""
GGap Architecture figure for SC2026 (engineering narrative).

Drawn at EXACT IEEE double-column print size — see _style.py.
matplotlib pt == LaTeX pt; the IEEE template includes figures with
\\centerline{\\includegraphics{...}} (no width override), so the
script must save at exact print dimensions.

Single integrated full-width panel with three stacked sections:
  1. Level 1: Inter-GPU Parallelism (MPI, 1 rank/GPU) — 3 GPU boxes with sites
  2. Level 2: Intra-GPU Parallelism — agents mapped to thread/block/CU hierarchy
  3. Fused Kernel per Tick — horizontal priority strip with 10 priority steps
     where thread-grid icons show which threads are active at each priority

Outputs:
  papers/SC2026/figs/ggap_architecture.pdf  (vector — for \\includegraphics in LaTeX)
  papers/SC2026/figs/ggap_architecture.png  (raster preview at 600 DPI for dev)

Usage:
    python ggap_architecture.py [--dpi 600]
"""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

from _style import (
    COL_DOUBLE, figsize_double, apply_rcparams,
    F_TITLE, F_HEAD, F_LABEL, F_BODY, F_SMALL,
    LW_THICK, LW_MED, LW_THIN, LW_HAIR,
    C_SITE, C_GAP, C_TREE,
    C_HPC, C_IDLE, C_EXPAND, C_DISP, C_TEXT,
)

# Chassis unification: every non-agent structural color in this figure
# uses ONE dark navy (C_HPC). Site/Gap/Tree are the only highlights.
# These aliases keep the existing in-body code untouched.
C_MPI = C_THREAD = C_CU = C_BARRIER = C_HPC

apply_rcparams()

FIGS_DIR = Path(__file__).resolve().parent.parent / "figs"

FIG_W = COL_DOUBLE
FIG_H = 3.0                          # content-driven height (see plan)
FIG_SIZE = figsize_double(FIG_H)

# Local-only colour: pale GPU box background — internal to this figure,
# not part of the shared paper palette.
C_GPU_BG = "#EBF4FF"


def rbox(ax, x, y, w, h, ec, fc="none", lw=LW_THICK, rad=0.03, zo=1, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={rad}",
        edgecolor=ec, facecolor=fc, linewidth=lw, zorder=zo, linestyle=ls))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dpi", type=int, default=600,
                        help="Raster DPI for the PNG preview output.")
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.set_aspect("equal")
    ax.axis("off")

    L = 0.05
    LR = FIG_W - 0.05  # full-width panel
    CX = (L + LR) / 2

    # ==================================================================
    # HPC SYSTEM LABEL (top) — in-figure title removed; use LaTeX \caption.
    # Bold dark "section header" reads as the figure's conceptual
    # subtitle, paralleling the "Multi-priority kernel pipeline" header
    # below.
    # ==================================================================
    hpc_label_y = FIG_H - 0.12
    ax.text(CX, hpc_label_y, "Two-level parallelism on an Exascale HPC system",
            ha="center", va="center", fontsize=6, fontweight="bold",
            color=C_TEXT)

    # ==================================================================
    # LAYOUT COMPUTATION (top-down from HPC label)
    # ==================================================================
    # Level 2: Inter-GPU (directly below HPC label with small gap)
    l2_box_h = 0.78
    l2_box_top = hpc_label_y - 0.12
    l2_box_y = l2_box_top - l2_box_h

    # Level 1: Intra-GPU (below Level 2 with gap for "zoom" arrow).
    l1_box_h = 0.78
    l1_box_top = l2_box_y - 0.15
    l1_box_y = l1_box_top - l1_box_h

    # Fused kernel strip (below Level 1 with gap for the bold
    # "Multi-priority kernel pipeline" header to breathe.
    fk_box_h = 0.78
    fk_box_top = l1_box_y - 0.22
    fk_box_y = fk_box_top - fk_box_h

    # ==================================================================
    # HPC → Level 2 angled connectors
    # ==================================================================
    l2_box_w = LR - L
    # META-BOX for "two-level parallelism" — encompasses Box 1 (top),
    # the zoom-into-GPU arrow gap, and Box 2 (bottom). Replaces what used
    # to be two separate outlined boxes. Same outline style as before;
    # the Level 1 / Level 2 banners inside act as section headers.
    meta_box_y = l1_box_y                           # bottom of former Box 2
    meta_box_h = (l2_box_y + l2_box_h) - l1_box_y   # full span
    rbox(ax, L, meta_box_y, l2_box_w, meta_box_h,
         C_HPC, fc=C_HPC + "06", lw=LW_THICK, rad=0.02, zo=0)
    ax.text(L + l2_box_w / 2, l2_box_y + l2_box_h - 0.02,
            "Level 1: Inter-rank MPI \u2014 site's ensemble colocated (1 rank/GPU)",
            ha="center", va="top", fontsize=6, fontweight="bold",
            color=C_MPI, zorder=1)

    # 3 GPU boxes inside
    gpu_w = 1.70
    gpu_h = 0.50
    gpu_gap_space = 0.28
    total_gpus_w = 3 * gpu_w + 2 * gpu_gap_space
    gx_start = CX - total_gpus_w / 2
    content_top = l2_box_y + l2_box_h - 0.10
    content_bot = l2_box_y + 0.10
    gpu_y = (content_top + content_bot) / 2 - gpu_h / 2
    gpu_xs = [gx_start + i * (gpu_w + gpu_gap_space) for i in range(3)]

    for gx, lab in zip(gpu_xs, ["GPU 0", "GPU 1", "GPU N"]):
        rbox(ax, gx, gpu_y, gpu_w, gpu_h, C_MPI, fc=C_GPU_BG,
             lw=LW_MED, rad=0.015, zo=2)
        ax.text(gx + gpu_w / 2, gpu_y + gpu_h - 0.025, lab,
                ha="center", va="top", fontsize=6, fontweight="bold",
                color=C_MPI, zorder=3)

    # Box 1 bottom descriptor: per-site agent structure that gets colocated.
    # Symmetrical with Box 2 (GPU spec) and Box 3 (legend) bottom descriptors.
    # Counts shown as illustrative examples ("e.g.") — actual experiments vary.
    # White bbox + high zorder so the descriptor visually masks the
    # "zoom into one GPU" arrow that passes through this region.
    ax.text(L + l2_box_w / 2, l2_box_y + 0.05,
            "Per colocated site: 1 site agent + gap agents (e.g. 500/site) "
            "+ tree agents (e.g. 1000/gap)",
            ha="center", va="center", fontsize=6, fontstyle="italic",
            color=C_MPI, zorder=5,
            bbox=dict(facecolor="white", edgecolor="none", pad=1))

    # Site circles (labeled "S" inside). Coords scaled by 0.92x in y to
    # fit gpu_h = 0.50 (originally 0.54); sr fits the 6 pt "S" label.
    sr = 0.065
    s0_raw = [(0.28, 0.258), (0.70, 0.276), (0.34, 0.092),
              (1.02, 0.202), (1.24, 0.083)]
    s0 = [(gpu_xs[0] + x, gpu_y + y) for x, y in s0_raw]
    s1_raw = [(0.24, 0.276), (0.68, 0.294), (0.46, 0.110),
              (1.04, 0.239), (1.24, 0.092)]
    s1 = [(gpu_xs[1] + x, gpu_y + y) for x, y in s1_raw]
    s2_raw = [(0.36, 0.239), (1.00, 0.202), (0.58, 0.074), (1.18, 0.074)]
    s2 = [(gpu_xs[2] + x, gpu_y + y) for x, y in s2_raw]

    # Plain site circles with "S" label
    for sites in [s0, s1, s2]:
        for sx, sy in sites:
            ax.add_patch(plt.Circle((sx, sy), sr, fc=C_SITE,
                                    ec=C_SITE, lw=LW_THIN, zorder=4))
            ax.text(sx, sy, "S", ha="center", va="center",
                    fontsize=6, fontweight="bold", color="white", zorder=5)

    edge_color = C_MPI + "70"

    # Intra-GPU edges — only connect nearby sites within each GPU
    for edges, sites in [
        ([(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)], s0),
        ([(0, 1), (0, 2), (2, 3), (3, 4)], s1),
        ([(0, 1), (0, 2), (1, 3), (2, 3)], s2),
    ]:
        for i, j in edges:
            ax.plot([sites[i][0], sites[j][0]], [sites[i][1], sites[j][1]],
                    color=edge_color, lw=LW_THIN, zorder=3)

    # Cross-GPU edges — only right-edge of GPU_i to left-edge of GPU_{i+1}
    for sa, ia, sb, ib in [(s0, 3, s1, 0), (s0, 4, s1, 2),
                            (s1, 3, s2, 0), (s1, 4, s2, 2)]:
        ax.plot([sa[ia][0], sb[ib][0]], [sa[ia][1], sb[ib][1]],
                color=edge_color, lw=LW_THIN, zorder=3)

    # MPI labels
    for i in range(2):
        mx = (gpu_xs[i] + gpu_w + gpu_xs[i + 1]) / 2
        ax.text(mx, gpu_y + gpu_h / 2, "MPI", ha="center", va="center",
                fontsize=6, fontstyle="italic", color=C_MPI)

    # ==================================================================
    # Box 1 → Box 2 transition: vertical arrow with "zoom into one GPU"
    # label alongside, visually grouping the two boxes as the
    # "two-level parallelism" pair from the paper's abstract.
    # Arrow tail at the bottom edge of GPU 1 (the middle GPU, centered
    # at CX), so it visually emanates from that specific GPU. Tip just
    # above Box 2's top edge.
    # ==================================================================
    zoom_arrow_top_y = gpu_y                       # tail (bottom of GPU 1)
    zoom_arrow_bot_y = l1_box_y + l1_box_h + 0.02  # tip (just above Box 2)
    ax.annotate("",
                xy=(CX, zoom_arrow_bot_y),     # arrow tip (downward)
                xytext=(CX, zoom_arrow_top_y), # arrow tail
                arrowprops=dict(arrowstyle="-|>",
                                color=C_EXPAND, lw=1.0,
                                mutation_scale=8,
                                shrinkA=0, shrinkB=0),
                zorder=4)
    # Label alongside the arrow, positioned BELOW the white-bbox descriptor
    # so it sits in the visible portion of the arrow between the descriptor
    # and Box 2 (otherwise the descriptor's white bbox would mask it).
    label_y = (l2_box_y + zoom_arrow_bot_y) / 2 - 0.02
    ax.text(CX + 0.06, label_y,
            "zoom into one GPU",
            ha="left", va="center",
            fontsize=6, fontstyle="italic",
            color=C_EXPAND, zorder=6)

    # ==================================================================
    # LEVEL 1 BOX
    # ==================================================================
    l1_box_w = LR - L
    # (Box 2 individual outline removed — now part of the meta-box drawn above.)
    # Dashed divider line right above the Level 2 banner — visually splits
    # the meta-box into the upper "Level 1" half and the lower "Level 2" half.
    divider_y = l1_box_y + l1_box_h + 0.03
    ax.plot([L, LR], [divider_y, divider_y],
            color=C_HPC, lw=LW_THIN, ls="--", zorder=1)
    ax.text(L + l1_box_w / 2, l1_box_y + l1_box_h - 0.02,
            "Level 2: Intra-rank GPU threads \u2014 1 agent = 1 thread (grid-stride at scale)",
            ha="center", va="top", fontsize=6, fontweight="bold",
            color=C_THREAD, zorder=1)
    # System-spec subtitle: makes the magic 110 in "CU 109" self-explanatory.
    # Positioned below the CU row, inside the bottom margin of Level 2 box.
    ax.text(L + l1_box_w / 2, l1_box_y + 0.06,
            "AMD MI250X GCD: 110 CUs \u00b7 8 blocks/CU \u00b7 128 threads/block",
            ha="center", va="center", fontsize=6, fontstyle="italic",
            color=C_THREAD, zorder=1)

    # Horizontal agent bar — centered horizontally to match the GPU row above.
    # bar_total_w = total_gpus_w so the bar sits in the same horizontal
    # envelope as the 3 GPU boxes in Level 1, and bar_left = gx_start so
    # the bar's left edge aligns with the leftmost GPU box.
    bar_h = 0.10
    bar_total_w = total_gpus_w
    bar_left = gx_start
    bar_y = l1_box_y + 0.55  # bar bottom; positioned with ~0.11 gap below
                              # the Level 2 banner and ~0.13 gap above the
                              # CU row.

    # Bar widths are conceptual (Site < Gap << Tree), not calibrated to any
    # specific N_g / N_t — actual experiments use varying values. The visual
    # just conveys the breed-population ordering.
    site_w    = bar_total_w * 0.05
    gap_w_bar = bar_total_w * 0.25
    tree_w    = bar_total_w * 0.70

    # Single-word prefix label, right-aligned with breathing room from
    # the bar. The Level 2 banner + Box 1's per-site descriptor already
    # establish the per-GPU / across-sites scope, so the label stays minimal.
    ax.text(bar_left - 0.10, bar_y + bar_h / 2,
            "Agents:",
            ha="right", va="center",
            fontsize=6, fontstyle="italic",
            color=C_THREAD)

    bx = bar_left
    for color, w, label in [
        (C_SITE, site_w, "Site"),
        (C_GAP, gap_w_bar, "Gap"),
        (C_TREE, tree_w, "Tree"),
    ]:
        rbox(ax, bx, bar_y, w, bar_h, color, fc=color,
             lw=LW_THIN, rad=0.004, zo=3)
        ax.text(bx + w / 2, bar_y + bar_h / 2, label,
                ha="center", va="center", fontsize=6, fontweight="bold",
                color="white", zorder=4)
        bx += w


    # 3 CU boxes (mirror GPU width). Bar->CU gap proportional to
    # l1_box_h = 0.78.
    cu_w_box = gpu_w
    cu_gap_x = gpu_gap_space
    cu_y_top = l1_box_y + 0.42  # ~0.13 below the bar bottom; gives the
                                  # per-CU connector arrows visible travel.
    cu_h_box = 0.26
    cu_y_bot = cu_y_top - cu_h_box
    cu_total_w_val = 3 * cu_w_box + 2 * cu_gap_x
    cu_left_start = CX - cu_total_w_val / 2
    cu_xs = [cu_left_start + i * (cu_w_box + cu_gap_x) for i in range(3)]
    cu_labels = ["CU 0", "CU 1", "CU 109"]

    for ci, (cx, clab) in enumerate(zip(cu_xs, cu_labels)):
        rbox(ax, cx, cu_y_bot, cu_w_box, cu_h_box,
             C_CU, fc=C_CU + "08", lw=LW_MED, rad=0.01, zo=3)
        ax.text(cx + 0.04, cu_y_top - 0.015, clab,
                ha="left", va="top", fontsize=6, fontweight="bold",
                color=C_CU, zorder=5)
        n_blk_cols = 4
        n_blk_rows = 2
        blk_margin_x = 0.05
        blk_margin_top = 0.07  # smaller CU box → tighter top margin
        blk_margin_bot = 0.02
        blk_gap_x = 0.015
        blk_gap_y = 0.018
        blk_avail_w = cu_w_box - 2 * blk_margin_x
        blk_avail_h = cu_h_box - blk_margin_top - blk_margin_bot
        blk_w = (blk_avail_w - (n_blk_cols - 1) * blk_gap_x) / n_blk_cols
        blk_h = (blk_avail_h - (n_blk_rows - 1) * blk_gap_y) / n_blk_rows

        for r in range(n_blk_rows):
            for c in range(n_blk_cols):
                bxx = cx + blk_margin_x + c * (blk_w + blk_gap_x)
                byy = cu_y_bot + blk_margin_bot + (n_blk_rows - 1 - r) * (blk_h + blk_gap_y)
                rbox(ax, bxx, byy, blk_w, blk_h,
                     C_THREAD + "70", fc=C_THREAD + "10",
                     lw=LW_HAIR, rad=0.003, zo=4)
                # Vertical divider lines inside each block — densified
                # so the resulting per-block "thread columns" are roughly
                # the same width as the thread cells in the priority
                # strip in Box 3 (~0.037 in each), giving the two boxes
                # a consistent visual texture.
                n_div = 10
                line_pad = 0.004
                for k in range(1, n_div + 1):
                    line_x = bxx + k * blk_w / (n_div + 1)
                    ax.plot([line_x, line_x],
                            [byy + line_pad, byy + blk_h - line_pad],
                            color=C_THREAD + "70", lw=0.15, zorder=4.5)

    # Ellipsis between CU 1 and CU 109
    ell_cu_x = (cu_xs[1] + cu_w_box + cu_xs[2]) / 2
    ax.text(ell_cu_x, cu_y_top - cu_h_box / 2, "\u00b7\u00b7\u00b7",
            ha="center", va="center", fontsize=14,
            color=C_CU + "70", zorder=4)

    # Vertical arrows from agent bar to CUs (longer now that bar→CU gap is ~0.22 in).
    arrow_top = bar_y - 0.008
    arrow_bot = cu_y_top + 0.01
    n_per_cu = 8
    for cx in cu_xs:
        for k in range(n_per_cu):
            ax_x = cx + cu_w_box * (k + 0.5) / n_per_cu
            ax.annotate("",
                        xy=(ax_x, arrow_bot), xytext=(ax_x, arrow_top),
                        arrowprops=dict(arrowstyle="-|>", color=C_THREAD + "55",
                                        lw=LW_HAIR), zorder=2)


    # ==================================================================
    # Box 2 → Box 3 transition: just a small italic label, no visual
    # connector. Temporal/execution drill-down indicator.
    # ==================================================================
    tick_label_y = (l1_box_y + fk_box_y + fk_box_h) / 2
    # Bold dark "section header" so the inter-box connector reads as a
    # real section title, paralleling the top "Two-level parallelism..." label.
    ax.text(CX, tick_label_y, "Multi-priority kernel pipeline (inside each tick)",
            ha="center", va="center",
            fontsize=6, fontweight="bold",
            color=C_TEXT)

    # ==================================================================
    # FUSED KERNEL BOX (bottom)
    # ==================================================================
    fk_box_x = L
    fk_box_w = LR - L
    rbox(ax, fk_box_x, fk_box_y, fk_box_w, fk_box_h,
         C_BARRIER, fc=C_BARRIER + "06", lw=LW_THICK, rad=0.02, zo=0)
    ax.text(fk_box_x + fk_box_w / 2, fk_box_y + fk_box_h - 0.02,
            "Fused kernel per tick: 10 priorities in order \u2192 1 launch via on-device grid barriers",
            ha="center", va="top", fontsize=6, fontweight="bold",
            color=C_BARRIER, zorder=1)

    # 10 priority steps arranged horizontally
    priorities = [
        ("P0", "Gap"),  ("P1", "Site"), ("P2", "Gap"),  ("P3", "Tree"),
        ("P4", "Gap"),  ("P5", "Tree"), ("P6", "Gap"),  ("P7", "Tree"),
        ("P8", "Gap"),  ("P9", "Site"),
    ]
    bc = {"Gap": C_GAP, "Tree": C_TREE, "Site": C_SITE}

    # Active cells per breed (out of 50, in a 10x5 icon).
    # Ratio Site:Gap:Tree ≈ 1:3:28 preserved as closely as possible
    # given 50 isn't a multiple of 32.
    fill_count = {"Site": 2, "Gap": 5, "Tree": 43}

    n_p = len(priorities)
    margin_x = 0.14
    avail_w = fk_box_w - 2 * margin_x
    gap_between = 0.10  # was 0.04 — wider gaps so short sync arrows fit
    step_w = (avail_w - (n_p - 1) * gap_between) / n_p

    # Priority strip sized for fk_box_h = 0.78.
    step_icon_y_top = fk_box_y + fk_box_h - 0.24
    step_icon_h = 0.38
    step_icon_y_bot = step_icon_y_top - step_icon_h
    step_breed_label_y = step_icon_y_bot - 0.05  # unused now

    step_x_start = fk_box_x + margin_x

    # Short sequential-flow arrows in each gap between adjacent priorities.
    # Tail at right edge of priority i, tip at left edge of priority i+1.
    # IMPORTANT: shrinkA=shrinkB=0 — matplotlib defaults are 2pt each end,
    # which on a 0.10 in gap leaves the arrow physically pulled inward and
    # not touching either icon. Setting both to 0 lets the arrow span the
    # whole gap. The arrow passes through the red grid-barrier line at the
    # midpoint and clearly reaches the next step function.
    short_arrow_y = (step_icon_y_top + step_icon_y_bot) / 2
    for _i in range(n_p - 1):
        icon_right_x = step_x_start + _i * (step_w + gap_between) + step_w
        next_icon_left_x = step_x_start + (_i + 1) * (step_w + gap_between)
        ax.annotate("",
                    xy=(next_icon_left_x, short_arrow_y),       # arrow tip
                    xytext=(icon_right_x, short_arrow_y),       # arrow tail
                    arrowprops=dict(arrowstyle="->",
                                    color=C_HPC, lw=1.2,
                                    mutation_scale=8,
                                    shrinkA=0, shrinkB=0),
                    zorder=6)

    for i, (pnum, breed) in enumerate(priorities):
        sx = step_x_start + i * (step_w + gap_between)
        color = bc[breed]

        # Thread-grid icon: 10 cols × 5 rows = 50 cells. Smaller cell count
        # leaves room for wider gap_between (see below) so we can fit short
        # arrows / sync indicators between adjacent priority icons.
        n_cols = 10
        n_rows = 5
        icon_margin = 0.012
        icon_w_avail = step_w - 2 * icon_margin
        icon_h_avail = step_icon_h - 2 * icon_margin
        cell_gap = 0.008  # was 0.0015 — visible breathing room around cells
        cell_w = (icon_w_avail - (n_cols - 1) * cell_gap) / n_cols
        cell_h = (icon_h_avail - (n_rows - 1) * cell_gap) / n_rows

        n_fill = fill_count[breed]
        cell_idx = 0
        for r in range(n_rows):
            for c in range(n_cols):
                cx_cell = sx + icon_margin + c * (cell_w + cell_gap)
                cy_cell = (step_icon_y_bot + icon_margin
                           + (n_rows - 1 - r) * (cell_h + cell_gap))
                if cell_idx < n_fill:
                    rbox(ax, cx_cell, cy_cell, cell_w, cell_h,
                         color, fc=color, lw=0.15, rad=0.001, zo=5)
                else:
                    rbox(ax, cx_cell, cy_cell, cell_w, cell_h,
                         C_IDLE, fc=C_IDLE + "50", lw=0.12, rad=0.001, zo=5)
                cell_idx += 1

        # Grid barrier indicator between steps — emphasized in red so it
        # reads as a sync point distinct from the navy chassis around it.
        if i < n_p - 1:
            bar_line_x = sx + step_w + gap_between / 2
            ax.plot([bar_line_x, bar_line_x],
                    [step_icon_y_bot, step_icon_y_top],
                    color=C_DISP, lw=1.5, zorder=5)

    # Legend at bottom of fused kernel box
    # Legend split into two text calls so the "|=grid barrier" item can
    # use the same red (C_DISP) as the actual barriers in the strip above.
    # Both text calls share the same y and the same x boundary; navy
    # part ends at the boundary (ha=right), red part starts at it (ha=left),
    # so the legend reads as one continuous line with two colors.
    navy_part = ("each cell = 1 thread   "
                 "\u25a0 active (Site/Gap/Tree)   "
                 "\u25a1 idle at barrier   ")
    red_part = "|=grid barrier"
    char_w = 0.033  # rough per-char width at 6 pt italic sans-serif
    center_x = fk_box_x + fk_box_w / 2
    boundary_x = center_x + (len(navy_part) - len(red_part)) * char_w / 2
    ax.text(boundary_x, fk_box_y + 0.03, navy_part,
            ha="right", va="bottom", fontsize=6, fontstyle="italic",
            color=C_HPC)
    ax.text(boundary_x, fk_box_y + 0.03, red_part,
            ha="left", va="bottom", fontsize=6, fontstyle="italic",
            color=C_DISP)

    # ==================================================================
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    # Save BOTH PDF (vector — for \includegraphics in LaTeX) and PNG
    # (raster preview for dev review). No bbox_inches='tight' so the
    # saved files are exactly FIG_SIZE — fits IEEE column width with
    # no overflow into the gutter.
    out_base = FIGS_DIR / "ggap_architecture"
    # PDF for LaTeX: transparent background so the figure blends with the
    # page color regardless of document-level styling.
    fig.savefig(f"{out_base}.pdf", format="pdf",
                transparent=True, edgecolor="none")
    # PNG for dev preview: keep white background for readability against any
    # IDE / browser background.
    fig.savefig(f"{out_base}.png", format="png", dpi=args.dpi,
                facecolor="white", edgecolor="none")
    print(f"Saved: {out_base}.{{pdf,png}}  ({FIG_W}\u00d7{FIG_H} in)")


if __name__ == "__main__":
    main()
