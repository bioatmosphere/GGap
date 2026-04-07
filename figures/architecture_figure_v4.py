"""
GGap Architecture Figure v4 for SC2026 Application Paper.
IEEE double-column format: 7.16 x ~6.2 inches.

Single integrated full-width panel with three stacked sections:
  1. Level 2: Inter-GPU Parallelism (MPI, 1 rank/GPU) — 3 GPU boxes with sites
  2. Level 1: Intra-GPU Parallelism — agents mapped to thread/block/CU hierarchy
  3. Fused Kernel per Tick — horizontal priority strip with 10 priority steps
     where thread-grid icons show which threads are active at each priority

Usage:
    python architecture_figure_v4.py [--format png|svg] [--dpi 600]
"""
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

FIG_W = 7.16
FIG_H = 3.8

C = dict(
    site="#C53030", gap="#C05621", tree="#2F855A",
    mpi="#2B6CB0", barrier="#718096", nclosure="#B7791F",
    white="#FFFFFF", text="#1A202C", gpu_bg="#EBF4FF",
    thread="#4A5568", cu="#6B46C1", hpc="#1A365D",
    idle="#CBD5E0",  # light grey for idle threads
)


def rbox(ax, x, y, w, h, ec, fc="none", lw=1.0, rad=0.03, zo=1, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={rad}",
        edgecolor=ec, facecolor=fc, linewidth=lw, zorder=zo, linestyle=ls))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", default="png", choices=["png", "svg"])
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.set_aspect("equal")
    ax.axis("off")

    L = 0.05
    LR = FIG_W - 0.05  # full-width panel
    CX = (L + LR) / 2

    # ==================================================================
    # HPC SYSTEM LABEL (top) — in-figure title removed; use LaTeX \caption
    # ==================================================================
    hpc_label_y = FIG_H - 0.12
    ax.text(CX, hpc_label_y, "Exascale HPC System (e.g., Frontier)",
            ha="center", va="center", fontsize=8, fontweight="bold",
            color=C["hpc"],
            bbox=dict(boxstyle="round,pad=0.10", fc=C["hpc"] + "08",
                      ec=C["hpc"] + "40", lw=0.6))

    # ==================================================================
    # LAYOUT COMPUTATION (top-down from HPC label)
    # ==================================================================
    # Level 2: Inter-GPU (directly below HPC label with small gap)
    l2_box_h = 1.00
    l2_box_top = hpc_label_y - 0.14
    l2_box_y = l2_box_top - l2_box_h

    # Level 1: Intra-GPU (below Level 2 with gap for "zoom" arrow)
    l1_box_h = 1.20
    l1_box_top = l2_box_y - 0.18
    l1_box_y = l1_box_top - l1_box_h

    # Fused kernel strip (below Level 1 with gap for "per tick" arrow)
    fk_box_h = 0.64
    fk_box_top = l1_box_y - 0.18
    fk_box_y = fk_box_top - fk_box_h

    # ==================================================================
    # HPC → Level 2 angled connectors
    # ==================================================================
    l2_box_w = LR - L
    rbox(ax, L, l2_box_y, l2_box_w, l2_box_h,
         C["mpi"], fc=C["mpi"] + "06", lw=1.0, rad=0.02, zo=0)
    ax.text(L + l2_box_w / 2, l2_box_y + l2_box_h - 0.02,
            "Level 2: Inter-GPU Parallelism (MPI, 1 rank/GPU)",
            ha="center", va="top", fontsize=8, fontweight="bold",
            color=C["mpi"], zorder=1)

    # 3 GPU boxes inside
    gpu_w = 1.70
    gpu_h = 0.54
    gpu_gap_space = 0.28
    total_gpus_w = 3 * gpu_w + 2 * gpu_gap_space
    gx_start = CX - total_gpus_w / 2
    content_top = l2_box_y + l2_box_h - 0.14
    content_bot = l2_box_y + 0.14
    gpu_y = (content_top + content_bot) / 2 - gpu_h / 2
    gpu_xs = [gx_start + i * (gpu_w + gpu_gap_space) for i in range(3)]

    # HPC → L2 angled lines
    hpc_line_y_start = hpc_label_y - 0.10
    hpc_line_y_mid = l2_box_y + l2_box_h + 0.06
    for gx in gpu_xs:
        gcx = gx + gpu_w / 2
        gpu_top_y = gpu_y + gpu_h
        ax.plot([gcx, gcx], [hpc_line_y_mid, gpu_top_y + 0.01],
                color=C["hpc"] + "50", lw=0.5, zorder=1)
    ax.plot([gpu_xs[0] + gpu_w / 2, gpu_xs[2] + gpu_w / 2],
            [hpc_line_y_mid, hpc_line_y_mid],
            color=C["hpc"] + "50", lw=0.5, zorder=1)
    ax.plot([CX, CX], [hpc_line_y_start, hpc_line_y_mid],
            color=C["hpc"] + "50", lw=0.5, zorder=1)

    for gx, lab in zip(gpu_xs, ["GPU 0", "GPU 1", "GPU N"]):
        rbox(ax, gx, gpu_y, gpu_w, gpu_h, C["mpi"], fc=C["gpu_bg"],
             lw=0.6, rad=0.015, zo=2)
        ax.text(gx + gpu_w / 2, gpu_y + gpu_h - 0.03, lab,
                ha="center", va="top", fontsize=8, fontweight="bold",
                color=C["mpi"], zorder=3)

    # Site circles (labeled "S" inside)
    sr = 0.078
    s0_raw = [(0.28, 0.28), (0.70, 0.30), (0.34, 0.10),
              (1.02, 0.22), (1.24, 0.09)]
    s0 = [(gpu_xs[0] + x, gpu_y + y) for x, y in s0_raw]
    s1_raw = [(0.24, 0.30), (0.68, 0.32), (0.46, 0.12),
              (1.04, 0.26), (1.24, 0.10)]
    s1 = [(gpu_xs[1] + x, gpu_y + y) for x, y in s1_raw]
    s2_raw = [(0.36, 0.26), (1.00, 0.22), (0.58, 0.08), (1.18, 0.08)]
    s2 = [(gpu_xs[2] + x, gpu_y + y) for x, y in s2_raw]

    # Plain site circles with "S" label
    for sites in [s0, s1, s2]:
        for sx, sy in sites:
            ax.add_patch(plt.Circle((sx, sy), sr, fc=C["site"] + "15",
                                    ec=C["site"], lw=0.5, zorder=4))
            ax.text(sx, sy, "S", ha="center", va="center",
                    fontsize=8, fontweight="bold", color=C["site"], zorder=5)

    edge_color = C["mpi"] + "70"

    # Intra-GPU edges — only connect nearby sites within each GPU
    for edges, sites in [
        ([(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)], s0),
        ([(0, 1), (0, 2), (2, 3), (3, 4)], s1),
        ([(0, 1), (0, 2), (1, 3), (2, 3)], s2),
    ]:
        for i, j in edges:
            ax.plot([sites[i][0], sites[j][0]], [sites[i][1], sites[j][1]],
                    color=edge_color, lw=0.55, zorder=3)

    # Cross-GPU edges — only right-edge of GPU_i to left-edge of GPU_{i+1}
    for sa, ia, sb, ib in [(s0, 3, s1, 0), (s0, 4, s1, 2),
                            (s1, 3, s2, 0), (s1, 4, s2, 2)]:
        ax.plot([sa[ia][0], sb[ib][0]], [sa[ia][1], sb[ib][1]],
                color=edge_color, lw=0.55, zorder=3)

    # MPI labels
    for i in range(2):
        mx = (gpu_xs[i] + gpu_w + gpu_xs[i + 1]) / 2
        ax.text(mx, gpu_y + gpu_h / 2, "MPI", ha="center", va="center",
                fontsize=8, fontweight="bold", color=C["mpi"])

    # Callout OUTSIDE Level 2 box (left side), pointing into one GPU's site
    callout_target = s0[0]  # top-left site of GPU 0
    ax.annotate("site data\ncolocated",
                xy=(callout_target[0] - sr * 0.6,
                    callout_target[1]),
                xytext=(L - 0.02, callout_target[1]),
                ha="left", va="center",
                fontsize=8, fontstyle="italic", fontweight="bold",
                color=C["site"] + "DD",
                arrowprops=dict(arrowstyle="->", color=C["site"] + "80",
                                lw=0.6),
                linespacing=1.1, zorder=7,
                annotation_clip=False)


    # ==================================================================
    # "zoom into one GPU" arrow (Level 2 → Level 1)
    # ==================================================================
    zoom_arr_start = l2_box_y - 0.03
    zoom_arr_end = l1_box_y + l1_box_h + 0.02
    gpu1_cx = gpu_xs[1] + gpu_w / 2
    ax.annotate("", xy=(gpu1_cx, zoom_arr_end),
                xytext=(gpu1_cx, zoom_arr_start),
                arrowprops=dict(arrowstyle="-|>", color=C["mpi"],
                                lw=0.8, ls="--"), zorder=2)

    # ==================================================================
    # LEVEL 1 BOX
    # ==================================================================
    l1_box_w = LR - L
    rbox(ax, L, l1_box_y, l1_box_w, l1_box_h,
         C["thread"], fc=C["thread"] + "06", lw=1.0, rad=0.02, zo=0)
    ax.text(L + l1_box_w / 2, l1_box_y + l1_box_h - 0.02,
            "Level 1: Intra-GPU Parallelism (Threads)",
            ha="center", va="top", fontsize=8, fontweight="bold",
            color=C["thread"], zorder=1)

    # Horizontal agent bar (wide)
    bar_h = 0.12
    bar_total_w = LR - L - 0.40
    bar_left = L + 0.20
    bar_y = l1_box_y + l1_box_h - 0.42

    site_w = bar_total_w * 0.025
    gap_w_bar = bar_total_w * 0.08
    tree_w = bar_total_w * 0.895

    bx = bar_left
    for color, w, label in [
        (C["site"], site_w, "Site"),
        (C["gap"], gap_w_bar, "Gap"),
        (C["tree"], tree_w, "Tree"),
    ]:
        rbox(ax, bx, bar_y, w, bar_h, color, fc=color + "20",
             lw=0.5, rad=0.004, zo=3)
        if w >= site_w:
            ax.text(bx + w / 2, bar_y + bar_h / 2, label,
                    ha="center", va="center", fontsize=8, fontweight="bold",
                    color=color, zorder=4)
        bx += w

    ax.text(bar_left + bar_total_w / 2, bar_y + bar_h + 0.015,
            "N\u209b sites \u00d7 500 gaps \u00d7 1000 trees per GPU",
            ha="center", va="bottom", fontsize=8, fontweight="bold",
            color=C["thread"])

    # 3 CU boxes (mirror GPU width)
    cu_w_box = gpu_w
    cu_gap_x = gpu_gap_space
    cu_y_top = bar_y - 0.14
    cu_h_box = 0.42
    cu_y_bot = cu_y_top - cu_h_box
    cu_total_w_val = 3 * cu_w_box + 2 * cu_gap_x
    cu_left_start = CX - cu_total_w_val / 2
    cu_xs = [cu_left_start + i * (cu_w_box + cu_gap_x) for i in range(3)]
    cu_labels = ["CU 0", "CU 1", "CU 109"]

    for ci, (cx, clab) in enumerate(zip(cu_xs, cu_labels)):
        rbox(ax, cx, cu_y_bot, cu_w_box, cu_h_box,
             C["cu"], fc=C["cu"] + "08", lw=0.6, rad=0.01, zo=3)
        ax.text(cx + 0.04, cu_y_top - 0.02, clab,
                ha="left", va="top", fontsize=8, fontweight="bold",
                color=C["cu"], zorder=5)
        ax.text(cx + cu_w_box - 0.04, cu_y_top - 0.02,
                "8 blocks", ha="right", va="top", fontsize=8,
                fontstyle="italic", color=C["cu"] + "AA", zorder=5)

        n_blk_cols = 4
        n_blk_rows = 2
        blk_margin_x = 0.05
        blk_margin_top = 0.14
        blk_margin_bot = 0.04
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
                     C["thread"] + "70", fc=C["thread"] + "10",
                     lw=0.35, rad=0.003, zo=4)

                # 4x4 thread grid inside every block
                n_thr_cols = 4
                n_thr_rows = 4
                tm = 0.004
                t_avail_w = blk_w - 2 * tm
                t_avail_h = blk_h - 2 * tm
                t_cw = (t_avail_w - (n_thr_cols - 1) * 0.001) / n_thr_cols
                t_ch = (t_avail_h - (n_thr_rows - 1) * 0.001) / n_thr_rows
                for tr in range(n_thr_rows):
                    for tc in range(n_thr_cols):
                        tx = bxx + tm + tc * (t_cw + 0.001)
                        ty = byy + tm + (n_thr_rows - 1 - tr) * (t_ch + 0.001)
                        rbox(ax, tx, ty, t_cw, t_ch,
                             C["thread"], fc=C["thread"] + "30",
                             lw=0.1, rad=0.0005, zo=5)

    # Ellipsis between CU 1 and CU 109
    ell_cu_x = (cu_xs[1] + cu_w_box + cu_xs[2]) / 2
    ax.text(ell_cu_x, cu_y_top - cu_h_box / 2, "\u00b7\u00b7\u00b7",
            ha="center", va="center", fontsize=14,
            color=C["cu"] + "70", zorder=4)

    # Vertical arrows from agent bar to CUs
    arrow_top = bar_y - 0.008
    arrow_bot = cu_y_top + 0.01
    n_per_cu = 6
    for cx in cu_xs:
        for k in range(n_per_cu):
            ax_x = cx + cu_w_box * (k + 0.5) / n_per_cu
            ax.annotate("",
                        xy=(ax_x, arrow_bot), xytext=(ax_x, arrow_top),
                        arrowprops=dict(arrowstyle="-|>", color=C["thread"] + "55",
                                        lw=0.4), zorder=2)

    label_y = (arrow_top + arrow_bot) / 2
    ax.text(bar_left + bar_total_w / 2, label_y, "1 agent = 1 thread",
            ha="center", va="center", fontsize=8, fontweight="bold",
            color=C["thread"],
            bbox=dict(boxstyle="round,pad=0.18", fc="white",
                      ec=C["thread"] + "60", lw=0.4))


    # ==================================================================
    # "per tick" arrow (Level 1 → Fused Kernel)
    # ==================================================================
    per_tick_arr_start = l1_box_y - 0.03
    per_tick_arr_end = fk_box_y + fk_box_h + 0.02
    ax.annotate("", xy=(CX, per_tick_arr_end),
                xytext=(CX, per_tick_arr_start),
                arrowprops=dict(arrowstyle="-|>", color=C["barrier"],
                                lw=0.8, ls="--"), zorder=2)
    ax.text(CX + 0.10, (per_tick_arr_start + per_tick_arr_end) / 2,
            "per tick", ha="left", va="center",
            fontsize=8, fontstyle="italic", fontweight="bold",
            color=C["barrier"])

    # ==================================================================
    # FUSED KERNEL BOX (bottom)
    # ==================================================================
    fk_box_x = L
    fk_box_w = LR - L
    rbox(ax, fk_box_x, fk_box_y, fk_box_w, fk_box_h,
         C["barrier"], fc=C["barrier"] + "06", lw=1.0, rad=0.02, zo=0)
    ax.text(fk_box_x + fk_box_w / 2, fk_box_y + fk_box_h - 0.02,
            "Fused Kernel per Tick  (1 launch, on-device grid barriers)",
            ha="center", va="top", fontsize=8, fontweight="bold",
            color=C["barrier"], zorder=1)

    # 10 priority steps arranged horizontally
    priorities = [
        ("P0", "Gap"),  ("P1", "Site"), ("P2", "Gap"),  ("P3", "Tree"),
        ("P4", "Gap"),  ("P5", "Tree"), ("P6", "Gap"),  ("P7", "Tree"),
        ("P8", "Gap"),  ("P9", "Site"),
    ]
    bc = {"Gap": C["gap"], "Tree": C["tree"], "Site": C["site"]}

    # Active cells per breed (out of 32, in an 8x4 icon)
    # Site = 1 cell, Gap = 3 cells, Tree = 32 - 4 = 28 cells
    fill_count = {"Site": 1, "Gap": 3, "Tree": 28}

    n_p = len(priorities)
    margin_x = 0.14
    avail_w = fk_box_w - 2 * margin_x
    gap_between = 0.04
    step_w = (avail_w - (n_p - 1) * gap_between) / n_p

    step_icon_y_top = fk_box_y + fk_box_h - 0.32
    step_icon_h = 0.18
    step_icon_y_bot = step_icon_y_top - step_icon_h
    step_breed_label_y = step_icon_y_bot - 0.05  # unused now

    step_x_start = fk_box_x + margin_x

    # Horizontal priority-flow arrow spanning the priority strip (above icons)
    arrow_y = step_icon_y_top + 0.06
    arrow_x0 = step_x_start
    arrow_x1 = step_x_start + n_p * step_w + (n_p - 1) * gap_between
    ax.annotate("",
                xy=(arrow_x1, arrow_y), xytext=(arrow_x0, arrow_y),
                arrowprops=dict(arrowstyle="-|>",
                                color=C["barrier"], lw=1.0),
                zorder=4)
    ax.text((arrow_x0 + arrow_x1) / 2, arrow_y + 0.02,
            "10 priorities \u2192 executed in order",
            ha="center", va="bottom", fontsize=8, fontstyle="italic",
            color=C["barrier"])

    for i, (pnum, breed) in enumerate(priorities):
        sx = step_x_start + i * (step_w + gap_between)
        color = bc[breed]

        # Thread-grid icon: 8 cols × 4 rows = 32 cells (same icon size, finer resolution)
        n_cols = 8
        n_rows = 4
        icon_margin = 0.012
        icon_w_avail = step_w - 2 * icon_margin
        icon_h_avail = step_icon_h - 2 * icon_margin
        cell_gap = 0.0015
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
                         color, fc=color + "80", lw=0.15, rad=0.001, zo=5)
                else:
                    rbox(ax, cx_cell, cy_cell, cell_w, cell_h,
                         C["idle"], fc=C["idle"] + "50", lw=0.12, rad=0.001, zo=5)
                cell_idx += 1

        # Grid barrier indicator between steps
        if i < n_p - 1:
            bar_line_x = sx + step_w + gap_between / 2
            ax.plot([bar_line_x, bar_line_x],
                    [step_icon_y_bot, step_icon_y_top],
                    color=C["barrier"], lw=1.0, zorder=5)

    # Legend at bottom of fused kernel box
    ax.text(fk_box_x + fk_box_w / 2, fk_box_y + 0.04,
            "|=grid barrier   "
            "\u25a0 active threads (\u221d agents of active breed)   "
            "\u25a1 idle at barrier",
            ha="center", va="bottom", fontsize=8, fontstyle="italic",
            color=C["barrier"])

    # ==================================================================
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    out = f"architecture_figure_v4.{args.format}"
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight", pad_inches=0.02,
                facecolor="white", edgecolor="none")
    print(f"Saved: {out}  ({FIG_W}\u00d7{FIG_H} in)")


if __name__ == "__main__":
    main()
