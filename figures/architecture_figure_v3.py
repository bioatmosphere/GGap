"""
GGap Architecture Figure v3 for SC2026 Application Paper.
IEEE double-column format: 7.16 x 3.0 inches (~1/3 page).

Two-panel layout (2/3 + 1/3):
  Left  — Two-level parallelism (HPC system → MPI GPUs → threads)
  Right — Priority kernel pipeline with grid barriers

Usage:
    python architecture_figure_v3.py [--format png|pdf|svg] [--dpi 600]
"""
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

FIG_W = 7.16
FIG_H = 5.5

C = dict(
    site="#C53030", gap="#C05621", tree="#2F855A",
    mpi="#2B6CB0", barrier="#718096", nclosure="#B7791F",
    white="#FFFFFF", text="#1A202C", gpu_bg="#EBF4FF",
    thread="#4A5568", cu="#6B46C1", hpc="#1A365D",
)

SPLIT = FIG_W * 2 / 3


def rbox(ax, x, y, w, h, ec, fc="none", lw=1.0, rad=0.03, zo=1, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={rad}",
        edgecolor=ec, facecolor=fc, linewidth=lw, zorder=zo, linestyle=ls))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.set_aspect("equal")
    ax.axis("off")

    L = 0.05
    LR = SPLIT - 0.10
    LEFT_CX = (L + LR) / 2

    # ==================================================================
    # LEFT PANEL (a)
    # ==================================================================
    ax.text((L + LR) / 2, FIG_H - 0.02, "Two-Level Parallelism",
            ha="center", va="top", fontsize=9, fontweight="bold", color=C["text"])

    # ─────────────────────────────────────────
    # HPC label at top with angled lines down to Level 2 box
    # ─────────────────────────────────────────
    hpc_label_y = FIG_H - 0.20
    hpc_cx = LEFT_CX
    ax.text(hpc_cx, hpc_label_y, "Exascale HPC System (e.g., Frontier)",
            ha="center", va="center", fontsize=7, fontweight="bold",
            color=C["hpc"],
            bbox=dict(boxstyle="round,pad=0.10", fc=C["hpc"] + "08",
                      ec=C["hpc"] + "40", lw=0.6))

    # Level 2 box (moved down to give HPC label space)
    l2_box_x = L
    l2_box_w = LR - L
    l2_box_h = 2.05
    l2_box_y = hpc_label_y - 0.20 - l2_box_h

    rbox(ax, l2_box_x, l2_box_y, l2_box_w, l2_box_h,
         C["mpi"] + "30", fc=C["mpi"] + "04", lw=0.5, rad=0.015, zo=0)
    ax.text(l2_box_x + l2_box_w / 2, l2_box_y + l2_box_h - 0.02,
            "Level 2: Inter-GPU Parallelism (MPI, 1 rank/GPU)",
            ha="center", va="top", fontsize=7, fontweight="bold",
            color=C["mpi"], zorder=1)

    # 3 GPU boxes inside
    gpu_w = 1.20
    gpu_h = 1.05
    gpu_gap_space = 0.18
    total_gpus_w = 3 * gpu_w + 2 * gpu_gap_space
    gx_start = LEFT_CX - total_gpus_w / 2
    # Center the GPU row vertically inside L2 box content area
    # Content area: above colocation note (l2_box_y + 0.05) and below title (l2_box_y + l2_box_h - 0.10)
    content_top = l2_box_y + l2_box_h - 0.10
    content_bot = l2_box_y + 0.10
    gpu_y = (content_top + content_bot) / 2 - gpu_h / 2
    gpu_xs = [gx_start + i * (gpu_w + gpu_gap_space) for i in range(3)]

    # 90° angled lines from HPC label down to each GPU box
    hpc_line_y_start = hpc_label_y - 0.10
    hpc_line_y_mid = l2_box_y + l2_box_h + 0.02  # horizontal run level
    for gx in gpu_xs:
        gcx = gx + gpu_w / 2
        gpu_top = gpu_y + gpu_h
        # Vertical down from mid level to GPU top
        ax.plot([gcx, gcx], [hpc_line_y_mid, gpu_top + 0.01],
                color=C["hpc"] + "50", lw=0.5, zorder=1)
    # Horizontal bar connecting all three verticals
    ax.plot([gpu_xs[0] + gpu_w / 2, gpu_xs[2] + gpu_w / 2],
            [hpc_line_y_mid, hpc_line_y_mid],
            color=C["hpc"] + "50", lw=0.5, zorder=1)
    # Single vertical from HPC label down to horizontal bar
    ax.plot([hpc_cx, hpc_cx], [hpc_line_y_start, hpc_line_y_mid],
            color=C["hpc"] + "50", lw=0.5, zorder=1)

    for gx, lab in zip(gpu_xs, ["GPU 0", "GPU 1", "GPU N"]):
        rbox(ax, gx, gpu_y, gpu_w, gpu_h, C["mpi"], fc=C["gpu_bg"],
             lw=0.6, rad=0.015, zo=2)
        ax.text(gx + gpu_w / 2, gpu_y + gpu_h - 0.02, lab,
                ha="center", va="top", fontsize=6, fontweight="bold",
                color=C["mpi"], zorder=3)

    # Site circles with "S"
    sr = 0.10
    s0_raw = [(0.28, 0.72), (0.58, 0.78), (0.32, 0.30),
              (0.80, 0.58), (0.92, 0.24)]
    s0 = [(gpu_xs[0] + x, gpu_y + y) for x, y in s0_raw]
    s1_raw = [(0.22, 0.76), (0.70, 0.82), (0.42, 0.36),
              (0.96, 0.34), (0.26, 0.16)]
    s1 = [(gpu_xs[1] + x, gpu_y + y) for x, y in s1_raw]
    s2_raw = [(0.34, 0.66), (0.78, 0.60), (0.48, 0.24), (0.90, 0.20)]
    s2 = [(gpu_xs[2] + x, gpu_y + y) for x, y in s2_raw]

    for sites in [s0, s1, s2]:
        for sx, sy in sites:
            ax.add_patch(plt.Circle((sx, sy), sr, fc=C["site"] + "15",
                                    ec=C["site"], lw=0.45, zorder=4))
            ax.text(sx, sy, "S", ha="center", va="center",
                    fontsize=6, fontweight="bold", color=C["site"], zorder=5)

    # Intra-GPU edges
    for edges, sites in [
        ([(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)], s0),
        ([(0, 1), (0, 2), (2, 3), (2, 4)], s1),
        ([(0, 1), (0, 2), (1, 3), (2, 3)], s2),
    ]:
        for i, j in edges:
            ax.plot([sites[i][0], sites[j][0]], [sites[i][1], sites[j][1]],
                    color=C["site"] + "35", lw=0.35, zorder=3)

    # Cross-GPU edges (same style as intra-GPU; data still flows via MPI)
    for sa, ia, sb, ib in [(s0, 3, s1, 0), (s0, 4, s1, 4),
                            (s1, 1, s2, 0), (s1, 3, s2, 2)]:
        ax.plot([sa[ia][0], sb[ib][0]], [sa[ia][1], sb[ib][1]],
                color=C["site"] + "35", lw=0.35, zorder=3)

    # MPI labels
    for i in range(2):
        mx = (gpu_xs[i] + gpu_w + gpu_xs[i + 1]) / 2
        ax.text(mx, gpu_y + gpu_h / 2, "MPI", ha="center", va="center",
                fontsize=6, fontweight="bold", color=C["mpi"])

    # Colocation note inside Level 2 box (just above bottom edge)
    coloc_y = l2_box_y + 0.025
    ax.text(l2_box_x + l2_box_w / 2, coloc_y,
            "Each \u25cb = 1 site (site agent + 500 gaps + 500K trees, all colocated on the same GPU)",
            ha="center", va="bottom", fontsize=6, fontstyle="italic",
            color=C["site"] + "CC")

    ey = l2_box_y - 0.03  # placeholder for arrow positioning

    # ─────────────────────────────────────────
    # Down arrow from GPU 1 to Level 1
    # ─────────────────────────────────────────
    gpu1_cx = gpu_xs[1] + gpu_w / 2

    # Level 1 box
    l1_box_x = L
    l1_box_w = LR - L
    l1_box_h = 2.05
    l1_box_y = 0.04
    rbox(ax, l1_box_x, l1_box_y, l1_box_w, l1_box_h,
         C["thread"] + "30", fc=C["thread"] + "04", lw=0.5, rad=0.015, zo=0)
    ax.text(l1_box_x + l1_box_w / 2, l1_box_y + l1_box_h - 0.02,
            "Level 1: Intra-GPU Parallelism (Threads)",
            ha="center", va="top", fontsize=7, fontweight="bold",
            color=C["thread"], zorder=1)

    # Arrow from GPU 1 down to Level 1 box (short, between the two boxes)
    arr_start = l2_box_y - 0.02
    arr_end = l1_box_y + l1_box_h + 0.01
    ax.annotate("", xy=(gpu1_cx, arr_end), xytext=(gpu1_cx, arr_start),
                arrowprops=dict(arrowstyle="-|>", color=C["mpi"],
                                lw=0.8, ls="--"), zorder=2)
    ax.text(gpu1_cx + 0.10, (arr_start + arr_end) / 2,
            "zoom into one GPU", ha="left", va="center",
            fontsize=6, fontstyle="italic", fontweight="bold",
            color=C["mpi"])

    # === HORIZONTAL agent bar (well below the title) ===
    bar_h = 0.18
    bar_y = l1_box_y + l1_box_h - 0.50
    bar_left = gx_start
    bar_total_w = total_gpus_w

    site_w = bar_total_w * 0.025  # slightly larger so "Site" label fits
    gap_w = bar_total_w * 0.08
    tree_w = bar_total_w * 0.895

    bx = bar_left
    for color, w, label, n_lines in [
        (C["site"], site_w, "Site", 2),
        (C["gap"], gap_w, "Gap", 6),
        (C["tree"], tree_w, "Tree", 40),
    ]:
        rbox(ax, bx, bar_y, w, bar_h, color, fc=color + "20",
             lw=0.45, rad=0.003, zo=3)
        for li in range(1, n_lines):
            lx = bx + w * li / n_lines
            ax.plot([lx, lx], [bar_y + 0.008, bar_y + bar_h - 0.008],
                    color=color + "20", lw=0.2, zorder=3)
        # Always label
        if w >= site_w:
            ax.text(bx + w / 2, bar_y + bar_h / 2, label,
                    ha="center", va="center", fontsize=6, fontweight="bold",
                    color=color, zorder=4)
        bx += w

    ax.text(bar_left + bar_total_w / 2, bar_y + bar_h + 0.025,
            "All agents on one GPU  (N\u209b sites \u00d7 500 gaps/site \u00d7 1000 trees/gap)",
            ha="center", va="bottom", fontsize=6, fontweight="bold",
            color=C["thread"])

    # ──────────────────────────────────────────────────
    # NESTED: 3 CU boxes, each with 2x4 blocks; agent→thread arrows (illustrative)
    # ──────────────────────────────────────────────────
    # Mirror GPU box geometry: same width and gap
    n_cu = 3
    cu_w_box = gpu_w
    cu_gap_x = gpu_gap_space
    cu_y_top = bar_y - 0.30
    cu_h_box = 0.78
    cu_y_bot = cu_y_top - cu_h_box
    # Center the CU row, same as GPU row
    cu_total_w = 3 * cu_w_box + 2 * cu_gap_x
    cu_left_start = LEFT_CX - cu_total_w / 2

    cu_xs = [cu_left_start + i * (cu_w_box + cu_gap_x) for i in range(n_cu)]
    cu_labels = ["CU 0", "CU 1", "CU 109"]

    for ci, (cx, clab) in enumerate(zip(cu_xs, cu_labels)):
        rbox(ax, cx, cu_y_bot, cu_w_box, cu_h_box,
             C["cu"], fc=C["cu"] + "08", lw=0.6, rad=0.01, zo=3)
        ax.text(cx + 0.03, cu_y_top - 0.02, clab,
                ha="left", va="top", fontsize=6, fontweight="bold",
                color=C["cu"], zorder=5)
        # "8 blocks/CU" label inside CU box, top-right corner
        ax.text(cx + cu_w_box - 0.03, cu_y_top - 0.02,
                "8 blocks", ha="right", va="top", fontsize=6,
                fontstyle="italic", color=C["cu"] + "AA", zorder=5)

        # 2 rows × 4 cols = 8 blocks
        n_blk_cols = 4
        n_blk_rows = 2
        blk_margin_x = 0.04
        blk_margin_top = 0.10
        blk_margin_bot = 0.04
        blk_gap_x = 0.012
        blk_gap_y = 0.015
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
                     lw=0.3, rad=0.003, zo=4)

                # Inside the FIRST block of CU 1 (middle, top-left block), draw thread cells
                if ci == 1 and r == 0 and c == 0:
                    n_thr_cols = 2
                    n_thr_rows = 4
                    tm = 0.005
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
                                 lw=0.15, rad=0.001, zo=5)

    # Ellipsis between CU 1 and CU 109
    ell_cu_x = (cu_xs[1] + cu_w_box + cu_xs[2]) / 2
    ax.text(ell_cu_x, cu_y_top - cu_h_box / 2, "\u00b7\u00b7\u00b7",
            ha="center", va="center", fontsize=14,
            color=C["cu"] + "70", zorder=4)

    # Annotation pointing into the leftmost block of CU 1 (middle CU)
    blk1_cx = cu_xs[1] + 0.04 + ((cu_w_box - 0.08) / 4) / 2
    blk1_cy = cu_y_bot + 0.04 + (cu_h_box - 0.14) / 4
    ax.annotate("128 threads/block",
                xy=(blk1_cx + 0.05, blk1_cy + 0.06),
                xytext=(blk1_cx + 0.30, blk1_cy + 0.18),
                fontsize=6, fontstyle="italic", color=C["thread"],
                arrowprops=dict(arrowstyle="-", color=C["thread"] + "70",
                                lw=0.35), zorder=6)

    # === Vertical arrows from agent bar to each CU (6 per CU = 18 total) ===
    arrow_top = bar_y - 0.005
    arrow_bot = cu_y_top + 0.01
    n_per_cu = 6
    for cx in cu_xs:
        for k in range(n_per_cu):
            # Spread 6 arrows across the width of each CU
            ax_x = cx + cu_w_box * (k + 0.5) / n_per_cu
            ax.annotate("",
                        xy=(ax_x, arrow_bot), xytext=(ax_x, arrow_top),
                        arrowprops=dict(arrowstyle="-|>", color=C["thread"] + "55",
                                        lw=0.4), zorder=2)

    # "1 agent = 1 thread" label centered between bar and CU row
    label_y = (arrow_top + arrow_bot) / 2
    ax.text(bar_left + bar_total_w / 2, label_y, "1 agent = 1 thread",
            ha="center", va="center", fontsize=6, fontweight="bold",
            color=C["thread"],
            bbox=dict(boxstyle="round,pad=0.15", fc="white",
                      ec=C["thread"] + "60", lw=0.4))

    # Total summary at bottom
    ax.text(bar_left + bar_total_w / 2, cu_y_bot - 0.10,
            "110 CUs \u00d7 8 blocks/CU \u00d7 128 threads/block = 112,640 concurrent threads",
            ha="center", va="top", fontsize=6, fontweight="bold",
            color=C["thread"])

    # ==================================================================
    # DIVIDER
    # ==================================================================
    ax.plot([SPLIT, SPLIT], [0.04, FIG_H - 0.04],
            color=C["text"] + "12", lw=0.5, zorder=1)

    # ==================================================================
    # RIGHT PANEL (b)
    # ==================================================================
    rp_l = SPLIT + 0.05
    rp_r = FIG_W - 0.03

    ax.text((rp_l + rp_r) / 2, FIG_H - 0.02,
            "Fused Kernel per Tick",
            ha="center", va="top", fontsize=9, fontweight="bold",
            color=C["text"])

    # Tick header
    tick_y = FIG_H - 0.32
    ax.text(rp_l + 0.06, tick_y, "Tick T",
            ha="left", va="center", fontsize=7, fontweight="bold",
            color=C["text"],
            bbox=dict(boxstyle="round,pad=0.12", fc=C["barrier"] + "15",
                      ec=C["barrier"] + "60", lw=0.5))

    # Pipeline box - emphasized solid border to show "one kernel"
    pipe_t = tick_y - 0.16
    pipe_b = 0.32
    pipe_w = rp_r - rp_l - 0.16  # leave room for return arrow on right
    rbox(ax, rp_l, pipe_b, pipe_w, pipe_t - pipe_b,
         C["barrier"], fc=C["barrier"] + "06", lw=0.8, rad=0.018, zo=1)
    # "1 kernel launch" badge on the left side of the pipe box
    ax.text(rp_l - 0.02, (pipe_t + pipe_b) / 2,
            "1 kernel launch",
            ha="center", va="center", fontsize=6, fontweight="bold",
            color=C["barrier"], rotation=90,
            bbox=dict(boxstyle="round,pad=0.10", fc="white",
                      ec=C["barrier"] + "60", lw=0.4))

    priorities = [
        ("P0",  "Gap",  "Aggregate litter & LAI"),
        ("P1",  "Site", "Soil biogeochem (365d)"),
        ("P2",  "Gap",  "Relay climate to gaps"),
        ("P3",  "Tree", "Env. response, pot. growth"),
        ("P4",  "Gap",  "Aggregate N demand"),
        ("P5",  "Tree", "Seedbank, renewal"),
        ("P6",  "Gap",  "Density-based recruit"),
        ("P7",  "Tree", "N-limited growth, mort."),
        ("P8",  "Gap",  "Aggregate N consumed"),
        ("P9",  "Site", "Soil N balance, leaching"),
        ("P10", "Site", "Seed dispersal (MPI)"),
    ]
    bc = {"Gap": C["gap"], "Tree": C["tree"], "Site": C["site"]}

    n = len(priorities)
    row_h = (pipe_t - pipe_b - 0.06) / n
    badge_w = 0.32
    badge_h = row_h * 0.62

    for i, (pnum, breed, desc) in enumerate(priorities):
        ry = pipe_t - 0.03 - (i + 0.5) * row_h
        color = bc[breed]
        px = rp_l + 0.08
        rbox(ax, px, ry - badge_h / 2, badge_w, badge_h, color, fc=color,
             lw=0, rad=0.012, zo=4)
        ax.text(px + badge_w / 2, ry, pnum, ha="center", va="center",
                fontsize=6, fontweight="bold", color="white", zorder=5)
        dx = px + badge_w + 0.06
        ax.text(dx, ry, desc, ha="left", va="center",
                fontsize=6, color=C["text"] + "CC", zorder=4)
        if i < n - 1:
            # Down arrow from this priority badge to the next priority badge
            arrow_top_y = ry - badge_h / 2 - 0.008
            arrow_bot_y = ry - row_h + badge_h / 2 + 0.008
            ax.annotate("",
                        xy=(px + badge_w / 2, arrow_bot_y),
                        xytext=(px + badge_w / 2, arrow_top_y),
                        arrowprops=dict(arrowstyle="-|>",
                                        color=C["barrier"] + "90",
                                        lw=0.7),
                        zorder=3)

    # === Curved return arrow: bottom of pipeline back to top, labeled "Tick T+1" ===
    # Use FancyArrowPatch with arc connection
    from matplotlib.patches import FancyArrowPatch
    return_x = rp_l + pipe_w + 0.02
    return_top_y = pipe_t - 0.02
    return_bot_y = pipe_b + 0.02

    arc = FancyArrowPatch(
        (return_x, return_bot_y), (return_x, return_top_y),
        connectionstyle="arc3,rad=0.45",
        arrowstyle="-|>", mutation_scale=10,
        color=C["barrier"], lw=1.0, zorder=4)
    ax.add_patch(arc)

    # "Tick T+1" label on the curve
    ax.text(return_x + 0.20, (return_top_y + return_bot_y) / 2,
            "Tick\nT+1", ha="left", va="center",
            fontsize=6, fontweight="bold", color=C["barrier"],
            linespacing=1.1)

    # Bottom annotation
    ax.text(rp_l + pipe_w / 2, pipe_b - 0.04,
            "\u2193 = grid barrier (on-device, no CPU sync)",
            ha="center", va="top", fontsize=6, fontstyle="italic",
            color=C["barrier"])

    # ==================================================================
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    out = f"architecture_figure_v3.{args.format}"
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight", pad_inches=0.02,
                facecolor="white", edgecolor="none")
    print(f"Saved: {out}  ({FIG_W}\u00d7{FIG_H} in)")


if __name__ == "__main__":
    main()
