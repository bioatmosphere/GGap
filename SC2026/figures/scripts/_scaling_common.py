"""
Shared helpers for the SC2026 scaling-figure scripts.

Each individual plot script (e.g. weak_scaling_efficiency.py) imports from
this module so the per-script files can stay focused on plot logic.

Convention: each plot script produces a single standalone figure (no panels)
sized for inclusion as a LaTeX subfigure. LaTeX handles panel layout via the
`subcaption` package; matplotlib should not draw subplot grids or in-figure
panel labels like (a) / (b) / (c).
"""
import csv
from pathlib import Path

import numpy as np

# ============================================================================
# Figure sizing — standalone subfigures meant for LaTeX subcaption layout
# ============================================================================

# Default standalone subfigure size. LaTeX subfigure environments will scale
# these to fit a column or half-column slot; this size keeps text readable
# both as the original PDF and after scaling. Height tightened in two
# steps: 2.6 → 2.0 → 1.8 in to reduce visual weight on the page when 2
# panels are paired in a stacked single-column \begin{figure} layout. The
# 1.8 in floor is the practical minimum at FONT_LABEL = 6 pt: the rotated
# y-axis label "Speedup vs. 8-GPU baseline" (~1.26 in tall at 6 pt) just
# fits in the plot interior (~1.4 in after x-axis margins). Going below
# 1.8 in would clip the y-axis label.
FIG_W = 3.3   # inches
FIG_H = 1.8   # inches (was 2.0, originally 2.6)

# Wider variant for stacked-bar breakdown plots that need legend space.
FIG_W_WIDE = 4.0
FIG_H_WIDE = 2.8


# ============================================================================
# Color palette (Material Design, matches architecture_figure_v4.py style)
# ============================================================================

COLOR_WEAK_A = "#E53935"   # red    — communication-heavy weak scaling
COLOR_WEAK_B = "#1E88E5"   # blue   — compute-heavy weak scaling
COLOR_STRONG = "#43A047"   # green  — strong scaling
COLOR_REF    = "#9E9E9E"   # gray   — light reference (e.g., gridlines)
COLOR_IDEAL  = "#374151"   # dark slate-gray — ideal/baseline reference lines
COLOR_XRANK  = "#FB8C00"   # orange — cross-rank fraction overlay

# ============================================================================
# Shared font sizes (IEEE single-column figures, ~3.5" wide)
# ============================================================================
# Centralized so weak and strong scaling plots stay visually identical.
# All set to 6 pt — the IEEE conference figure floor — so the rotated
# y-axis labels (e.g. "Speedup vs. 8-GPU baseline") fit cleanly inside
# the new compact FIG_H = 2.0 in plot height without overflowing the top
# edge, and so labels / ticks / legends / annotations all share one
# consistent visual weight across the figure pair.
FONT_LABEL = 6   # x/y axis labels
FONT_TICK  = 6   # x/y tick labels
FONT_LEG   = 6   # legend text
FONT_PILL  = 6   # in-plot annotation pill (e.g., "sustained ≈ 41 ms/tick")
FONT_ANNOT = 6   # smaller annotations (e.g., per-bar efficiency labels)

# Ideal-reference line styling (dashed slate-gray, used by both weak and
# strong scaling efficiency/speedup plots)
IDEAL_LW = 1.5

# Stacked-decomposition palette (matches plot_weak_scaling_breakdown.py)
DECOMP_ORDER = [
    "GPU Execution", "MPI Pack", "MPI Exchange", "MPI Unpack",
    "Data Prep", "Write Back", "Kernel Args",
]
DECOMP_COLORS = {
    "GPU Execution": "#2196F3",
    "MPI Pack":      "#FF5722",
    "MPI Exchange":  "#F44336",
    "MPI Unpack":    "#E91E63",
    "Data Prep":     "#FF9800",
    "Write Back":    "#4CAF50",
    "Kernel Args":   "#607D8B",
}

# Setup-phase palette.
#
# "First Tick" and "Steady State" deliberately reuse `C_GAP` (amber) and
# `C_TREE` (teal) from `_style.py` so the weak/strong scaling plots use
# the SAME yellow and green that the architecture figure (ggap_architecture.py)
# and hierarchy figure (site_hierarchy.py / site_hierarchy_v2.py) use for the
# Gap and Tree agent boxes. This keeps the paper visually coherent — the same
# color-coding throughout all figures.
from _style import C_GAP, C_TREE

PHASE_ORDER = [
    "Model Creation", "Load Globals", "Site Init",
    "Connectivity", "GPU Setup", "First Tick", "Steady State",
]
PHASE_COLORS = {
    "Model Creation": "#90A4AE",
    "Load Globals":   "#9575CD",
    "Site Init":      "#64B5F6",
    "Connectivity":   "#4FC3F7",
    "GPU Setup":      "#4DD0E1",
    "First Tick":     C_GAP,    # amber #E8A838 — matches Gap agents
    "Steady State":   C_TREE,   # teal  #3A9E78 — matches Tree agents
}

# Expected GPU counts per the experiment design (used to mark missing rows).
EXPECTED_STRONG = [8, 16, 32, 64, 128, 256, 512]


# ============================================================================
# Path auto-detection
# ============================================================================

def get_default_paths(script_file):
    """Return (csv_dir, figs_dir, md_dir) defaults inferred from script_file.

    Layout: SC2026/figures/scripts/ (this file lives here)
            SC2026/figures/figs/
            SC2026/figures/md/
            SC2026/scaling_analysis/results/  (CSV data)
    """
    script_dir = Path(script_file).resolve().parent   # SC2026/figures/scripts/
    figures_dir = script_dir.parent                    # SC2026/figures/
    sc2026_dir = figures_dir.parent                    # SC2026/

    csv_dir = sc2026_dir / "scaling_analysis" / "results"
    figs_dir = figures_dir / "figs"
    md_dir = figures_dir / "md"
    return csv_dir, figs_dir, md_dir


# ============================================================================
# CSV reading and metric derivation
# ============================================================================

def read_csv(csv_path):
    """Read CSV and average duplicate runs per GPU count.

    Same averaging convention as scaling_analysis/scripts/plot_weak_scaling_breakdown.py:11-35.
    Returns (sorted_gpu_list, {gpu: averaged_row_dict}).
    """
    data = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            g = int(row["num_gpus"])
            data.setdefault(g, [])
            parsed = {}
            for k, v in row.items():
                try:
                    parsed[k] = float(v)
                except (ValueError, TypeError):
                    parsed[k] = 0.0
            data[g].append(parsed)
    gpus_sorted = sorted(data.keys())
    avg = {}
    for g in gpus_sorted:
        runs = data[g]
        avg[g] = {k: sum(r.get(k, 0.0) for r in runs) / len(runs) for k in runs[0]}
    return gpus_sorted, avg


def agents_per_site(num_gaps, maxtrees):
    """1 site agent + num_gaps gap agents + num_gaps*maxtrees tree agents."""
    return 1 + num_gaps + num_gaps * maxtrees


def cross_rank_pct(grid_height, sites_per_gpu):
    """(grid_height * 3 * 2) / (sites_per_gpu * 8) * 100."""
    return (grid_height * 6.0) / (sites_per_gpu * 8.0) * 100.0


def derive_metrics(gpus, avg, baseline_gpu, scaling_kind):
    """Compute speedup, efficiency, throughput per GPU count.

    Both weak and strong scaling efficiency are computed from `simulation_time`
    (= first_tick + steady_state). This is the wall-clock time for the entire
    simulate() call and includes the one-time first-tick warmup (GPU buffer
    construction, ghost topology discovery, communication map build) amortized
    across the run. This is more honest than using only the steady-state
    `mean_tick_time` because first_tick grows mildly with rank count and is a
    real per-simulation cost.

    Throughput stays based on `mean_tick_time` (steady-state), since throughput
    is the sustained agent-update rate — a different concept from amortized
    per-tick wall time.

    scaling_kind: 'weak' or 'strong'
    """
    metrics = {}
    t_base_sim = avg[baseline_gpu]["simulation_time"]
    for g in gpus:
        rec = avg[g]
        n_gaps = int(rec["num_gaps"])
        maxtrees = int(rec["maxtrees"])
        total_sites = int(rec["total_sites"])
        aps = agents_per_site(n_gaps, maxtrees)
        total_agents = total_sites * aps
        mean_tick = rec["mean_tick_time"]
        sim_time = rec["simulation_time"]
        throughput = total_agents / mean_tick if mean_tick > 0 else 0.0

        if scaling_kind == "weak":
            efficiency = (t_base_sim / sim_time) * 100 if sim_time > 0 else 0.0
            speedup = None
        else:  # strong
            speedup = t_base_sim / sim_time if sim_time > 0 else 0.0
            efficiency = speedup / (g / baseline_gpu) * 100

        metrics[g] = {
            "total_sites": total_sites,
            "total_agents": total_agents,
            "sites_per_gpu": int(rec["sites_per_gpu"]),
            "agents_per_gpu": aps * int(rec["sites_per_gpu"]),
            "sim_time": sim_time,
            "first_tick": rec["first_tick_time"],
            "steady_state": rec["steady_state_time"],
            "mean_tick": mean_tick,
            "throughput": throughput,
            "efficiency": efficiency,
            "speedup": speedup,
            "mpi_fraction": rec.get("mpi_fraction", 0.0) * 100,
            "gpu_exec_fraction": rec.get("gpu_execution_fraction", 0.0) * 100,
        }
    return metrics


def stacked_decomposition_data(gpus, avg):
    """Per-tick decomposition in milliseconds, returned as dict of arrays."""
    out = {}
    out["GPU Execution"] = np.array([(avg[g]["mean_gpu_compute"] + avg[g]["mean_gpu_sync"]) * 1000 for g in gpus])
    out["MPI Pack"]      = np.array([avg[g]["mean_mpi_gpu_pack"] * 1000 for g in gpus])
    out["MPI Exchange"]  = np.array([avg[g]["mean_mpi_exchange"] * 1000 for g in gpus])
    out["MPI Unpack"]    = np.array([avg[g]["mean_mpi_gpu_unpack"] * 1000 for g in gpus])
    out["Data Prep"]     = np.array([avg[g]["mean_data_prep"] * 1000 for g in gpus])
    out["Write Back"]    = np.array([avg[g]["mean_write_back"] * 1000 for g in gpus])
    out["Kernel Args"]   = np.array([avg[g]["mean_kernel_args_build"] * 1000 for g in gpus])
    return out


# ============================================================================
# Plot helpers
# ============================================================================

def setup_log2_xaxis(ax, expected_gpus, label="Number of GPUs", rotation=45):
    """Apply log2 x-axis with the given GPU counts as ticks. Uses the
    shared FONT_LABEL/FONT_TICK constants so all plots stay consistent.

    Default rotation=45 matches the weak-scaling efficiency plots' x-tick
    style for cross-plot visual consistency."""
    ax.set_xscale("log", base=2)
    ax.set_xticks(expected_gpus)
    ax.set_xticklabels([str(g) for g in expected_gpus],
                       fontsize=FONT_TICK, rotation=rotation)
    ax.set_xlabel(label, fontsize=FONT_LABEL)


def save_figure(fig, out_pdf, out_png, dpi):
    """Save the figure as PNG (white bg, for dev preview) and PDF (transparent
    bg, for LaTeX inclusion).

    Transparent PDF lets LaTeX subfigure layouts blend cleanly with the page
    background regardless of any document-level coloring.
    """
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight", transparent=True)


def add_cli_args(parser, plot_name):
    """Add the standard --csv-dir / --figs-dir / --dpi flags."""
    parser.add_argument("--csv-dir", type=str, default=None,
                        help="CSV input dir (default: <script>/../results)")
    parser.add_argument("--figs-dir", type=str, default=None,
                        help="Output dir for figures (default: <repo>/SC2026/figures/figs)")
    parser.add_argument("--dpi", type=int, default=600,
                        help="PNG DPI (default 600 for paper)")
    return parser
