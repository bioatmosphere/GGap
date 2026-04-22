#!/usr/bin/env python3
"""Generate scaling_results.md — the source-of-truth writeup for the paper.

Reads the three scaling CSVs and writes a comprehensive Markdown document
covering executive summary, hardware stack, all results tables, per-tick
decomposition, setup-phase breakdown, narrative paragraphs, methodology
notes, and data provenance. Figures are referenced by the per-plot file
names produced by the standalone subfigure scripts in this directory.

Usage:
    python papers/SC2026/src/generate_results_md.py
"""
import argparse
from pathlib import Path

from _scaling_common import (
    EXPECTED_STRONG,
    read_csv, derive_metrics, cross_rank_pct, get_default_paths,
)


def fmt_int(x):
    return f"{int(x):,}"


def fmt_float(x, prec=2):
    if x is None:
        return "—"
    return f"{x:.{prec}f}"


# ============================================================================
# Table writers
# ============================================================================

def write_table_weak(name, metrics, expected_gpus, csv_path):
    lines = [
        f"### {name}",
        "",
        f"Source: `{csv_path}`",
        "",
        "| GPUs | Total Sites | Total Agents | Sim Time (s) | First Tick (s) | Steady (s) | Mean Tick (ms) | Throughput (B upd/s) | Efficiency (%) | MPI % |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for g in expected_gpus:
        if g in metrics:
            m = metrics[g]
            lines.append(
                f"| {g} | {fmt_int(m['total_sites'])} | {fmt_int(m['total_agents'])} | "
                f"{fmt_float(m['sim_time'], 1)} | {fmt_float(m['first_tick'], 1)} | "
                f"{fmt_float(m['steady_state'], 1)} | {fmt_float(m['mean_tick']*1000, 2)} | "
                f"{fmt_float(m['throughput']/1e9, 2)} | {fmt_float(m['efficiency'], 1)} | "
                f"{fmt_float(m['mpi_fraction'], 3)} |"
            )
        else:
            lines.append(f"| {g} | — | — | — | — | — | — | — | — | — |")
    lines.append("")
    return "\n".join(lines)


def write_table_strong(name, metrics, avg, expected_gpus, csv_path):
    lines = [
        f"### {name}",
        "",
        f"Source: `{csv_path}`",
        "",
        "| GPUs | Sites/GPU | Sim Time (s) | Mean Tick (ms) | Speedup | Efficiency (%) | Cross-rank % | MPI % |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for g in expected_gpus:
        if g in metrics:
            m = metrics[g]
            spg = int(avg[g]["sites_per_gpu"])
            xrank = cross_rank_pct(int(avg[g]["grid_height"]), spg)
            lines.append(
                f"| {g} | {spg} | {fmt_float(m['sim_time'], 1)} | "
                f"{fmt_float(m['mean_tick']*1000, 2)} | {fmt_float(m['speedup'], 2)}× | "
                f"{fmt_float(m['efficiency'], 1)} | {fmt_float(xrank, 1)} | "
                f"{fmt_float(m['mpi_fraction'], 3)} |"
            )
        else:
            lines.append(f"| {g} | — | — | — | — | — | — | — |")
    lines.append("")
    return "\n".join(lines)


def write_decomp_table(name, gpus, avg):
    lines = [
        f"### {name} — Per-tick Time Breakdown (μs)",
        "",
        "| GPUs | GPU Exec | MPI Pack | MPI Exchg | MPI Unpack | Data Prep | Write Back | Kernel Args | Total |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for g in gpus:
        rec = avg[g]
        gpu_exec = (rec["mean_gpu_compute"] + rec["mean_gpu_sync"]) * 1e6
        mpi_pack = rec["mean_mpi_gpu_pack"] * 1e6
        mpi_exch = rec["mean_mpi_exchange"] * 1e6
        mpi_unpk = rec["mean_mpi_gpu_unpack"] * 1e6
        data_prep = rec["mean_data_prep"] * 1e6
        write_back = rec["mean_write_back"] * 1e6
        kern_args = rec["mean_kernel_args_build"] * 1e6
        total = gpu_exec + mpi_pack + mpi_exch + mpi_unpk + data_prep + write_back + kern_args
        lines.append(
            f"| {g} | {gpu_exec:,.1f} | {mpi_pack:,.2f} | {mpi_exch:,.2f} | "
            f"{mpi_unpk:,.2f} | {data_prep:,.2f} | {write_back:,.2f} | "
            f"{kern_args:,.2f} | {total:,.1f} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_setup_table(name, gpus, avg):
    lines = [
        f"### {name} — End-to-End Phase Breakdown (s)",
        "",
        "| GPUs | Model Create | Load Globals | Site Init | Connectivity | GPU Setup | First Tick | Steady State | Total Sim |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for g in gpus:
        rec = avg[g]
        lines.append(
            f"| {g} | {rec['model_creation_time']:.3f} | {rec['load_globals_time']:.2f} | "
            f"{rec['site_init_time']:.1f} | {rec['connectivity_time']:.2f} | "
            f"{rec['gpu_setup_time']:.2f} | {rec['first_tick_time']:.1f} | "
            f"{rec['steady_state_time']:.1f} | {rec['simulation_time']:.1f} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_headline(metrics_a, metrics_s, gpus_a, gpus_s):
    peak_a = max(gpus_a)
    peak_s = max(gpus_s)

    ma = metrics_a[peak_a]
    ms = metrics_s[peak_s]
    # Per-GPU throughput: total agents / mean_tick / N. In weak scaling this is
    # essentially constant across GPU counts (sustained per-GPU rate). For
    # strong scaling it's not a meaningful headline number because per-GPU work
    # shrinks with N — we report "—" for strong.
    a_per_gpu_mups = ma["throughput"] / peak_a / 1e6
    s_per_gpu_str = "—"
    # Sustained per-tick (steady-state average across all measured GPU counts).
    # For weak scaling this is a meaningful single number — per-tick stays
    # constant by design. For strong scaling per-tick varies dramatically with
    # GPU count (95 → 19 ms), so the average isn't meaningful; report "—".
    a_per_tick_avg_ms = sum(metrics_a[g]["mean_tick"] for g in gpus_a) / len(gpus_a) * 1000
    s_per_tick_avg_str = "—"

    # The weak-scaling experiment is the headline (full UVAFME per-site
    # fidelity, 10 sites/GPU, complete 8 → 2,048 GPU sweep). There is no
    # longer a separate Weak B configuration in this paper.
    lines = [
        "### Headline Numbers",
        "",
        "| Metric | **Weak Scaling (10 sites/GPU)** | **Strong Scaling (2,048 sites)** |",
        "|---|---|---|",
        f"| Peak GPUs | {peak_a} | {peak_s} |",
        f"| Peak total sites | {fmt_int(ma['total_sites'])} | {fmt_int(ms['total_sites'])} |",
        f"| Peak total agents | {fmt_int(ma['total_agents'])} | {fmt_int(ms['total_agents'])} |",
        f"| Peak total throughput | {ma['throughput']/1e9:.2f} B upd/s | {ms['throughput']/1e9:.2f} B upd/s |",
        f"| Sustained per-GPU throughput | {a_per_gpu_mups:.0f} M upd/s/GPU | {s_per_gpu_str} |",
        f"| Sustained steady-state per-tick (avg) | ~{a_per_tick_avg_ms:.0f} ms | {s_per_tick_avg_str} |",
        f"| Mean tick at peak GPU count | {ma['mean_tick']*1000:.2f} ms | {ms['mean_tick']*1000:.2f} ms |",
        f"| Parallel efficiency at peak | {ma['efficiency']:.1f}% | {ms['efficiency']:.1f}% |",
    ]
    if ms["speedup"]:
        lines.append(f"| Speedup at peak | — | {ms['speedup']:.1f}× |")
    lines.append("")
    return "\n".join(lines)


# ============================================================================
# Main entry point
# ============================================================================

def main():
    csv_dir, _, md_dir = get_default_paths(__file__)
    parser = argparse.ArgumentParser(description="Generate scaling_results.md from CSVs")
    parser.add_argument("--csv-dir", type=str, default=str(csv_dir),
                        help="CSV input directory (default: <script>/../results)")
    parser.add_argument("--md-dir", type=str, default=str(md_dir),
                        help="Markdown output directory (default: <repo>/papers/SC2026/md)")
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir)
    md_dir = Path(args.md_dir)
    md_dir.mkdir(parents=True, exist_ok=True)

    weak_a_csv = csv_dir / "weak_scaling.csv"
    strong_csv = csv_dir / "strong_scaling_b.csv"

    print("=== Reading CSVs ===")
    print(f"  Weak: {weak_a_csv}")
    gpus_a, avg_a = read_csv(weak_a_csv)
    print(f"  Strong: {strong_csv}")
    gpus_s, avg_s = read_csv(strong_csv)
    missing_s = sorted(set(EXPECTED_STRONG) - set(gpus_s))
    if missing_s:
        print(f"  WARNING: strong_scaling missing GPUs {missing_s}")

    metrics_a = derive_metrics(gpus_a, avg_a, baseline_gpu=gpus_a[0], scaling_kind="weak")
    metrics_s = derive_metrics(gpus_s, avg_s, baseline_gpu=gpus_s[0], scaling_kind="strong")

    peak_a = max(gpus_a)
    peak_s = max(gpus_s)

    # Per-GPU and per-tick aggregates referenced in multiple sections (§1
    # headline, §1 key claims, §9.1, §9.4) — compute once up front so the
    # section blocks below can reference them directly.
    a_per_gpu_mups = metrics_a[peak_a]['throughput'] / peak_a / 1e6
    a_per_tick_avg_ms = sum(metrics_a[g]['mean_tick'] for g in gpus_a) / len(gpus_a) * 1000

    md = []
    md.append("# GGap Scaling Results — SC2026 Application Track\n")
    md.append("This document is the source of truth for all numbers in the scaling section of "
              "the SC2026 paper. Figures referenced live in `../figs/` (each plot is a standalone "
              "PDF for inclusion as a LaTeX subfigure). Regenerate by running:\n")
    md.append("```bash\n# Each plot is its own script:\npython papers/SC2026/src/weak_scaling_efficiency.py\npython papers/SC2026/src/phase_breakdown.py\n# ... etc.\n# And the markdown:\npython papers/SC2026/src/generate_results_md.py\n```\n")

    # 1. Executive summary
    md.append("## 1. Executive Summary\n")
    md.append(write_headline(metrics_a, metrics_s, gpus_a, gpus_s))
    md.append("**Key claims for the paper:**\n")
    md.append(f"- Weak Scaling demonstrates "
              f"**{metrics_a[peak_a]['efficiency']:.0f}% parallel efficiency at {peak_a} GPUs** "
              f"({fmt_int(metrics_a[peak_a]['total_agents'])} agents = "
              f"{fmt_int(metrics_a[peak_a]['total_sites'])} sites at 10 sites/GPU under full "
              f"UVAFME per-site fidelity), sustaining "
              f"~{a_per_gpu_mups:.0f} M agent-updates/sec/GPU under a stress-test "
              f"communication regime (37.5% cross-rank edges).")
    md.append(f"- Strong Scaling delivers a **{metrics_s[peak_s]['speedup']:.1f}× speedup** "
              f"from {gpus_s[0]} → {peak_s} GPUs ({metrics_s[peak_s]['efficiency']:.0f}% parallel "
              f"efficiency at the largest scale). The efficiency drop is correlated with the per-GPU "
              f"partition shrinking from 256 → 4 sites: at the smallest partitions each GPU kernel "
              f"underutilizes the MI250X GCD (kernel launch overhead becomes a larger fraction of "
              f"GPU time), and the cross-rank edge fraction climbs from 1.2% → 75%.")
    md.append(f"- Across all experiments, GPU execution is ≥99% of per-tick wall time and "
              f"MPI ghost-exchange consumes "
              f"<{max(metrics_a[peak_a]['mpi_fraction'], metrics_s[peak_s]['mpi_fraction']):.3f}% "
              f"of per-tick time, demonstrating that GPU-aware MPI on Slingshot-11 is effectively "
              f"overlapped with computation.\n")

    # 2. Hardware
    md.append("## 2. Hardware & Software Stack\n")
    md.append("- **Platform:** OLCF Frontier (Oak Ridge National Laboratory)")
    md.append("- **Per node:** AMD EPYC 7A53 (64 cores) + 4× AMD MI250X GPUs (8 GCDs/node)")
    md.append("- **Interconnect:** HPE Slingshot-11, 4× 25 GB/s injection bandwidth/node")
    md.append("- **GPU-aware MPI:** Cray MPICH with `MPICH_GPU_SUPPORT_ENABLED=1`")
    md.append("- **Software:** PrgEnv-gnu 8.6.0, ROCm 6.4.1, Python 3.13, CuPy (ROCm), mpi4py")
    md.append("- **Per-rank affinity:** 7 CPU cores/task, `--gpu-bind=closest`")
    md.append("- **SAGESim config:** `SAGESIM_NUM_SMS=110` (matches MI250X GCD CU count)\n")

    # 3. Experiment matrix
    md.append("## 3. Experiment Matrix\n")
    md.append("| Experiment | Sites/GPU | Gaps/site | Trees/gap | Grid H | Cross-rank % | GPU Range (complete) | Paper section |")
    md.append("|---|---:|---:|---:|---:|---:|---|---|")
    md.append("| Weak Scaling (comm-heavy)          |  10   | 500 | 1,000 |  5 | 37.5%       | 8–2,048 (9/9) | **Main (§4, §9.1)** |")
    md.append("| Strong Scaling (fixed 2,048 sites) | 256→4 | 200 |   500 |  4 | 1.2%→75.0%  | 8–512 (7/7)   | **Main (§5, §9.2)** |")
    md.append("")

    # 4. Weak scaling — main result
    md.append("## 4. Weak Scaling — Main Result\n")
    md.append("Full UVAFME per-site fidelity (500 gaps × 1,000 trees), 10 sites/GPU, "
              "37.5% cross-rank density. This is the headline weak-scaling configuration "
              "for the main paper, with all 9 rank counts (8 → 2,048 GPUs) complete.\n")
    md.append(write_table_weak("Weak Scaling — 10 sites/GPU", metrics_a, gpus_a, "weak_scaling.csv"))

    # 5. Strong scaling
    md.append("## 5. Strong Scaling Results\n")
    md.append(write_table_strong("Strong Scaling — 2,048 sites fixed", metrics_s, avg_s, gpus_s, "strong_scaling_b.csv"))

    # 6. Per-tick decomposition
    md.append("## 6. Per-Tick Time Decomposition\n")
    md.append("All values in microseconds (μs). `GPU Exec` = `mean_gpu_compute + mean_gpu_sync` "
              "(kernel launch overhead + actual GPU work). MPI columns are the GPU-aware MPI "
              "ghost-cell exchange. `Total` is the average per-tick wall time. Decomposition "
              "tables for the Weak Scaling and Strong Scaling experiments are co-located "
              "here for reference.\n")
    md.append(write_decomp_table("Weak Scaling", gpus_a, avg_a))
    md.append(write_decomp_table("Strong", gpus_s, avg_s))

    # 7. Setup-phase
    md.append("## 7. Setup-Phase Breakdown\n")
    md.append("Setup phases are one-time costs amortized over the simulation run. Setup is "
              "dominated by Python-side site initialization and the first-tick buffer build "
              "(ghost topology discovery + GPU buffer allocation). Two figures use the "
              "setup-phase data: (1) Fig.~`setup_amortization` plots setup-fraction-of-total "
              "vs. simulation length using the **largest weak-scaling 2,048-GPU configuration** "
              "as the representative steady-state cost — see §9.3 for the temporal-capability "
              "story. (2) Fig.~`phase_breakdown` shows 7 stacked bars across the strong-"
              "scaling sweep (sites/GPU = 4 → 256, one per rank count from 8 → 512 GPUs), "
              "demonstrating how each phase **scales with per-rank workload** as the partition "
              "grows. The complementary weak-scaling claim — that per-rank cost is constant "
              "across the 8 → 2,048 GPU sweep — is conveyed by the flat efficiency line in "
              "Fig.~`weak_scaling_efficiency` and by the §6 per-tick decomposition table; "
              "the four merged phases vary by <5% across that sweep. See §9.2 for the "
              "phase-breakdown narrative.\n")
    md.append(write_setup_table("Weak Scaling (representative for amortization)", gpus_a, avg_a))

    # 8. Figures — list each standalone subfigure, split into main vs supplementary
    md.append("## 8. Figures (standalone subfigures for LaTeX `subcaption`)\n")
    md.append("All figures are saved as paired PDF (paper) + PNG (preview) at 600 DPI in "
              "`../figs/` (sibling of this `md/` folder). Each panel is its own file so LaTeX can "
              "compose them via `\\begin{subfigure}` / `\\subfloat`.\n")
    md.append("### 8.1 Main paper figures\n")
    md.append("Four standalone single-column figures, each making one focused claim:\n")
    md.append("- `weak_scaling_efficiency` — Parallel efficiency vs. rank count line plot. "
              "Green measured efficiency line at ~96–101% across 8 → 2,048 GPUs, with a "
              "horizontal dashed reference at 100% (ideal weak scaling). Headline corner "
              "pill: `96.2% efficiency / ~41 ms/tick sustained / at 2,048 GPUs`. Mirrors "
              "`strong_scaling_speedup` in chart type (line plot + corner pill) for visual "
              "symmetry.")
    md.append("- `strong_scaling_speedup` — Log-log speedup curve with ideal-linear "
              "reference (left axis, green = measured speedup, gray dashed = ideal) and "
              "cross-rank edge fraction on right twin axis (orange dashed). The colored "
              "y-axis labels carry the line-color → metric mapping, so the figure has no "
              "legend or in-plot annotation; headline numbers live in the LaTeX caption. "
              "**Suggested caption:** *Strong scaling on 2{,}048 fixed sites: 12.4× speedup "
              "at 512 GPUs (19.3% parallel efficiency). Left axis (green) plots measured "
              "speedup against the gray-dashed ideal-linear reference; the vertical gap "
              "between the two curves IS the parallel-efficiency loss. Right axis (orange "
              "dashed) shows the cross-rank edge fraction climbing from 1.2% to 75% as the "
              "per-GPU partition shrinks from 256 to 4 sites.*")
    md.append("- `phase_breakdown` — **7 stacked bars** across the strong-scaling sweep "
              "(sites/GPU = 4 / 8 / 16 / 32 / 64 / 128 / 256, one per rank count from "
              "8 → 512 GPUs). Each bar is decomposed into Initialize / GPU Setup / First "
              "Tick / Steady State. Bars shrink dramatically left → right as per-GPU work "
              "decreases (setup-dominated 8-GPU case → steady-state-dominated 512-GPU case). "
              "**Suggested caption:** *Phase breakdown of total wall-clock time across the "
              "strong-scaling sweep, showing how each phase scales with per-GPU workload as "
              "the partition shrinks from 256 to 4 sites/GPU. Under the complementary "
              "weak-scaling experiment (10 sites/GPU, not shown), all four merged phases "
              "vary by <5% across the 8 → 2,048 GPU sweep — confirmed by the §6 per-tick "
              "decomposition table and the flat efficiency line in `weak_scaling_efficiency`. "
              "Per-tick MPI ghost-exchange is constant at ~5 μs/rank throughout.*")
    md.append("- `setup_amortization` — Setup fraction of total wall time vs. simulation "
              "length, computed from the largest weak-scaling configuration (2,048 GPUs); "
              "pure computational, no use-case markers — production use cases are "
              "referenced in §9.3 text.\n")
    md.append("### 8.2 Supplementary figures\n")
    md.append("None. All main-paper figures are listed in §8.1 above.\n")

    # 9. Narrative
    md.append("## 9. Narrative for the Paper\n")

    # 9.0 — capability framing (CONUS baseline + 4-figure preview)
    md.append("### 9.0 Why scaling matters for GGap\n")
    md.append("Today's GGap CONUS production run uses ~1,400 sites at 1-year temporal resolution "
              "for ~1,000 simulated years (~1,000 ticks). This is small in both dimensions: "
              "spatially because 1,400 sites barely capture 25 km grid coverage over the "
              "continental US, and temporally because annual time stepping cannot resolve "
              "sub-year ecosystem dynamics (drought stress, fire-weather coupling, phenology "
              "timing). The scaling analysis below characterizes whether GGap can support "
              "**higher spatial resolution** (more sites at the same fidelity) and **higher "
              "temporal resolution** (much shorter time steps over comparable simulated "
              "horizons), or both.\n")
    md.append("The main-paper figures answer four independent questions. "
              "Fig.~`weak_scaling_efficiency` proves the framework scales to ~15× more "
              "sites than the current CONUS baseline at near-perfect efficiency under full "
              "UVAFME per-site fidelity. Fig.~`strong_scaling_speedup` proves wall time can "
              "be shrunk for a fixed problem so per-scenario turnaround drops. "
              "Fig.~`phase_breakdown` shows wall-clock time scales with per-GPU workload via "
              "7 stacked strong-scaling bars; the orthogonal claim that per-rank cost is "
              "constant in N at fixed workload is conveyed by `weak_scaling_efficiency`'s "
              "flat efficiency line and quantified textually in §9.2 (var <5% across "
              "8 → 2,048 GPUs). Fig.~`setup_amortization` proves that "
              "one-time setup costs become negligible for long-horizon production runs at "
              "any temporal resolution finer than yearly. Together the four results argue "
              "that GGap's scaling unlocks production-quality high-resolution forest "
              "simulation that was previously infeasible.\n")

    md.append("### 9.1 Weak scaling — full-fidelity, communication-heavy\n")
    conus_baseline_sites = 1400
    sites_ratio = metrics_a[peak_a]['total_sites'] / conus_baseline_sites
    md.append(f"**Spatial capability vs. CONUS baseline.** Today's CONUS production run uses "
              f"~{conus_baseline_sites:,} sites; the framework sustains "
              f"**{metrics_a[peak_a]['efficiency']:.0f}% parallel efficiency at {peak_a} GPUs** "
              f"({fmt_int(metrics_a[peak_a]['total_sites'])} sites = "
              f"{fmt_int(metrics_a[peak_a]['total_agents'])} agents, "
              f"~{sites_ratio:.0f}× more sites than current CONUS), demonstrating that spatial "
              f"resolution is not capacity-limited at full per-site fidelity. The methodology "
              f"and per-tick numbers behind this claim are detailed below.\n")
    md.append(f"We evaluate weak scaling on OLCF Frontier with the **full-fidelity, "
              f"communication-heavy configuration**: 10 sites per GPU with 500 gaps × 1,000 trees "
              f"per site (matching production UVAFME runs). With grid height 5 and 1D column-slab "
              f"partitioning, each rank exchanges 30 of its 80 site-edges across rank boundaries — "
              f"a **37.5% cross-rank fraction** that stress-tests the communication subsystem. "
              f"From {gpus_a[0]} to {peak_a} GPUs (a {peak_a // gpus_a[0]}× scale-up to "
              f"{fmt_int(metrics_a[peak_a]['total_agents'])} agents), parallel efficiency stays at "
              f"**{metrics_a[peak_a]['efficiency']:.1f}%** (Fig.~`weak_scaling_efficiency`). "
              f"The model sustains a steady-state per-tick wall time of "
              f"**~{a_per_tick_avg_ms:.0f} ms/tick** across the entire 256× scale-up "
              f"(variation < 5%; full range "
              f"{min(metrics_a[g]['mean_tick'] for g in gpus_a)*1000:.1f}–"
              f"{max(metrics_a[g]['mean_tick'] for g in gpus_a)*1000:.1f} ms), and "
              f"each GPU sustains **~{a_per_gpu_mups:.0f} M agent-updates/sec**, "
              f"totaling **{metrics_a[peak_a]['throughput']/1e9:.0f} B agent-updates/sec at "
              f"{peak_a} GPUs**. Fig.~`weak_scaling_efficiency` plots parallel efficiency "
              f"vs. rank count as a single line: a horizontal dashed reference at 100% marks "
              f"ideal weak scaling, and the green measured-efficiency line stays at "
              f"~96–101% across the entire 8 → 2,048 GPU sweep. The line dips slightly to "
              f"96.2% at 1024 and 2048 GPUs — visible as the right end of the curve falling "
              f"~4 percentage points below the 100% reference — and we attribute this small "
              f"drop to mild first-tick growth (~53 s → ~56 s), not to steady-state per-tick "
              f"cost; see §9.3 for the first-tick anatomy. The headline numbers "
              f"(96.2% efficiency, ~{a_per_tick_avg_ms:.0f} ms/tick sustained, at "
              f"{peak_a} GPUs) are carried by the corner pill in the lower-left of the "
              f"figure. Steady-state per-tick MPI ghost-exchange remains constant at "
              f"~5 μs/rank across the entire range.\n")

    md.append("### 9.2 Strong scaling — time-to-solution\n")
    md.append(f"Strong scaling on a fixed problem of 2,048 sites "
              f"(~{fmt_int(metrics_s[gpus_s[0]]['total_agents'])} agents). "
              f"Fig.~`strong_scaling_speedup` shows the combined speedup-and-cross-rank "
              f"picture: going from {gpus_s[0]} → {peak_s} GPUs achieves a "
              f"**{metrics_s[peak_s]['speedup']:.1f}× speedup** "
              f"({metrics_s[peak_s]['efficiency']:.1f}% parallel efficiency at the end of "
              f"the curve), with the vertical gap between the measured (green) and "
              f"ideal-linear (gray dashed) curves directly visualizing the efficiency loss; "
              f"the orange right-axis line tracks the cross-rank edge fraction climbing "
              f"from 1.2% to 75% as the per-GPU partition shrinks. Mean tick time falls "
              f"from {metrics_s[gpus_s[0]]['mean_tick']*1000:.1f} ms at {gpus_s[0]} GPUs to "
              f"{metrics_s[peak_s]['mean_tick']*1000:.1f} ms at {peak_s} GPUs, with GPU "
              f"execution remaining >99% of total per-tick wall time across the entire range. "
              f"At the most extreme partition (4 sites/GPU, single-column slab), each GPU "
              f"still updates ~400K agents per tick in "
              f"{metrics_s[peak_s]['mean_tick']*1000:.2f} ms, showing the runtime retains "
              f"usable throughput even at minimum granularity.\n")
    md.append("**Per-tick non-GPU overhead is flat across the entire sweep.** Direct "
              "instrumentation shows that the per-tick cost outside the fused GPU kernel — "
              "MPI ghost-exchange (the SAGESim `data_prep` wrapper around `exchange_ghost_data()`, "
              "covering GPU pack, GPU-aware Slingshot Isend/Irecv/Waitall, and GPU unpack), CuPy "
              "kernel-arg dispatch, and double-buffer write-back — is constant at "
              "**~5 μs + ~14 μs + ~16 μs ≈ 35 μs** across all seven configurations. This is "
              "**<0.2% of even the smallest tick (19 ms at 512 GPUs)**, and confirms the runtime "
              "imposes no rank-count tax on overhead. The strong-scaling efficiency drop is "
              "therefore **GPU-kernel-granularity bound**, not framework-bound: as sites/GPU "
              "shrinks from 256 to 4, per-agent compute cost rises ~13× because CuPy kernel "
              "launch overhead and intra-GCD parallelism limits become a larger fraction of "
              "the per-GPU work.\n")
    md.append("**Why MPI cost stays constant despite the rising cross-rank fraction.** "
              "Fig.~`strong_scaling_speedup`'s right axis shows the cross-rank edge fraction "
              "climbing from 1.2% to 75% as the per-GPU partition shrinks. This rise is "
              "purely a *relative* effect — boundary divided by interior. The **absolute** "
              "cross-rank edge count per rank is fixed at **24** (= `grid_height × 3 × 2` = "
              "`4 × 3 × 2` for the 1D column-slab partition with constant `grid_height = 4`), "
              "and each rank exchanges only with 2 peers (left/right neighbor with periodic "
              "wraparound) regardless of total rank count. Every rank therefore packs and "
              "exchanges a constant ~960 B per tick across the entire 8 → 512 GPU sweep, costing "
              "the constant ~5 μs MPI time reported above. The rising cross-rank fraction is a "
              "relative indicator of partition shrinkage, not a driver of communication cost — "
              "and the constancy of measured MPI time confirms this directly.\n")
    md.append("**Scope and limitations.** This strong-scaling experiment characterizes "
              "time-to-solution for the GGap model's natural 8-neighbor Moore connectivity under "
              "1D column-slab partitioning with `grid_height = 4`. As GPU count grows, per-rank "
              "compute work shrinks while per-rank cross-rank edges remain fixed, so the "
              "experiment isolates GPU kernel-granularity effects rather than network effects. "
              "Other partition topologies (2D decomposition with O(√N) peers per rank, k-NN "
              "connectivity, larger ghost halos) would induce different cross-rank scaling and "
              "are left to future work.\n")

    # Setup-phase breakdown paragraph (the strong-scaling 4-segment shrinkage story)
    s_init_8   = (avg_s[gpus_s[0]]['model_creation_time']
                  + avg_s[gpus_s[0]]['load_globals_time']
                  + avg_s[gpus_s[0]]['site_init_time']
                  + avg_s[gpus_s[0]]['connectivity_time'])
    s_init_max = (avg_s[peak_s]['model_creation_time']
                  + avg_s[peak_s]['load_globals_time']
                  + avg_s[peak_s]['site_init_time']
                  + avg_s[peak_s]['connectivity_time'])
    s_gpu_setup_8   = avg_s[gpus_s[0]]['gpu_setup_time']
    s_gpu_setup_max = avg_s[peak_s]['gpu_setup_time']
    s_first_8   = avg_s[gpus_s[0]]['first_tick_time']
    s_first_max = avg_s[peak_s]['first_tick_time']
    s_steady_8   = avg_s[gpus_s[0]]['steady_state_time']
    s_steady_max = avg_s[peak_s]['steady_state_time']
    s_total_8   = s_init_8 + s_gpu_setup_8 + s_first_8 + s_steady_8
    s_total_max = s_init_max + s_gpu_setup_max + s_first_max + s_steady_max
    s_setup_pct_8 = (s_init_8 + s_gpu_setup_8 + s_first_8) / s_total_8 * 100
    s_steady_pct_max = s_steady_max / s_total_max * 100
    spg_max = peak_s // gpus_s[0]  # 64×
    md.append(f"**Wall time scales with per-GPU workload.** "
              f"Fig.~`phase_breakdown` decomposes the strong-scaling sweep into 7 stacked "
              f"bars (one per rank count from 8 → 512 GPUs at sites/GPU = 256 → 4), each "
              f"split into Initialize / GPU Setup / First Tick / Steady State. Initialize "
              f"(host-side Python agent construction, dominated by site init) and GPU Setup "
              f"(write-buffer allocation) **scale near-linearly with the per-rank agent "
              f"count** — as sites/GPU shrinks {spg_max}× (256 → 4), Initialize falls from "
              f"~{s_init_8:.0f} s to ~{s_init_max:.0f} s and GPU Setup from "
              f"~{s_gpu_setup_8:.0f} s to ~{s_gpu_setup_max:.0f} s. First Tick scales "
              f"**sub-linearly** (~{s_first_8/s_first_max:.0f}× shrinkage from "
              f"~{s_first_8:.0f} s to ~{s_first_max:.0f} s) because it bundles per-rank "
              f"buffer construction with smaller per-rank components for kernel JIT and the "
              f"MPI communication-map handshake. Steady State scales **least** "
              f"(~{s_steady_8/s_steady_max:.0f}× shrinkage from ~{s_steady_8:.0f} s to "
              f"~{s_steady_max:.0f} s), reflecting the GPU kernel-granularity bound discussed "
              f"above. Bars shrink from ~{s_total_8:.0f} s at {gpus_s[0]} GPUs "
              f"(~{s_setup_pct_8:.0f}% setup-dominated) to ~{s_total_max:.0f} s at {peak_s} "
              f"GPUs (~{s_steady_pct_max:.0f}% steady-state-dominated), and the crossover "
              f"where steady state catches up with setup is directly visible.\n")
    md.append(f"**Wall time does NOT scale with rank count when communication is constant.** "
              f"The complementary weak-scaling claim is conveyed by `weak_scaling_efficiency`'s "
              f"flat efficiency line at ~96–101% across the 8 → 2,048 GPU sweep, and is "
              f"backed by the §6 per-tick decomposition table: **all four merged phases "
              f"(Initialize / GPU Setup / First Tick / Steady State) vary by <5% across the "
              f"8 → 2,048 GPU sweep at fixed 10 sites/GPU**. Per-tick MPI ghost-exchange is "
              f"verified to stay at ~5 μs/rank across the entire range (see §9.3), so the "
              f"constant-communication condition holds throughout. Combined with "
              f"Fig.~`phase_breakdown`, this establishes **wall time = f(per-rank workload) "
              f"alone, with no rank-count tax**.\n")

    md.append("### 9.3 Setup amortization — temporal capability for long-horizon production\n")
    # Use the largest measured weak-scaling configuration as the representative
    # for setup amortization (the weak-scaling experiment is the main-paper
    # one with the complete 8 → 2,048 GPU sweep).
    g_rep = gpus_a[-1]
    avg_rep = avg_a
    setup_total = sum(avg_rep[g_rep][k] for k in [
        "model_creation_time", "load_globals_time", "partitioning_time",
        "site_init_time", "connectivity_time", "gpu_setup_time", "first_tick_time",
    ])
    site_init_s = avg_rep[g_rep]["site_init_time"]
    first_tick_s = avg_rep[g_rep]["first_tick_time"]
    mean_tick_s = avg_rep[g_rep]["mean_tick_time"]
    target = 5.0
    crossover_ticks = setup_total * (100 - target) / target / mean_tick_s

    def setup_frac_at(n_ticks):
        return setup_total / (setup_total + n_ticks * mean_tick_s) * 100

    pct_1k     = setup_frac_at(1_000)
    pct_36k5   = setup_frac_at(36_500)
    pct_365k   = setup_frac_at(365_000)
    pct_876k   = setup_frac_at(876_000)

    md.append(f"Today's CONUS run uses 1,000 yearly ticks. At higher temporal resolutions the "
              f"tick count grows substantially: a centennial run at daily resolution is 36,500 "
              f"ticks; a millennial run at daily resolution is 365,000 ticks; an "
              f"hourly-resolution centennial run is 876,000 ticks. Fig.~`setup_amortization` "
              f"shows that the one-time setup cost (~{setup_total:.0f} s for the representative "
              f"weak-scaling {g_rep}-GPU configuration, dominated by site initialization "
              f"({site_init_s:.0f} s of host-side Python agent construction) and the first-tick "
              f"buffer build ({first_tick_s:.0f} s of ghost-cell topology discovery + GPU buffer "
              f"allocation)) drops below 5% of total wall time at "
              f"**~{int(crossover_ticks):,} ticks**. Reading the curve at typical use cases:\n")
    md.append(f"- **1,000 ticks (current CONUS, yearly)** — setup is ~{pct_1k:.0f}% of total "
              f"wall time; our benchmark is heavily setup-dominated.")
    md.append(f"- **36,500 ticks (centennial daily)** — setup falls to ~{pct_36k5:.0f}%.")
    md.append(f"- **365,000 ticks (millennial daily)** — setup falls to ~{pct_365k:.1f}%.")
    md.append(f"- **876,000 ticks (centennial hourly)** — setup is "
              f"~{pct_876k:.1f}% (essentially zero).\n")
    md.append("**Any production simulation at daily resolution or finer for centennial-or-longer "
              "horizons makes setup negligible.** This is the key temporal-capability claim of "
              "the paper: GGap is *not* setup-bound for the science use cases that motivate it; "
              "the setup-bound regime only contains the short benchmarking runs reported in this "
              "very section. Reducing site-init and first-tick costs remains a clear engineering "
              "target for future work, but is not on the critical path for production science "
              "use cases.\n")

    md.append("**Anatomy of the first tick — why setup costs what it does.** The first-tick "
              "cost consists of four one-time SAGESim operations that are cached for "
              "steady-state ticks and never repeated (see `sagesim/model.py:1340-1398`):\n")
    md.append("1. **GPU buffer construction** (`sagesim/model.py:1014-1254`) — host→device "
              "transfer of every agent property tensor (breed IDs, ragged neighbor CSR, scalars, "
              "vectors), allocation of double-buffering write buffers, creation of "
              "agent-id ↔ local-index hash maps, and one MPI allreduce to synchronize property "
              "widths across ranks. Cost is **O(N_local × num_properties)** — for the weak-scaling experiment that is "
              "~5 M agents × ~20 properties per GPU.")
    md.append("2. **Ghost topology discovery** (`sagesim/gpu_kernels.py:31`, called from "
              "`sagesim/model.py:1349`) — each rank scans its CSR neighbor lists to identify the "
              "unique remote agent IDs it needs as ghosts. CPU-side vectorized scan, no MPI yet. "
              "**O(local_edges)**.")
    md.append("3. **Communication map build** (`sagesim/gpu_kernels.py:517-846`, called from "
              "`sagesim/model.py:1377`) — two-phase MPI handshake: an Alltoall to exchange request "
              "counts (`gpu_kernels.py:626`), then Isend/Irecv pairs to exchange the actual "
              "remote-agent-id lists (`:629-647`). The rank then builds per-peer pack/unpack index "
              "arrays and pre-allocates persistent send/recv GPU buffers. "
              "**O(P × peers_per_rank)** for the handshake and **O(boundary_size)** for index "
              "construction. SAGESim's 1D column-slab partition keeps `peers_per_rank` constant at "
              "2 regardless of total rank count, so this scales gracefully.")
    md.append("4. **First ghost exchange + first kernel launch** — pack/MPI/unpack runs once to "
              "fill ghost slots, and CuPy NVRTC compiles the fused step function from PTX to "
              "binary on the first `@jit.rawkernel` launch (`sagesim/model.py:1466`). Subsequent "
              "ticks reuse the cached binary.\n")
    md.append("In steady state (`sagesim/model.py:1386-1398`) only step 4 repeats — buffers, hash "
              "maps, comm maps, and the compiled kernel are all cached. That is why steady ticks "
              "are ~41 ms vs ~53 s for tick 1 under weak scaling. The mild growth of first_tick from "
              "~53 s → ~56 s at 1024+ GPUs (visible as the orange-segment growth in "
              "`weak_scaling_efficiency`) is the source of the small efficiency drop; we "
              "attribute it to the O(P) Alltoall handshake in step 3 dominating the otherwise "
              "constant per-rank work as P grows.\n")

    md.append("**Steady-state MPI is constant in weak scaling** (verified, not just predicted). "
              "Per-rank cross-rank edges (30) and MPI peer count (2, by 1D column-slab "
              "partitioning with periodic wraparound) are independent of total rank count, and "
              "SAGESim's per-tick exchange uses non-blocking Isend/Irecv only (no per-tick "
              "collective operations). Measured ghost-exchange wall time — the full SAGESim "
              "`data_prep` wrapper around `exchange_ghost_data()`, covering GPU pack, GPU-aware "
              "Slingshot Isend/Irecv/Waitall, and GPU unpack — stays at **~5 μs/rank ± 10%** "
              "across the full 8 → 2,048 GPU range, confirming the GPU-aware Slingshot-11 path "
              "scales as the methodology predicts. The small efficiency drop at 1024+ GPUs is "
              "therefore attributable to first-tick growth, not steady-state communication.\n")

    # 9.4 — synthesis tying the three capabilities together
    md.append("### 9.4 Synthesis: what GGap unlocks\n")
    md.append(f"Combining the three capability claims: the **spatial capability** "
              f"(Fig.~`weak_scaling_efficiency`, ~{sites_ratio:.0f}× more sites than current "
              f"CONUS at {metrics_a[peak_a]['efficiency']:.0f}% efficiency under full UVAFME "
              f"per-site fidelity), the **temporal capability** "
              f"(Fig.~`setup_amortization`, setup amortized for daily-or-finer resolution at "
              f"centennial horizons), and the **time-to-solution capability** "
              f"(Fig.~`strong_scaling_speedup`, {metrics_s[peak_s]['speedup']:.1f}× wall-time "
              f"reduction for fixed problems) collectively argue that GGap's scaling unlocks a "
              f"class of high-resolution, long-horizon, large-domain forest simulations that "
              f"were previously infeasible with the current 1,400-site / 1-year-per-tick CONUS "
              f"baseline. Fig.~`phase_breakdown` provides the per-phase characterization "
              f"showing wall time = f(per-rank work) under strong scaling, complemented by "
              f"the §9.2 textual claim that all four phases vary by <5% across the weak-"
              f"scaling 8 → 2,048 GPU sweep. With steady-state per-tick pinned at "
              f"~{a_per_tick_avg_ms:.0f} ms/tick in the weak-scaling regime, scaling to "
              f"~{fmt_int(metrics_a[peak_a]['total_sites'])} sites at sub-yearly resolution "
              f"for centennial horizons is a straightforward extension of the measurements "
              f"reported here.\n")

    # 10. Methodology
    md.append("## 10. Methodology Notes\n")
    md.append("**Parallel efficiency definitions** (both weak and strong are computed from "
              "`simulation_time = first_tick_time + steady_state_time`, the full wall-clock cost "
              "of the `simulate()` call including the one-time first-tick warmup amortized over "
              "the run; this is more honest than reporting steady-state-only because `first_tick` "
              "grows mildly with rank count and is a real per-simulation cost):\n")
    md.append("- *Weak scaling*: `eff(N) = T_baseline_sim / T_N_sim × 100`. Ideal = 100%.")
    md.append("- *Strong scaling*: `speedup(N) = T_baseline_sim / T_N_sim`; "
              "`eff(N) = speedup(N) / (N / N_baseline) × 100`. Ideal = 100%.\n")
    md.append("**Throughput**: `total_agents / mean_tick_time` in agent-updates/second. "
              "Throughput uses the steady-state per-tick time (not the amortized one) because it "
              "measures *sustained* per-tick capability — a separate concept from full-run "
              "wall-clock efficiency.\n")
    md.append("**Total agents**: `total_sites × (1 + num_gaps × (1 + maxtrees))` "
              "(1 site agent + N gap agents + N×M tree agents per site).\n")
    md.append("**Cross-rank fraction**: `(grid_height × 6) / (sites_per_gpu × 8)` — derived "
              "from the 8-neighbor Moore connectivity and 1D column-slab partition.\n")
    md.append("**Duplicates**: where the CSV contains repeated runs for the same GPU count, "
              "all numeric columns are averaged before plotting (same convention as the existing "
              "`plot_weak_scaling_breakdown.py` script).\n")
    md.append("**Mean tick window**: SAGESim's `verbose_timing` separates tick 1 (buffer build, "
              "ghost topology discovery, communication map build) from ticks 2..N. `mean_tick_time` "
              "is the average of ticks 2..N (steady state) and is the figure used in all efficiency "
              "and throughput calculations.\n")
    md.append("**`gpu_execution`**: equals `mean_gpu_compute + mean_gpu_sync`. The GPU kernel "
              "launches asynchronously, so `gpu_compute` only captures CPU-side launch overhead "
              "(~0.5 ms) — actual kernel execution completes during `gpu_sync` when the CPU calls "
              "`stream.synchronize()`.\n")

    # 11. Provenance
    md.append("## 11. Data Provenance\n")
    md.append("| Source CSV | Used by | GPU counts present | Missing |")
    md.append("|---|---|---|---|")
    md.append(f"| `weak_scaling.csv` | weak_scaling_efficiency, setup_amortization, §4, §6, "
              f"§9.1, §9.2 textual claim, §9.3 | {gpus_a} | none |")
    md.append(f"| `strong_scaling_b.csv` | strong_scaling_speedup, phase_breakdown, §5, §6, "
              f"§9.2 | {gpus_s} | {missing_s if missing_s else 'none'} |")
    md.append("")
    md.append("**To refresh:** drop new rows into the relevant CSV and re-run the affected plot "
              "scripts in `papers/SC2026/src/`, then re-run "
              "`python papers/SC2026/src/generate_results_md.py` to regenerate this document.\n")

    md_path = md_dir / "scaling_results.md"
    md_path.write_text("\n".join(md))
    print(f"  Wrote {md_path}")

    # Stdout summary
    print("\n=== Summary ===")
    print(f"Weak: peak {peak_a} GPUs, {metrics_a[peak_a]['efficiency']:.1f}% eff, "
          f"{metrics_a[peak_a]['throughput']/1e9:.2f} B upd/s, "
          f"{fmt_int(metrics_a[peak_a]['total_agents'])} agents")
    print(f"Strong: peak {peak_s} GPUs, {metrics_s[peak_s]['speedup']:.1f}× speedup, "
          f"{metrics_s[peak_s]['efficiency']:.1f}% eff")
    print("\nDone.")


if __name__ == "__main__":
    main()
