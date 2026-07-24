# GGap — Tennessee region example

A single example that reproduces the downscaled **Tennessee (TN)** forest-gap
simulation used in the paper, and regenerates the paper-comparison figure for site 978
(S. Appalachia). This folder holds the run driver, the snapshot extractors, and the
plotting pipeline; the CONUS input data is read from the repo's
`SC2026/conus_simulation/input_data/` (present in any clone — not duplicated here).

**Region:** lat 34.9–36.7, lon −90.5 to −81.5 → **54 sites**.
**Configuration (defaults):** 500 gaps × 1000 trees per site, **1000 years**, seed 42.

---

## 1. Install dependencies

**a) SAGESim** (the GPU agent-based framework). This also brings `numpy`, `networkx`,
and `awkward`:

```bash
git clone <SAGESim-URL>
cd SAGESim && pip install -e . && cd ..
```

**b) Hardware-matched runtime packages.** `cupy` and `mpi4py` are **not** bundled —
they depend on your specific CUDA/ROCm version and MPI implementation, so install them
to match your machine:

```bash
pip install cupy-cuda12x mpi4py     # CUDA 12.x
# pip install cupy-cuda11x mpi4py   # CUDA 11.x
# (ROCm: install the ROCm cupy build instead)
```

`mpi4py` needs an MPI runtime present first (e.g. OpenMPI: `conda install -c conda-forge openmpi`).

**c) METIS — only for a multi-GPU run.** Multi-rank runs partition sites across GPUs
with the `gpmetis` binary. A single-GPU run skips partitioning and does **not** need it.

```bash
conda install -c conda-forge metis   # or: apt-get install metis
```

**d) Clone GGap** (this repo) — no `pip install` of GGap itself is required; `run.py`
puts the `gap` package on the path automatically.

```bash
git clone <GGap-URL>
cd GGap/SC2026/tn_example
```

---

## 2. Run the simulation

The number of GPUs used equals the number of MPI ranks, i.e. the `-n` you pass to
`mpirun`. Each rank binds itself to its own GPU automatically (rank *i* → GPU *i*).

```bash
# 2 GPUs (recommended)
mpirun -n 2 python run.py

# 1 GPU
python run.py
```

All parameters already default to the TN configuration, so `run.py` needs no arguments.
Every parameter is still overridable, e.g. `python run.py --years 500`. Run
`python run.py --help` to see them.

Output (snapshots + per-rank metadata) is written to `./results/`.

> **GPU memory:** with 2 GPUs the 54 sites split ~27 per GPU (~15M tree-agents each),
> which fits a 16 GB card. All 54 sites on a **single** 16 GB GPU (~30M agents) may run
> out of memory — prefer 2 GPUs.

---

## 3. Reproduce the paper-comparison figure

This pipeline turns the raw snapshots from step 2 into the site-978 reproduction figure.
Site 978 (S. Appalachia) is one of the paper's 10 representative sites **and** lies inside
the TN box, so its two panels compare directly against the paper's published figure.

```bash
# one-time: plotting-only deps (NOT needed to run the simulation itself)
pip install matplotlib pandas pillow

# 1. Extract site 978's full time series across ALL snapshots (not just the last year)
python extract_timeseries.py            # default: --sites 978

# 2. Plot our two panels: size distribution @ yr 1000  |  species composition over time
python plot_tn_site978.py

# 3. Compose side-by-side against the paper's published panel #7
python compose_comparison.py
```

**Output:** `figures/tn_site978_vs_paper.png` (and `.pdf`) — our panels beside the paper's,
showing the same size-distribution shape and dominant species assemblage.

> Aside: `python extract.py` is a separate convenience that writes the *final-year* species
> tables for **all 54 sites** (`results/species/site_NNNN/species_data.csv`). It is **not**
> part of the figure pipeline above (which needs the full time series from `extract_timeseries.py`).

---

## Expected results & runtime

- `figures/tn_site978_vs_paper.png` — the paper-vs-ours side-by-side for site 978.
- Wall time ≈ **15–20 min** on 2× NVIDIA Tesla P100 (16 GB) for step 2: ~3–4 min setup/build,
  ~14 min simulation (the first reported interval includes a one-time CUDA kernel-compile
  warmup). The figure pipeline (step 3) adds ~2–3 min (time-series extraction + plotting).

## Notes

- **Determinism:** the seed is fixed and identical on all ranks, and the RNG is
  partition-invariant, so results do not depend on the number of GPUs.
- **CUDA-aware MPI:** if your MPI is *not* CUDA-aware, leave `MPICH_GPU_SUPPORT_ENABLED`
  unset (setting it makes SAGESim pass GPU pointers to a non-CUDA-aware MPI and segfault).
  On a CUDA-aware MPI (e.g. Cray MPICH) it may be set.
