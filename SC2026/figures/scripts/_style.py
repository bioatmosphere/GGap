"""
Shared style for SC2026 paper figures (IEEE conference template).

All figures should be drawn at EXACT IEEE print size so that
matplotlib pt == LaTeX pt and \\includegraphics{file} (no width
spec, as in the template) renders unchanged.

Import at the top of every figure script:
    from _style import (
        COL_SINGLE, COL_DOUBLE, figsize_single, figsize_double,
        F_TITLE, F_HEAD, F_LABEL, F_BODY, F_SMALL,
        LW_THICK, LW_MED, LW_THIN, LW_HAIR,
        C_SITE, C_GAP, C_TREE,
        C_MPI, C_THREAD, C_CU, C_HPC, C_BARRIER, C_IDLE,
        C_TEXT, C_DISP, C_EXPAND,
        apply_rcparams,
    )
    apply_rcparams()

NOTE on the Site/Gap/Tree palette below: these hex values mirror the
constants in `site_hierarchy.py` (the coworker-owned figure). If that
file's palette ever changes, re-sync these values manually so the two
figures stay visually consistent.
"""

# ── IEEE conference column widths (inches) ──────────────────────
COL_SINGLE = 3.487       # IEEEtran conference single-column text width
COL_DOUBLE = 7.16        # IEEEtran conference text width (full-page span)

# ── Font tiers (PRINT pt — matches template's 8 pt label rule) ──
F_TITLE = 9.0    # section banners
F_HEAD  = 8.0    # primary box labels (use as default)
F_LABEL = 8.0    # legends, axis labels, major annotations
F_BODY  = 7.5    # secondary annotations
F_SMALL = 7.0    # minimum acceptable; use sparingly

# ── Line widths (PRINT pt) ──────────────────────────────────────
LW_THICK = 1.0   # primary box outlines
LW_MED   = 0.7   # secondary outlines, arrows
LW_THIN   = 0.5   # IEEE legibility floor for solid lines
LW_HAIR  = 0.4   # decorative dotted lines (brackets etc.)

# ── Site / Gap / Tree palette (mirrors site_hierarchy.py) ───────
C_SITE = '#2B5C8A'   # steel blue
C_GAP  = '#E8A838'   # amber
C_TREE = '#3A9E78'   # teal

# ── Engineering / supporting colours ────────────────────────────
C_MPI     = '#2B6CB0'
C_THREAD  = '#4A5568'
C_CU      = '#6B46C1'
C_HPC     = '#1A365D'
C_BARRIER = '#718096'
C_IDLE    = '#CBD5E0'
C_TEXT    = '#1A202C'
C_DISP    = '#C44E52'
C_EXPAND  = '#888888'


# ── Helpers ─────────────────────────────────────────────────────
def figsize_single(height_in):
    """figsize tuple for a single-column figure of given height (inches)."""
    return (COL_SINGLE, height_in)


def figsize_double(height_in):
    """figsize tuple for a double-column (full-width) figure."""
    return (COL_DOUBLE, height_in)


def apply_rcparams():
    """Set matplotlib defaults so per-figure code stays clean."""
    import matplotlib as mpl
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
        'font.size': F_HEAD,
        'axes.titlesize': F_TITLE,
        'axes.labelsize': F_LABEL,
        'xtick.labelsize': F_BODY,
        'ytick.labelsize': F_BODY,
        'legend.fontsize': F_BODY,
        'figure.titlesize': F_TITLE,
        'savefig.dpi': 600,
        'pdf.fonttype': 42,    # embed TrueType, not Type 3 — IEEE-friendly
        'ps.fonttype':  42,
    })
