"""
GGap - GPU-accelerated Forest Gap Dynamics Model
Integrates UVAFME forest processes with SAGESim agent framework.

Species data is loaded from CSV file when GAPModel is initialized.
Access via model.species_list after creating the model.
"""

__version__ = "0.1.0"

# Lazy imports for GPU-dependent modules
def _get_gap_model():
    """Lazy import of GAPModel (requires cupy)."""
    from gap.gap_model import GAPModel
    return GAPModel

def _get_tree_breed():
    """Lazy import of TreeBreed (requires cupy)."""
    from gap.tree_breed import TreeBreed
    return TreeBreed

# Make available through __getattr__ for backward compatibility
def __getattr__(name):
    if name == "GAPModel":
        return _get_gap_model()
    elif name == "TreeBreed":
        return _get_tree_breed()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    "GAPModel",
    "TreeBreed",
]
