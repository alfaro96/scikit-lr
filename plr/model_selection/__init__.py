"""
    The module "plr.model_selection" includes several methods
    for properly analyzing classifiers.
"""

# =============================================================================
# Imports
# =============================================================================

# Validation
from .validation import cross_val_split

# =============================================================================
# Public objects
# =============================================================================

# Methods
__all__ = ["cross_val_split"]
