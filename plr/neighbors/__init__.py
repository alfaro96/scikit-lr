"""
    The module "plr.neighbors" includes estimators
    to solve the Partial Label Ranking Problem with the
    nearest neighbors paradigm.
"""

# =============================================================================
# Imports
# =============================================================================

# Nearest neighbors
from .partial_label_ranking import KNeighborsPartialLabelRanker

# =============================================================================
# Public objects
# =============================================================================

# Classes
__all__ = ["KNeighborsPartialLabelRanker"]
