"""
    The module "plr.tree" includes estimators
    to solve the Partial Label Ranking Problem with
    decision trees.
"""

# =============================================================================
# Imports
# =============================================================================

# Greedy
from .greedy import DecisionTreePartialLabelRanker

# =============================================================================
# Public objects
# =============================================================================

# Classes
__all__ = ["DecisionTreePartialLabelRanker"]
