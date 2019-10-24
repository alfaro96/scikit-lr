"""
The :mod:`plr.tree` module includes decision tree-based models for
Label Ranking and Partial Label Ranking.
"""

# =============================================================================
# Imports
# =============================================================================

# Local application
from ._classes import (
    BaseDecisionTree, DecisionTreeLabelRanker, DecisionTreePartialLabelRanker)


# =============================================================================
# Public objects
# =============================================================================

# Available classes
__all__ = [
    "BaseDecisionTree",
    "DecisionTreeLabelRanker", "DecisionTreePartialLabelRanker"
]
