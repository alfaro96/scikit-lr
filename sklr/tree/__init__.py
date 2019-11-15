"""
The :mod:`sklr.tree` module includes decision tree-based models for
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

# Set the classes that are accessible
# from the module sklr.tree
__all__ = [
    "BaseDecisionTree",
    "DecisionTreeLabelRanker", "DecisionTreePartialLabelRanker"
]
