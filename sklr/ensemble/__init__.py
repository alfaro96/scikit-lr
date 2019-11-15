"""
The :mod:`sklr.ensemble` module includes ensemble-based methods for
Label Ranking and Partial Label Ranking problems.
"""


# =============================================================================
# Imports
# =============================================================================

# Local application
from ._base import BaseEnsemble
from ._bagging import BaggingLabelRanker, BaggingPartialLabelRanker
from ._forest import RandomForestLabelRanker, RandomForestPartialLabelRanker
from ._weight_boosting import AdaBoostLabelRanker


# =============================================================================
# Public objects
# =============================================================================

# Set the classes that are accessible
# from the module sklr.ensemble
__all__ = [
    "BaggingLabelRanker", "BaggingPartialLabelRanker",
    "BaseEnsemble",
    "RandomForestLabelRanker", "RandomForestPartialLabelRanker",
    "AdaBoostLabelRanker"
]
