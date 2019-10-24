"""
The :mod:`plr.ensemble` module includes ensemble-based methods for
Label Ranking and Partial Label Ranking problems.
"""


# =============================================================================
# Imports
# =============================================================================

# Local application
from ._base import BaseEnsemble
from ._bagging import BaggingLabelRanker, BaggingPartialLabelRanker
from ._forest import RandomForestLabelRanker, RandomForestPartialLabelRanker
from ._weight_boosting import AdaBoostLabelRanker, AdaBoostPartialLabelRanker


# =============================================================================
# Public objects
# =============================================================================

# All classes
__all__ = [
    "BaggingLabelRanker", "BaggingPartialLabelRanker",
    "BaseEnsemble",
    "RandomForestLabelRanker", "RandomForestPartialLabelRanker",
    "AdaBoostLabelRanker", "AdaBoostPartialLabelRanker"
]
