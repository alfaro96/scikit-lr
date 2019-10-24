"""
The :mod:`plr.neighbors` module implements
the k-nearest neighbors algorithm.
"""


# =============================================================================
# Imports
# =============================================================================

# Local application
from ._base import VALID_METRICS
from ._dist_metrics import DistanceMetric
from ._label_ranking import KNeighborsLabelRanker
from ._partial_label_ranking import KNeighborsPartialLabelRanker


# =============================================================================
# Public objects
# =============================================================================

# All classes
__all__ = [
    "DistanceMetric",
    "KNeighborsLabelRanker",
    "KNeighborsPartialLabelRanker"
]

# All constants
__all__ += ["VALID_METRICS"]
