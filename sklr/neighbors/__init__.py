"""The :mod:`sklr.neighbors` module implements nearest neighbors estimators."""


# =============================================================================
# Imports
# =============================================================================

# Local application
from ._dist_metrics import DistanceMetric
from ._label_ranking import KNeighborsLabelRanker
from ._partial_label_ranking import KNeighborsPartialLabelRanker


# =============================================================================
# Module public objects
# =============================================================================
__all__ = [
    "DistanceMetric", "KNeighborsLabelRanker", "KNeighborsPartialLabelRanker"
]
