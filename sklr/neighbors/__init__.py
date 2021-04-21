"""The :mod:`sklr.neighbors` module implements nearest neighbors estimators."""


# =============================================================================
# Imports
# =============================================================================

# Local application
from ._label_ranking import KNeighborsLabelRanker
from ._partial_label_ranking import KNeighborsPartialLabelRanker


# =============================================================================
# Module public objects
# =============================================================================

__all__ = ["KNeighborsLabelRanker", "KNeighborsPartialLabelRanker"]
