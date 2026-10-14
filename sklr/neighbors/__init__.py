"""Label rankers based on the nearest neighbors of each sample."""

from sklr.neighbors._label_ranking import KNeighborsLabelRanker
from sklr.neighbors._partial_label_ranking import KNeighborsPartialLabelRanker

__all__ = ["KNeighborsLabelRanker", "KNeighborsPartialLabelRanker"]
