"""Label rankers that decompose label ranking into standard learning problems."""

from sklr.reduction._labelwise import LabelwiseLabelRanker
from sklr.reduction._pairwise import PairwiseLabelRanker

__all__ = ["LabelwiseLabelRanker", "PairwiseLabelRanker"]
