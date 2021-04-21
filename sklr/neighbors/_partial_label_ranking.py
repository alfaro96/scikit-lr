"""Nearest neighbors partial label ranking."""


# =============================================================================
# Imports
# =============================================================================

# Third party
from sklearn.neighbors._base import KNeighborsMixin
from sklearn.neighbors._base import NeighborsBase as BaseNeighbors

# Local application
from ..base import PartialLabelRankerMixin
from ._base import _predict_k_neighbors


# =============================================================================
# Classes
# =============================================================================

class KNeighborsPartialLabelRanker(KNeighborsMixin,
                                   PartialLabelRankerMixin,
                                   BaseNeighbors):
    """Partial label ranker implementing the k-nearest neighbors vote."""

    def __init__(self,
                 n_neighbors=5,
                 *,
                 weights="uniform",
                 algorithm="auto",
                 leaf_size=30,
                 p=2,
                 metric="minkowski",
                 metric_params=None,
                 n_jobs=None,
                 **kwargs):
        """Constructor."""
        super(KNeighborsPartialLabelRanker, self).__init__(n_neighbors,
                                                           algorithm=algorithm,
                                                           leaf_size=leaf_size,
                                                           p=p,
                                                           metric=metric,
                                                           metric_params=metric_params,  # noqa 
                                                           n_jobs=n_jobs,
                                                           **kwargs)

        self.weights = weights

    def fit(self, X, Y):
        """Fit the k-nearest neighbors partial label ranker from the
        training dataset."""
        return super(KNeighborsPartialLabelRanker, self)._fit(X, Y)

    def predict(self, X):
        """Predict the target partial rankings for the provided data."""
        return _predict_k_neighbors(self, X)
