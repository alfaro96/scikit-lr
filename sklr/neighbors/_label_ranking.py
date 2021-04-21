"""Nearest neighbors label ranking."""


# =============================================================================
# Imports
# =============================================================================

# Third party
from sklearn.neighbors._base import KNeighborsMixin
from sklearn.neighbors._base import NeighborsBase as BaseNeighbors

# Local application
from ..base import LabelRankerMixin
from ._base import _predict_k_neighbors


# =============================================================================
# Classes
# =============================================================================

class KNeighborsLabelRanker(KNeighborsMixin,
                            LabelRankerMixin,
                            BaseNeighbors):
    """Label ranker implementing the k-nearest neighbors vote."""

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
        super(KNeighborsLabelRanker, self).__init__(n_neighbors,
                                                    algorithm=algorithm,
                                                    leaf_size=leaf_size,
                                                    p=p,
                                                    metric=metric,
                                                    metric_params=metric_params,  # noqa 
                                                    n_jobs=n_jobs,
                                                    **kwargs)

        self.weights = weights

    def fit(self, X, Y):
        """Fit the k-nearest neighbors label ranker from the training
        dataset."""
        return super(KNeighborsLabelRanker, self)._fit(X, Y)

    def predict(self, X):
        """Predict the target rankings for the provided data."""
        return _predict_k_neighbors(self, X)
