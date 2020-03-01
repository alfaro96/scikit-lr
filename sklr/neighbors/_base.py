"""Base classes for all nearest neighbors estimators."""


# =============================================================================
# Imports
# =============================================================================

# Standard
from abc import ABC
from abc import abstractmethod
from numbers import Integral

# Third party
import numpy as np

# Local application
from ..base import BaseEstimator
from ._dist_metrics import DistanceMetric


# =============================================================================
# Constants
# =============================================================================

# Define the set of valid weights
VALID_WEIGHTS = {"uniform", "distance"}


# =============================================================================
# Methods
# =============================================================================

def _get_weights(Y, dist, weights):
    """Get the weights from an array of rankings
    and distances and a parameter weights."""
    sample_weight = np.ones(dist.shape, dtype=np.float64)

    # Weight the samples according with the number of
    # ranked classes on each nearest neighbor and so it
    # is possible to obtain a more reliable prediction
    sample_weight *= np.mean(np.isfinite(Y), axis=2)

    if weights == "distance":
        # If user attempts to classify a point that
        # was zero-distance from one or more training
        # points, those training points are weighted
        # as one and the other points as zero
        with np.errstate(divide="ignore"):
            sample_weight *= 1 / dist
        inf_mask = np.isinf(sample_weight)
        inf_row = np.any(inf_mask, axis=1)
        sample_weight[inf_row] = inf_mask[inf_row]

    return sample_weight


# =============================================================================
# Classes
# =============================================================================

class BaseNeighbors(BaseEstimator, ABC):
    """Base class for nearest neighbors estimators."""

    @abstractmethod
    def __init__(self, n_neighbors, weights, metric, p):
        """Constructor."""
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.p = p

    def fit(self, X, Y):
        """Fit the neighbors estimator on the training data and rankings."""
        (X, Y) = self._validate_train_data(X, Y)

        if not isinstance(self.n_neighbors, (Integral, np.integer)):
            raise TypeError("The number of nearest neighbors does "
                            "not take an integer value. Got {0}."
                            .format(type(self.n_neighbors)))
        elif self.n_neighbors <= 0:
            raise ValueError("The number of nearest neighbors "
                             "must be greater than zero. Got {0}."
                             .format(self.n_neighbors))
        elif self.n_neighbors > self.n_samples_in_:
            raise ValueError("The number of nearest neighbors must "
                             "be less than or equal to the number "
                             "of samples. Got {0} number of samples "
                             "and {1} number of nearest neighbors."
                             .format(self.n_samples_in_, self.n_neighbors))

        if self.weights not in VALID_WEIGHTS:
            raise ValueError("Unknown weights: {0}. Expected one of {1}."
                             .format(self.weights, list(VALID_WEIGHTS)))

        self._metric = DistanceMetric.get_metric(self.metric, p=self.p)

        (self._X, self._Y) = (X, Y)

        return self

    def _pairwise_distances(self, X, reduce_func):
        """Generate a distance matrix between the input data
        and the training input samples with function reduction."""
        return reduce_func(self._metric.pairwise(X1=X, X2=self._X))


class KNeighborsMixin:
    """Mixin for k-nearest neighbors searches."""

    def _k_neighbors_reduce_func(self, dist):
        """Reduce a chunk of distances to nearest neighbors."""
        sample_range = np.arange(dist.shape[0])[:, None]

        # Use "np.argpartition" instead of "np.argsort" to obtain
        # the indexes of the k-nearest neighbors because it is more
        # efficient when the number of nearest neighbors to take is,
        # by far, less than the number of samples (the general case)
        neigh_ind = np.argpartition(dist, self.n_neighbors - 1, axis=1)
        neigh_ind = neigh_ind[:, :self.n_neighbors]

        # "np.argpartition" does not guarantee sorted order,
        # so sort again but using a smaller number of samples
        neigh_idx = np.argsort(dist[sample_range, neigh_ind])
        neigh_ind = neigh_ind[sample_range, neigh_idx]

        neigh_dist = dist[sample_range, neigh_ind]

        return (neigh_ind, neigh_dist)

    def _k_neighbors(self, X):
        """Return indexes of and distances to the neighbors of each point."""
        X = self._validate_test_data(X)

        # Reduce the distance matrix to extract the indexes of the nearest
        # neighbors for each point (also returning the neighbors distances)
        return self._pairwise_distances(X, self._k_neighbors_reduce_func)

    def predict(self, X):
        """Predict rankings for the provided data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features), dtype=np.float64
            The input samples.

        Returns
        -------
        Y : ndarray of shape (n_samples, n_classes), dtype=np.int64
            The predicted rankings.
        """
        (neigh_ind, neigh_dist) = self._k_neighbors(X)

        # The sample weights are computed even if the uniform weighting
        # function is employed because they are weighted according with
        # the number of missed classes on the nearest neighbors rankings
        neigh_Y = self._Y[neigh_ind]
        neigh_weights = _get_weights(neigh_Y, neigh_dist, self.weights)

        # The rankings of the nearest neighbors for each input sample are
        # aggregated to obtain the predictions (using the sample weights)
        Y = np.array([self._rank_algorithm.aggregate(Y, sample_weight)
                      for (Y, sample_weight) in zip(neigh_Y, neigh_weights)])

        return Y
