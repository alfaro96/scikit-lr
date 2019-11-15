"""Base and mixin classes for nearest neighbors."""


# =============================================================================
# Imports
# =============================================================================

# Standard
from abc import ABC, abstractmethod
from functools import partial
from numbers import Integral
from warnings import warn

# Third party
import numpy as np

# Local application
from ..base import BaseEstimator, is_label_ranker
from ._dist_metrics import DistanceMetric
from ._dist_metrics import METRIC_MAPPING
from ..utils.validation import check_array, check_is_fitted


# =============================================================================
# Constants
# =============================================================================

# The metrics that can be employed
# to compute the nearest neighbors
VALID_METRICS = list(METRIC_MAPPING.keys())


# =============================================================================
# Classes
# =============================================================================

def _check_weights(weights):
    """Check to make sure weights are valid."""
    # Check whether the provided weight function is valid
    if weights not in {"uniform", "distance"}:
        raise ValueError("Weights not recognized. Should be 'uniform' "
                         "or 'distance'. Got '{}'.".format(weights))


def _get_weights(Y, dist, weights):
    """Get the weights from an array of rankings
    and distances and a parameter ``weights``."""
    # Initialize the samples weights to a uniform distribution
    sample_weight = np.ones(dist.shape, dtype=np.float64)

    # Weight the samples according to the number
    # of missing classes on each nearest neighbor.
    # This will allow to obtain a more reliable prediction
    sample_weight *= np.mean(~np.isnan(Y), axis=2)

    # Weight the samples by the inverse of the distance
    if weights == "distance":
        # If user attempts to classify a point that
        # was zero distance from one or more training
        # points, those training points are weighted
        # as 1.0 and the other points as 0.0
        with np.errstate(divide="ignore"):
            sample_weight = 1 / sample_weight
        inf_mask = np.isinf(sample_weight)
        inf_row = np.any(inf_mask, axis=1)
        sample_weight[inf_row] = inf_mask[inf_row]

    # Return the sample weights
    return sample_weight


# =============================================================================
# Classes
# =============================================================================

# =============================================================================
# Base nearest neighbors
# =============================================================================
class BaseNeighbors(BaseEstimator, ABC):
    """Base class for nearest neighbors estimators."""

    @abstractmethod
    def __init__(self, n_neighbors=None, weights="uniform", p=2,
                 metric="minkowski"):
        """Constructor."""
        # Initialize the hyperparameters
        self.n_neighbors = n_neighbors
        self.p = p
        self.weights = weights
        self.metric = metric

    def _check_metric(self):
        """Check that the metric is correct."""
        # Obtain the metric object
        # from the string identifier
        if self.metric == "precomputed":
            pass
        else:
            self._metric = DistanceMetric.get_metric(
                metric=self.metric, p=self.p)

    def fit(self, X, Y):
        """Fit the model using X as training data and Y as target rankings.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features) or
                (n_samples, n_samples) if metric="precomputed".
            Training data.

        Y : np.ndarray of shape (n_samples, n_classes)
            Target rankings.

        Returns
        -------
        self : object
        """
        # Validate the training data
        (X, Y) = self._validate_training_data(X, Y)

        # Check that the metric is correct
        self._check_metric()

        # Check that the weights are correct
        _check_weights(self.weights)

        # If the metric is precomputed, then, ensure
        # that the training data is an square matrix
        if (self.metric == "precomputed" and
                self.n_samples_ != self.n_features_):
            raise ValueError("Precomputed matrix must be a square matrix. "
                             "Input is a {}x{} matrix."
                             .format(self.n_samples_, self.n_features_))

        # Ensure that the number of nearest neighbors
        # is an integer type greater than zero and less
        # than or equal the number of samples
        if not isinstance(self.n_neighbors, (Integral, np.integer)):
            raise TypeError("n_neighbors does not take an "
                            "integer value. Got {}."
                            .format(type(self.n_neighbors)))
        elif self.n_neighbors <= 0:
            raise ValueError("Expected n_neighbors > 0. Got {}."
                             .format(self.n_neighbors))
        elif self.n_neighbors > self.n_samples_:
            raise ValueError("Expected n_neighbors <= n_samples. "
                             "Got n_samples = {} and n_neighbors = {}."
                             .format(self.n_samples_, self.n_neighbors))

        # Store the training instances
        # (or distances) and the rankings
        (self._X, self._Y) = (X, Y)

        # Return the fitted nearest neighbors estimator
        return self

    def _check_pairwise_arrays(self, X):
        """Check that the size of the second dimension (number of features)
        of the input array is equal to the one the training input samples,
        or the equivalent check for a precomputed distance matrix
        (number of samples)."""
        # If the distances are precomputed, then, check
        # that the second dimension of the input array
        # is equal to the number of training input samples
        if self.metric == "precomputed":
            if X.shape[1] != self.n_samples_:
                raise ValueError("Precomputed metric requires shape "
                                 "(n_queries, n_indexed). "
                                 "Got {} for {} indexed."
                                 .format(X.shape, self.n_samples_))
        # Otherwise, check that the second dimension of the input array
        # is equal to the number of features on the training input samples
        else:
            if X.shape[1] != self.n_features_:
                raise ValueError("Incompatible dimension for X and Y "
                                 "matrices: X.shape[1] = {} while "
                                 "Y.shape[1] = {}."
                                 .format(X.shape[1], self.n_features_))

    def _pairwise_distances(self, X, reduce_func):
        """Generate a distance matrix between the input data
        and the training input samples with function reduction."""
        # Check the input data to ensure
        # that it is properly formatted
        # with respect to the training data
        self._check_pairwise_arrays(X)

        # Compute the distances between the input data and the
        # training input samples if they are not precomputed
        if self.metric != "precomputed":
            dist = self._metric.pairwise(X1=X, X2=self._X)
        else:
            dist = X

        # Apply the reduction function to the distances
        result = reduce_func(dist)

        # Return the result
        return result


# =============================================================================
# K-Nearest Neighbors Mixin
# =============================================================================
class KNeighborsMixin:
    """Mixin for k-nearest neighbors searches."""

    def _kneighbors_reduce_func(self, dist, n_neighbors, return_distance):
        """Reduce a chunk of distances to nearest neighbors."""
        # Initialize a sample range to extract
        # the k-nearest neighbors via indexing
        sample_range = np.arange(dist.shape[0])[:, None]

        # Instead of using "np.argsort" to obtain the indexes
        # of the k-nearest neighbors, "np.argpartition" is
        # employed, since it is more efficient when
        # k << n_samples (which is the general case)
        neigh_ind = np.argpartition(
            dist, n_neighbors - 1, axis=1)[:, :n_neighbors]

        # However, "np.argpartition" does not
        # guarantee sorted order, so sort again
        # but using a smaller number of samples
        neigh_ind = neigh_ind[
            sample_range,
            np.argsort(dist[sample_range, neigh_ind])]

        # Obtain the result to be returned, taking
        # into account if the distance must be returned
        if return_distance:
            result = (neigh_ind, dist[sample_range, neigh_ind])
        else:
            result = neigh_ind

        # Return the nearest neighbors
        # and the distances (if corresponds)
        return result

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        """Find the k-neighbors of a point.

        Return indexes of and distances to the neighbors of each point.

        Parameters
        ----------
        X : {None, np.ndarray} of shape (n_queries, n_features),
                or (n_queries, n_indexes) if metric == "precomputed",
                optional (default=None)
            The query point or points.
            If not provided, neighbors of each indexed point are returned.
            In this case, the query point is not considered its own neighbor.

        n_neighbors : {None, int}, optional (default=None)
            Number of nearest neighbors to get (default is the value
            passed to the constructor).

        return_distance : bool, optional (default=True)
            If False, distances will not be returned.

        Returns
        -------
        neigh_ind : np.ndarray of shape (n_queries, n_neighbors)
            Indexes of the nearest points in the population matrix.

        neigh_dist : np.ndarray of shape (n_queries, n_neighbors)
            Array representing the lengths to points, only present if
            ``return_distance=True``.

        Raises
        ------
        TypeError
            If the number of nearest neighbors does not take an integet value.

        ValueError
            If the number of nearest neighbors is less than or equal zero.
            If the number of nearest neighbors is greater than the sample size.
        """
        # Check if the model is fitted
        check_is_fitted(self)

        # Ensure that the number of nearest neighbors
        # is either None (to use the number of nearest
        # neighbors of the model) or an integer type greater
        # than zero and less than or equal the number of samples
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        if not isinstance(n_neighbors, (Integral, np.integer)):
            raise TypeError("n_neighbors does not take an "
                            "integer value. Got {}."
                            .format(type(self.n_neighbors)))
        elif n_neighbors <= 0:
            raise ValueError("Expected n_neighbors > 0. Got {}."
                             .format(n_neighbors))

        # Check the input data, to know if the nearest neighbors
        # of the training input samples must be computed
        query_is_train = X is None

        # If the distances of the training
        # input samples must be computed,
        # store them into the corresponding
        # variable and increase the number
        # of nearest neighbors to account
        # for the sample itself being returned,
        # which is removed later
        if query_is_train:
            # Store the training input samples
            X = self._X
            # Increase the number of nearest neighbors
            n_neighbors += 1
        # Otherwise, check the query points to
        # know if they are properly formatted
        else:
            X = check_array(X, dtype=np.float64)

        # Ensures that the number of nearest neighbors is less
        # or equal than the number of training input samples
        if n_neighbors > self.n_samples_:
            raise ValueError("Expected n_neighbors <= n_samples. "
                             "Got n_samples = {} and n_neighbors = {}."
                             .format(self.n_samples_, n_neighbors))

        # Obtain the number of
        # samples of the input data
        n_samples = X.shape[0]

        # Compute the nearest neighbors

        # Create a variable with a partial function that
        # initializes the parameters for the reduction
        # function that transform distances to nearest neighbors
        reduce_func = partial(
            self._kneighbors_reduce_func,
            n_neighbors=n_neighbors,
            return_distance=return_distance)

        # Gather the result containing the nearest neighbors
        # of the input data and their distances (if corresponds)
        result = self._pairwise_distances(X, reduce_func)

        # Once the result is know, it is important to
        # remove the sample itself being returned when
        # the query is the training input data
        if query_is_train:
            # Check whether the distances
            # are included in the result
            if return_distance:
                (neigh_ind, dist) = result
            else:
                neigh_ind = result
            # Create a mask to remove the sample
            # itself from their nearest neighbors
            sample_range = np.arange(n_samples, dtype=np.intp)[:, None]
            mask = neigh_ind != sample_range
            # Corner case: When the number of duplicates
            # are more than the number of neighbors, the
            # first will not be the sample, but a duplicate.
            # In that case mask the first duplicate
            dup_gr_nbrs = np.all(mask, axis=1)
            mask[:, 0][dup_gr_nbrs] = False
            # Remove the sample itself from the nearest neighbors
            neigh_ind = np.reshape(
                neigh_ind[mask], (n_samples, n_neighbors - 1))
            # Reconstitute the result taking into account
            # whether the distances must be also returned
            if return_distance:
                dist = np.reshape(dist[mask], (n_samples, n_neighbors - 1))
                result = (neigh_ind, dist)
            else:
                result = neigh_ind

        # Return the result
        return result
