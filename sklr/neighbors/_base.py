"""Base functions for nearest neighbors."""


# =============================================================================
# Imports
# =============================================================================

# Third party
from sklearn.neighbors._base import _get_weights as get_weights
import numpy as np


# =============================================================================
# Functions
# =============================================================================

def _get_weights(Y, dist, weights):
    """Get the weights from an array of rankings and distances."""
    sample_weight = get_weights(dist, weights)

    # Assign a constant value if applying a uniform weighting
    # to weigh the samples by the number of available classes
    sample_weight = sample_weight if sample_weight else 1

    mask = Y != -1
    sample_weight *= np.mean(mask, axis=2)

    return sample_weight


def _aggregate_neighbors(estimator, neigh_Y, neigh_weights):
    """Aggregate the nearest neighbors rankings according to the weights."""
    aggregate = estimator._rank_algorithm.aggregate
    neigh_data = zip(neigh_Y, neigh_weights)

    Y = [aggregate(Y, sample_weight) for Y, sample_weight in neigh_data]

    return np.array(Y)


def _predict_k_neighbors(estimator, X):
    """Predict using a k-nearest neighbors estimator."""
    X = estimator._validate_data(X, reset=False)

    neigh_dist, neigh_ind = estimator.kneighbors(X)

    neigh_Y = estimator._y[neigh_ind]
    neigh_weights = _get_weights(neigh_Y, neigh_dist, estimator.weights)

    return _aggregate_neighbors(estimator, neigh_Y, neigh_weights)
