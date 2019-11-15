"""Testing for the distance metrics."""


# =============================================================================
# Imports
# =============================================================================

# Standard
from itertools import product

# Third party
from scipy.spatial.distance import cdist
import numpy as np
import pytest

# Local application
from sklr.neighbors import DistanceMetric
from sklr.utils import check_random_state


# =============================================================================
# Initialization
# =============================================================================

# The seed and the random state generator
# to always obtain the same results and,
# so, ensure that the tests carried out
# are always the same
seed = 198075
random_state = check_random_state(seed)

# The arrays of data. Since they will be used in all
# the tests, they are globally defined to ensure
# that the same data is used along tests
n_features = 2
n_samples_X1 = 5
n_samples_X2 = 5

X1 = random_state.random_sample((n_samples_X1, n_features))
X2 = random_state.random_sample((n_samples_X2, n_features))

# The different metrics to be tested and possible
# combinations of hyperparametes for them
METRICS_HYPERPARAMS = {
    "euclidean": {},
    "chebyshev": {},
    "minkowski": {"p": (1.0, 1.5, 2.0, 3.0)}
}


# =============================================================================
# Testing
# =============================================================================

def check_pdist(metric, hyperparams_comb, dist_true):
    """Check the pdist method."""
    # Obtain the distance metric according to the string
    # identifier and the hyperparameter combination
    dist_metric = DistanceMetric.get_metric(metric, **hyperparams_comb)

    # Compute the pairwise distance of the data in X1
    dist_pred = dist_metric.pairwise(X1)

    # Assert that the real and the predicted
    # cross-pairwise distances are the same
    np.testing.assert_array_almost_equal(x=dist_pred, y=dist_true)


def check_cdist(metric, hyperparams_comb, dist_true):
    """Check the cdist method."""
    # Obtain the distance metric according to the string
    # identifier and the hyperparameter combination
    dist_metric = DistanceMetric.get_metric(metric, **hyperparams_comb)

    # Compute the cross-pairwise distance
    # between the data in X1 and X2
    dist_pred = dist_metric.pairwise(X1, X2)

    # Assert that the real and the predicted
    # cross-pairwise distances are the same
    np.testing.assert_array_almost_equal(x=dist_pred, y=dist_true)


@pytest.mark.parametrize("metric", METRICS_HYPERPARAMS)
@pytest.mark.pdist
def test_pdist(metric):
    """Test the pdist method."""
    # Obtain the hyperparameters
    # to be tested for this metric
    hyperparams = METRICS_HYPERPARAMS[metric]
    hyperparams_names = hyperparams.keys()

    # Test the different combinations of hyperparameters
    for hyperparams_values in product(*hyperparams.values()):
        # Obtain the combination of hyperparameters
        hyperparams_comb = dict(zip(hyperparams_names, hyperparams_values))
        # Obtain the true distance
        dist_true = cdist(X1, X1, metric, **hyperparams_comb)
        # Check that the true distance and
        # the predicted ones are the same
        check_pdist(metric, hyperparams_comb, dist_true)


@pytest.mark.parametrize("metric", METRICS_HYPERPARAMS)
@pytest.mark.cdist
def test_cdist(metric):
    """Test the cdist method."""
    # Obtain the hyperparameters
    # to be tested for this metric
    hyperparams = METRICS_HYPERPARAMS[metric]
    hyperparams_names = hyperparams.keys()

    # Test the different combinations of hyperparameters
    for hyperparams_values in product(*hyperparams.values()):
        # Obtain the combination of hyperparameters
        hyperparams_comb = dict(zip(hyperparams_names, hyperparams_values))
        # Obtain the true distance
        dist_true = cdist(X1, X2, metric, **hyperparams_comb)
        # Check that the true distance and
        # the predicted ones are the same
        check_cdist(metric, hyperparams_comb, dist_true)
