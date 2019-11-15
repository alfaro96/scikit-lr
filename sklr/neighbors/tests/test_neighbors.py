"""Testing for the neighbors estimators."""


# =============================================================================
# Imports
# =============================================================================

# Third party
import numpy as np
import pytest

# Local application
from sklr.neighbors import (
    DistanceMetric, KNeighborsLabelRanker, KNeighborsPartialLabelRanker)
from sklr.neighbors import VALID_METRICS
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

# The following data and rankings are needed
# along the different test methods. Thus,
# they are globally declared

# The training data and the training rankings. The
# training data is obtained by taking random values
# from a normal distribution and the training rankings
# by taking random permutations of five classes
X_train = random_state.randn(20, 5)
Y_train = np.array([np.random.permutation(5) + 1 for _ in range(20)])

# The test data and the test rankings (both
# (obtained in the same way that the
# training data and the training rankings)
X_test = random_state.randn(10, 5)
Y_test = np.array([np.random.permutation(5) + 1 for _ in range(10)])

# The following classes are required
# for a few of test methods to work. Even
# if they can be locally declared on each
# test method, the extra memory overhead
# of declaring it in the global scope
# should not be an issue
KNeighborsRankers = [KNeighborsLabelRanker, KNeighborsPartialLabelRanker]


# =============================================================================
# Testing
# =============================================================================

@pytest.mark.not_correct_n_neighbors_gets_raised
@pytest.mark.parametrize("KNeighborsRanker", KNeighborsRankers)
def test_not_correct_n_neighbors_gets_raised(KNeighborsRanker):
    """Test whether the n_neighbors hyperparameter is properly formatted."""
    # Initialize this k-nearest neighbors estimator
    # that will be used to test this method
    estimator = KNeighborsRanker()

    # Assert that an exception is raised when the number of
    # nearest neighbors of this estimator is not an integer type
    with pytest.raises(TypeError):
        estimator.set_hyperparams(n_neighbors=None).fit(X_train, Y_train)

    # Assert that an exception is raised when the
    # number of nearest neighbors of this estimator
    # is not an strictly positive integer
    with pytest.raises(ValueError):
        estimator.set_hyperparams(n_neighbors=0).fit(X_train, Y_train)

    # Assert that an exception is raised when the
    # number of nearest neighbors of this estimator
    # is greater than the number of samples
    with pytest.raises(ValueError):
        estimator.set_hyperparams(n_neighbors=50).fit(X_train, Y_train)

    # Assert the same that above but for the the "k_neighbors" method

    # Assert that an exception is raised when the number of nearest
    # neighbors passed to the"k_neighbors" method is not an integer type
    with pytest.raises(TypeError):
        (estimator.set_hyperparams(n_neighbors=5)
                  .fit(X_train, Y_train)
                  .kneighbors(X_test, n_neighbors="foo"))

    # Assert that an exception is raised when the number
    # of nearest neighbors passed to the "k_neighbors"
    # method is not an strictly positive integer
    with pytest.raises(ValueError):
        (estimator.set_hyperparams(n_neighbors=5)
                  .fit(X_train, Y_train)
                  .kneighbors(X_test, n_neighbors=0))

    # Assert that an exception is raised when the number
    # of nearest neighbors passed to the "k_neighbors"
    # method is greater than the number of samples
    with pytest.raises(ValueError):
        (estimator.set_hyperparams(n_neighbors=5)
                  .fit(X_train, Y_train)
                  .kneighbors(X_test, n_neighbors=50))


@pytest.mark.not_correct_shape_gets_raised
@pytest.mark.parametrize("KNeighborsRanker", KNeighborsRankers)
def test_not_correct_shape_gets_raised(KNeighborsRanker):
    """Test that an exception is raised with a non properly shaped data."""
    # Initialize the k-nearest neighbors estimator
    # that will be used to test this method
    estimator = KNeighborsRanker(metric="precomputed")

    # Assert that an exception is raised when using a precomputed
    # matrix whose first dimension is not the same than the second one
    with pytest.raises(ValueError):
        estimator.fit(X_train, Y_train)

    # Assert that an exception is raised when the input data
    # has not the same second dimension that the training one
    with pytest.raises(ValueError):
        estimator.fit(X_train[:5], Y_train[:5]).kneighbors(X_test[:, :3])

    # Assert that an exception is raised when the test data
    # has not the same number of features that the training one
    with pytest.raises(ValueError):
        (estimator.set_hyperparams(metric="euclidean")
                  .fit(X_train, Y_train)
                  .kneighbors(X_test[:, :3]))


@pytest.mark.not_correct_weights_gets_raised
@pytest.mark.parametrize("KNeighborsRanker", KNeighborsRankers)
def test_not_correct_weights_gets_raised(KNeighborsRanker):
    """Test that an exception is raised with a non properly weight."""
    # Initialize the k-nearest neighbors estimator
    # that will be used to test this method
    estimator = KNeighborsRanker()

    # Assert that an exception is raised when the weight
    # function used in this estimator is not available
    with pytest.raises(ValueError):
        estimator.set_hyperparams(weights="foo").fit(X_train, Y_train)


@pytest.mark.precomputed
@pytest.mark.parametrize("KNeighborsRanker", KNeighborsRankers)
def test_precomputed(KNeighborsRanker):
    """Test nearest neighbors estimators with precomputed distance matrix."""
    # Initialize the k-nearest neighbors estimators
    # that will be used to test this method
    neigh_X = KNeighborsRanker(n_neighbors=3)
    neigh_D = KNeighborsRanker(n_neighbors=3, metric="precomputed")

    # Compute the pairwise distance between the training input samples and
    # the distance between the training input samples and the test samples
    # (use the Euclidean distance since the remaining ones has been tested)
    DXX = DistanceMetric.get_metric("euclidean").pairwise(X_train)
    DYX = DistanceMetric.get_metric("euclidean").pairwise(X_test, X_train)

    # Obtain the nearest neighbors distance and indexes of the
    # test input samples regarding the training input samples
    # without and with the precomputed distance matrix
    (ind_X, dist_X) = neigh_X.fit(X_train, Y_train).kneighbors(X_test)
    (ind_D, dist_D) = neigh_D.fit(DXX, Y_train).kneighbors(DYX)

    # Assert that the indexes and distances are the same
    np.testing.assert_array_equal(x=ind_X, y=ind_D)
    np.testing.assert_array_equal(x=dist_X, y=dist_D)

    # Obtain the predictions without and with the precomputed matrix
    pred_X = neigh_X.fit(X_train, Y_train).predict(X_test)
    pred_D = neigh_D.fit(DXX, Y_train).predict(DYX)

    # Assert that the predictions obtained without
    # and with the precomputed matrix are the same
    np.testing.assert_array_equal(x=pred_X, y=pred_D)

    # Ensure that it also works with the
    # training input samples themselves,
    # but now using the distance as weight
    # function to fully test the code
    neigh_X = neigh_X.set_hyperparams(weights="distance")
    neigh_D = neigh_D.set_hyperparams(weights="distance")

    # Returning the distance

    # Obtain the nearest neighbors indexes and
    # distances of each training input sample
    # regarding the other instances without
    # using the precomputed distance matrix
    (ind_X, dist_X) = neigh_X.fit(X_train, Y_train).kneighbors()

    # The same with the precomputed one
    (ind_D, dist_D) = neigh_D.fit(DXX, Y_train).kneighbors()

    # Assert that the indexes and distances are the same
    np.testing.assert_array_equal(x=ind_X,  y=ind_D)
    np.testing.assert_array_equal(x=dist_X, y=dist_D)

    # Obtain the predictions without and with the precomputed matrix
    pred_X = neigh_X.fit(X_train, Y_train).predict(X_test)
    pred_D = neigh_D.fit(DXX, Y_train).predict(DYX)

    # Assert that the predictions obtained without
    # and with the precomputed matrix are the same
    np.testing.assert_array_equal(x=pred_X, y=pred_D)

    # Without returning the distance

    # Obtain the nearest neighbors indexes of
    # each training input sample regarding
    # the other instances without using
    # the precomputed distance matrix
    ind_X = neigh_X.fit(X_train, Y_train).kneighbors(return_distance=False)

    # The same with the precomputed one
    ind_D = neigh_D.fit(DXX, Y_train).kneighbors(return_distance=False)

    # Assert that the indexes and distances are the same
    np.testing.assert_array_equal(x=ind_X,  y=ind_D)

    # Obtain the predictions without and with the precomputed matrix
    pred_X = neigh_X.fit(X_train, Y_train).predict(X_test)
    pred_D = neigh_D.fit(DXX, Y_train).predict(DYX)

    # Assert that the predictions obtained without
    # and with the precomputed matrix are the same
    np.testing.assert_array_equal(x=pred_X, y=pred_D)
