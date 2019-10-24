"""Testing for the neighbors estimators."""


# =============================================================================
# Imports
# =============================================================================

# Third party
import numpy as np
import pytest

# Local application
from plr.datasets import load_iris
from plr.exceptions import NotFittedError
from plr.neighbors import (
    DistanceMetric, KNeighborsLabelRanker, KNeighborsPartialLabelRanker)
from plr.neighbors import VALID_METRICS
from plr.utils import check_random_state


# =============================================================================
# Initialization
# =============================================================================

# The seed to always obtain the same results
seed = 198075

# The random number generator
random_state = check_random_state(seed)

# The random training input samples and rankings
X_train = random_state.randn(20, 5)
Y_train = np.array([np.random.permutation(5) + 1 for _ in range(20)])

# The random test input samples
X_test = random_state.randn(10, 5)
Y_test = np.array([np.random.permutation(5) + 1 for _ in range(10)])


# =============================================================================
# Testing
# =============================================================================

@pytest.mark.n_neighbors_datatype
def test_n_neighbors_datatype():
    """Test whether the n_neighbors hyperparameter is strictly positive."""
    # Assert that an exception is raised when the
    # number of nearest neighbors is not an integer type
    with pytest.raises(TypeError):
        model = KNeighborsLabelRanker(n_neighbors=None)
        model.fit(X_train, Y_train)

    with pytest.raises(TypeError):
        model = KNeighborsLabelRanker()
        clf = model.fit(X_train, Y_train)
        clf.kneighbors(n_neighbors="foo")

    # Assert that an exception is raised when the
    # number of nearest neighbors is not strictly positive
    with pytest.raises(ValueError):
        model = KNeighborsLabelRanker(n_neighbors=0)
        clf = model.fit(X_train, Y_train)

    with pytest.raises(ValueError):
        model = KNeighborsLabelRanker()
        clf = model.fit(X_train, Y_train)
        clf.kneighbors(n_neighbors=0)

    # Assert that an exception is raised when the number
    # of nearest neighbors is greater than the number of samples
    with pytest.raises(ValueError):
        model = KNeighborsLabelRanker(n_neighbors=50)
        model.fit(X_train, Y_train)

    with pytest.raises(ValueError):
        model = KNeighborsLabelRanker()
        clf = model.fit(X_train, Y_train)
        clf.kneighbors(n_neighbors=50)

    # Assert that no exception is raised with a proper value
    model = KNeighborsLabelRanker()
    clf = model.fit(X_train, Y_train)
    clf.kneighbors(n_neighbors=5)


@pytest.mark.not_fitted_error_gets_raised
def test_not_fitted_error_gets_raised():
    """Test that an exception is raised when using a non fitted estimator."""
    # Assert that an exception is raised when trying
    # to use an estimator when it is not fitted
    with pytest.raises(NotFittedError):
        KNeighborsLabelRanker().kneighbors()


@pytest.mark.not_correct_shape_gets_raised
def test_not_correct_shape_gets_raised():
    """Test that an exception is raised with a non properly shaped data."""
    # Assert that an exception is raised when the input data
    # has not the same second dimension that the training one
    with pytest.raises(ValueError):
        model = KNeighborsLabelRanker()
        clf = model.fit(X_train, Y_train)
        clf.kneighbors(X_train[:, 1:])

    # Assert that an exception is raised when using a precomputed
    # matrix whose first dimension is not the same than the second one
    with pytest.raises(ValueError):
        model = KNeighborsLabelRanker(metric="precomputed")
        clf = model.fit(X_train, Y_train)


@pytest.mark.not_correct_weights_gets_raised
def test_not_correct_weights_gets_raised():
    """Test that an exception is raised with a non properly weight."""
    # Assert that an exception is raised when the weight is not valid
    with pytest.raises(ValueError):
        model = KNeighborsLabelRanker(weights="foo")
        model.fit(X_train, Y_train)


@pytest.mark.precomputed
def test_precomputed():
    """Test nearest neighbors estimators with precomputed distance matrix."""
    # Initialize the estimators
    neigh_X = KNeighborsLabelRanker(n_neighbors=3)
    neigh_D = KNeighborsLabelRanker(n_neighbors=3, metric="precomputed")

    # Compute the pairwise distance between the training input samples and
    # the distance between the training input samples and the test samples
    DXX = DistanceMetric.get_metric("euclidean").pairwise(X_train)
    DYX = DistanceMetric.get_metric("euclidean").pairwise(X_test, X_train)

    # Obtain the nearest neighbors distance and indexes of the
    # test input samples regarding the training input samples
    # without using the precomputed distance matrix
    (ind_X, dist_X) = neigh_X.fit(X_train, Y_train).kneighbors(X_test)

    # The same with the precomputed matrix
    (ind_D, dist_D) = neigh_D.fit(DXX, Y_train).kneighbors(DYX)

    # Assert that the indexes and distances are the same
    np.testing.assert_array_equal(x=ind_X, y=ind_D)
    np.testing.assert_array_equal(x=dist_X, y=dist_D)

    # Ensure that it also works with the
    # training input samples themselves

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

    # Assert that an exception is raised
    # when the matrix is not of correct shape
    with pytest.raises(ValueError):
        neigh_D.fit(DXX, Y_train).kneighbors(X_train)

    # Obtain the predictions without and with the precomputed matrix
    pred_X = neigh_X.fit(X_train, Y_train).predict(X_test)
    pred_D = neigh_D.fit(DXX, Y_train).predict(DYX)

    # Assert that the predictions obtained without
    # and with the precomputed matrix are the same
    np.testing.assert_array_equal(x=pred_X, y=pred_D)


@pytest.mark.neighbors_iris
def test_neighbors_iris():
    """Sanity checks on the iris dataset."""
    # Load the iris dataset
    iris = load_iris()

    # Obtain training and test indexes
    idx = random_state.permutation(iris.data_lr.shape[0])
    train_size, test_size = int(0.8 * idx.shape[0]), int(0.2 * idx.shape[0])
    train_idx, test_idx = idx[:train_size], idx[test_size:]

    # Initialize the model and the Label Ranker
    model = KNeighborsLabelRanker(n_neighbors=1)
    clf = model.fit(iris.data_lr, iris.ranks_lr)

    # Assert that the points themselves are in the decision boundary
    np.testing.assert_array_equal(
        x=clf.predict(iris.data_lr),
        y=iris.ranks_lr)

    # Assert that a minimum score is achieved
    model = KNeighborsLabelRanker(n_neighbors=5, weights="distance")
    clf = model.fit(iris.data_lr[train_idx], iris.ranks_lr[train_idx])
    assert clf.score(iris.data_lr[test_idx], iris.ranks_lr[test_idx]) > 0.95

    # Initialize the model and the Partial Label Ranker
    model = KNeighborsPartialLabelRanker(n_neighbors=1)
    clf = model.fit(iris.data_plr, iris.ranks_plr)

    # Assert that the points themselves are in the decision boundary
    np.testing.assert_array_equal(
        x=clf.predict(iris.data_plr),
        y=iris.ranks_plr)

    # Assert that a minimum score is achieved
    model = KNeighborsPartialLabelRanker(n_neighbors=5, weights="distance")
    clf = model.fit(iris.data_plr[train_idx], iris.ranks_plr[train_idx])
    assert clf.score(iris.data_plr[test_idx], iris.ranks_plr[test_idx]) > 0.95
