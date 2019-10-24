"""Testing for input validations methods."""


# =============================================================================
# Imports
# =============================================================================

# Third party
import numpy as np
import pytest

# Local application
from plr.datasets import load_iris
from plr.exceptions import NotFittedError
from plr.neighbors import KNeighborsLabelRanker
from plr.tree import DecisionTreeLabelRanker
from plr.utils import (
    check_array, check_is_fitted, check_consistent_length,
    check_random_state, check_X_Y, has_fit_parameter)


# =============================================================================
# Initialization
# =============================================================================

# The seed to always obtain the same results
seed = 198075


# =============================================================================
# Testing
# =============================================================================

@pytest.mark.check_array
def test_check_array():
    """Test the check_array method."""
    # Assert that an error is raised when the object is not an array instance
    with pytest.raises(TypeError):
        check_array(None)

    # Assert that an error is raised when the data type
    # of the provided array is not integer or floating
    with pytest.raises(TypeError):
        check_array(np.array([[None]]))

    # Assert that an error is raised when
    # forcing the array to be finite and it is not
    with pytest.raises(ValueError):
        check_array(np.array([[np.nan]]))
    with pytest.raises(ValueError):
        check_array(np.array([[np.inf]]))

    # Assert that an error is raised when
    # forcing the array to be 2-D and it is not
    with pytest.raises(ValueError):
        check_array(np.array([1, 2, 3]))

    # Obtain the data type of the transformed array
    array = check_array(
        array=np.array([1.0]),
        force_all_finite=False,
        ensure_2d=False,
        dtype=np.int64)

    # Assert that the array is properly transformed
    assert np.all(array == np.array([1], dtype=np.int64))


@pytest.mark.check_is_fitted
def test_check_is_fitted():
    """Test the check_is_fitted method."""
    # Assert that an error is raised the object is not an estimator instance
    with pytest.raises(TypeError):
        check_is_fitted("KNeighborsLabelRanker")

    # Assert that an error is raised when the estimator is not an object
    with pytest.raises(TypeError):
        check_is_fitted(KNeighborsLabelRanker)

    # Initialize an estimator and a dataset to ensure
    # that no error is raised when an estimator is fitted
    X = np.array([[1], [2]])
    Y = np.array([[1, 2, 3], [2, 3, 1]])
    estimator = KNeighborsLabelRanker(n_neighbors=1).fit(X, Y)

    # Asser that an error is not raised when the
    # instance is an estimator and it is already fitted
    assert check_is_fitted(estimator) is None


@pytest.mark.check_consistent_length
def test_check_consistent_length():
    """Test the check_consistent_length method."""
    # Assert that an error is raised when the objects are not arrays
    with pytest.raises(TypeError):
        check_consistent_length([None])

    # Assert that an error is raised when
    # the arrays does not have the same length
    with pytest.raises(ValueError):
        check_consistent_length(np.array([1, 2]), np.array([1]))

    # Assert that an error is not raised when the arrays have the same length
    assert check_consistent_length(np.array([1]), np.array([2])) is None


@pytest.mark.check_random_state
def test_check_random_state():
    """Test the check_random_state method."""
    # Assert that an error is raised
    # when the seed cannot be used for seeding
    with pytest.raises(ValueError):
        check_random_state("")

    # Assert that no error is raised and the correct
    # objects are returned when the seed can be used for seeding
    assert isinstance(
        check_random_state(None), np.random.RandomState)
    assert isinstance(
        check_random_state(np.random), np.random.RandomState)
    assert isinstance(
        check_random_state(seed), np.random.RandomState)
    assert isinstance(
        check_random_state(np.random.RandomState(seed)), np.random.RandomState)


@pytest.mark.check_X_Y
def test_check_X_Y():
    """Test the check_X_Y method."""
    # Initialize the input samples and rankings
    X = np.array([[1, 2, 3], [1, 2, 3]])
    Y = np.array([[1, 2, 3], [1, 2, 3]])

    # Validate and convert them
    (X_converted, Y_converted) = check_X_Y(X=X, Y=Y)

    # Assert that they are properly returned
    np.testing.assert_array_equal(X, X_converted)
    np.testing.assert_array_equal(Y, Y_converted)


@pytest.mark.has_fit_parameter
def test_has_fit_parameter():
    """Test the has_fit_parameter method."""
    # Assert that the KNeighborsLabelRanker does not have
    # sample weight parameter in the fit method. Also,
    # assert that the DecisionTreeLabelRanker has the
    # sample weight parameter in the fit method
    assert not has_fit_parameter(KNeighborsLabelRanker, "sample_weight")
    assert has_fit_parameter(DecisionTreeLabelRanker, "sample_weight")
