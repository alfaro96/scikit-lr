"""Testing for input validations methods."""


# =============================================================================
# Imports
# =============================================================================

# Third party
import numpy as np
import pytest

# Local application
from sklr.exceptions import NotFittedError
from sklr.neighbors import KNeighborsLabelRanker
from sklr.tree import DecisionTreeLabelRanker
from sklr.utils import (
    check_array, check_is_fitted, check_consistent_length,
    check_random_state, check_X_Y, has_fit_parameter)


# =============================================================================
# Initialization
# =============================================================================

# The seed to always obtain the same results and, so,
# ensure that the tests carried out are always the same
seed = 198075


# =============================================================================
# Testing
# =============================================================================

@pytest.mark.check_array
def test_check_array():
    """Test the check_array method."""
    # Assert that an error is raised when
    # the object is not a NumPy array
    with pytest.raises(TypeError):
        check_array(None)

    # Assert that an error is raised when the data type
    # of the provided array is not integer or floating
    with pytest.raises(TypeError):
        check_array(np.array([[None]]))

    # Assert that an error is raised when forcing the
    # array to be finite and it contains infinite data
    with pytest.raises(ValueError):
        check_array(np.array([[np.nan]]))
    with pytest.raises(ValueError):
        check_array(np.array([[np.inf]]))

    # Assert that an error is raised when forcing the
    # array to be 2-D and it is of a different dimension
    with pytest.raises(ValueError):
        check_array(np.array([1, 2, 3]))

    # Obtain a properly formatted array to
    # ensure that it is properly transformed
    array = check_array(
        array=np.array([1.0]),
        force_all_finite=False,
        ensure_2d=False,
        dtype=np.int64)

    # Assert that the returned array is properly formatted ensuring
    # that an integer array with the same values is returned
    assert np.all(array == np.array([1], dtype=np.int64))


@pytest.mark.check_is_fitted
def test_check_is_fitted():
    """Test the check_is_fitted method."""
    # Initialize an estimator and a dataset to ensure
    # that no error is raised when an estimator is fitted
    X = np.array([[1], [2]])
    Y = np.array([[1, 2, 3], [2, 3, 1]])
    model = DecisionTreeLabelRanker(random_state=seed)
    clf = model.fit(X, Y)

    # Assert that an error is raised the
    # object is not an estimator instance
    with pytest.raises(TypeError):
        check_is_fitted("DecisionTreeLabelRanker")

    # Assert that an error is raised when the
    # estimator is a class instead of an object
    with pytest.raises(TypeError):
        check_is_fitted(DecisionTreeLabelRanker)

    # AsserT that an error is not raised when the
    # instance is an estimator already fitted
    check_is_fitted(clf)


@pytest.mark.check_consistent_length
def test_check_consistent_length():
    """Test the check_consistent_length method."""
    # Assert that an error is raised when
    # the objects are not NumPy arrays
    with pytest.raises(TypeError):
        check_consistent_length([None])

    # Assert that an error is raised when the
    # NumPy arrays does not have the same length
    with pytest.raises(ValueError):
        check_consistent_length(np.array([1, 2]), np.array([1]))

    # Assert that an error is not raised
    # when the NumPy arrays have the same length
    # (also ensuring that None values are passed)
    check_consistent_length(np.array([1]), np.array([2]), None)


@pytest.mark.check_random_state
def test_check_random_state():
    """Test the check_random_state method."""
    # Assert that an error is raised when
    # the seed cannot be used for seeding
    with pytest.raises(ValueError):
        check_random_state("")

    # Assert that no error is raised and that the correct
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
    # Initialize the input samples and rankings that
    # will be tested for being properly checked
    X = np.array([[0], [1]])
    Y = np.array([[1, 2, 3], [1, 2, 3]])

    # Validate and convert them
    (X_converted, Y_converted) = check_X_Y(X, Y)

    # Assert that they are properly returned by checking
    # that the values are the same than the original ones
    np.testing.assert_array_equal(X, X_converted)
    np.testing.assert_array_equal(Y, Y_converted)


@pytest.mark.has_fit_parameter
def test_has_fit_parameter():
    """Test the has_fit_parameter method."""
    # Assert that the DecisionTreeLabelRanker has the
    # sample weight parameter in the fit method. Also,
    # assert that the KNeighborsLabelRanker does not have
    # the sample weight parameter in the fit method
    assert has_fit_parameter(DecisionTreeLabelRanker, "sample_weight")
    assert not has_fit_parameter(KNeighborsLabelRanker, "sample_weight")
