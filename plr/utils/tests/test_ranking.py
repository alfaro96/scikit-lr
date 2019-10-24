"""Testing for ranking util methods."""


# =============================================================================
# Imports
# =============================================================================

# Third party
from scipy.stats import rankdata
import numpy as np
import pytest

# Local application
from plr.utils import (
    unique_rankings, check_label_ranking_targets,
    check_partial_label_ranking_targets, type_of_targets,
    is_ranking_without_ties, is_ranking_with_ties, rank_data)


# =============================================================================
# Testing
# =============================================================================

@pytest.mark.unique_rankings
def test_unique_rankings():
    """Test the unique_rankings method."""
    # Initialize a set of finite rankings and the unique ones
    Y = np.array([[1, 2, 3], [1, 2, 3], [3, 2, 1], [3, 2, 1]])
    Y_unique = np.array([[1, 2, 3], [3, 2, 1]])

    # Assert that the unique finite rankings are the same
    np.testing.assert_array_equal(x=Y_unique, y=unique_rankings(Y))

    # Initialize a set of infinite rankings and the unique ones
    Y = np.array([[1, 2, 3], [1, 2, np.nan], [3, 2, 1], [1, 2, np.nan]])
    Y_unique = np.array([[1, 2, 3], [1, 2, np.nan], [3, 2, 1]])

    # Assert that the unique infinite rankings are the same
    np.testing.assert_array_equal(x=Y_unique, y=unique_rankings(Y))

    # Assert that an error is raised for "unknown" rankings
    with pytest.raises(ValueError):
        unique_rankings(np.array([[1, 2, 4]]))


@pytest.mark.check_label_ranking_targets
def test_check_label_ranking_targets():
    """Test the check_label_ranking_targets method."""
    # Assert that an error is not raised when
    # the rankings are Label Ranking targets
    assert check_label_ranking_targets(
        np.array([[1, 2, 3], [3, 2, 1], [2, 1, 3]])) is None
    assert check_label_ranking_targets(
        np.array([[1, 2, np.nan], [1, 2, np.nan], [2, 1, 3]])) is None

    # Assert that an error is raised when the
    # rankings are not Label Ranking targets
    with pytest.raises(ValueError):
        assert check_label_ranking_targets(
            np.array([[1, 2, 4], [3, 2, 1], [2, 1, 3]]))


@pytest.mark.check_partial_label_ranking_targets
def test_check_partial_label_ranking_targets():
    """Test the check_partial_label_ranking_targets method."""
    # Assert that an error is not raised when the
    # rankings are Partial Label Ranking targets
    assert check_partial_label_ranking_targets(
        np.array([[1, 2, 2], [2, 2, 1], [2, 1, 2]])) is None
    assert check_partial_label_ranking_targets(
        np.array([[1, 2, np.nan], [1, 1, np.nan], [2, 1, 2]])) is None

    # Assert that an error is raised when the
    # rankings are not Partial Label Ranking targets
    with pytest.raises(ValueError):
        check_partial_label_ranking_targets(
            np.array([[1, 2, 4], [2, 2, 1], [2, 1, 2]]))
    with pytest.raises(ValueError):
        check_partial_label_ranking_targets(
            np.array([[1, np.nan, 4], [np.nan, 2, np.nan], [2, np.nan, 2]]))


@pytest.mark.is_ranking_without_ties
def test_is_ranking_without_ties():
    """Test the is_ranking_without_ties method."""
    # Assert that a 2-D array is not a ranking without ties
    assert not is_ranking_without_ties(np.array([[1, 2, 3], [2, 1, 3]]))

    # Assert that a ranking with ties is not a ranking without ties
    assert not is_ranking_without_ties(np.array([1, 1, 2]))
    assert not is_ranking_without_ties(np.array([1, 1, np.nan]))

    # Assert that a ranking without ties is a ranking without ties
    assert is_ranking_without_ties(np.array([1, 2, 3]))
    assert is_ranking_without_ties(np.array([1, 2, np.nan]))

    # Assert that rankings with all infinite
    # classes are rankings without ties
    assert is_ranking_without_ties(np.array([np.nan, np.nan, np.nan]))
    assert is_ranking_without_ties(np.array([np.inf, np.inf, np.inf]))


@pytest.mark.is_ranking_with_ties
def test_is_ranking_with_ties():
    """Test the is_ranking_with_ties method."""
    # Assert that 2-D array is not a ranking with ties
    assert not is_ranking_with_ties(np.array([[1, 2, 3], [2, 1, 3]]))

    # Assert that a ranking without ties is a ranking with ties
    assert is_ranking_with_ties(np.array([1, 2, 3]))
    assert is_ranking_with_ties(np.array([1, 2, np.nan]))

    # Assert that a ranking with ties is a ranking with ties
    assert is_ranking_with_ties(np.array([1, 1, 2]))
    assert is_ranking_with_ties(np.array([1, 1, np.nan]))

    # Assert that rankings with all infinite
    # classes are rankings without ties
    assert is_ranking_with_ties(np.array([np.nan, np.nan, np.nan]))
    assert is_ranking_with_ties(np.array([np.inf, np.inf, np.inf]))


@pytest.mark.parametrize("method", ["ordinal", "dense", "foo"])
@pytest.mark.rank_data
def test_rank_data(method):
    """Test the rank_data method."""
    # Initialize the data to be tested
    data_finite = np.array([2, 0, 3, 2])
    data_infinite = np.array([2, np.nan, 3, np.nan])

    # Assert that an error is raised for the "foo" method
    if method == "foo":
        with pytest.raises(ValueError):
            rank_data(data_finite, method)
    # For the rest of methods, assert that the obtained rankings are correct
    else:
        np.testing.assert_array_equal(
            x=rankdata(data_finite, method),
            y=rank_data(data_finite, method))
        np.testing.assert_array_equal(
            x=rankdata(data_infinite, method),
            y=rank_data(data_infinite, method, check_input=False))
