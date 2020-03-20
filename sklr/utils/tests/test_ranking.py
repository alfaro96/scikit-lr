"""Testing for ranking util methods."""


# =============================================================================
# Imports
# =============================================================================

# Third party
from scipy.stats import rankdata
import numpy as np
import pytest

# Local application
from sklr.utils import (
    check_label_ranking_targets, check_partial_label_ranking_targets,
    is_ranking_without_ties, is_ranking_with_ties, rank_data)


# =============================================================================
# Initialization
# =============================================================================

# Initialize a set of rankings for the Label Ranking problem and the Partial
# Label Ranking problem. Even if they are not used in all the methods, they
# will be employed in most of them. Since they are just toy rankings, the
# extra memory overhead should not be an issue
Y_lr = np.array([[1, 2, 3], [1, 2, 3], [3, 2, 1]])
Y_random_lr = np.array([[1, np.nan, 2], [1, np.nan, 2], [1, 2, np.nan]])
Y_top_lr = np.array([[1, 2, np.inf], [1, 2, np.inf], [np.inf, 2, 1]])
Y_plr = np.array([[1, 1, 2], [1, 2, 2], [1, 1, 2]])
Y_random_plr = np.array([[1, np.nan, 2], [1, np.nan, 2], [1, 1, np.nan]])
Y_top_plr = np.array([[1, 1, np.inf], [1, 2, np.inf], [1, 1, np.inf]])
Y_unknown = np.array([[1, 2, 4]])


# =============================================================================
# Testing
# =============================================================================

@pytest.mark.check_label_ranking_targets
def test_check_label_ranking_targets():
    """Test the check_label_ranking_targets method."""
    # Assert that an error is not raised when
    # the rankings are Label Ranking targets
    # (even if some of the classes are randomly missed)
    check_label_ranking_targets(Y_lr)
    check_label_ranking_targets(Y_random_lr)
    check_label_ranking_targets(Y_top_lr)

    # Assert that an error is raised when the
    # rankings are not Label Ranking targets
    # (either Partial Label Ranking targets
    # or unknown targets)
    with pytest.raises(ValueError):
        check_label_ranking_targets(Y_plr)
    with pytest.raises(ValueError):
        check_label_ranking_targets(Y_random_plr)
    with pytest.raises(ValueError):
        check_label_ranking_targets(Y_top_plr)
    with pytest.raises(ValueError):
        check_label_ranking_targets(Y_unknown)


@pytest.mark.check_partial_label_ranking_targets
def test_check_partial_label_ranking_targets():
    """Test the check_partial_label_ranking_targets method."""
    # Assert that an error is not raised when the
    # rankings are Partial Label Ranking targets (even
    # if some of the classes are randomly or top-k missed)
    check_partial_label_ranking_targets(Y_lr)
    check_partial_label_ranking_targets(Y_random_lr)
    check_partial_label_ranking_targets(Y_top_lr)
    check_partial_label_ranking_targets(Y_plr)
    check_partial_label_ranking_targets(Y_random_plr)
    check_partial_label_ranking_targets(Y_top_plr)

    # Assert that an error is raised when the
    # rankings are not Partial Label Ranking targets
    # (just check for unknown rankings, since Label
    # Ranking targets are also Partial Label Ranking targets)
    with pytest.raises(ValueError):
        check_partial_label_ranking_targets(Y_unknown)


@pytest.mark.is_ranking_without_ties
def test_is_ranking_without_ties():
    """Test the is_ranking_without_ties method."""
    # Assert that a 2-D array is not a ranking without ties
    # (use the Label Ranking targets)
    assert not is_ranking_without_ties(Y_lr)

    # Assert that a ranking with ties is not a ranking without ties
    # (use all types of Partial Label Ranking targets)
    assert not is_ranking_without_ties(Y_plr[0])
    assert not is_ranking_without_ties(Y_random_plr[2])
    assert not is_ranking_without_ties(Y_top_plr[0])

    # Assert that a ranking without ties is a ranking without ties
    # (use both types of Label Ranking targets)
    assert is_ranking_without_ties(Y_lr[0])
    assert is_ranking_without_ties(Y_random_lr[0])
    assert is_ranking_without_ties(Y_top_lr[0])


@pytest.mark.is_ranking_with_ties
def test_is_ranking_with_ties():
    """Test the is_ranking_with_ties method."""
    # Assert that a 2-D array is not a ranking with ties
    # (use the Partial Label Ranking targets)
    assert not is_ranking_with_ties(Y_plr)

    # Assert that a ranking without ties is not a ranking with ties
    # (use all types of Label Ranking targets)
    assert is_ranking_with_ties(Y_lr[0])
    assert is_ranking_with_ties(Y_random_lr[0])
    assert is_ranking_with_ties(Y_top_lr[0])

    # Assert that a ranking with ties is a ranking with ties
    # (use all types of Label Ranking targets)
    assert is_ranking_with_ties(Y_plr[0])
    assert is_ranking_with_ties(Y_random_plr[0])
    assert is_ranking_with_ties(Y_top_plr[0])


@pytest.mark.parametrize("method", ["ordinal", "dense", "foo"])
@pytest.mark.rank_data
def test_rank_data(method):
    """Test the rank_data method."""
    # Initialize the bunch of data to be tested. In fact,
    # it is desired to test finite and infinite values to
    # ensure that the missing values are ranked in the last positions
    data_finite = np.array([2, 0, 3, 2])
    data_infinite = np.array([2, np.nan, 3, np.nan])

    # Assert that an error is raised for the
    # "foo" method, since it is not available
    if method == "foo":
        with pytest.raises(ValueError):
            rank_data(data_finite, method)
    # For the rest of methods, assert that the obtained rankings
    # are correct employing the rankdata method provided by SciPy
    else:
        # Finite data
        np.testing.assert_array_equal(
            x=rankdata(data_finite, method),
            y=rank_data(data_finite, method))
        # Infinite data
        np.testing.assert_array_equal(
            x=rankdata(data_infinite, method),
            y=rank_data(data_infinite, method))
