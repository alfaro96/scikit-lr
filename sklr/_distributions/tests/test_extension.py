import numpy as np
import pytest
from numpy.testing import assert_array_equal

from sklr._distributions._extension import complete_rankings
from sklr._distributions.tests._oracles import (
    complete_ranking,
    extensions,
    kendall_distance,
    random_rankings,
)


def _random_center(n_labels, rng):
    """Draw a random complete ranking to use as the center."""
    return rng.permutation(n_labels).astype(np.intp) + 1


@pytest.mark.parametrize("n_labels", [2, 3, 4, 5, 6])
def test_complete_rankings_is_most_probable_extension(n_labels):
    """Check that the completed rankings are the closest extensions to the center."""
    rng = np.random.RandomState(0)
    y = random_rankings(50, n_labels, missing=0.5, random_state=0)
    for row in y:
        center = _random_center(n_labels, rng)
        completed = complete_rankings(row[np.newaxis], center)[0]
        candidates = list(extensions(row))
        assert any(np.array_equal(completed, z) for z in candidates)
        assert kendall_distance(completed, center) == min(
            kendall_distance(z, center) for z in candidates
        )


@pytest.mark.parametrize("n_labels", [2, 5, 12])
def test_complete_rankings_proposition(n_labels):
    """Check the completed rankings against Proposition 1 of [cheng_decision_2009]."""
    rng = np.random.RandomState(0)
    y = random_rankings(100, n_labels, missing=0.6, random_state=0)
    center = _random_center(n_labels, rng)
    expected = np.array([complete_ranking(row, center) for row in y])
    assert_array_equal(complete_rankings(y, center), expected)


@pytest.mark.parametrize(
    "y, center, expected",
    [
        # All the unranked labels go after the second ranked one, in the order of
        # the center, which is not the order of their indices
        (
            [1, 2, np.nan, np.nan, np.nan, np.nan],
            [1, 2, 6, 5, 4, 3],
            [1, 2, 6, 5, 4, 3],
        ),
        # The gaps after the first and the third ranked labels are equally good,
        # and the first one is chosen
        ([1, 2, 3, np.nan], [1, 4, 2, 3], [1, 3, 4, 2]),
    ],
)
def test_complete_rankings_ties(y, center, expected):
    """Check how the ties between gaps and between inserted labels are broken."""
    completed = complete_rankings(
        np.array([y], dtype=np.float64), np.array(center, dtype=np.intp)
    )
    assert_array_equal(completed[0], expected)


def test_complete_rankings_same_gap_many_labels():
    """Check the order of many labels inserted in the same gap."""
    n_labels = 30
    y = np.full((1, n_labels), np.nan)
    y[0, :2] = [1, 2]
    center = np.r_[1, 2, np.arange(n_labels, 2, -1)].astype(np.intp)
    assert_array_equal(complete_rankings(y, center)[0], center)


def test_complete_rankings_complete_and_sparse_rows():
    """Check that complete rankings are kept and others follow an agreeing center."""
    center = np.array([3, 1, 2, 4], dtype=np.intp)
    y = np.array([[2, 1, 4, 3], [np.nan] * 4, [np.nan, 1, np.nan, np.nan]])
    completed = complete_rankings(y, center)
    assert_array_equal(completed, [[2, 1, 4, 3], center, center])
