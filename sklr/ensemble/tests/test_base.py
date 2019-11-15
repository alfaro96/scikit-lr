"""Testing for the base module."""


# =============================================================================
# Imports
# =============================================================================

# Standard
from numbers import Integral

# Third party
import numpy as np
import pytest

# Local application
from sklr.datasets import load_iris
from sklr.ensemble import BaggingLabelRanker
from sklr.ensemble._base import _set_random_states
from sklr.neighbors import KNeighborsLabelRanker
from sklr.tree import DecisionTreeLabelRanker
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

# A toy example to check that the
# base ensemble is properly working

# Data
X = np.array([
    [-2, -1],
    [-1, -1],
    [-1, -2],
    [1, 1],
    [1, 2],
    [2, 1]])

# Rankings
Y = np.array([
    [1, 2, 3],
    [1, 2, 3],
    [1, 2, 3],
    [2, 1, 3],
    [2, 1, 3],
    [2, 1, 3]])


# =============================================================================
# Testing
# =============================================================================

@pytest.mark.base
def test_base():
    """Test BaseEnsemble methods."""
    # Initialize the ensemble
    # using three base estimators
    ensemble = BaggingLabelRanker(n_estimators=3, random_state=seed)

    # Fit the ensemble to the training dataset for
    # being able to manually set the estimators
    clf = ensemble.fit(X, Y)

    # Initialize an empty list of estimators
    # that will be manually created
    clf.estimators_ = []

    # Manually create an estimator without random state,
    # another two with an integer random state and another
    # one without appending to the list of estimators
    clf._make_estimator()
    clf._make_estimator(random_state=seed)
    clf._make_estimator(random_state=seed + 1)
    clf._make_estimator(append=False)

    # Assert that the length of the
    # list of estimators is correct
    assert 3 == len(ensemble.estimators_)

    # Assert that the random states have
    # been properly set for each estimator
    assert clf.estimators_[0].random_state is None
    assert isinstance(clf.estimators_[1].random_state, (Integral, np.integer))
    assert isinstance(clf.estimators_[2].random_state, (Integral, np.integer))

    # Assert that the random state of the
    # second and third estimator are different
    assert clf.estimators_[1].random_state != clf.estimators_[2].random_state


@pytest.mark.base_zero_n_estimators
def test_base_zero_n_estimators():
    """Test that instantiating a BaseEnsemble
    with n_estimators<=0 raises a ValueError."""
    # Initialize the ensemble
    # using zero base estimators
    ensemble = BaggingLabelRanker(n_estimators=0, random_state=seed)

    # Assert that an error is raised when the number
    # of estimators is less than or equal zero
    with pytest.raises(ValueError):
        ensemble.fit(X, Y)


@pytest.mark.base_not_int_n_estimators
def test_base_not_int_n_estimators():
    """Test that instantiating a BaseEnsemble
    without an integer raises a TypeError."""
    # Initialize the ensemble using a
    # number of estimators of str type
    ensemble = BaggingLabelRanker(n_estimators="foo", random_state=seed)

    # Assert that an error is raised when the
    # number of estimators is not an integer type
    with pytest.raises(TypeError):
        ensemble.fit(X, Y)


@pytest.mark.set_random_states
def test_set_random_states():
    """Test the _set_random_states method."""
    # Smoke test, KNeighborsLabelRanker does not have random
    # state so that the process of setting the seed is bypassed
    _set_random_states(KNeighborsLabelRanker(), random_state=seed)

    # Assert that the random state as None in the base estimator
    # still sets a random state in the base estimator
    estimator1 = DecisionTreeLabelRanker(random_state=None)
    assert estimator1.random_state is None
    _set_random_states(estimator1, None)
    assert isinstance(estimator1.random_state, (Integral, np.integer))

    # Assert that the random state fixes results in consistent
    # initialization, that is, even if another random state
    # is set in the initialization of the base estimator, the
    # random state passed to this method sets the random state
    estimator2 = DecisionTreeLabelRanker(random_state=None)
    _set_random_states(estimator1, random_state=seed)
    assert isinstance(estimator1.random_state, (Integral, np.integer))
    _set_random_states(estimator2, random_state=seed)
    assert estimator1.random_state == estimator2.random_state
