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
from plr.datasets import load_iris
from plr.ensemble import BaggingLabelRanker
from plr.ensemble._base import _set_random_states
from plr.neighbors import KNeighborsLabelRanker
from plr.tree import DecisionTreeLabelRanker
from plr.utils import check_random_state


# =============================================================================
# Initialization
# =============================================================================

# Initialize a seed to always obtain the same results
seed = 198075

# The random number generator
random_state = check_random_state(seed)


# =============================================================================
# Testing
# =============================================================================

@pytest.mark.base
def test_base():
    """Test BaseEnsemble methods."""
    # Initialize the ensemble
    ensemble = BaggingLabelRanker(n_estimators=3)

    # Load the iris dataset for the sake of testing
    iris = load_iris()

    # Fit the ensemble
    clf = ensemble.fit(iris.data_lr, iris.ranks_lr)

    # Initialize an empty list of estimators that will be manually created
    clf.estimators_ = []

    # Manually create the estimators:
    #   - The first without random state.
    #   - The second and the third with an integer random state.
    #   - The fourth, without appending to the list of estimators.
    clf._make_estimator()
    clf._make_estimator(random_state=random_state)
    clf._make_estimator(random_state=random_state)
    clf._make_estimator(append=False)

    # Assert that the length of estimators is correct
    assert 3 == len(ensemble.estimators_)

    # Assert that the random states have been properly set
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
    # Load the iris dataset
    iris = load_iris()

    # Assert that an error is raised when the number
    # of estimators is less than or equal zero
    with pytest.raises(ValueError):
        BaggingLabelRanker(n_estimators=0).fit(iris.data_lr, iris.ranks_lr)


@pytest.mark.base_not_int_n_estimators
def test_base_not_int_n_estimators():
    """Test that instantiating a BaseEnsemble
    without an integer raises a TypeError."""
    # Load the iris dataset
    iris = load_iris()

    # Assert that an error is raised when
    # the number of estimators is not integer
    with pytest.raises(TypeError):
        BaggingLabelRanker(n_estimators="0").fit(iris.data_lr, iris.ranks_lr)
    with pytest.raises(TypeError):
        BaggingLabelRanker(n_estimators=0.0).fit(iris.data_lr, iris.ranks_lr)


@pytest.mark.set_random_states
def test_set_random_states():
    """Test the _set_random_states method."""
    # Smoke test, KNeighborsLabelRanker does not have random state
    _set_random_states(KNeighborsLabelRanker(), random_state=seed)

    # Assert that the random state as None still sets the value
    estimator1 = DecisionTreeLabelRanker(random_state=None)
    assert estimator1.random_state is None
    _set_random_states(estimator1, None)
    assert isinstance(estimator1.random_state, (Integral, np.integer))

    # Assert that the random state fixes
    # results in consistent initialization
    estimator2 = DecisionTreeLabelRanker(random_state=None)
    _set_random_states(estimator1, random_state=seed)
    assert isinstance(estimator1.random_state, (Integral, np.integer))
    _set_random_states(estimator2, random_state=seed)
    assert estimator1.random_state == estimator2.random_state
