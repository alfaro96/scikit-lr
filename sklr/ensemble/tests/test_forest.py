"""Testing for the forest module."""


# =============================================================================
# Imports
# =============================================================================

# Standard
from itertools import product, chain

# Third party
import numpy as np
import pytest

# Local application
from sklr.ensemble import (
    RandomForestLabelRanker, RandomForestPartialLabelRanker)
from sklr.ensemble._forest import _get_n_samples_bootstrap
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

# The following variables are required for some
# test methods to work. Even if they will not be
# used in all the tests, they are globally declared
# to avoid that they are defined along several methods.
# The extra memory overhead should not be an issue

# The criteria for the Label Ranking problem
# and the Partial Label Ranking problem
LR_CRITERIA = ["mallows"]
PLR_CRITERIA = ["disagreements", "distance", "entropy"]
CRITERIA = [*LR_CRITERIA, *PLR_CRITERIA]

# The distances required
# for the Mallows criterion
DISTANCES = ["kendall"]

# The splitters that can be used to split
# an internal node of the decision tree
SPLITTERS = ["binary", "frequency", "width"]

# The forest rankers
LR_FORESTS = [RandomForestLabelRanker]
PLR_FORESTS = [RandomForestPartialLabelRanker]
FORESTS = [*LR_FORESTS, *PLR_FORESTS]

# The possible combinations of forest
# rankers, criteria, splitters and distance
COMBINATIONS_LR = product(LR_FORESTS, LR_CRITERIA, SPLITTERS, DISTANCES)
COMBINATIONS_PLR = product(PLR_FORESTS, PLR_CRITERIA, SPLITTERS, DISTANCES)
COMBINATIONS = list(chain(COMBINATIONS_LR, COMBINATIONS_PLR))

# A toy example to check that the forest rankers
# are properly working. In fact, initialize two datasets,
# one with training data and rankings and another one with
# test data and rankings

# Training

# Data
X_train = np.array([
    [-2, -1],
    [-1, -1],
    [-1, -2],
    [1, 1],
    [1, 2],
    [2, 1]])

# Rankings
Y_train = np.array([
    [1, 2, 3],
    [1, 2, 3],
    [1, 2, 3],
    [2, 1, 3],
    [2, 1, 3],
    [2, 1, 3]])

# Test

# Data
X_test = np.array([
    [-1, -1],
    [2, 2],
    [3, 2]])

# Rankings
Y_test = np.array([
    [1, 2, 3],
    [2, 1, 3],
    [2, 1, 3]])


# =============================================================================
# Testing
# =============================================================================

@pytest.mark.toy_example
@pytest.mark.parametrize(
    "RandomForestRanker,criterion,splitter,distance", COMBINATIONS)
def test_toy_example(RandomForestRanker, criterion, splitter, distance):
    """Test the forest rankers on a toy dataset."""
    # Initialize the forest ranker using the given
    # criterion, splitter and (if correponds) distance,
    # Also, set the maximum number of samples to consider
    # to three quarters of the number of training samples
    if RandomForestRanker is RandomForestLabelRanker:
        model = RandomForestRanker(n_estimators=10,
                                   criterion=criterion,
                                   distance=distance,
                                   splitter=splitter,
                                   max_samples=0.75,
                                   random_state=seed)
    else:
        model = RandomForestRanker(n_estimators=10,
                                   criterion=criterion,
                                   splitter=splitter,
                                   max_samples=0.75,
                                   random_state=seed)

    # Initialize a sample weight that will be
    # used to fit the different forest rankers
    sample_weight = random_state.randint(10, size=(X_train.shape[0]))

    # Fit the forest ranker to the training
    # dataset also using the sample weight
    clf = model.fit(X_train, Y_train, sample_weight)

    # Obtain the predictions of the decision tree ranker
    Y_pred = clf.predict(X_test)

    # Assert that the predictions are correct
    np.testing.assert_array_equal(Y_pred, Y_test)

    # Now, apply the same procedure
    # but only using one feature
    model = model.set_hyperparams(max_features=1)
    clf = model.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    np.testing.assert_array_equal(Y_pred, Y_test)

    # The same as before but without bootstrapping
    # for the sake of fully test the code (coverage)

    # All features
    model = model.set_hyperparams(bootstrap=False, max_features=None)
    clf = model.fit(X_train, Y_train, sample_weight)
    Y_pred = clf.predict(X_test)
    np.testing.assert_array_equal(Y_pred, Y_test)

    # One feature
    model = model.set_hyperparams(bootstrap=False, max_features=1)
    clf = model.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    np.testing.assert_array_equal(Y_pred, Y_test)


@pytest.mark.max_samples_exceptions
def test_max_samples_exceptions():
    """Test invalid max_samples values."""
    # Assert that an exception is raised when
    # the number of bootstrapped samples is
    # an integer type less than or equal zero
    with pytest.raises(ValueError):
        _get_n_samples_bootstrap(100, -1)

    # Assert that an exception is raised when the
    # number of bootstrapped samples is an integer
    # type greater than the number of samples
    with pytest.raises(ValueError):
        _get_n_samples_bootstrap(100, 1000)

    # Assert that an exception is raised when
    # the number of bootstrapped samples is
    # a floating type less than or equal zero
    with pytest.raises(ValueError):
        _get_n_samples_bootstrap(100, 0.0)

    # Assert that an exception is raised when
    # the number of bootstrapped samples is
    # a floating type greater than or equal one
    with pytest.raises(ValueError):
        _get_n_samples_bootstrap(100, 1.0)

    # Assert that an exception is raised when
    # the number of bootstrapped samples is
    # not an integer or floating type
    with pytest.raises(TypeError):
        _get_n_samples_bootstrap(100, "foo")

    # Assert that no exception is raised
    # with properly formatted samples
    assert _get_n_samples_bootstrap(100, None) == 100
    assert _get_n_samples_bootstrap(100, 50) == 50
    assert _get_n_samples_bootstrap(100, 0.5) == 50
