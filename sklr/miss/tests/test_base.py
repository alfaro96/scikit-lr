"""Testing of base classes for missing classes."""


# =============================================================================
# Imports
# =============================================================================

# Third party
import numpy as np
import pytest

# Local application
from sklr.miss import SimpleMisser


# =============================================================================
# Initialization
# =============================================================================

# The seed to always obtain the same results and, so,
# ensure that the tests carried out are always the same
seed = 198075

# Initialize a bunch of rankings from
# which the classes will be missed
Y = np.array([[1, 2, 3], [2, 2, 1], [2, 2, 1]])

# Initialize the different percentages
# of missed classes to be tested
percentages = [0.0, 0.3, 0.6]


# =============================================================================
# Testing
# =============================================================================

@pytest.mark.simple_misser_random
def test_simple_misser_random():
    """Test the SimpleMisser class with the random strategy."""
    # Initialize the misser employed to miss the classes
    mis = SimpleMisser(strategy="random", random_state=seed)

    # Initialize the true and rankings obtained after missing
    # a zero, thirty and sixty percentage of classes
    Yt_0_true = np.array(
        [[1, 2, 3], [2, 2, 1], [2, 2, 1]])
    Yt_30_true = np.array(
        [[1, 2, np.nan], [2, np.nan, 1], [np.nan, 1, np.nan]])
    Yt_60_true = np.array(
        [[1, 2, np.nan], [2, np.nan, 1], [np.nan, np.nan, np.nan]])

    # Initialize a list with each one of the rankings to be tested
    Yts = [Yt_0_true, Yt_30_true, Yt_60_true]

    # Assert that the transformed rankings
    # with the missed classes are correct
    for (percentage, Yt_true) in zip(percentages, Yts):
        # Obtain the predicted rankings with the
        # specified percentage of missed classes
        Yt_pred = mis.set_hyperparams(percentage=percentage).fit_transform(Y)
        # Assert that the true and the predicted rankings with
        # the specified percentage of missed classes are the same
        np.testing.assert_allclose(Yt_pred, Yt_true, equal_nan=True)


@pytest.mark.simple_misser_top
def test_simple_misser_top():
    """Test the SimpleMisser class with the top strategy."""
    # Initialize the misser employed to miss the classes
    mis = SimpleMisser(strategy="top", random_state=seed)

    # Initialize the true and rankings obtained after missing
    # a zero, thirty and sixty percentage of classes
    Yt_0_true = np.array(
        [[1, 2, 3], [2, 2, 1], [2, 2, 1]])
    Yt_30_true = np.array(
        [[1, 2, 3], [2, 2, 1], [2, 2, 1]])
    Yt_60_true = np.array(
        [[1, 2, np.inf], [2, np.inf, 1], [2, np.inf, 1]])

    # Initialize a list with each one of the rankings to be tested
    Yts = [Yt_0_true, Yt_30_true, Yt_60_true]

    # Assert that the transformed rankings
    # with the missed classes are correct
    for (percentage, Yt_true) in zip(percentages, Yts):
        # Obtain the predicted rankings with the
        # specified percentage of missed classes
        Yt_pred = mis.set_hyperparams(percentage=percentage).fit_transform(Y)
        # Assert that the true and the predicted rankings with
        # the specified percentage of missed classes are the same
        np.testing.assert_allclose(Yt_pred, Yt_true)


@pytest.mark.simple_misser_gets_raised
def test_simple_misser_gets_raised():
    """Test thta the SimpleMisser class raises the corresponding errors."""
    # Assert that an error is raised
    # when the strategy is not supported
    with pytest.raises(ValueError):
        SimpleMisser(strategy="foo", random_state=seed).fit(Y)

    # Assert that an error is raised when the
    # percentage is not in the desired interval
    with pytest.raises(ValueError):
        SimpleMisser(percentage=-5, random_state=seed).fit(Y)
    with pytest.raises(ValueError):
        SimpleMisser(percentage=5, random_state=seed).fit(Y)

    # The following tests are required for all the
    # transformers, since they are errors that must
    # be thrown when validating the training data and
    # test data. Therefore, they are only checked once

    # Assert that an error is raised when the target types
    # of the input rankings is not a subset of the fitted rankings
    with pytest.raises(ValueError):
        SimpleMisser(random_state=seed).fit(Y).transform(np.array([[0, 0, 0]]))

    # Assert that an error is raised when the number of classes
    # of the input rankings is different than the rankings used to fit
    with pytest.raises(ValueError):
        SimpleMisser(random_state=seed).fit(Y).transform(np.array([[1, 2]]))
