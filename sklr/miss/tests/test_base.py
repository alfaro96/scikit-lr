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


# =============================================================================
# Testing
# =============================================================================

@pytest.mark.simple_misser
def test_simple_misser():
    """Test the SimpleMisser class."""
    # Initialize the misser employed to miss the classes
    mis = SimpleMisser(random_state=seed)

    # Initialize the rankings from
    # which the classes will be missed
    Y = np.array([[1, 2, 3], [2, 2, 1], [2, 2, 1]])

    # Initialize the different percentages
    # of missed classes to be tested
    percentages = [0.0, 0.3, 0.6]

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

    # Assert that the transformed rankings
    # with the missed classes are correct
    for (percentage, Yt_true) in zip(percentages, Yts):
        # Obtain the predicted rankings with the
        # specified percentage of missed classes
        Yt_pred = mis.set_hyperparams(percentage=percentage).fit_transform(Y)
        # Assert that the true and the predicted rankings with
        # the specified percentage of missed classes are the same
        np.testing.assert_allclose(Yt_pred, Yt_true, equal_nan=True)
