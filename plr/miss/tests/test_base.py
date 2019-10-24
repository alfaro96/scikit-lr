"""Testing of base classes for missing classes."""


# =============================================================================
# Imports
# =============================================================================

# Third party
import numpy as np
import pytest

# Local application
from plr.miss import SimpleMisser


# =============================================================================
# Initialization
# =============================================================================

# Initialize a seed to always obtain the same results
seed = 198075

# The rankings
Y = np.array([[1, 2, 3], [2, 2, 1], [2, 2, 1]])

# The percentages
percentages = [0.0, 0.3, 0.6]

# The randomly missed rankings
Y_random = [
    np.array([[1, 2, 3], [2, 2, 1], [2, 2, 1]]),
    np.array([[1, 2, np.nan], [2, np.nan, 1], [np.nan, 1, np.nan]]),
    np.array([[1, 2, np.nan], [2, np.nan, 1], [np.nan, np.nan, np.nan]])
]

# The top-k missed rankings
Y_top = [
    np.array([[1, 2, 3], [2, 2, 1], [2, 2, 1]]),
    np.array([[1, 2, np.inf], [2, np.inf, 1], [2, np.inf, 1]]),
    np.array([[1, np.inf, np.inf], [np.inf, np.inf, 1], [np.inf, np.inf, 1]])
]


# =============================================================================
# Testing
# =============================================================================

@pytest.mark.simple_misser
def test_simple_misser():
    """Test the SimpleMisser class."""
    # Assert that an error is raised when the strategy is not supported
    with pytest.raises(ValueError):
        SimpleMisser(strategy="foo").fit(Y)

    # Assert that an error is raised when the
    # percentage is not in the desired interval
    with pytest.raises(ValueError):
        SimpleMisser(percentage=1.5).fit(Y)

    # Assert that an error is raised when the target types
    # of the input rankings is not a subset of the fitted rankings
    with pytest.raises(ValueError):
        SimpleMisser().fit(Y).transform(np.array([[0, 0, 0]]))

    # Assert that an error is raised when the number of classes
    # of the input rankings is different than the rankings used to fit
    with pytest.raises(ValueError):
        SimpleMisser().fit(Y).transform(np.array([[1, 2]]))

    # Assert that an error is raised when the
    # rankings already contain missed classes
    with pytest.raises(ValueError):
        SimpleMisser().fit_transform(Y_random[1])

    # Assert that the randomly missed rankings are correct
    for (percentage, Y_true) in zip(percentages, Y_random):
        mis = SimpleMisser(percentage, strategy="random", random_state=seed)
        Y_pred = mis.fit_transform(Y)
        np.testing.assert_allclose(Y_pred, Y_true, equal_nan=True)

    # Assert that the top-k missed rankings are correct
    for (percentage, Y_true) in zip(percentages, Y_top):
        mis = SimpleMisser(percentage, strategy="top")
        Y_pred = mis.fit_transform(Y)
        np.testing.assert_array_equal(Y_pred, Y_true)
