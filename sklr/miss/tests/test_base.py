"""Testing of the base classes for missing classes from rankings."""


# =============================================================================
# Imports
# =============================================================================

# Third party
import numpy as np
import pytest

# Local application
from sklr.miss import SimpleMisser


# =============================================================================
# Testing
# =============================================================================

class TestSimpleMisser:
    """Test the SimpleMisser class."""

    def setup(self):
        """Setup the attributes for testing the misser."""
        # Initialize the different percentages of
        # classes to be missed from the rankings
        self.percentages = [0.0, 0.3, 0.6]

        # Initialize a seed to always
        # obtain the same results
        self.seed = 198075

        # Initialize a unique misser object to save memory
        self.misser = SimpleMisser(random_state=self.seed)

        # Initialize a set of rankings used to miss
        # the classes with the different strategies
        self.Y = np.array([[1, 2, 3], [2, 2, 1], [2, 2, 1]])

    def test_gets_raised(self):
        """Test that the errors handled by the misser are raised."""
        # Assert that a value error is raised when
        # the provided strategy is not allowed
        with pytest.raises(ValueError):
            (self.misser.set_hyperparams(strategy="foo")
                        .fit(self.Y))

        # Set a valid strategy in the misser since
        # the following calls will raise errors
        # in the percentage of classes to be missed
        self.misser.set_hyperparams(strategy="random")

        # Assert that a type error is raised when the percentage
        # of classes to be missed is not integer of floating
        with pytest.raises(TypeError):
            (self.misser.set_hyperparams(percentage="foo")
                        .fit(self.Y))

        # Assert that a value error is raised when the
        # percentage of classes to be missed is greater
        # or equal than zero and less or equal than one
        with pytest.raises(ValueError):
            (self.misser.set_hyperparams(percentage=-5)
                 .fit(self.Y))
        with pytest.raises(ValueError):
            (self.misser.set_hyperparams(percentage=5)
                        .fit(self.Y))

        # Set a valid percentage of classes to be missed
        # and fit the misser, since the following calls
        # will raise errors during transform method
        (self.misser.set_hyperparams(percentage=0.0)
                    .fit(self.Y))

        # Assert that a value error is raised when
        # the number of classes of the input rankings
        # is different than the fitted number of classes
        with pytest.raises(ValueError):
            self.misser.transform(np.array([[1, 2]]))

    def test_random(self):
        """Test the misser with the random strategy."""
        # Initialize the rankings that must be obtained after
        # randomly missing the classes with each percentage
        Yt_true_0 = np.array(
            [[1, 2, 3], [2, 2, 1], [2, 2, 1]])
        Yt_true_30 = np.array(
            [[1, 2, np.nan], [2, np.nan, 1], [np.nan, 1, np.nan]])
        Yt_true_60 = np.array(
            [[1, 2, np.nan], [2, np.nan, 1], [np.nan, np.nan, np.nan]])

        # Initialize a list with the previous arrays
        # to use the same code for all of them
        Yts_true = [Yt_true_0, Yt_true_30, Yt_true_60]

        # Set the random strategy in the misser object
        self.misser.set_hyperparams(strategy="random")

        # Check that the rankings that must be obtained and
        # the rankings that have been transformed with each
        # percentage of classes to be missed is the same
        for (Yt_true, percentage) in zip(Yts_true, self.percentages):
            # Fit the misser to the set of rankings according with this
            # percentage of classes to be missed and transform them
            Yt_pred = (self.misser.set_hyperparams(percentage=percentage)
                                  .fit_transform(self.Y))
            # Assert that the rankings that must be obtained and
            # the rankings that have been transformed are the same
            np.testing.assert_allclose(Yt_pred, Yt_true, equal_nan=True)

    def test_top(self):
        """Test the misser with the top strategy."""
        # Initialize the rankings that must be obtained after
        # missing the classes out of top-k with each percentage
        Yt_true_0 = np.array(
            [[1, 2, 3], [2, 2, 1], [2, 2, 1]])
        Yt_true_30 = np.array(
            [[1, 2, 3], [2, 2, 1], [2, 2, 1]])
        Yt_true_60 = np.array(
            [[1, 2, np.inf], [2, np.inf, 1], [2, np.inf, 1]])

        # Initialize a list with the previous arrays
        # to use the same code for all of them
        Yts_true = [Yt_true_0, Yt_true_30, Yt_true_60]

        # Set the top strategy to the misser object
        self.misser.set_hyperparams(strategy="top")

        # Check that the rankings that must be obtained and
        # the rankings that have been transformed with each
        # percentage of classes to be missed is the same
        for (Yt_true, percentage) in zip(Yts_true, self.percentages):
            # Fit the misser to the set of rankings according with this
            # percentage of classes to be missed and transform them
            Yt_pred = (self.misser.set_hyperparams(percentage=percentage)
                                  .fit_transform(self.Y))
            # Assert that the rankings that must be obtained and
            # the rankings that have been transformed are the same
            np.testing.assert_allclose(Yt_pred, Yt_true)
