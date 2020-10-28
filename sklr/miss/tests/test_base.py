"""Testing of the base classes for missing classes from rankings."""


# =============================================================================
# Imports
# =============================================================================

# Third party
import numpy as np
import pytest

# Local application
from sklr.miss import SimpleMisser
from sklr.datasets import load_iris


# =============================================================================
# Testing
# =============================================================================

class TestSimpleMisser:
    """Testing of the class to miss classes from rankings."""

    def setup(self):
        """Setup the attributes for testing."""
        # The strategy will be set in the corresponding testing method
        self.misser = SimpleMisser(probability=0.6, random_state=198075)

        (_, self.Y) = load_iris(problem="partial_label_ranking")

    def test_gets_raised(self):
        """Test that the handled errors are raised."""
        with pytest.raises(ValueError, match="supported strategies"):
            SimpleMisser(strategy="foo").fit(self.Y)

        with pytest.raises(TypeError, match="integer or floating"):
            SimpleMisser(probability="foo").fit(self.Y)

        with pytest.raises(ValueError, match="less than or equal one"):
            SimpleMisser(probability=1.5).fit(self.Y)

        with pytest.raises(ValueError, match="greater than or equal zero"):
            SimpleMisser(probability=-1.5).fit(self.Y)

    def test_random(self):
        """Test the random strategy.

        In particular, asserts that the specified percentage of classes are
        deleted (up to one decimal of precision).
        """
        self.probability = self.misser.probability
        self.random_misser = self.misser.set_params(strategy="random")

        Yt = self.random_misser.fit_transform(self.Y)
        nan_mean = np.mean(np.isnan(Yt))

        assert np.isclose(nan_mean, self.probability, rtol=1e-1)

    def test_top(self):
        """Test the top strategy.

        In particular, asserts that the latest ranked classes are deleted
        (maintaining the specified arbitrariness).
        """
        self.top_misser = self.misser.set_params(strategy="top")

        Yt = self.top_misser.fit_transform(self.Y)
        idx_last_ranked = np.argmax(self.Y, axis=1)

        assert np.all(np.isinf(np.choose(idx_last_ranked, Yt.T)))
