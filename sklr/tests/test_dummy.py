"""Testing of dummy estimators."""


# =============================================================================
# Imports
# =============================================================================

# Third party
import pytest
import numpy as np

# Local application
from sklr.datasets import load_iris
from sklr.dummy import VALID_STRATEGIES
from sklr.dummy import DummyLabelRanker, DummyPartialLabelRanker


# =============================================================================
# Methods
# =============================================================================

def _check_dummy_score(dummy_model, strategy, X, Y):
    """Check that the score is too low for "real" problems."""
    dummy_model = dummy_model.set_params(strategy=strategy)

    assert dummy_model.fit(X, Y).score(X, Y) < 0.5


# =============================================================================
# Testing
# =============================================================================

class TestDummyLabelRanker:
    """Testing of the Dummy Label Ranker."""

    def setup(self):
        """Setup the attributes for testing."""
        # The strategy will be set at the corresponding testing method
        self.dummy_model = DummyLabelRanker(constant=np.array([1, 2, 3]))

        (self.X, self.Y) = load_iris(problem="label_ranking")

    def test_gets_raised(self):
        """Test that the handled errors are raised.

        Note that these assertions about raised errors are common for
        all dummy estimators. Therefore, they will be tested only here.
        """
        with pytest.raises(ValueError, match="Unknown strategy type"):
            DummyLabelRanker(strategy="foo").fit(self.X, self.Y)

        with pytest.raises(ValueError, match="ranking has to be specified"):
            DummyLabelRanker(strategy="constant").fit(self.X, self.Y)

        with pytest.raises(ValueError, match="ranking should have shape"):
            (DummyLabelRanker(strategy="constant", constant=np.zeros(2))
             .fit(self.X, self.Y))

        with pytest.raises(ValueError, match="ranking is not the target type"):
            (DummyLabelRanker(strategy="constant", constant=np.zeros(3))
             .fit(self.X, self.Y))

    @pytest.mark.parametrize("strategy", VALID_STRATEGIES)
    def test_score(self, strategy):
        """Test that the score is too low for "real" problems."""
        _check_dummy_score(self.dummy_model, strategy, self.X, self.Y)


class TestDummyPartialLabelRanker:
    """Testing of the Dummy Partial Label Ranker."""

    def setup(self):
        """Setup the attributes for testing."""
        # The strategy will be set at the corresponding testing method
        self.dummy_model = DummyPartialLabelRanker(constant=np.ones(3))

        (self.X, self.Y) = load_iris(problem="partial_label_ranking")

    @pytest.mark.parametrize("strategy", VALID_STRATEGIES)
    def test_score(self, strategy):
        """Test that the score is too low for "real" problems."""
        _check_dummy_score(self.dummy_model, strategy, self.X, self.Y)
