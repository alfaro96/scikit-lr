"""Testing of nearest neighbors estimators."""


# =============================================================================
# Imports
# =============================================================================

# Third party
import pytest

# Local application
from sklr.datasets import load_iris
from sklr.neighbors._base import VALID_WEIGHTS
from sklr.neighbors import KNeighborsLabelRanker, KNeighborsPartialLabelRanker


# =============================================================================
# Constants
# =============================================================================

# Define a list of valid number of nearest
# neighbors to assess the estimator scores
VALID_NEIGHBORS = [3, 4, 5, 6, 7, 8, 9, 10]


# =============================================================================
# Methods
# =============================================================================

def _check_knn_score(knn_model, X, Y):
    """Check the score for "real" problems."""
    assert knn_model.fit(X, Y).score(X, Y) > 0.9


# =============================================================================
# Testing
# =============================================================================

class TestKNeighborsLabelRanker:
    """Testing of the K-Nearest Neighbors Label Ranker."""

    def setup(self):
        """Setup the attributes for testing."""
        (self.X, self.Y) = load_iris(problem="label_ranking")

    def test_gets_raised(self):
        """Test that the handled errors are raised.

        Note that these assertions about raised errors are common for all
        k-nearest neighbors estimators. Therefore, they will be tested only
        here.
        """
        with pytest.raises(TypeError, match="not take an integer value"):
            KNeighborsLabelRanker(n_neighbors="foo").fit(self.X, self.Y)

        with pytest.raises(ValueError, match="must be greater than zero"):
            KNeighborsLabelRanker(n_neighbors=0).fit(self.X, self.Y)

        with pytest.raises(ValueError, match="less than or equal"):
            KNeighborsLabelRanker(n_neighbors=151).fit(self.X, self.Y)

        with pytest.raises(ValueError, match="Unknown weights"):
            KNeighborsLabelRanker(weights="foo").fit(self.X, self.Y)

        with pytest.raises(ValueError, match="Unknown metric"):
            KNeighborsLabelRanker(metric="foo").fit(self.X, self.Y)

    @pytest.mark.parametrize("n_neighbors", VALID_NEIGHBORS)
    @pytest.mark.parametrize("weights", VALID_WEIGHTS)
    def test_score(self, n_neighbors, weights):
        """Test the score for "real" problems."""
        _check_knn_score(
            KNeighborsLabelRanker(n_neighbors, weights), self.X, self.Y)


class TestKNeighborsPartialLabelRanker:
    """Testing of the K-Nearest Neighbors Partial Label Ranker."""

    def setup(self):
        """Setup the attributes for testing."""
        (self.X, self.Y) = load_iris(problem="partial_label_ranking")

    @pytest.mark.parametrize("n_neighbors", VALID_NEIGHBORS)
    @pytest.mark.parametrize("weights", VALID_WEIGHTS)
    def test_score(self, n_neighbors, weights):
        """Test the score for "real" problems."""
        _check_knn_score(
            KNeighborsPartialLabelRanker(n_neighbors, weights), self.X, self.Y)
