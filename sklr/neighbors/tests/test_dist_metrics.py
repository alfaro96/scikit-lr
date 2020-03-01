"""Testing of the distance metric wrapper."""


# =============================================================================
# Imports
# =============================================================================

# Third party
from scipy.spatial.distance import cdist
import numpy as np
import pytest

# Local application
from sklr.utils import check_random_state
from sklr.neighbors import DistanceMetric


# =============================================================================
# Constants
# =============================================================================

# Map the metric identifier to the hyperparameters to be tested
# (using Minkowski since it will return more efficient methods)
METRIC_HYPERPARAMS = {
    "minkowski": {"p": [1.0, 2.0, 3.0, 4.0, 5.0]}
}


# =============================================================================
# Methods
# =============================================================================

def get_items(mapping):
    """Obtain all key and value pairs."""
    for key in mapping:
        for sub_key in mapping[key]:
            for value in mapping[key][sub_key]:
                yield (key, value)


# =============================================================================
# Testing
# =============================================================================

class TestDistanceMetric:
    """Test of the distance metric wrapper."""

    def setup(self):
        """Setup the attributes for testing."""
        self.random_state = check_random_state(198075)

        self.X1 = self.random_state.random_sample((2, 2))
        self.X2 = self.random_state.random_sample((5, 2))

    def check_pairwise(self, metric, p, X1, X2):
        """Check the pairwise method."""
        dist_metric = DistanceMetric.get_metric(metric, p=p)

        dist_pred = dist_metric.pairwise(X1, X2)
        dist_true = cdist(X1, X2, metric, p=p)

        np.testing.assert_array_equal(dist_pred, dist_true)

    @pytest.mark.parametrize("metric,p", get_items(METRIC_HYPERPARAMS))
    def test_pdist(self, metric, p):
        """Test the pdist method."""
        self.check_pairwise(metric, p, self.X1, self.X1)

    @pytest.mark.parametrize("metric,p", get_items(METRIC_HYPERPARAMS))
    def test_cdist(self, metric, p):
        """Test the cdist method."""
        self.check_pairwise(metric, p, self.X1, self.X2)
