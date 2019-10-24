"""Testing for the Partial Label Ranking metrics."""


# =============================================================================
# Imports
# =============================================================================

# Third party
import numpy as np
import pytest

# Local application
from plr.metrics import penalized_kendall_distance, tau_x_score


# =============================================================================
# Initialization
# =============================================================================

# The rankings
Y_true = np.array([[1, 2, 2, 3, 3], [1, 2, 2, 3, 3], [1, 2, 2, 3, 3]])
Y_pred = np.array([[1, 2, 2, 3, 3], [1, 3, 2, 2, 3], [2, 3, 3, 1, 1]])

# The sample weights
sample_weights = (np.array([1/3, 1/3, 1/3]), np.array([0.5, 0.3, 0.2]))

# The penalized Kendall distances
penalized_kendall_distances = (
    (0.3, 3.0, np.array([0, 0.3, 0.6]), np.array([0, 3, 6])),
    (0.21, 2.1, np.array([0, 0.3, 0.6]), np.array([0, 3, 6]))
)

# The Tau-x scores
tau_x_scores = (0.4, 0.58)


# =============================================================================
# Testing
# =============================================================================

@pytest.mark.penalized_kendall_distance
@pytest.mark.parametrize(
    "dist_true,sample_weight",
    zip(penalized_kendall_distances, sample_weights))
def test_penalized_kendall_distance(dist_true, sample_weight):
    """Test the penalized_kendall_distance method."""
    # Compute the normalized and penalized Kendall distance
    dist_pred = penalized_kendall_distance(
        Y_true, Y_pred, normalize=True, sample_weight=sample_weight)

    # Assert that the normalized and penalized Kendall distances are the same
    np.testing.assert_almost_equal(dist_true[0], dist_pred)

    # Compute the penalized Kendall distance
    dist_pred = penalized_kendall_distance(
        Y_true, Y_pred, normalize=False, sample_weight=sample_weight)

    # Assert that the penalized Kendall distances are the same
    np.testing.assert_almost_equal(dist_true[1], dist_pred)

    # Compute the array of normalized and penalized Kendall distances
    dists_pred = penalized_kendall_distance(
        Y_true, Y_pred, normalize=True, sample_weight=sample_weight,
        return_dists=True)[1]

    # Assert that the arrays of normalized and
    # penalized Kendall distances are the same
    np.testing.assert_array_almost_equal(dists_pred, dist_true[2])

    # Compute the array of penalized Kendall distances
    dists_pred = penalized_kendall_distance(
        Y_true, Y_pred, normalize=False, sample_weight=sample_weight,
        return_dists=True)[1]

    # Assert that the arrays of penalized Kendall distances are the same
    np.testing.assert_array_almost_equal(dists_pred, dist_true[3])


@pytest.mark.tau_x_score
@pytest.mark.parametrize(
    "score_true,sample_weight", zip(tau_x_scores, sample_weights))
def test_tau_x_score(score_true, sample_weight):
    """Test the tau_x_score method."""
    # Compute the Tau-x score
    score_pred = tau_x_score(Y_true, Y_pred, sample_weight)

    # Assert that the Tau-x scores are the same
    np.testing.assert_almost_equal(score_true, score_pred)
