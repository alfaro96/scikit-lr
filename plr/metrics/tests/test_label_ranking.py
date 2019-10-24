"""Testing for Label Ranking metrics."""


# =============================================================================
# Imports
# =============================================================================

# Third party
import numpy as np
import pytest

# Local application
from plr.metrics import kendall_distance, tau_score


# =============================================================================
# Initialization
# =============================================================================

# The rankings
Y_true = np.array([[1, 2, 3, 4, 5], [1, 3, 2, 5, 4], [1, 2, 3, 5, 4]])
Y_pred = np.array([[1, 3, 2, 5, 4], [1, 4, 2, 3, 5], [3, 4, 5, 1, 2]])

# The sample weights
sample_weights = (np.array([1/3, 1/3, 1/3]), np.array([0.5, 0.3, 0.2]))

# The Kendall distances
kendall_distances = (
    (11/30, 11/3, np.array([0.2, 0.2, 0.7]), np.array([2, 2, 7])),
    (0.3, 3.0, np.array([0.2, 0.2, 0.7]), np.array([2, 2, 7]))
)

# The Tau scores
tau_scores = (4/15, 0.4)


# =============================================================================
# Testing
# =============================================================================


@pytest.mark.kendall_distance
@pytest.mark.parametrize(
    "dist_true,sample_weight", zip(kendall_distances, sample_weights))
def test_kendall_distance(dist_true, sample_weight):
    """Test the kendall_distance method."""
    # Compute the normalized Kendall distance
    dist_pred = kendall_distance(
        Y_true, Y_pred, normalize=True, sample_weight=sample_weight)

    # Assert that the normalized Kendall distances are the same
    np.testing.assert_almost_equal(dist_true[0], dist_pred)

    # Compute the Kendall distance
    dist_pred = kendall_distance(
        Y_true, Y_pred, normalize=False, sample_weight=sample_weight)

    # Assert that the Kendall distances are the same
    np.testing.assert_almost_equal(dist_true[1], dist_pred)

    # Compute the array of normalized Kendall distances
    dists_pred = kendall_distance(
        Y_true, Y_pred, normalize=True, sample_weight=sample_weight,
        return_dists=True)[1]

    # Assert that the arrays of normalized Kendall distances are the same
    np.testing.assert_array_almost_equal(dists_pred, dist_true[2])

    # Compute the array of Kendall distances
    dists_pred = kendall_distance(
        Y_true, Y_pred, normalize=False, sample_weight=sample_weight,
        return_dists=True)[1]

    # Assert that the arrays of Kendall distances are the same
    np.testing.assert_array_almost_equal(dists_pred, dist_true[3])


@pytest.mark.tau_score
@pytest.mark.parametrize(
    "score_true,sample_weight",
    zip(tau_scores, sample_weights))
def test_tau_score(score_true, sample_weight):
    """Test the tau_score method."""
    # Compute the Tau score
    score_pred = tau_score(Y_true, Y_pred, sample_weight)

    # Assert that the Tau scores are the same
    np.testing.assert_almost_equal(score_true, score_pred)
