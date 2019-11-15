"""Testing for Label Ranking metrics."""


# =============================================================================
# Imports
# =============================================================================

# Third party
import numpy as np
import pytest

# Local application
from sklr.metrics import kendall_distance, tau_score


# =============================================================================
# Initialization
# =============================================================================

# Initialize the true and the predicted
# rankings to test the different methods
Y_true = np.array([[1, 2, 3, 4, 5], [1, 3, 2, 5, 4], [1, 2, 3, 5, 4]])
Y_pred = np.array([[1, 3, 2, 5, 4], [1, 4, 2, 3, 5], [3, 4, 5, 1, 2]])


# =============================================================================
# Testing
# =============================================================================

@pytest.mark.kendall_distance
def test_kendall_distance():
    """Test the kendall_distance method."""
    # Initialize the true and the predicted Kendall
    # distance, without and with normalization
    kendall_distance_true = 11/3
    kendall_distance_norm_true = 11/30
    kendall_distance_pred = kendall_distance(
        Y_true, Y_pred, normalize=False, return_dists=False)
    kendall_distance_norm_pred = kendall_distance(
        Y_true, Y_pred, normalize=True, return_dists=False)

    # Initialize the true and the predicted Kendall
    # distances, without and with normalization
    kendall_distances_true = np.array([2, 2, 7])
    kendall_distances_norm_true = np.array([0.2, 0.2, 0.7])
    (_, kendall_distances_pred) = kendall_distance(
        Y_true, Y_pred, normalize=False, return_dists=True)
    (_, kendall_distances_norm_pred) = kendall_distance(
        Y_true, Y_pred, normalize=True, return_dists=True)

    # Assert that the true and the predicted Kendall
    # distance without and with normalization is correct
    np.testing.assert_almost_equal(
        kendall_distance_pred, kendall_distance_true)
    np.testing.assert_almost_equal(
        kendall_distance_norm_pred, kendall_distance_norm_true)

    # Assert that the true and the predicted Kendall
    # distances without and with normalization are correct
    np.testing.assert_array_almost_equal(
        kendall_distances_pred, kendall_distances_true)
    np.testing.assert_almost_equal(
        kendall_distances_norm_pred, kendall_distances_norm_true)


@pytest.mark.tau_score
def test_tau_score():
    """Test the tau_score method."""
    # Initialize the true and the predicted Tau score
    tau_score_true = 4 / 15
    tau_score_pred = tau_score(Y_true, Y_pred)

    # Assert that the true and the
    # predicted Tau scores are the same
    np.testing.assert_almost_equal(
        tau_score_pred, tau_score_true)
