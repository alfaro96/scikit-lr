"""Testing for the Partial Label Ranking metrics."""


# =============================================================================
# Imports
# =============================================================================

# Third party
import numpy as np
import pytest

# Local application
from sklr.metrics import penalized_kendall_distance, tau_x_score


# =============================================================================
# Initialization
# =============================================================================


# Initialize the true and the predicted
# rankings to test the different methods
Y_true = np.array([[1, 2, 2, 3, 3], [1, 2, 2, 3, 3], [1, 2, 2, 3, 3]])
Y_pred = np.array([[1, 2, 2, 3, 3], [1, 3, 2, 2, 3], [2, 3, 3, 1, 1]])


# =============================================================================
# Testing
# =============================================================================

@pytest.mark.penalized_kendall_distance
def test_penalized_kendall_distance():
    """Test the penalized_kendall_distance method."""
    # Initialize the true and the predicted penalized
    # Kendall distance, without and with normalization
    penalized_kendall_distance_true = 3.0
    penalized_kendall_distance_norm_true = 0.3
    penalized_kendall_distance_pred = penalized_kendall_distance(
        Y_true, Y_pred, normalize=False, return_dists=False)
    penalized_kendall_distance_norm_pred = penalized_kendall_distance(
        Y_true, Y_pred, normalize=True, return_dists=False)

    # Initialize the true and the predicted penalized
    # Kendall distances, without and with normalization
    penalized_kendall_distances_true = np.array([0, 3, 6])
    penalized_kendall_distances_norm_true = np.array([0, 0.3, 0.6])
    (_, penalized_kendall_distances_pred) = penalized_kendall_distance(
        Y_true, Y_pred, normalize=False, return_dists=True)
    (_, penalized_kendall_distances_norm_pred) = penalized_kendall_distance(
        Y_true, Y_pred, normalize=True, return_dists=True)

    # Assert that the true and the predicted penalized Kendall
    # distance without and with normalization is correct
    np.testing.assert_almost_equal(
        penalized_kendall_distance_pred,
        penalized_kendall_distance_true)
    np.testing.assert_almost_equal(
        penalized_kendall_distance_norm_pred,
        penalized_kendall_distance_norm_true)

    # Assert that the true and the predicted penalized Kendall
    # distances without and with normalization are correct
    np.testing.assert_array_almost_equal(
        penalized_kendall_distances_pred,
        penalized_kendall_distances_true)
    np.testing.assert_almost_equal(
        penalized_kendall_distances_norm_pred,
        penalized_kendall_distances_norm_true)


@pytest.mark.tau_x_score
def test_tau_x_score():
    """Test the tau_x_score method."""
    # Initialize the true and the predicted Tau-x score
    tau_x_score_true = 0.4
    tau_x_score_pred = tau_x_score(Y_true, Y_pred)

    # Assert that the true and the
    # predicted Tau-x scores are the same
    np.testing.assert_almost_equal(
        tau_x_score_pred, tau_x_score_true)
