"""
    Testing for the "ranking" module (plr.metrics).
"""

# =============================================================================
# Imports
# =============================================================================

# NumPy
import numpy as np

# PLR
from plr.metrics import kendall_distance, tau_x_score

# Pytest
import pytest

# =============================================================================
# Initialization
# =============================================================================

# Seed
seed = 23473

# Initialize the bucket orders
Y_1 = np.array([[1, 2, 2, 3, 3],
                [1, 2, 2, 3, 3],
                [1, 2, 2, 3, 3]],
                dtype = np.intp)

Y_2 = np.array([[1, 2, 2, 3, 3],
                [1, 3, 2, 2, 3],
                [2, 3, 3, 1, 1]],
                dtype = np.intp)

# Initialize the sample weights
sample_weight_1 = None
sample_weight_2 = np.array([1.0, 0.0, 0.0], dtype = np.float64)
sample_weight_3 = np.array([0.5, 0.3, 0.2], dtype = np.float64)

# Obtain the Kendall distances
kendall_distance_sample_weight_1      = kendall_distance(Y_true = Y_1, Y_pred = Y_2, sample_weight = sample_weight_1, normalize = False)
kendall_distance_sample_weight_2      = kendall_distance(Y_true = Y_1, Y_pred = Y_2, sample_weight = sample_weight_2, normalize = False, check_input = False)
kendall_distance_sample_weight_3      = kendall_distance(Y_true = Y_1, Y_pred = Y_2, sample_weight = sample_weight_3, normalize = False, check_input = False)
kendall_distance_sample_weight_1_norm = kendall_distance(Y_true = Y_1, Y_pred = Y_2, sample_weight = sample_weight_1, normalize = True)
kendall_distance_sample_weight_2_norm = kendall_distance(Y_true = Y_1, Y_pred = Y_2, sample_weight = sample_weight_2, normalize = True, check_input = False)
kendall_distance_sample_weight_3_norm = kendall_distance(Y_true = Y_1, Y_pred = Y_2, sample_weight = sample_weight_3, normalize = True, check_input = False)

# Obtain the Tau x scores
tau_x_score_sample_weight_1 = tau_x_score(Y_true = Y_1, Y_pred = Y_2, sample_weight = sample_weight_1)
tau_x_score_sample_weight_2 = tau_x_score(Y_true = Y_1, Y_pred = Y_2, sample_weight = sample_weight_2, check_input = False)
tau_x_score_sample_weight_3 = tau_x_score(Y_true = Y_1, Y_pred = Y_2, sample_weight = sample_weight_3, check_input = False)

# =============================================================================
# Testing methods
# =============================================================================

@pytest.mark.kendall_distance
def test_kendall_distance():
    """
        Test the Kendall distance for different sample weighting.
    """
    # Equally weighted
    np.testing.assert_almost_equal(kendall_distance_sample_weight_1,
                                    3.0,
                                    err_msg = "The Kendall distance for the given bucket orders with the samples equally weighted is not correct.")
    
    np.testing.assert_almost_equal(kendall_distance_sample_weight_1_norm,
                                    0.3,
                                    err_msg = "The Kendall distance (with normalization) for the given bucket orders with the samples equally weighted is not correct.")

    # Unbalanced weighted
    np.testing.assert_almost_equal(kendall_distance_sample_weight_2,
                                    0.0,
                                    err_msg = "The Kendall distance for the given bucket orders with the samples unbalanced weighted is not correct.")

    np.testing.assert_almost_equal(kendall_distance_sample_weight_2_norm,
                                    0.0,
                                    err_msg = "The Kendall distance (with normalization) for the given bucket orders with the samples unbalanced weighted is not correct.")

    # Normally weighted
    np.testing.assert_almost_equal(kendall_distance_sample_weight_3,
                                    2.1,
                                    err_msg = "The Kendall distance for the given bucket orders with the samples normally weighted is not correct.")

    np.testing.assert_almost_equal(kendall_distance_sample_weight_3_norm,
                                    0.21,
                                    err_msg = "The Kendall distance (with normalization) for the given bucket orders with the samples normally weighted is not correct.")
    
@pytest.mark.tau_x_score
def test_tau_x_score():
    """
        Test the Tau x coefficient for different sample weighting.
    """
    # Equally weighted
    np.testing.assert_almost_equal(tau_x_score_sample_weight_1,
                                    0.4,
                                    err_msg = "The Tau x score for the given bucket orders with the samples equally weighted is not correct.")

    # Unbalanced weighted
    np.testing.assert_almost_equal(tau_x_score_sample_weight_2,
                                    1.0,
                                    err_msg = "The Tau x score for the given bucket orders with the samples unbalanced weighted is not correct.")
    
    # Normally weighted
    np.testing.assert_almost_equal(tau_x_score_sample_weight_3,
                                    0.58,
                                    err_msg = "The Tau x score for the given bucket orders with the samples normally weighted is not correct.")


def test_true_pred_sample_weight_check_input():
    """
        Test the input checking for the true and predicted bucket orders and sample weight.
    """
    # The true bucket orders are not a NumPy array
    with pytest.raises(TypeError):
        kendall_distance(Y_true = None, Y_pred = Y_2)
    with pytest.raises(TypeError):
        tau_x_score(Y_true = None, Y_pred = Y_2)

    # The true bucket orders are not a 2-D NumPy array
    with pytest.raises(ValueError):
        kendall_distance(Y_true = Y_1[0], Y_pred = Y_2)
    with pytest.raises(ValueError):
        tau_x_score(Y_true = Y_1[0], Y_pred = Y_2)

    # The predicted bucket orders are not a NumPy array
    with pytest.raises(TypeError):
        kendall_distance(Y_true = Y_1, Y_pred = None)
    with pytest.raises(TypeError):
        tau_x_score(Y_true = Y_1, Y_pred = None)

    # The predicted bucket orders are not a 2-D NumPy array
    with pytest.raises(ValueError):
        kendall_distance(Y_true = Y_1, Y_pred = Y_2[0])
    with pytest.raises(ValueError):
        tau_x_score(Y_true = Y_1, Y_pred = Y_2[0])

    # The sample weight is not a NumPy array
    with pytest.raises(TypeError):
        kendall_distance(Y_true = Y_1, Y_pred = Y_2, sample_weight = TypeError)
    with pytest.raises(TypeError):
        tau_x_score(Y_true = Y_1, Y_pred = Y_2, sample_weight = TypeError)

    # The sample weight is not a 1-D NumPy array
    with pytest.raises(ValueError):
        kendall_distance(Y_true = Y_1, Y_pred = Y_2, sample_weight = sample_weight_2[None, :])
    with pytest.raises(ValueError):
        tau_x_score(Y_true = Y_1, Y_pred = Y_2, sample_weight = sample_weight_2[None, :])

    # All the weights are zero
    with pytest.raises(ValueError):
        kendall_distance(Y_true = Y_1, Y_pred = Y_2, sample_weight = sample_weight_2 * 0)
    with pytest.raises(ValueError):
        tau_x_score(Y_true = Y_1, Y_pred = Y_2, sample_weight = sample_weight_2 * 0)

    # The bucket orders and sample weight have not the same length
    with pytest.raises(ValueError):
        kendall_distance(Y_true = Y_1[:-1], Y_pred = Y_2, sample_weight = sample_weight_2)
    with pytest.raises(ValueError):
        kendall_distance(Y_true = Y_1, Y_pred = Y_2[:-1], sample_weight = sample_weight_2[:-1])
    with pytest.raises(ValueError):
        kendall_distance(Y_true = Y_1, Y_pred = Y_2, sample_weight = sample_weight_2[:-1])
    with pytest.raises(ValueError):
        tau_x_score(Y_true = Y_1[:-1], Y_pred = Y_2, sample_weight = sample_weight_2)
    with pytest.raises(ValueError):
        tau_x_score(Y_true = Y_1, Y_pred = Y_2[:-1], sample_weight = sample_weight_2[:-1])
    with pytest.raises(ValueError):
        tau_x_score(Y_true = Y_1, Y_pred = Y_2, sample_weight = sample_weight_2[:-1])

    # The bucket orders and sample weight cannot be converted to double
    with pytest.raises(ValueError):
        kendall_distance(Y_true = np.array([["Testing"]]), Y_pred = np.array([["Coverage"]]))
    with pytest.raises(ValueError):
        tau_x_score(Y_true = np.array([["Testing"]]), Y_pred = np.array([["Coverage"]]))
