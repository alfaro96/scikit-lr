"""
    Testing for the "probability" module (plr.metrics).
"""

# =============================================================================
# Imports
# =============================================================================

# NumPy
import numpy as np

# PLR
from plr.metrics import bhattacharyya_score, bhattacharyya_distance

# Pytest
import pytest

# =============================================================================
# Initialization
# =============================================================================

# Seed
seed = 23473

# Initialize the probability distributions
prob_dists_1 = np.array([[0.00, 0.10, 0.20, 0.30, 0.40],
                         [0.15, 0.15, 0.30, 0.00, 0.40],
                         [0.40, 0.30, 0.20, 0.10, 0.00]],
                         dtype = np.float64)

prob_dists_2 = np.array([[0.00, 0.20, 0.30, 0.40, 0.10],
                         [0.40, 0.15, 0.30, 0.00, 0.15],
                         [0.00, 0.20, 0.10, 0.40, 0.30]],
                         dtype = np.float64)

# Initialize the sample weights
sample_weight_1 = None
sample_weight_2 = np.array([1.0, 0.0, 0.0], dtype = np.float64)
sample_weight_3 = np.array([0.5, 0.3, 0.2], dtype = np.float64)

# Obtain the Bhattacharyya distances
bhat_distance_sample_weight_1 = bhattacharyya_distance(probs_true = prob_dists_1, probs_pred = prob_dists_2, sample_weight = sample_weight_1)
bhat_distance_sample_weight_2 = bhattacharyya_distance(probs_true = prob_dists_1, probs_pred = prob_dists_2, sample_weight = sample_weight_2)
bhat_distance_sample_weight_3 = bhattacharyya_distance(probs_true = prob_dists_1, probs_pred = prob_dists_2, sample_weight = sample_weight_3)

# Obtain the Bhattacharyya scores
bhat_score_sample_weight_1 = bhattacharyya_score(probs_true = prob_dists_1, probs_pred = prob_dists_2, sample_weight = sample_weight_1)
bhat_score_sample_weight_2 = bhattacharyya_score(probs_true = prob_dists_1, probs_pred = prob_dists_2, sample_weight = sample_weight_2)
bhat_score_sample_weight_3 = bhattacharyya_score(probs_true = prob_dists_1, probs_pred = prob_dists_2, sample_weight = sample_weight_3)

# =============================================================================
# Testing methods
# =============================================================================

@pytest.mark.bhattacharyya_distance
def test_bhattacharyya_distance():
    """
        Test the Bhattacharrya distance for different sample weighting.
    """
    # Equally weighted
    np.testing.assert_almost_equal(bhat_distance_sample_weight_1,
                                   0.2217910259,
                                   err_msg = "The Bhattacharyya distance for the given probability distributions with the samples equally weighted is not correct.")

    # Unbalanced weighted
    np.testing.assert_almost_equal(bhat_distance_sample_weight_2,
                                   0.06958537698,
                                   err_msg = "The Bhattacharyya distance for the given probability distributions with the samples unbalanced weighted is not correct.")

    # Normally weighted
    np.testing.assert_almost_equal(bhat_distance_sample_weight_3,
                                   0.1601486261,
                                   err_msg = "The Bhattacharyya distance for the given probability distributions with the samples normally weighted is not correct.")

@pytest.mark.bhattacharyya_score
def test_bhattacharyya_score():
    """
        Test the Bhattacharrya coefficient for different sample weighting.
    """
    # Equally weighted
    np.testing.assert_almost_equal(bhat_score_sample_weight_1,
                                   0.8196829237,
                                   err_msg = "The Bhattacharyya score for the given probability distributions with the samples equally weighted is not correct.")

    # Unbalanced weighted
    np.testing.assert_almost_equal(bhat_score_sample_weight_2,
                                   0.932780492,
                                   err_msg = "The Bhattacharyya score for the given probability distributions with the samples unbalanced weighted is not correct.")
    
    # Normally weighted
    np.testing.assert_almost_equal(bhat_score_sample_weight_3,
                                   0.8656336967,
                                   err_msg = "The Bhattacharyya score for the given probability distributions with the samples normally weighted is not correct.")

def test_probs_dists_check_input():
    """
        Test the input checking for the provided probability distributions.
    """
    # The probability distributions are not a NumPy array
    with pytest.raises(TypeError):
        bhattacharyya_distance(probs_true = None, probs_pred = None)
    with pytest.raises(TypeError):
        bhattacharyya_score(probs_true = None, probs_pred = None)

    # The probability distributions are not a 2-D NumPy array
    with pytest.raises(ValueError):
        bhattacharyya_distance(probs_true = prob_dists_1[0], probs_pred = prob_dists_2[0])
    with pytest.raises(ValueError):
        bhattacharyya_score(probs_true = prob_dists_1[0], probs_pred = prob_dists_2[0])

    # The probability distributions do not sum one
    with pytest.raises(ValueError):
        bhattacharyya_distance(probs_true = prob_dists_1 * 2, probs_pred = prob_dists_2 * 2)
    with pytest.raises(ValueError):
        bhattacharyya_score(probs_true = prob_dists_1 * 2, probs_pred = prob_dists_2 * 2)
