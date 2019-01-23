"""
    This module gathers metrics for probability distributions.
"""

# =============================================================================
# Imports
# =============================================================================

# NumPy
import numpy as np

# PLR
from ..utils.validation import check_prob_dists, check_true_pred_sample_weight
from ._calculator       import bhattacharyya_distance_calculator, bhattacharyya_score_calculator

# =============================================================================
# Public objects
# =============================================================================

# Methods
__all__ = ["bhattacharyya_distance",
           "bhattacharyya_score"]

# =============================================================================
# Methods
# =============================================================================

def bhattacharyya_distance(probs_true,
                           probs_pred,
                           sample_weight = None,
                           check_input   = True):
    """
        Obtains the average Bhattacharyya distance between two arrays of probability distributions.
        The Bhattacharyya distance is an approximated measurement of the spread between
        two probability distributions.
        
        Parameters
        ----------
            probs_true: np.ndarray
                Ground truth of right probability distributions.
                
            probs_pred: np.ndarray
                Predicted probability distributions.

            sample_weight: {None, np.ndarray} (default = None)
                The sample weight of each instance. If "None", the samples are equally weighted.
            
            check_input: boolean (default = True)
                    Allow to bypass several input checking.
        
            Returns
            -------
                distance: double
                    Mean Bhattacharyya distance obtained from the arrays of probability distributions.
    """
    # Check the input parameters (if corresponds)
    if check_input:
        probs_true = check_prob_dists(probs_true)
        probs_pred = check_prob_dists(probs_pred)
        (probs_true, probs_pred, sample_weight) = check_true_pred_sample_weight(probs_true, probs_pred, sample_weight, np.float64)

    # Initialize the fake sample weight
    fake_sample_weight = np.ones(1, dtype = np.float64)

    # Obtain the distance
    distance = bhattacharyya_distance_calculator(probs_true = probs_true, probs_pred = probs_pred, sample_weight = sample_weight, fake_sample_weight = fake_sample_weight)

    # Return the obtained Bhattacharrya distance between the probability distributions
    return distance

def bhattacharyya_score(probs_true,
                        probs_pred,
                        sample_weight = None,
                        check_input   = True):
    """
        Obtains the average Bhattacharyya coefficient between two arrays of probability distributions.
        The Bhattacharyya coefficient is an approximated measurement of the amount of overlap between
        two probability distributions.
        
        Parameters
        ----------
            probs_true: np.ndarray
                Ground truth of right probability distributions.
                
            probs_pred: np.ndarray
                Predicted probability distributions.
            
            sample_weight: {None, np.ndarray} (default = None)
                The sample weight of each instance. If "None", the samples are equally weighted.

            check_input: boolean (default = True)
                    Allow to bypass several input checking.
        
            Returns
            -------
                coefficient: double
                    Mean Bhattacharyya coefficient obtained from the arrays of probability distributions.
    """
    # Check the input parameters (if corresponds)
    if check_input:
        probs_true = check_prob_dists(probs_true)
        probs_pred = check_prob_dists(probs_pred)
        (probs_true, probs_pred, sample_weight) = check_true_pred_sample_weight(probs_true, probs_pred, sample_weight, np.float64)

    # Obtain the coefficient
    coefficient = bhattacharyya_score_calculator(probs_true = probs_true, probs_pred = probs_pred, sample_weight = sample_weight)

    # Return the obtained Bhattacharrya coefficient between the probability distributions
    return coefficient
