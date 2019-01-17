"""
    This module gathers metrics for rankings.
"""

# =============================================================================
# Imports
# =============================================================================

# NumPy
import numpy as np

# PLR
from ..utils.validation import check_true_pred_sample_weight
from ._calculator       import kendall_distance_calculator, tau_x_score_calculator

# =============================================================================
# Public objects
# =============================================================================

# Methods
__all__ = ["kendall_distance",
           "tau_x_score"]

# =============================================================================
# Global methods
# =============================================================================

def kendall_distance(Y_true,
                     Y_pred,
                     sample_weight = None,
                     normalize     = False):
    """
        Computes the Kendall distance between two bucket orders using a penalization factor.
        It measures the spread between bucket orders.

        Parameters
        ----------
            Y_true: np.ndarray
                Ground truth of right rankings.
                
            Y_pred: np.ndarray
                Predicted rankings.

            sample_weight: {None, np.ndarray} (default = None)
                The sample weight of each instance. If "None", the samples are equally weighted.

            normalize: bool (default = False)
                Whether to normalize the distance.
                
        Returns
        -------
            distance: double
                Mean Kendall distance for the given rankings.
    """
    # Check the input parameters
    (Y_true, Y_pred, sample_weight) = check_true_pred_sample_weight(Y_true, Y_pred, sample_weight, np.intp)

    # Obtain the distance
    distance = kendall_distance_calculator(Y_true = Y_true, Y_pred = Y_pred, sample_weight = sample_weight, normalize = normalize)

    # Return the obtained Kendall distance between the rankings
    return distance

def tau_x_score(Y_true,
                Y_pred,
                sample_weight = None):
    """
        Computes the Kendall's Tau x rank correlation coefficient. It measures
        the amount of overlap between rankings.
        
        Parameters
        ----------
            Y_true: np.ndarray
                Ground truth of right rankings.
                
            Y_pred: np.ndarray
                Predicted rankings.

            sample_weight: {None, np.ndarray} (default = None)
                The sample weight of each instance. If "None", the samples are equally weighted.
                
        Returns
        -------
            coefficient: double
                Mean Kendall's Tau x rank correlation coefficient for the given rankings.
    """ 
    # Check the input parameters
    (Y_true, Y_pred, sample_weight) = check_true_pred_sample_weight(Y_true, Y_pred, sample_weight, np.intp)
    
    # Obtain the coefficient
    coefficient = tau_x_score_calculator(Y_true = Y_true, Y_pred = Y_pred, sample_weight = sample_weight)

    # Return the obtained coefficient between the rankings
    return coefficient
