"""
    This module gathers the object to solve the
    Optimal Bucket Order Problem.
"""

# =============================================================================
# Imports
# =============================================================================

# NumPy
import numpy as np

# PLR
from ._builder          import OptimalBucketOrderProblemBuilder, UtopianMatrixBuilder, DANIIndexesBuilder
from .matrix            import PairOrderMatrix
from .order             import BucketOrder
from ..utils.validation import check_is_fitted, check_is_type, check_random_state

# =============================================================================
# Public objects
# =============================================================================

# Algorithms
ALGORITHMS = {
                "bpa_original_sg":  0,
                "bpa_original_mp":  1,
                "bpa_original_mp2": 2,
                "bpa_lia_sg":       3,
                "bpa_lia_mp":       4,
                "bpa_lia_mp2":      5
             }

REVERSE_ALGORITHMS = {
                        value: key for (key, value) in ALGORITHMS.items()
                     }

# Classes
__all__ = ["OptimalBucketOrderProblem"]

# =============================================================================
# Optimal Bucket Order Problem
# =============================================================================
class OptimalBucketOrderProblem:
    """
        Store the information of the bucket order obtained after solving the OBOP.
    """
    
    def __init__(self,
                 algorithm    = "bpa_lia_mp2",
                 beta         = 0.25,
                 random_state = None):
        """
            Constructor of OptimalBucketOrderProblem class.
            
            Parameters
            ----------
                algorithm: string, optional (default = "bpa_lia_mp2")
                    Algorithm employed to solve the OBOP. Supported algorithms
                    are "bpa_original_sg", "bpa_original_mp", "bpa_original_mp2"
                    for the original Bucket Pivot Algorithm with single pivot, multi-pivot
                    and multi-pivot with two stages (respectively) and "bpa_lia_sg", "bpa_lia_mp" and "bpa_lia_mp2"
                    for the Bucket Pivot Algorithm with least indecision assumpution with single pivot, multi-pivot
                    and multi-pivot with two stages (respectively).

                beta: float, optional (default = 0.25)
                    To make the decision about the precedence relation of each item w.r.t. the pivot.
                    
                random_state: {None, int, RandomState instance}, optional (default = None)
                    - If int, "random_state" is the seed used by the random number generator.
                    - If RandomState instance, "random_state" is the random number generator.
                    - If None, the random number generator is the "RandomState" instance used
                      by "np.random".
        
            Returns
            -------
                self: object
                    Current object initialized.
        """
        # Initialize the hyperparameters of the current object
        self.algorithm    = algorithm
        self.beta         = beta
        self.random_state = random_state
        
    def fit(self,
            pair_order_matrix,
            check_input = True):
        """
            Fit the model to obtain the corresponding bucket order, using the given pair order matrix.
                
            Parameters
            ----------
                pair_order_matrix: instance of PairOrderMatrix
                    The pair order matrix employed to solve the OBOP.

                check_input: boolean (default = True)
                    Allow to bypass several input checking.
        
            Returns
            -------
                self: object
                    Current object already trained.
        """
        # Check the input parameters (if corresponds)
        if check_input:
            check_is_type(pair_order_matrix, PairOrderMatrix)
            check_is_fitted(pair_order_matrix, "matrix_")

        # Initialize the attributes of the current object
        self.y_ = np.zeros(pair_order_matrix.n_classes_, dtype = np.intp)

        # Obtain the random state object
        random_state = check_random_state(self.random_state)

        # Initialize the utopian matrix and DANI indexes
        utopian_matrix = np.zeros((pair_order_matrix.n_classes_, pair_order_matrix.n_classes_), dtype = np.float64)
        dani_indexes   = np.zeros(pair_order_matrix.n_classes_,                                 dtype = np.float64)

        # Check if the utopian matrix and the DANI indexes must be computed
        if "lia" in self.algorithm:
            UtopianMatrixBuilder().build(pair_order_matrix = pair_order_matrix.matrix_,
                                         utopian_matrix    = utopian_matrix)

            DANIIndexesBuilder().build(pair_order_matrix = pair_order_matrix.matrix_,
                                       utopian_matrix    = utopian_matrix,
                                       dani_indexes      = dani_indexes)

        # Build the bucket order
        OptimalBucketOrderProblemBuilder(algorithm    = ALGORITHMS[self.algorithm],
                                         beta         = self.beta,
                                         random_state = random_state).build(items             = pair_order_matrix.items_,
                                                                            y                 = self.y_,
                                                                            pair_order_matrix = pair_order_matrix.matrix_,
                                                                            utopian_matrix    = utopian_matrix,
                                                                            dani_indexes      = dani_indexes)
        
        # Return the already trained object
        return self
