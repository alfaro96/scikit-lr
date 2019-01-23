"""
    This module gathers the objects needed to
    solve the Optimal Bucket Order Problem.
"""

# =============================================================================
# Imports
# =============================================================================

# NumPy
import numpy as np

# PLR
from ._builder          import PairOrderMatrixBuilder, UtopianMatrixBuilder, AntiUtopianMatrixBuilder
from ._builder          import distance, normalize_matrix
from ..utils.validation import check_is_type, check_is_fitted, check_Y_sample_weight

# Misc
from abc      import ABC, abstractmethod
from warnings import warn

# =============================================================================
# Public objects
# =============================================================================

# Classes
__all__ = ["PairOrderMatrix",
           "UtopianMatrix",
           "AntiUtopianMatrix"]

# =============================================================================
# Base matrix
# =============================================================================
class Matrix(ABC):
    """
        Base class for matrix.
    """

    def distance(self,
                 other,
                 check_input = True):
        """
            Compute the distance between any pair of matrices.
            
            Parameters
            ----------
                other: instance of "Matrix"
                    The other matrix object.

                check_input: boolean (default = True)
                    Allow to bypass several input checking.
        
            Returns
            -------
                distance: double
                    Distance between the matrices.
        """
        # Check the input parameters (if corresponds)
        if check_input:
            check_is_fitted(self, "matrix_")
            check_is_type(other, Matrix)
            check_is_fitted(other, "matrix_")

        # Return the obtained distance between the matrices
        return distance(matrix_1 = self.matrix_, matrix_2 = other.matrix_)

# =============================================================================
# Pair order matrix
# =============================================================================
class PairOrderMatrix(Matrix):
    """
        Store the information about the precedences and pair order matrix (normalization of the previous one)
        for a given set of bucket orders.
    """

    def fit(self,
            Y             = None,
            sample_weight = None,
            precedences   = None,
            check_input   = True):
        """
            Fit the model to obtain the precedences and pair order matrix from the given set of bucket orders.
            It is also possible to fit the model from an already computed precedences matrix.
            
            Parameters
            ----------
                Y: {None, np.ndarray} (default = None)
                    The set of bucket orders.

                sample_weight: {None, np.ndarray} (default = None)
                    The sample weight of each instance. If "None", the samples are equally weighted.

                precedences: {None, np.ndarray} (default = None)
                    The precedences matrix.

                check_input: boolean (default = True)
                    Allow to bypass several input checking.
        
            Returns
            -------
                self: PairOrderMatrix
                    Current object already trained.
        """
        # Check the input parameters (if corresponds)
        if check_input:
            if isinstance(Y, type(None)) and isinstance(precedences, type(None)):
                raise ValueError("At least one way to build the pair order matrix object must be given. Check the input parameters.")
            if not isinstance(Y, type(None)) and not isinstance(precedences, type(None)):
                warn("Both ways to build the pair order matrix object has been given. The set of bucket orders and sample weight will be employed.")
            if not isinstance(Y, type(None)):
                (Y, sample_weight) = check_Y_sample_weight(Y, sample_weight)
            if not isinstance(precedences, type(None)):
                if not isinstance(precedences, np.ndarray):
                    raise TypeError("The precedences matrix must be a NumPy array, got '{}'.".format(type(precedences).__name__)) 
                if precedences.ndim != 3:
                    raise ValueError("The precedences matrix must be a 3-D NumPy array, got {}-D NumPy array.".format(precedences.ndim))
                if precedences.shape[0] != precedences.shape[1] or precedences.shape[2] != 2:
                    raise ValueError("The precedences matrix has shape '{}', while a square matrix (with an inner NumPy array of 2 elements) is required.".format(precedences.shape))

        # Build the pair order matrix accordingly, taking into account that if both ways
        # are given, the bucket orders and sample weight are used
        if not isinstance(Y, type(None)):
            # Initialize the attributes of the current object
            (self.n_samples_, self.n_classes_) = Y.shape
            self.items_                        = list(range(self.n_classes_))
            self.precedences_                  = np.zeros((self.n_classes_, self.n_classes_, 2), dtype = np.float64)
            self.matrix_                       = np.zeros((self.n_classes_, self.n_classes_),    dtype = np.float64)

            # Build the attributes of the pair order matrix
            PairOrderMatrixBuilder().build(Y                 = Y,
                                           sample_weight     = sample_weight,
                                           precedences       = self.precedences_,
                                           pair_order_matrix = self.matrix_)

        # Otherwise, if the bucket orders are not given, use the precedences matrix
        # Since it is known that at least one of them is provided
        else:
            # Initialize the attributes of the current object
            self.n_samples_    = precedences[0][0][1]
            self.n_classes_    = precedences.shape[0]
            self.items_        = list(range(self.n_classes_))
            self.precedences_  = precedences
            self.matrix_       = np.zeros((self.n_classes_, self.n_classes_), dtype = np.float64)

            # Normalize the precedences matrix to obtain the pair order matrix
            normalize_matrix(precedences = precedences, pair_order_matrix = self.matrix_)

        # Return the current object already trained
        return self

# =============================================================================
# Utopian matrix
# =============================================================================
class UtopianMatrix(Matrix):
    """       
        Store the information about the employed pair order matrix, utopian matrix, utopia-value and utopicity.
    """

    def fit(self,
            pair_order_matrix,
            check_input = True):
        """
            Fit the model to obtain the utopian matrix, utopia-value and
            utopicity from the given pair order matrix.
            
            Parameters
            ----------
                pair_order_matrix: instance of PairOrderMatrix
                    Pair order matrix employed to build the utopian matrix.

                check_input: boolean (default = True)
                    Allow to bypass several input checking.
        
            Returns
            -------
                self: UtopianMatrix
                    Current object already trained.
        """
        # Check the input parameters (if corresponds)
        if check_input:
            check_is_type(pair_order_matrix, PairOrderMatrix)
            check_is_fitted(pair_order_matrix, "matrix_")

        # Initialize the attributes of the current object
        self.parent_ = pair_order_matrix
        self.matrix_ = np.zeros((self.parent_.n_classes_, self.parent_.n_classes_), dtype = np.float64)

        # Build the utopian matrix
        UtopianMatrixBuilder().build(pair_order_matrix = self.parent_.matrix_,
                                     utopian_matrix    = self.matrix_)

        # Obtain the utopia-value and utopicity
        self.value_     = self.distance(other = self.parent_, check_input = False)
        self.utopicity_ = (0.25 * self.parent_.n_classes_ * (self.parent_.n_classes_ - 1) - self.value_) / \
                          (0.25 * self.parent_.n_classes_ * (self.parent_.n_classes_ - 1))
        
        # Return the current object already trained
        return self

# =============================================================================
# Anti-utopian matrix
# =============================================================================
class AntiUtopianMatrix(Matrix):
    """       
        Store the information about the parent pair order matrix, anti-utopian matrix and anti-utopia value.
    """

    def fit(self,
            pair_order_matrix,
            check_input = True):
        """
            Fit the model to obtatin the anti-utopian matrix and anti-utopia value
            from the given pair order matrix.
            
            Parameters
            ----------
                pair_order_matrix: instance of PairOrderMatrix
                    Pair order matrix employed to build the anti-utopian matrix.

                check_input: boolean (default = True)
                    Allow to bypass several input checking.
        
            Returns
            -------
                self: AntiUtopianMatrix
                    Current object already trained.
        """
        # Check the input parameters (if corresponds)
        if check_input:
            check_is_type(pair_order_matrix, PairOrderMatrix)
            check_is_fitted(pair_order_matrix, "matrix_")

        # Initialize the attributes of the current object
        self.parent_ = pair_order_matrix
        self.matrix_ = np.zeros((self.parent_.n_classes_, self.parent_.n_classes_), dtype = np.float64)

        # Build the anti-utopian matrix
        AntiUtopianMatrixBuilder().build(pair_order_matrix   = self.parent_.matrix_,
                                         anti_utopian_matrix = self.matrix_)

        # Obtain the anti-utopia value
        self.value_ = self.distance(other = self.parent_, check_input = False)
        
        # Return the current object already trained
        return self
