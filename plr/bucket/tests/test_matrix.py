"""
    Testing for the "matrix" module (plr.bucket).
"""

# =============================================================================
# Imports
# =============================================================================

# NumPy
import numpy as np

# PLR
from plr.bucket     import PairOrderMatrix, UtopianMatrix, AntiUtopianMatrix
from plr.exceptions import NotFittedError

# Pytest
import pytest

# =============================================================================
# Initialization
# =============================================================================

# Seed
seed = 23473

# Initialize the bucket orders
Y_comp = np.array([[1, 2, 3, 4, 5],
                   [4, 2, 1, 5, 3],
                   [1, 2, 2, 1, 1],
                   [2, 3, 3, 2, 1],
                   [1, 2, 3, 2, 2]],
                  dtype = np.float64)

Y_mis = np.array([[1, np.nan, np.nan, 2, 3],
                  [3, 1, np.nan, np.nan, 2],
                  [1, 2, np.nan, 1, 1],
                  [2, np.nan, np.nan, 2, 1],
                  [1, np.nan, np.nan, 2, 2]],
                 dtype = np.float64)

Y_top = np.array([[1, 2, np.inf, np.inf, np.inf],
                  [np.inf, 2, 1, np.inf, 3], 
                  [1, np.inf, np.inf, 1, 1],
                  [2, np.inf, np.inf, np.inf, 1],
                  [1, 2, np.inf, 2, 2]],
                 dtype = np.float64)

# Initialize the sample weights
sample_weight_1 = np.array([0.20, 0.20, 0.20, 0.20, 0.20], dtype = np.float64)
sample_weight_2 = np.array([1.00, 0.00, 0.00, 0.00, 0.00], dtype = np.float64)
sample_weight_3 = np.array([0.05, 0.30, 0.35, 0.10, 0.20], dtype = np.float64)

# Initialize a precedences matrix
precedences = np.array([[[0, 5], [4, 0], [4, 0], [3, 2], [2, 1]],
                        [[1, 0], [0, 5], [2, 2], [2, 1], [2, 1]],
                        [[1, 0], [1, 2], [0, 5], [2, 0], [2, 0]],
                        [[0, 2], [2, 1], [3, 0], [0, 5], [1, 2]],
                        [[2, 1], [2, 1], [3, 0], [2, 2], [0, 5]]],
                        dtype = np.float64)

# Train the objects

# Pair order matrices
pair_order_matrix_comp        = PairOrderMatrix().fit(Y = Y_comp)
pair_order_matrix_mis         = PairOrderMatrix().fit(Y = Y_mis)
pair_order_matrix_top         = PairOrderMatrix().fit(Y = Y_top)
pair_order_matrix_weighted_1  = PairOrderMatrix().fit(Y = Y_comp, sample_weight = sample_weight_1, check_input = False)
pair_order_matrix_weighted_2  = PairOrderMatrix().fit(Y = Y_comp, sample_weight = sample_weight_2, check_input = False)
pair_order_matrix_weighted_3  = PairOrderMatrix().fit(Y = Y_comp, sample_weight = sample_weight_3, check_input = False)
pair_order_matrix_precedences = PairOrderMatrix().fit(precedences = precedences, check_input = False)

# Utopian matrix
utopian_matrix = UtopianMatrix().fit(pair_order_matrix = pair_order_matrix_comp, check_input = False)

# Anti-utopian matrix
anti_utopian_matrix = AntiUtopianMatrix().fit(pair_order_matrix = pair_order_matrix_comp, check_input = False)

# Distance
distance = pair_order_matrix_comp.distance(other = pair_order_matrix_top, check_input = False)

# =============================================================================
# Testing
# =============================================================================

# =============================================================================
# Testing for the class Matrix
# =============================================================================
@pytest.mark.matrix
class TestMatrix(object):
    """
        Test several common method for the subclasses of the "Matrix" class.
    """

    def test_distance(self):
        """
            Test the distance between matrices.
        """
        np.testing.assert_equal(distance,
                                0.6,
                                err_msg = "The distance between the provided matrices is not correct.")
    
    def test_check_input_distance(self):
        """
          Test the input checking for the distance method of the Matrix class.  
        """
        # The "self" matrix is not fitted
        with pytest.raises(NotFittedError):
            PairOrderMatrix().distance(PairOrderMatrix())

        # The provided matrix is not type of "Matrix"
        with pytest.raises(TypeError):
            pair_order_matrix_comp.distance(Y_comp)
        
        # The provided matrix is not fitted
        with pytest.raises(NotFittedError):
            pair_order_matrix_comp.distance(PairOrderMatrix())

# =============================================================================
# Testing for the class PairOrderMatrix
# =============================================================================
@pytest.mark.pairordermatrix
class TestPairOrderMatrix(object):
    """
        Test the methods of the "PairOrderMatrix" class.
    """

    def test_fit_without_sample_weight(self):
        """
            Test the "PairOrderMatrix" objects obtained after fitting without sample weight and complete, missed and top-k bucket orders.
        """
        # Complete
        np.testing.assert_array_equal(pair_order_matrix_comp.precedences_,
                                      np.array([[[0, 5], [4, 0], [4, 0], [3, 2], [2, 1]],
                                                [[1, 0], [0, 5], [2, 2], [2, 1], [2, 1]],
                                                [[1, 0], [1, 2], [0, 5], [2, 0], [2, 0]],
                                                [[0, 2], [2, 1], [3, 0], [0, 5], [1, 2]],
                                                [[2, 1], [2, 1], [3, 0], [2, 2], [0, 5]]],
                                                dtype = np.float64),
                                      err_msg = "The precedences matrix for the complete bucket orders is not correct.")

        np.testing.assert_array_almost_equal(pair_order_matrix_comp.matrix_,
                                             np.array([[0.5, 0.8, 0.8, 0.8, 0.5],
                                                       [0.2, 0.5, 0.6, 0.5, 0.5],
                                                       [0.2, 0.4, 0.5, 0.4, 0.4],
                                                       [0.2, 0.5, 0.6, 0.5, 0.4],
                                                       [0.5, 0.5, 0.6, 0.6, 0.5]],
                                                       dtype = np.float64),
                                             err_msg = "The pair order matrix for the complete bucket orders is not correct.")

        # Missing
        np.testing.assert_array_equal(pair_order_matrix_mis.precedences_,
                                      np.array([[[0, 5], [4, 3], [5, 5], [3, 3], [2, 1]],
                                                [[4, 3], [0, 5], [5, 5], [4, 4], [4, 3]],
                                                [[5, 5], [5, 5], [0, 5], [5, 5], [5, 5]],
                                                [[1, 3], [5, 4], [5, 5], [0, 5], [2, 3]],
                                                [[2, 1], [4, 3], [5, 5], [2, 3], [0, 5]]],
                                                dtype = np.float64),
                                      err_msg = "The precedences matrix for the bucket orders with missing classes is not correct.")

        np.testing.assert_array_almost_equal(pair_order_matrix_mis.matrix_,
                                             np.array([[0.5,   0.5,  0.5, 45/70, 0.5],
                                                       [0.5,   0.5,  0.5, 6/13,  0.5],
                                                       [0.5,   0.5,  0.5, 0.5,   0.5],
                                                       [25/70, 7/13, 0.5, 0.5,   0.5],
                                                       [0.5,   0.5,  0.5, 0.5,   0.5]],
                                                       dtype = np.float64),
                                             err_msg = "The pair order matrix for the bucket orders with missing classes is not correct.")
                                        
        # Top-k  
        np.testing.assert_array_equal(pair_order_matrix_top.precedences_,
                                      np.array([[[0, 5], [4, 0], [4, 0], [3, 2], [2, 1]],
                                                [[1, 0], [0, 5], [2, 2], [2, 2], [2, 1]],
                                                [[1, 0], [1, 2], [0, 5], [1, 2], [1, 1]],
                                                [[0, 2], [1, 2], [2, 2], [0, 5], [0, 3]],
                                                [[2, 1], [2, 1], [3, 1], [2, 3], [0, 5]]], dtype = np.float64),
                                      err_msg = "The precedences matrix for the bucket orders applying top-k is not correct.")

        np.testing.assert_array_almost_equal(pair_order_matrix_top.matrix_,
                                             np.array([[0.5, 0.8, 0.8, 0.8, 0.5],
                                                       [0.2, 0.5, 0.6, 0.6, 0.5],
                                                       [0.2, 0.4, 0.5, 0.4, 0.3],
                                                       [0.2, 0.4, 0.6, 0.5, 0.3],
                                                       [0.5, 0.5, 0.7, 0.7, 0.5]],
                                                       dtype = np.float64),
                                             err_msg = "The pair order matrix for the bucket orders applying top-k is not correct.")

    def test_fit_with_sample_weight(self):
        """
            Test the "PairOrderMatrix" objects after fitting the model with several sample weight.
        """
        # Equally weighted
        np.testing.assert_array_equal(pair_order_matrix_weighted_1.precedences_,
                                      np.array([[[0, 5], [4, 0], [4, 0], [3, 2], [2, 1]],
                                                [[1, 0], [0, 5], [2, 2], [2, 1], [2, 1]],
                                                [[1, 0], [1, 2], [0, 5], [2, 0], [2, 0]],
                                                [[0, 2], [2, 1], [3, 0], [0, 5], [1, 2]],
                                                [[2, 1], [2, 1], [3, 0], [2, 2], [0, 5]]],
                                                dtype = np.float64),
                                      err_msg = "The precedences matrix with the samples equally weighted is not correct.")

        np.testing.assert_array_almost_equal(pair_order_matrix_weighted_1.matrix_,
                                             np.array([[0.5, 0.8, 0.8, 0.8, 0.5],
                                                       [0.2, 0.5, 0.6, 0.5, 0.5],
                                                       [0.2, 0.4, 0.5, 0.4, 0.4],
                                                       [0.2, 0.5, 0.6, 0.5, 0.4],
                                                       [0.5, 0.5, 0.6, 0.6, 0.5]],
                                                       dtype = np.float64),
                                             err_msg = "The pair order matrix with the samples equally weighted is not correct.")

        # Unbalanced weighted
        np.testing.assert_array_equal(pair_order_matrix_weighted_2.precedences_,
                                      np.array([[[0, 5], [5, 0], [5, 0], [5, 0], [5, 0]],
                                                [[0, 0], [0, 5], [5, 0], [5, 0], [5, 0]],
                                                [[0, 0], [0, 0], [0, 5], [5, 0], [5, 0]],
                                                [[0, 0], [0, 0], [0, 0], [0, 5], [5, 0]],
                                                [[0, 0], [0, 0], [0, 0], [0, 0], [0, 5]]],
                                                dtype = np.float64),
                                      err_msg = "The precedences matrix with the samples unbalanced weighted is not correct.")

        np.testing.assert_array_almost_equal(pair_order_matrix_weighted_2.matrix_,
                                             np.array([[0.5, 1.0, 1.0, 1.0, 1.0],
                                                       [0.0, 0.5, 1.0, 1.0, 1.0],
                                                       [0.0, 0.0, 0.5, 1.0, 1.0],
                                                       [0.0, 0.0, 0.0, 0.5, 1.0],
                                                       [0.0, 0.0, 0.0, 0.0, 0.5]],
                                                       dtype = np.float64),
                                             err_msg = "The pair order matrix with the samples unbalanced weighted is not correct.")


        # Normally weighted
        np.testing.assert_array_equal(pair_order_matrix_weighted_3.precedences_,
                                      np.array([[[0.00, 5.00], [3.50, 0.00], [3.50, 0.00], [2.75, 2.25], [1.25, 1.75]],
                                                [[1.50, 0.00], [0.00, 5.00], [1.25, 2.25], [1.75, 1.00], [1.75, 1.00]],
                                                [[1.50, 0.00], [1.50, 2.25], [0.00, 5.00], [1.75, 0.00], [1.75, 0.00]],
                                                [[0.00, 2.25], [2.25, 1.00], [3.25, 0.00], [0.00, 5.00], [0.25, 2.75]],
                                                [[2.00, 1.75], [2.25, 1.00], [3.25, 0.00], [2.00, 2.75], [0.00, 5.00]]],
                                                dtype = np.float64),
                                      err_msg = "The precedences matrix with the samples normally weighted is not correct.")

        np.testing.assert_array_almost_equal(pair_order_matrix_weighted_3.matrix_,
                                             np.array([[0.500, 0.700, 0.700, 0.775, 0.425],
                                                       [0.300, 0.500, 0.475, 0.450, 0.450],
                                                       [0.300, 0.525, 0.500, 0.350, 0.350],
                                                       [0.225, 0.550, 0.650, 0.500, 0.325],
                                                       [0.575, 0.550, 0.650, 0.675, 0.500]],
                                                       dtype = np.float64),
                                             err_msg = "The pair order matrix with the samples normally weighted is not correct.")

    def test_fit_precedences(self):
        """
            Test the "PairOrderMatrix" object after fitting the model from an already computed precedences matrix.
        """
        np.testing.assert_array_equal(pair_order_matrix_precedences.precedences_,
                                      np.array([[[0, 5], [4, 0], [4, 0], [3, 2], [2, 1]],
                                                [[1, 0], [0, 5], [2, 2], [2, 1], [2, 1]],
                                                [[1, 0], [1, 2], [0, 5], [2, 0], [2, 0]],
                                                [[0, 2], [2, 1], [3, 0], [0, 5], [1, 2]],
                                                [[2, 1], [2, 1], [3, 0], [2, 2], [0, 5]]],
                                                dtype = np.float64),
                                      err_msg = "The precedences matrix for the one given is not correct.")

        np.testing.assert_array_almost_equal(pair_order_matrix_precedences.matrix_,
                                             np.array([[0.5, 0.8, 0.8, 0.8, 0.5],
                                                       [0.2, 0.5, 0.6, 0.5, 0.5],
                                                       [0.2, 0.4, 0.5, 0.4, 0.4],
                                                       [0.2, 0.5, 0.6, 0.5, 0.4],
                                                       [0.5, 0.5, 0.6, 0.6, 0.5]],
                                                       dtype = np.float64),
                                             err_msg = "The pair order matrix for the given precedences matrix is not correct.")

    def test_fit_check_input(self):
        """
            Test the input checking for the "fit" method of the "PairOrderMatrix" class.
        """
        # No way provided
        with pytest.raises(ValueError):
            PairOrderMatrix().fit()

        # Both ways provided
        with pytest.warns(UserWarning):
            PairOrderMatrix().fit(Y = Y_comp, sample_weight = sample_weight_1, precedences = precedences)

        # The bucket orders are not a NumPy array
        with pytest.raises(TypeError):
            PairOrderMatrix().fit(Y = TypeError)

        # The bucket orders are not a 2-D NumPy array
        with pytest.raises(ValueError):
            PairOrderMatrix().fit(Y = Y_comp[0])

        # The sample weight is not a NumPy array
        with pytest.raises(TypeError):
            PairOrderMatrix().fit(Y = Y_comp, sample_weight = TypeError)

        # The sample weight is not a 1-D NumPy array
        with pytest.raises(ValueError):
            PairOrderMatrix().fit(Y = Y_comp, sample_weight = sample_weight_1[None, :])

        # All the weights are zero
        with pytest.raises(ValueError):
            PairOrderMatrix().fit(Y = Y_comp, sample_weight = sample_weight_1 * 0)

        # The bucket orders and the sample weight have not the same length
        with pytest.raises(ValueError):
            PairOrderMatrix().fit(Y = Y_comp, sample_weight = sample_weight_1[:-1])

        # The bucket orders and sample weight cannot be converted to double
        with pytest.raises(ValueError):
            PairOrderMatrix().fit(Y = np.array([["Testing"]]))

        # The precedences matrix is not a NumPy array
        with pytest.raises(TypeError):
            PairOrderMatrix().fit(precedences = TypeError)

        # The precedences matrix is not a NumPy array
        with pytest.raises(TypeError):
            PairOrderMatrix().fit(precedences = TypeError)

        # The precedences matrix is not a 3-D NumPy array
        with pytest.raises(ValueError):
            PairOrderMatrix().fit(precedences = precedences[0])

        # The precedences matrix is not square
        with pytest.raises(ValueError):
            PairOrderMatrix().fit(precedences = precedences[:-1])

# =============================================================================
# Testing for the class UtopianMatrix
# =============================================================================
@pytest.mark.utopianmatrix
class TestUtopianMatrix(object):
    """
        Test the methods of the "UtopianMatrix" class.
    """

    def test_fit(self):
        """
            Test the "UtopianMatrix" object after fitting the model with a "PairOrderMatrix" object.
        """
        np.testing.assert_array_almost_equal(utopian_matrix.matrix_,
                                             np.array([[0.5, 1.0, 1.0, 1.0, 0.5],
                                                       [0.0, 0.5, 0.5, 0.5, 0.5],
                                                       [0.0, 0.5, 0.5, 0.5, 0.5],
                                                       [0.0, 0.5, 0.5, 0.5, 0.5],
                                                       [0.5, 0.5, 0.5, 0.5, 0.5]],
                                                       dtype = np.float64),
                                             err_msg = "The utopian matrix for the given pair order matrix is not correct.")

        np.testing.assert_almost_equal(utopian_matrix.value_,
                                       2.0,
                                       err_msg = "The utopia-value for the given pair order matrix is not correct.")

        np.testing.assert_almost_equal(utopian_matrix.utopicity_,
                                       0.6,
                                       err_msg = "The utopicity for the given pair order matrix is not correct.")

    def test_fit_check_input(self):
        """
            Test the input checking for the "fit" method of the "UtopianMatrix" class.
        """
        # The pair order matrix is not type of "PairOrderMatrix"
        with pytest.raises(TypeError):
            UtopianMatrix().fit(Y_comp)

        # The pair order matrix is not fitted
        with pytest.raises(NotFittedError):
            UtopianMatrix().fit(PairOrderMatrix())

# =============================================================================
# Class for testing AntiUtopianMatrix
# =============================================================================
@pytest.mark.antiutopianmatrix
class TestAntiUtopianMatrix(object):
    """
        Test the methods of the "AntiUtopianMatrix" class.
    """

    def test_fit(self):
        """
            Test the "AntiUtopianMatrix" object after fitting the model with a "PairOrderMatrix" object.
        """
        np.testing.assert_array_almost_equal(anti_utopian_matrix.matrix_,
                                             np.array([[1.0, 0.0, 0.0, 0.0, 1.0],
                                                       [1.0, 1.0, 0.0, 1.0, 1.0],
                                                       [1.0, 1.0, 1.0, 1.0, 1.0],
                                                       [1.0, 1.0, 0.0, 1.0, 1.0],
                                                       [1.0, 1.0, 0.0, 0.0, 1.0]],
                                                       dtype = np.float64),
                                             err_msg = "The anti-utopian matrix for the given pair order matrix is not correct.")

        np.testing.assert_almost_equal(anti_utopian_matrix.value_,
                                       15.1,
                                       err_msg = "The anti-utopia value for the given pair order matrix is not correct.")

    def test_fit_check_input(self):
        """
            Test the input checking for the "fit" method of the "AntiUtopianMatrix" class.
        """
        # The pair order matrix is not type of "PairOrderMatrix"
        with pytest.raises(TypeError):
            AntiUtopianMatrix().fit(Y_comp)

        # The pair order matrix is not fitted
        with pytest.raises(NotFittedError):
            AntiUtopianMatrix().fit(PairOrderMatrix())
