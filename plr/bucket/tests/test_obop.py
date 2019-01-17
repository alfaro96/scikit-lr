"""
    Testing for the "obop" module (plr.bucket).
"""

# =============================================================================
# Imports
# =============================================================================

# NumPy
import numpy as np

# PLR
from plr.bucket     import PairOrderMatrix, OptimalBucketOrderProblem
from plr.exceptions import NotFittedError

# Pytest
import pytest

# =============================================================================
# Initialization
# =============================================================================

# Seed
seed = 23473

# Initialize the bucket orders and the pair order matrix
Y                 = np.array([[1, 2, 3, 4, 5], [4, 2, 1, 5, 3], [1, 2, 2, 1, 1], [2, 3, 3, 2, 1], [1, 2, 3, 2, 2]], dtype = np.float64)
pair_order_matrix = PairOrderMatrix().fit(Y)

# Initialize the objects employed to solve the OBOP

# Bucket Pivot Algorithm with a single pivot
obop_bpa_original_sg = OptimalBucketOrderProblem(algorithm    = "bpa_original_sg",
                                                 random_state = seed).fit(pair_order_matrix = pair_order_matrix,
                                                                          check_input       = False)
                                        
# Bucket Pivot Algorithm with multiple pivots
obop_bpa_original_mp = OptimalBucketOrderProblem(algorithm    = "bpa_original_mp",
                                                 random_state = seed).fit(pair_order_matrix = pair_order_matrix,
                                                                          check_input       = False)

# Bucket Pivot Algorithm with multiple pivots and two-stages
obop_bpa_original_mp2 = OptimalBucketOrderProblem(algorithm    = "bpa_original_mp2",
                                                  random_state = seed).fit(pair_order_matrix = pair_order_matrix,
                                                                           check_input       = False)

# Bucket Pivot Algorithm with least-indecision assumption and single pivot
obop_bpa_lia_sg = OptimalBucketOrderProblem(algorithm    = "bpa_lia_sg",
                                            random_state = seed).fit(pair_order_matrix = pair_order_matrix,
                                                                     check_input       = False)

# Bucket Pivot Algorithm with least-indecision assumption and multiple pivots
obop_bpa_lia_mp = OptimalBucketOrderProblem(algorithm    = "bpa_lia_mp",
                                            random_state = seed).fit(pair_order_matrix = pair_order_matrix,
                                                                     check_input       = False)

# Bucket Pivot Algorithm with least-indecision assumption, multiple pivots and two-stages
obop_bpa_lia_mp2 = OptimalBucketOrderProblem(algorithm    = "bpa_lia_mp2",
                                             random_state = seed).fit(pair_order_matrix = pair_order_matrix,
                                                                      check_input       = False)

# =============================================================================
# Testing for the class OptimalBucketOrderProblem
# =============================================================================
@pytest.mark.obop
class TestOptimalBucketOrderProblem(object):
    """
        Test the methods of the "OptimalBucketOrderProblem" class.
    """

    def test_fit_bpa_original_sg(self):
        """
            Test the Bucket Pivot algorithm with a single pivot.
        """
        np.testing.assert_array_equal(obop_bpa_original_sg.y_,
                                      np.array([1, 2, 2, 2, 2], dtype = np.intp),
                                      err_msg = "The bucket order obtained from the given pair order matrix solving the OBOP using the BPA with single pivot is not correct.")

    def test_fit_bpa_original_mp(self):
        """
            Test the Bucket Pivot algorithm with multiple pivots.
        """
        np.testing.assert_array_equal(obop_bpa_original_mp.y_,
                                      np.array([1, 2, 2, 2, 2], dtype = np.intp),
                                      err_msg = "The bucket order obtained from the given pair order matrix solving the OBOP using the BPA with multiple pivots is not correct.")

    def test_fit_bpa_original_mp2(self):
        """
            Test the Bucket Pivot algorithm with multiple pivots and two-stages.
        """
        np.testing.assert_array_equal(obop_bpa_original_mp2.y_,
                                      np.array([1, 1, 1, 1, 1], dtype = np.intp),
                                      err_msg = "The bucket order obtained from the given pair order matrix solving the OBOP using the BPA with multiple pivots and two-stages is not correct.")

    def test_fit_bpa_lia_sg(self):
        """
            Test the Bucket Pivot algorithm with least-indecision assumption (LIA) and a single pivot.
        """
        np.testing.assert_array_equal(obop_bpa_lia_sg.y_,
                                      np.array([1, 1, 1, 1, 1], dtype = np.intp),
                                      err_msg = "The bucket order obtained from the given pair order matrix solving the OBOP using the BPA-LIA with single pivot is not correct.")

    def test_fit_bpa_lia_mp(self):
        """
            Test the Bucket Pivot algorithm with least-indecision assumption (LIA) and multiple pivots.
        """
        np.testing.assert_array_equal(obop_bpa_lia_mp.y_,
                                      np.array([1, 1, 1, 1, 1], dtype = np.intp),
                                      err_msg = "The bucket order obtained from the given pair order matrix solving the OBOP using the BPA-LIA with multiple pivots is not correct.")

    def test_fit_bpa_lia_mp2(self):
        """
            Test the Bucket Pivot algorithm with least-indecision assumption (LIA), multiple pivots and a two-stages.
        """
        np.testing.assert_array_equal(obop_bpa_lia_mp2.y_,
                                      np.array([1, 1, 1, 1, 1], dtype = np.intp),
                                      err_msg = "The bucket order obtained from the given pair order matrix solving the OBOP using the BPA-LIA with multiple pivots is not correct.")

    def test_fit_check_input(self):
        """
            Test the input checking for the "fit" method of the "OptimalBucketOrderProblem" class.
        """
        # The pair order matrix is not type of "PairOrderMatrix"
        with pytest.raises(TypeError):
            OptimalBucketOrderProblem().fit(Y)

        # The pair order matrix is not fitted
        with pytest.raises(NotFittedError):
            OptimalBucketOrderProblem().fit(PairOrderMatrix())
