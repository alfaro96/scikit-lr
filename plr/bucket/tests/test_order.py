"""
    Testing for the "order" module (plr.bucket).
"""

# =============================================================================
# Imports
# =============================================================================

# NumPy
import numpy as np

# PLR
from plr.bucket import BucketOrder

# Pytest
import pytest

# =============================================================================
# Initialization
# =============================================================================

# Seed
seed = 23473

# Initialize the bucket orders
y_comp = np.array([1, 2, 3, 4, 5], dtype = np.float64)
y_mis  = np.array([1, np.nan, np.nan, 2, 3], dtype = np.float64)
y_top  = np.array([1, 2, np.inf, np.inf, np.inf], dtype = np.float64)

# Train the objects
bucket_order_comp = BucketOrder().fit(y_comp)
bucket_order_mis  = BucketOrder().fit(y_mis)
bucket_order_top  = BucketOrder().fit(y_top)

# =============================================================================
# Testing for the class BucketOrder
# =============================================================================
@pytest.mark.bucketorder
class TestBucketOrder(object):
    """
        Test the methods of the "BucketOrder" class.
    """

    def test_fit(self):
        """
            Test the "BucketOrder" objects obtained after fitting the model from the given complete, missed and top-k bucket orders.
        """
        # Complete  
        np.testing.assert_array_equal(bucket_order_comp.precedences_,
                                      np.array([[[0, 1], [1, 0], [1, 0], [1, 0], [1, 0]],
                                                [[0, 0], [0, 1], [1, 0], [1, 0], [1, 0]],
                                                [[0, 0], [0, 0], [0, 1], [1, 0], [1, 0]],
                                                [[0, 0], [0, 0], [0, 0], [0, 1], [1, 0]],
                                                [[0, 0], [0, 0], [0, 0], [0, 0], [0, 1]]], dtype = np.float64),
                                      err_msg = "The precedences for the given complete bucket order is not correct.")

        np.testing.assert_array_equal(bucket_order_comp.matrix_,
                                      np.array([[0.5, 1.0, 1.0, 1.0, 1.0],
                                                [0.0, 0.5, 1.0, 1.0, 1.0],
                                                [0.0, 0.0, 0.5, 1.0, 1.0],
                                                [0.0, 0.0, 0.0, 0.5, 1.0],
                                                [0.0, 0.0, 0.0, 0.0, 0.5]], dtype = np.float64),
                                      err_msg = "The bucket matrix for the given complete bucket order is not correct.")

        np.testing.assert_array_equal(bucket_order_comp.y_,
                                      np.array([1, 2, 3, 4, 5], dtype = np.float64),
                                      err_msg = "The bucket order for the given complete one is not correct.")
            
        # Missing
        np.testing.assert_array_equal(bucket_order_mis.precedences_,
                                      np.array([[[0, 1], [1, 1], [1, 1], [1, 0], [1, 0]],
                                                [[1, 1], [0, 1], [1, 1], [1, 1], [1, 1]],
                                                [[1, 1], [1, 1], [0, 1], [1, 1], [1, 1]],
                                                [[0, 0], [1, 1], [1, 1], [0, 1], [1, 0]],
                                                [[0, 0], [1, 1], [1, 1], [0, 0], [0, 1]]], dtype = np.float64),
                                      "The precedences for the bucket order with missing classes is not correct.")

        np.testing.assert_array_equal(bucket_order_mis.matrix_,
                                      np.array([[0.5, 0.5, 0.5, 1.0, 1.0],
                                                [0.5, 0.5, 0.5, 0.5, 0.5],
                                                [0.5, 0.5, 0.5, 0.5, 0.5],
                                                [0.0, 0.5, 0.5, 0.5, 1.0],
                                                [0.0, 0.5, 0.5, 0.0, 0.5]], dtype = np.float64),
                                      "The bucket matrix for the bucket order with missing classes is not correct.")

        np.testing.assert_allclose(bucket_order_mis.y_,
                                   np.array([1, np.nan, np.nan, 2, 3], dtype = np.float64),
                                   err_msg = "The bucket order for the one with missing classes is not correct.")

        # Top-k
        np.testing.assert_array_equal(bucket_order_top.precedences_,
                                      np.array([[[0, 1], [1, 0], [1, 0], [1, 0], [1, 0]],
                                                [[0, 0], [0, 1], [1, 0], [1, 0], [1, 0]],
                                                [[0, 0], [0, 0], [0, 1], [0, 1], [0, 1]],
                                                [[0, 0], [0, 0], [0, 1], [0, 1], [0, 1]],
                                                [[0, 0], [0, 0], [0, 1], [0, 1], [0, 1]]], dtype = np.float64),
                                      "The precedences for the bucket order applying top-k is not correct.")

        np.testing.assert_array_equal(bucket_order_top.matrix_,
                                      np.array([[0.5, 1.0, 1.0, 1.0, 1.0],
                                                [0.0, 0.5, 1.0, 1.0, 1.0],
                                                [0.0, 0.0, 0.5, 0.5, 0.5],
                                                [0.0, 0.0, 0.5, 0.5, 0.5],
                                                [0.0, 0.0, 0.5, 0.5, 0.5]], dtype = np.float64),
                                      "The bucket matrix for the bucket order applying top-k is not correct.")

        np.testing.assert_array_equal(bucket_order_top.y_,
                                      np.array([1, 2, np.inf, np.inf, np.inf], dtype = np.float64),
                                      err_msg = "The bucket order for the one applying top-k is not correct.")
