"""
    Testing for the "datasets" module ("plr.utils").
"""

# =============================================================================
# Imports
# =============================================================================

# NumPy
import numpy as np

# PLR
from plr.metrics import bhattacharyya_distance
from plr.utils   import ClusterProbability, MissRandom, TopK

# Pytest
import pytest

# =============================================================================
# Initialization
# =============================================================================

# Seed
seed = 23473

# Initialize the probability distributions
prob_dists = np.array([[0.065, 0.114, 0.156, 0.065, 0.081, 0.049, 0.032, 0.016, 0.016, 0.049, 0.081, 0.114, 0.016, 0.032, 0.049, 0.065],
                       [0.035, 0.124, 0.196, 0.045, 0.081, 0.049, 0.032, 0.026, 0.026, 0.049, 0.081, 0.124, 0.016, 0.032, 0.049, 0.035]],
                       dtype = np.float64)
    
# Transform the probability distributions to bucket orders using different thresholds
# In fact, in the first one, the metric is implicity given (for the test coverage)
Y_1 = ClusterProbability(threshold = 0.025, metric = bhattacharyya_distance).transform(prob_dists, check_input = False)
Y_2 = ClusterProbability(threshold = 0.05).transform(prob_dists, check_input = False)
Y_3 = ClusterProbability(threshold = 0.1).transform(prob_dists, check_input = False)

# Transform the bucket orders to incomplete ones in a random-way according to different percentages
Y_random_0   = MissRandom(perc = 0.0, random_state = seed).transform(Y_1, check_input = False)
Y_random_30  = MissRandom(perc = 0.3, random_state = seed).transform(Y_1, check_input = False)
Y_random_60  = MissRandom(perc = 0.6, random_state = seed).transform(Y_1, check_input = False)
Y_random_100 = MissRandom(perc = 1.0, random_state = seed).transform(Y_1, check_input = False)

# Transform the bucket orders using top-k according to different percentages

# Using the probabilities
Y_top_k_probs_0   = TopK(perc = 0.0).transform(Y_1, prob_dists, check_input = False)
Y_top_k_probs_30  = TopK(perc = 0.3).transform(Y_1, prob_dists, check_input = False)
Y_top_k_probs_60  = TopK(perc = 0.6).transform(Y_1, prob_dists, check_input = False)
Y_top_k_probs_100 = TopK(perc = 1.0).transform(Y_1, prob_dists, check_input = False)

# Using the bucket orders
Y_top_k_bucket_orders_0   = TopK(perc = 0.0).transform(Y_1, check_input = False)
Y_top_k_bucket_orders_30  = TopK(perc = 0.3).transform(Y_1, check_input = False)
Y_top_k_bucket_orders_60  = TopK(perc = 0.6).transform(Y_1, check_input = False)
Y_top_k_bucket_orders_100 = TopK(perc = 1.0).transform(Y_1, check_input = False)

# =============================================================================
# Class for testing ClusterProbability
# =============================================================================
@pytest.mark.cluster
class TestClusterProbability(object):
    """
        Test the methods of the "ClusterProbability" class.
    """

    def test_transform(self):
        """
            Test that the bucket orders obtained after transforming the the probability distributions are correct.
        """
        # Threshold of 0.025
        np.testing.assert_array_equal(Y_1,
                                      np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                                [3, 2, 1, 3, 2, 3, 3, 3, 3, 3, 2, 2, 3, 3, 3, 3]],
                                                dtype = np.intp),
                                      err_msg = "The bucket orders obtained after transforming the probability distributions using, as threshold, 0.025 are not correct.")
    
        # Threshold of 0.05
        np.testing.assert_array_equal(Y_2,
                                      np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
                                                dtype = np.intp),
                                      err_msg = "The bucket orders obtained after transforming the probability distributions using, as threshold, 0.05 are not correct.")
        # Threshold of 0.1
        np.testing.assert_array_equal(Y_3,
                                      np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
                                                dtype = np.intp),
                                      err_msg = "The bucket orders obtained after transforming the probability distributions using, as threshold, 0.1 are not correct.")

    def test_transform_check_input(self):
        """
            Test the input checking for the "transform" method of the "ClusterProbability" class.
        """
        # The probability distributions are not a NumPy array
        with pytest.raises(TypeError):
            ClusterProbability().transform(prob_dists = None)

        # The probability distributions are not a 2-D NumPy array
        with pytest.raises(ValueError):
            ClusterProbability().transform(prob_dists = prob_dists[0])

        # The probability distributions do not sum one
        with pytest.raises(ValueError):
            ClusterProbability().transform(prob_dists = prob_dists * 2)

# =============================================================================
# Class for testing MissRandom
# =============================================================================
@pytest.mark.missrandom
class TestMissRandom(object):
    """
        Test the methods of the "MissRandom" class.
    """

    def test_transform(self):
        """
            Test that the bucket orders obtained after the transform missing the classes in a random way are correct.
        """
        # First transform
        np.testing.assert_array_equal(Y_random_0,
                                      np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                                [3, 2, 1, 3, 2, 3, 3, 3, 3, 3, 2, 2, 3, 3, 3, 3]],
                                                dtype = np.float64),
                                      err_msg = "The bucket orders obtained missing labels in a random-way with probability 0.0 are not correct.")
    
        # Second transform
        np.testing.assert_array_equal(Y_random_30,
                                      np.array([[np.nan, 1, 1, 1, np.nan, 1, np.nan, np.nan, 1, 1, np.nan, np.nan, 1, np.nan, 1, np.nan],
                                                [3, 2, 1, np.nan, np.nan, np.nan, 3, 3, 3, 3, 2, np.nan, np.nan, np.nan, 3, 3]],
                                                dtype = np.float64),
                                      err_msg = "The bucket orders obtained missing labels in a random-way with probability 0.3 are not correct.")
        # Third transform
        np.testing.assert_array_equal(Y_random_60,
                                      np.array([[np.nan, 1, 1, np.nan, np.nan, 1, np.nan, np.nan, 1, 1, np.nan, np.nan, np.nan, np.nan, 1, np.nan],
                                                [np.nan, 1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 2, np.nan]],
                                                dtype = np.float64),
                                      err_msg = "The bucket orders obtained missing labels in a random-way with probability 0.6 are not correct.")

        # Fourth transform
        np.testing.assert_array_equal(Y_random_100,
                                      np.array([[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1, 1],
                                                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1, 1]],
                                                dtype = np.float64),
                                      err_msg = "The bucket orders obtained missing labels in a random-way with probability 1.0 are not correct.")

    def test_transform_check_input(self):
        """
            Test the input checking for the "transform" method of the "MissRandom" class.
        """
        # The bucket orders are not a NumPy array
        with pytest.raises(TypeError):
            MissRandom().transform(Y = None)

        # The bucket orders are not a 2-D NumPy array
        with pytest.raises(ValueError):
            MissRandom().transform(Y = Y_random_0[0])
        
        # The bucket orders cannot be converted to double
        with pytest.raises(ValueError):
            MissRandom().transform(Y = np.array([["Testing"], ["Coverage"]]))

# =============================================================================
# Class for testing TopK
# =============================================================================
@pytest.mark.topk
class TestTopK(object):
    """
        Test the methods of the "TopK" class.
    """

    def test_transform_probs(self):
        """
            Test that the bucket orders obtained after the transform applying the top-k process using the probability distributions are correct.
        """
        # First transform
        np.testing.assert_array_equal(Y_top_k_probs_0,
                                      np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                                [3, 2, 1, 3, 2, 3, 3, 3, 3, 3, 2, 2, 3, 3, 3, 3]],
                                                dtype = np.float64),
                                      err_msg = "The bucket orders obtained applying top-k with probability 0.0 using the probabilities are not correct.")
    
        # Second transform
        np.testing.assert_array_equal(Y_top_k_probs_30,
                                      np.array([[1, 1, 1, 1, 1, 1, np.inf, np.inf, np.inf, 1, 1, 1, np.inf, np.inf, 1, 1],
                                                [3, 2, 1, 3, 2, 3, np.inf, np.inf, np.inf, 3, 2, 2, np.inf, np.inf, 3, 3]],
                                                dtype = np.float64),
                                      err_msg = "The bucket orders obtained applying top-k with probability 0.3 using the probabilities are not correct.")
        # Third transform
        np.testing.assert_array_equal(Y_top_k_probs_60,
                                      np.array([[np.inf, 1, 1, np.inf, 1, np.inf, np.inf, np.inf, np.inf, np.inf, 1, 1, np.inf, np.inf, np.inf, np.inf],
                                                [np.inf, 2, 1, np.inf, 2, np.inf, np.inf, np.inf, np.inf, np.inf, 2, 2, np.inf, np.inf, np.inf, np.inf]],
                                                dtype = np.float64),
                                      err_msg = "The bucket orders obtained applying top-k with probability 0.6 using the probabilities are not correct.")

        # Fourth transform
        np.testing.assert_array_equal(Y_top_k_probs_100,
                                      np.array([[np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
                                                [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]],
                                                dtype = np.float64),
                                      err_msg = "The bucket orders obtained applying top-k with probability 1.0 using the probabilities are not correct.")

    def test_transform_bucket_orders(self):
        """
            Test that the bucket orders obtained after the transform applying the top-k process using only the rankings are correct.
        """
        # First transform
        np.testing.assert_array_equal(Y_top_k_bucket_orders_0,
                                      np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                                [3, 2, 1, 3, 2, 3, 3, 3, 3, 3, 2, 2, 3, 3, 3, 3]],
                                                dtype = np.float64),
                                      err_msg = "The bucket orders obtained applying top-k with probability 0.0 using the bucket orders are not correct.")
    
        # Second transform
        np.testing.assert_array_equal(Y_top_k_bucket_orders_30,
                                      np.array([[np.inf, np.inf, np.inf, np.inf, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                                [np.inf, 2, 1, np.inf, 2, np.inf, np.inf, 3, 3, 3, 2, 2, 3, 3, 3, 3]],
                                                dtype = np.float64),
                                      err_msg = "The bucket orders obtained applying top-k with probability 0.3 using the bucket orders are not correct.")
        # Third transform
        np.testing.assert_array_equal(Y_top_k_bucket_orders_60,
                                      np.array([[np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1, 1, 1, 1, 1, 1, 1],
                                                [np.inf, 2, 1, np.inf, 2, np.inf, np.inf, np.inf, np.inf, np.inf, 2, 2, np.inf, np.inf, 3, 3]],
                                                dtype = np.float64),
                                      err_msg = "The bucket orders obtained after applying top-k with a 0.6 percentage of missing labels using the bucket orders are not correct.")

        # Fourth transform
        np.testing.assert_array_equal(Y_top_k_bucket_orders_100,
                                      np.array([[np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
                                                [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]],
                                                dtype = np.float64),
                                      err_msg = "The bucket orders obtained after applying top-k with a 1.0 percentage of missing labels using the rankings are not correct.")

    def test_transform_check_input(self):
        """
            Test the input checking for the "transform" method of the "TopK" class.
        """
        # The bucket orders are not a NumPy array
        with pytest.raises(TypeError):
            TopK().transform(Y = None)

        # The bucket orders are not a 2-D NumPy array
        with pytest.raises(ValueError):
            TopK().transform(Y = Y_random_0[0])
        
        # The bucket orders cannot be converted to double
        with pytest.raises(ValueError):
            TopK().transform(Y = np.array([["Testing"], ["Coverage"]]))

        # The probability distributions are not a NumPy array
        with pytest.raises(TypeError):
            TopK().transform(Y = Y_random_0, prob_dists = TypeError)

        # The probability distributions are not a 2-D NumPy array
        with pytest.raises(ValueError):
            TopK().transform(Y = Y_random_0, prob_dists = prob_dists[0])

        # The probability distributions do not sum up one
        with pytest.raises(ValueError):
            TopK().transform(Y = Y_random_0, prob_dists = prob_dists * 2)

        # The bucket orders and the probability distributions have not the same length
        with pytest.raises(ValueError):
            TopK().transform(Y = Y_random_0, prob_dists = prob_dists[:1])

        # The bucket orders and the probability distributions cannot be converted to double
        with pytest.raises(ValueError):
            TopK().transform(Y = np.array([["Testing"], ["Coverage"]]), prob_dists = prob_dists)
