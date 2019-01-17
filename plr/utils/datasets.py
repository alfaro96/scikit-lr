"""
    This module gathers utils to transform datasets.
"""

# =============================================================================
# Imports
# =============================================================================

# NumPy
import numpy as np

# PLR
from ._transformer import ClusterProbabilityTransformer, MissRandomTransformer, TopKTransformer
from .validation   import check_prob_dists, check_random_state, check_X, check_Y, check_Y_prob_dists

# Misc
import re

# =============================================================================
# Public objects
# =============================================================================

# Classes
__all__ = ["ClusterProbability",
           "MissRandom",
           "TopK"]
           
# =============================================================================
# Cluster the probabilities into bucket orders
# =============================================================================
class ClusterProbability:
    """
        Holds the methods needed to transform probability distributions to bucket orders.
    """
    
    def __init__(self,
                 threshold = 0.05,
                 metric    = None):
        """
            Constructor of "ClusterProbability" class.
            
            Parameters
            ----------
                threshold: float, optional (default = 0.05)
                    Threshold value in the clustering algorithm.
            
                metric: object, optional (default = bhattacharyya_distance)
                    Function to decide how close are two probability distributions.
                    
            Returns
            -------
                self
                    Current object initialized.
        """
        # Avoid the circular import
        if isinstance(metric, type(None)):
            # Import
            from ..metrics.probability import bhattacharyya_distance
            # Get such method
            metric = bhattacharyya_distance
        # Initialize the hyperparameters of the current object
        self.threshold = threshold
        self.metric    = metric
        
    def transform(self,
                  prob_dists,
                  check_input = True):
        """
            Transform the probability distributions to bucket orders according to the hyperparameters.
            
            Parameters
            ----------
                prob_dists: np.ndarray
                    Set of probability distributions.

                check_input: boolean (default = True)
                    Allow to bypass several input checking.
                    
            Returns
            -------
                Y: np.ndarray
                    Bucket orders obtained from the given probability distributions.
        """
        # Check the input parameters (if corresponds)
        if check_input:
            prob_dists = check_prob_dists(prob_dists)

        # Initialize the attributes of the current object
        (self.n_samples_, self.n_classes_) = prob_dists.shape

        # Initialize some values to be employed
        matrices        = np.zeros((self.n_samples_, self.n_classes_, self.n_classes_), dtype = np.uint8)
        Y               = np.zeros((self.n_samples_, self.n_classes_),                  dtype = np.float64)
        prev_prob_dists = np.array(prob_dists,                                          dtype = np.float64)
        new_prob_dists  = np.array(prob_dists,                                          dtype = np.float64)

        # Transform the probabilities to bucket orders
        ClusterProbabilityTransformer(threshold = self.threshold,
                                      metric    = self.metric).transform(Y               = Y,
                                                                         prev_prob_dists = prev_prob_dists,
                                                                         new_prob_dists  = new_prob_dists,
                                                                         matrices        = matrices)

        # After transforming, set Y to integer
        Y = Y.astype(np.intp)

        # Return the obtained bucket orders from the probability distributions
        return Y

# =============================================================================
# Randomly miss classes
# =============================================================================
class MissRandom:
    """
        Holds the methods to miss classes in a random way according to the given percentage.
    """
    
    def __init__(self,
                 perc         = 0.0,
                 random_state = None):
        """
           Constructor of "MissRandom" class.

           Parameters
            ----------
                perc: float, optional (default = 0.0)
                    Percentage of missing classes.

                random_state: {None, int, RandomState instance}, optional (default = None)
                    - If int, random_state is the seed used by the random number generator.
                    - If RandomState instance, random_state is the random number generator.
                    - If None, the random number generator is the RandomState instance used
                      by np.random.
        """
        # Initializes the hyperparameters of the current object
        self.perc         = perc
        self.random_state = random_state

    def transform(self,
                  Y,
                  check_input = True):
        """
            Transform the bucket orders according to the given percentage of missing labels.
            
            Parameters
            ----------
                Y: np.ndarray
                    Set of bucket orders.

                check_input: boolean (default = True)
                    Allow to bypass several input checking.
                    
            Returns
            -------
                new_Y: np.ndarray
                    New bucket orders with missing classes.
        """
        # Check the input parameters (if corresponds)
        if check_input:
            Y = check_Y(Y)

        # Initialize the attributes of the current object
        (self.n_samples_, self.n_classes_) = Y.shape

        # Initialize some values to be employed
        new_Y = np.array(Y, dtype = np.float64)

        # Obtain the random state
        random_state = check_random_state(self.random_state)

        # Transform the bucket orders to incomplete ones according to the given percentage
        MissRandomTransformer(perc         = self.perc,
                              random_state = random_state).transform(new_Y)

        # Return the obtained bucket orders with the randomly missed classes
        return new_Y

# =============================================================================
# Top-k
# =============================================================================
class TopK:
    """
        Holds the methods needed to apply a top-k process to bucket orders.
    """

    def __init__(self,
                 perc = 0.0):
        """
            Constructor of "TopK" class.
            
            Parameters
            ----------
                perc: float, optional (default = 0.0)
                    Percentage of missing labels.
                    
            Returns
            -------
                self
                    Current object initialized.
        """
        # Initialize the hyperparameters of the model
        self.perc = perc

    def transform(self,
                  Y,
                  prob_dists  = None,
                  check_input = True):
        """
            Transform the given set of bucket orders to the one after applying the top-k process using the given percentage and
            (if given) the probabily distributions. If the probabilities are not given, just use the bucket orders.
            
            Parameters
            ----------
                Y: np.ndarray
                    Set of bucket orders.

                prob_dists: {None, np.ndarray} (default = None)
                    The probability distributions of the classes for each bucket order.

                check_input: boolean (default = True)
                    Allow to bypass several input checking.

            Returns
            -------
                new_Y: np.ndarray
                    New bucket orders after applying the top-k process.
        """
        # Check the parameters (if corresponds)
        if check_input:
            (Y, prob_dists) = check_Y_prob_dists(Y, prob_dists)

        # Initialize the attributes of the current object
        (self.n_samples_, self.n_classes_) = Y.shape

        # Initialize some values to be employed
        new_Y = np.array(Y, dtype = np.float64)

        # If the probability distributions are given, obtain the top-k classes according to them
        if not isinstance(prob_dists, type(None)):
            # Sort the probability distributions
            sorted_prob_dists = np.argsort(prob_dists)
            # Transform the bucket orders
            TopKTransformer(perc = self.perc).transform_from_probs(Y                 = new_Y,
                                                                   prob_dists        = prob_dists,
                                                                   sorted_prob_dists = sorted_prob_dists)
        # Otherwise, use only the bucket orders
        else:
            # Sort the bucket orders
            sorted_Y = np.argsort(-Y)
            # Transform the bucket orders
            TopKTransformer(perc = self.perc).transform_from_bucket_orders(Y        = new_Y,
                                                                           sorted_Y = sorted_Y)

        # Return the obtained bucket orders after applying top-k
        return new_Y
