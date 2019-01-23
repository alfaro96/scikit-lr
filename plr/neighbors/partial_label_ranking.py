"""
    This module gathers all the estimators to solve
    the Partial Label Ranking Problem with the nearest-neighbors paradigm.
"""

# =============================================================================
# Imports
# =============================================================================

# NumPy
import numpy as np

# PLR
from .base              import BaseKNeighborsPartialLabelRanker
from ..utils.validation import check_is_fitted, check_n_features, check_X, check_X_Y_sample_weight

# =============================================================================
# Public objects
# =============================================================================

# Classes
__all__ = ["KNeighborsPartialLabelRanker"]

# =============================================================================
# K-nearest neighbors
# =============================================================================
class KNeighborsPartialLabelRanker(BaseKNeighborsPartialLabelRanker):
    """
        Class to hold the methods needed to solve Partial Label Ranking Problem with the k-nearest neighbors paradigm.
    """

    def __init__(self,       
                 n_neighbors  = 5,
                 algorithm    = "kd_tree",
                 bucket       = "bpa_lia_mp2",
                 beta         = 0.25,
                 p            = 2,
                 leaf_size    = 30,
                 random_state = None):
        """
            Constructor of KNearestNeighborsPartialLabelRanker class.
            
            Parameters
            ----------
                n_neighbors: int, optional (default = 5)
                    Number of neighbors to use.
                       
                algorithm: string, optional (default = "kd_tree")
                    Algorithm used to compute the nearest neighbors:
                        - "brute" will use a brute-force search.
                        - "kd_tree" will use a KD-Tree.

                bucket: string, optional (default = "bpa_lia_mp2")
                    Algorithm that will be applied to the OBOP problem, i.e., the algorithm employed
                    to aggregate the bucket orders.

                beta: float, optional (default = 0.25)
                    The parameter to decide the precedence relation of each item w.r.t. the pivot.
            
                p: int, optional (default = 2)
                    Power parameter for the Minkowski metric.
                    When p = 1, this is equivalent to using Manhattan distance,
                    and Euclidean distance for p = 2. 
                    For arbitrary p, Minkowski distance is used.

                leaf_size: int, optional (default = 30)
                    Leaf size passed to KDTree. This can affect the speed of the construction and query,
                    as well as the memory required to store the tree. The optimal value depends on the nature of the problem.

                random_state: {None, int, RandomState instance}, optional (default = None)
                    - If int, random_state is the seed used by the random number generator.
                    - If RandomState instance, random_state is the random number generator.
                    - If None, the random number generator is the RandomState instance used
                      by np.random.
        
            Returns
            -------
                self: KNeighborsPartialLabelRanker
                    Current object initialized.
        """
        # Call to the constructor of the parent
        super().__init__(n_neighbors  = n_neighbors,
                         algorithm    = algorithm,
                         builder      = "knn",
                         bucket       = bucket,
                         beta         = beta,
                         p            = p,
                         leaf_size    = leaf_size,
                         random_state = random_state)

    def predict(self,
                X,
                Y             = None,
                sample_weight = None,
                check_input   = True):
        """
            Predict the bucket orders for the provided data.
            
            Parameters
            ----------
                X: np.ndarray
                    Dataset with the attributes.
            
                Y: {None, np.ndarray}, optional (default = None)
                    Rankings already sorted by distance.

                sample_weight: {None, np.ndarray}, optional (default = None)
                    Weights already sorted by distance.
                
                check_input: boolean (default = True)
                    Allow to bypass several input checking.
                    
            Returns
            -------
                predictions: np.ndarray
                    Predicted bucket orders for each instance.
        """
        # Check if the model is fitted
        check_is_fitted(self, "tree_")

        # Check the input parameters (if corresponds)
        if check_input:
            X = check_X(X)
            check_n_features(X.shape[1], self.n_features_)

        # Initialize some values from the input arrays
        n_samples = X.shape[0]

        # Initialize the predictions
        predictions = np.zeros((n_samples, self.n_classes_), dtype = np.intp)

        # If the bucket orders or sample weight are not provided, compute the distances and get the nearest neighbors
        if isinstance(Y, type(None)) or isinstance(sample_weight, type(None)):
            # Obtain the indexes of the k-nearest neighbors and, indexing, obtain the bucket orders and sample weight
            indexes         = self.knearestneighbors(X)
            nearest_buckets = self.Y_[indexes]
            nearest_weights = self.sample_weight_[indexes]
        # Otherwise, the bucket orders and sample weight are probably given (if the validation tests are passed)
        else:
            # Check the input parameters (if corresponds)
            if check_input:
                (_, Y, sample_weight) = check_X_Y_sample_weight(X, Y, sample_weight, Y_ndim = 3, sample_weight_ndim = 2)

            # Obtain the k-nearest neighbors, just indexing
            nearest_buckets = Y[:, :self.n_neighbors]
            nearest_weights = sample_weight[:, :self.n_neighbors]

        # Obtain the predictions solving the OBOP and using the corresponding builder
        self.builder_.predict(Y             = nearest_buckets,
                              sample_weight = nearest_weights,
                              predictions   = predictions)

        # Return the obtained predictions
        return predictions
    