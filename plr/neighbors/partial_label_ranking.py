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
from ..bucket.matrix    import PairOrderMatrix
from ..bucket.obop      import OptimalBucketOrderProblem
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

        # Obtain the predictions solving the OBOP
        predictions = np.array([(OptimalBucketOrderProblem(algorithm    = self.bucket,
                                                           beta         = self.beta,
                                                           random_state = self.random_state_)
                                                           .fit(pair_order_matrix = PairOrderMatrix()
                                                                                    .fit(Y             = nearest_buckets[sample],
                                                                                         sample_weight = nearest_weights[sample],
                                                                                         check_input   = False),
                                                                check_input       = False)).y_
                                for sample in range(n_samples)])

        # Return the predictions
        return predictions
    