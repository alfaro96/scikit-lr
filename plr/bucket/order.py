"""
    This module gathers the object to store
    the information about a bucket order.
"""

# =============================================================================
# Imports
# =============================================================================

# Numpy
import numpy as np

# PLR
from .matrix import PairOrderMatrix

# =============================================================================
# Public objects
# =============================================================================

# Classes
__all__ = ["BucketOrder"]

# =============================================================================
# Bucket order
# =============================================================================
class BucketOrder(PairOrderMatrix):
    """
        Store the information of a bucket order.
        It can be seen as a particular case of a pair order matrix with only one sample.
    """
    
    def fit(self,
            y,
            check_input = True):
        """
            Fit the model to obtain the precedences matrix and bucket matrix for the given bucket order.
            
            Parameters
            ----------
                y: np.ndarray
                    The bucket order.

                check_input: boolean (default = True)
                    Allow to bypass several input checking.
        
            Returns
            -------
                self: BucketOrder
                    Current object already trained.
        """     
        # Call to the fit object of the parent, checking the input (if corresponds)
        # Notice that "y[None, :]", has the same effect that "np.array([y])" but, in fact,
        # the first method is more efficcient, since instead of creating a new array,
        # a new view of the same one is returned
        super().fit(Y = y[None, :], sample_weight = np.array([1.0]), check_input = check_input)

        # Initialize the attributes of the current object    
        self.y_ = y

        # Return the current object already trained
        return self
