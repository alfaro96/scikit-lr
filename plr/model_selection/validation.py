"""
    This module gathers methods to validate a model.
"""

# =============================================================================
# Imports
# =============================================================================

# NumPy
import numpy as np

# PLR
from ..utils.validation import check_random_state, check_X_Y

# =============================================================================
# Public objects
# =============================================================================

# Methods
__all__ = ["cross_val_split"]

# =============================================================================
# Global methods
# =============================================================================

def cross_val_split(X,
                    Y,
                    n_repeats    = 5,
                    n_splits     = 10,
                    random_state = None):
    """
        Obtains the indexes to be used in a r x k-cv cross validation method.
        
        Parameters
        ----------
            X: np.ndarray
                The attributes.
                
            Y: np.ndarray:
                The bucket orders.
                
            n_repeats: int, optional (default = 5)
                Number of times that the original dataset is shuffled.
                
            n_splits: int, optional (default = 10)
                Number of folds on each cross validation iteration.
                
            random_state: {None, int, RandomState instance}, optional (default = None)
                    - If int, random_state is the seed used by the random number generator.
                    - If RandomState instance, random_state is the random number generator.
                    - If None, the random number generator is the RandomState instance used
                      by np.random.
                
        Returns
        -------
            indexes: np.ndarray
                The indexes to be used on each step of the cross validation method
    """
    # Check the input parameters
    (X, Y) = check_X_Y(X, Y)

    # Obtain the number of samples and the random generator
    n_samples    = X.shape[0]
    random_state = check_random_state(random_state)
    
    # Obtain the indexes to be used in each shuffle and the splits
    shuffles = np.array([random_state.permutation(n_samples) for _ in np.arange(n_repeats)])
    splits   = np.array(np.array_split(np.arange(n_samples), n_splits))

    # Obtain the indexes of the training and test datasets for each shuffle
    indexes = np.array([np.array([(shuffles[i, np.concatenate(np.r_[splits[:j],
                                                                 splits[j + 1:]])],
                                   shuffles[i, splits[j]])
                                  for j in np.arange(n_splits)])
                        for i in np.arange(n_repeats)])

    # Return the obtained indexes to apply the cross validation
    return indexes
