"""
    Testing for the "validation" module (plr.model_selection).
"""

# =============================================================================
# Imports
# =============================================================================

# NumPy
import numpy as np

# PLR
from plr.model_selection import cross_val_split

# Pytest
import pytest

# =============================================================================
# Initialization
# =============================================================================

# Seed
seed = 23473

# Random state
random_state = np.random.RandomState(seed)

# Initialize a normally distributed dataset (10000 samples with 10 attributes and 5 labels per bucket order)
X = random_state.standard_normal(size = (10000, 10))
Y = random_state.standard_normal(size = (10000, 5))

# Obtain the indexes for a 2x5cv method
indexes = cross_val_split(X            = X,
                          Y            = Y,
                          n_repeats    = 2,
                          n_splits     = 5,
                          random_state = seed)

# =============================================================================
# Testing methods
# =============================================================================

@pytest.mark.cross_val_split
def test_cross_val_split():
    """
        Test the splits obtained to apply a cross-validation method.
    """
    # Shape
    np.testing.assert_equal(indexes.shape,
                            (2, 5, 2),
                            err_msg = "The shape of the indexes obtained for the given cross-validation method is not correct.")

    # Number of samples
    np.testing.assert_equal(np.all(np.array([indexes[repeat][fold][0].shape[0] + indexes[repeat][fold][1].shape[0]
                                             for repeat in range(indexes.shape[0])
                                             for fold   in range(indexes.shape[1])]) == 10000),
                            True,
                            err_msg = "The length of the indexes obtained for the given cross-validation method is not correct.")

    # Different indexes in train and test
    np.testing.assert_equal(np.all(np.array([np.sum(np.intersect1d(indexes[repeat][fold][0], indexes[repeat][fold][1]))
                                             for repeat in range(indexes.shape[0])
                                             for fold   in range(indexes.shape[1])]) == 0),
                            True,
                            err_msg = "There are common indexes in the train and test folds.")

def test_X_Y_check_input():
    """
        Test the input checking for the attributes and bucket orders.
    """
    # The attributes are not a NumPy array
    with pytest.raises(TypeError):
        cross_val_split(X = None, Y = None)

    # The attributes are not a 2-D NumPy array
    with pytest.raises(ValueError):
        cross_val_split(X = X[0], Y = None)
    
    # The bucket orders are not a NumPy array
    with pytest.raises(TypeError):
        cross_val_split(X = X, Y = None)

    # The bucket orders are not a 2-D NumPy array
    with pytest.raises(ValueError):
        cross_val_split(X = X, Y = Y[0])

    # The attributes and bucket orders have not the same length
    with pytest.raises(ValueError):
        cross_val_split(X = X, Y = Y[:-1])

    # The attributes and bucket orders cannot be converted to double
    with pytest.raises(ValueError):
        cross_val_split(X = np.array([["Testing"]]), Y = np.array([["Coverage"]]))
