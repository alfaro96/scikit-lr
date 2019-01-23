"""
    Testing for the "spatial" module (plr.metrics).
"""

# =============================================================================
# Imports
# =============================================================================

# NumPy
import numpy as np

# PLR
from plr.metrics import minkowski

# Pytest
import pytest

# =============================================================================
# Initialization
# =============================================================================

# Seed
seed = 23473

# Initialize the arrays
u = np.array([0.0, 3.0, 4.0, 5.0, 3.0], dtype = np.float64)
v = np.array([1.0, 2.0, 2.0, 3.0, 3.0], dtype = np.float64)

# Obtain several Minkowski distances
manhattan = minkowski(u = u, v = v, p = 1)
euclidean = minkowski(u = u, v = v, p = 2, check_input = False)
arbitrary = minkowski(u = u, v = v, p = 4, check_input = False)

# =============================================================================
# Testing methods
# =============================================================================

@pytest.mark.minkowski
def test_minkowski():
    """
        Test the Minkdowski distance for different power parameters.
    """
    # Manhattan
    np.testing.assert_almost_equal(manhattan,
                                   6.0,
                                   err_msg = "The Minkdowski distance for the given arrays with a power parameter of one is not correct.")

    # Euclidean
    np.testing.assert_almost_equal(euclidean,
                                   3.1622776602,
                                   err_msg = "The Minkdowski distance for the given arrays with a power parameter of two is not correct.")

    # Arbitrary
    np.testing.assert_almost_equal(arbitrary,
                                   2.4147364028,
                                   err_msg = "The Minkdowski distance for the given arrays with a power parameter of four is not correct.")

def test_u_v_check_input():
    """
        Test the input checking for the input arrays.
    """
    # The arrays are not NumPy arrays
    with pytest.raises(TypeError):
        minkowski(u = None, v = None)
    with pytest.raises(TypeError):
        minkowski(u = u, v = None)

    # The arrays are not of the corresponding dimension
    with pytest.raises(ValueError):
        minkowski(u = u[None, :], v = None)
    with pytest.raises(ValueError):
        minkowski(u = u, v = v[None, :])

    # The arrays do not have the sample length
    with pytest.raises(ValueError):
        minkowski(u = u[:-1], v = v)

    # The arrays cannot be cast to double
    with pytest.raises(ValueError):
        minkowski(u = np.array(["Testing"]), v = np.array(["Coverage"]))

    # The power parameter is not an integer
    with pytest.raises(ValueError):
        minkowski(u = u, v = v, p = None)
