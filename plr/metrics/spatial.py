"""
    This module gathers metrics for spatial data.
"""

# =============================================================================
# Imports
# =============================================================================

# NumPy
import numpy as np

# PLR
from ._calculator       import minkowski_calculator
from ..utils.validation import check_arrays

# =============================================================================
# Public objects
# =============================================================================

# Methods
__all__ = ["minkowski"]

# =============================================================================
# Global methods
# =============================================================================

def minkowski(u,
              v,
              p = 2,
              check_input = True):
    """
        Computes the Minkowski distance between two 1-D arrays of spatial data.

        Parameters
        ----------
            u: np.ndarray
                Input array.
                
            v: np.ndarray
                Input array.

            p: int, optional (default = 2)
                The order of the norm of the difference.

            check_input: boolean (default = True)
                    Allow to bypass several input checking.
                
        Returns
        -------
            distance: double
                The Minkowski distance between the vectors 'u' and 'v'.
    """
    # Check the input parameters (if corresponds)
    if check_input:
        # Check the input arrays
        check_arrays(u, v)
        
        # Check the power parameter
        if not isinstance(p, int):
            raise ValueError("The data type of the power parameter must be 'int', got '{}'".format(type(p).__name__))

    # Obtain the distance
    distance = minkowski_calculator(u = u, v = v, p = p)

    # Return the obtained Minkowski distance between the arrays
    return distance
