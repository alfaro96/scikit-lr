# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False


# =============================================================================
# Imports
# =============================================================================

# Local application
from .._types cimport DTYPE_t_2D, INT64_t, INT64_t_2D, UINT8_t_2D, SIZE_t


# =============================================================================
# Enums
# =============================================================================

# The strategy to miss classes from rankings
ctypedef enum STRATEGY:
    RANDOM
    TOP
