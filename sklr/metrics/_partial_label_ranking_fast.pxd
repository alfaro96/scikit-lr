# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False


# =============================================================================
# Imports
# =============================================================================

# Local application
from .._types cimport (
    BOOL_t,
    DTYPE_t, DTYPE_t_1D,
    INT64_t, INT64_t_1D, INT64_t_2D,
    SIZE_t)


# =============================================================================
# Methods
# =============================================================================

cdef DTYPE_t _penalized_kendall_distance_fast(INT64_t_1D y_true,
                                              INT64_t_1D y_pred,
                                              BOOL_t normalize) nogil
