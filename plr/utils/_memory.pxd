# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False


# =============================================================================
# Imports
# =============================================================================

# Local application
from .._types cimport (
    DTYPE_t, DTYPE_t_1D, DTYPE_t_2D,
    INT64_t, INT64_t_1D, INT64_t_2D,
    SIZE_t)


# =============================================================================
# Methods
# =============================================================================

cdef void view_to_pointer_INT64_1D(
        INT64_t_1D view, INT64_t **pointer) nogil


cdef void view_to_pointer_INT64_2D(
        INT64_t_2D view, INT64_t **pointer) nogil


cdef void pointer_to_view_INT64_2D(
        INT64_t *pointer, INT64_t_2D view) nogil
