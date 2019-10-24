# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False


# =============================================================================
# Imports
# =============================================================================

# Local application
from .._types cimport DTYPE_t, DTYPE_t_1D, INT64_t, SIZE_t


# =============================================================================
# Structs
# =============================================================================

# Index tracking structure
cdef struct IndexedElement:
    SIZE_t index
    DTYPE_t value


# =============================================================================
# Methods
# =============================================================================

cdef void argsort_view(DTYPE_t_1D data, SIZE_t **order) nogil


cdef void argsort_pointer(DTYPE_t *data, SIZE_t **order,
                          INT64_t n_values) nogil
