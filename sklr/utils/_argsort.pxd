# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False


# =============================================================================
# Imports
# =============================================================================

# Local application
from .._types cimport BOOL_t, DTYPE_t, DTYPE_t_1D, INT64_t, SIZE_t


# =============================================================================
# Structs
# =============================================================================

# Index tracking data structure to know
# the index and the value of a pointer
cdef struct Indexed:
    SIZE_t index
    DTYPE_t value


# =============================================================================
# Methods
# =============================================================================

cdef SIZE_t* argsort(DTYPE_t *data, INT64_t n_values) nogil
