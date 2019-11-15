# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False


# =============================================================================
# Imports
# =============================================================================

# Local application
from .._types cimport INT64_t, SIZE_t


# =============================================================================
# Methods
# =============================================================================

cdef void copy_pointer_INT64_1D(INT64_t *input_pointer,
                                INT64_t *output_pointer,
                                INT64_t m) nogil


cdef void copy_pointer_INT64_2D(INT64_t *input_pointer,
                                INT64_t *output_pointer,
                                INT64_t m, INT64_t n) nogil
