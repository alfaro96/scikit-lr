# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False


# =============================================================================
# Imports
# =============================================================================

# Third party
from cython.operator cimport dereference as deref
from libc.stdlib cimport malloc


# =============================================================================
# Methods
# =============================================================================

cdef void view_to_pointer_INT64_1D(
        INT64_t_1D view, INT64_t **pointer) nogil:
    """Copy the contents of the 1-D integer memory view to the pointer."""
    # Initialize some values from the input arrays
    cdef INT64_t m = view.shape[0]

    # Define the indexes
    cdef SIZE_t i

    # Copy the contents
    for i in range(m):
        deref(pointer)[i] = view[i]


cdef void view_to_pointer_INT64_2D(
        INT64_t_2D view, INT64_t **pointer) nogil:
    """Copy the contents of the 2-D integer memory view to the pointer."""
    # Initialize some values from the input arrays
    cdef INT64_t m = view.shape[0]
    cdef INT64_t n = view.shape[1]

    # Define the indexes
    cdef SIZE_t i
    cdef SIZE_t j

    # Copy the contents
    for i in range(m):
        for j in range(n):
            deref(pointer)[i * n + j] = view[i][j]


cdef void pointer_to_view_INT64_2D(
        INT64_t *pointer, INT64_t_2D view) nogil:
    """Copy the contents of the 2-D integer pointer to the memory view."""
    # Initialize some values from the input arrays
    cdef INT64_t m = view.shape[0]
    cdef INT64_t n = view.shape[1]

    # Define the indexes
    cdef SIZE_t i
    cdef SIZE_t j

    # Copy the contents
    for i in range(m):
        for j in range(n):
            view[i, j] = pointer[i * n + j]
