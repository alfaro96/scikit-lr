# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False


# =============================================================================
# Methods
# =============================================================================

cdef void copy_pointer_INT64_1D(INT64_t *input_pointer,
                                INT64_t *output_pointer,
                                INT64_t m) nogil:
    """Copy the contents of a 1-D pointer to another one."""
    cdef SIZE_t i

    for i in range(m):
        output_pointer[i] = input_pointer[i]


cdef void copy_pointer_INT64_2D(INT64_t *input_pointer,
                                INT64_t *output_pointer,
                                INT64_t m, INT64_t n) nogil:
    """Copy the contents of a 2-D pointer to another one."""
    cdef SIZE_t i
    cdef SIZE_t j

    for i in range(m):
        for j in range(n):
            output_pointer[i * n + j] = input_pointer[i * n + j]
