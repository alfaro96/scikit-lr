# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False


# =============================================================================
# Imports
# =============================================================================

# Third party
from libc.math cimport isnan
from libc.stdlib cimport calloc, const_void, free, qsort


# =============================================================================
# Methods
# =============================================================================

cdef int _compare(const_void *a, const_void *b) nogil:
    """Compare function for sorting the indexes."""
    cdef DTYPE_t comparison

    cdef SIZE_t a_index = (<Indexed*> a).index
    cdef SIZE_t b_index = (<Indexed*> b).index
    cdef DTYPE_t a_value = (<Indexed*> a).value
    cdef DTYPE_t b_value = (<Indexed*> b).value

    cdef BOOL_t is_nan_a = isnan(a_value)
    cdef BOOL_t is_nan_b = isnan(b_value)

    # NaN values should be sorted after positive
    # infinity and, in case that both values are
    # NaN, sort first the one with lowest index
    # (and, therefore, randomness is mantained)
    if is_nan_a and is_nan_b:
        comparison = a_index - b_index
    elif is_nan_a and not is_nan_b:
        comparison = 1
    elif not is_nan_a and is_nan_b:
        comparison = -1
    # As before, in case that both finite values are
    # the same, sort first the one with lowest index
    else:
        if a_value == b_value:
            comparison = a_index - b_index
        else:
            comparison = a_value - b_value

    return -1 if comparison <= 0 else 1


cdef SIZE_t* argsort(DTYPE_t *data, INT64_t n_values) nogil:
    """Compute the indexes that would sort the array."""
    cdef Indexed *indexed = <Indexed*> calloc(n_values, sizeof(Indexed))
    cdef SIZE_t *order = <SIZE_t*> calloc(n_values, sizeof(SIZE_t))

    cdef SIZE_t index

    # Copy the indexes and the values from
    # the input data to the index tracking
    # array to know the sorted indexes
    for index in range(n_values):
        indexed[index].index = index
        indexed[index].value = data[index]

    qsort(indexed, n_values, sizeof(Indexed), _compare)

    # Copy the already sorted indexes from the
    # index tracking array to the output pointer
    for index in range(n_values):
        order[index] = indexed[index].index

    free(indexed)

    return order
