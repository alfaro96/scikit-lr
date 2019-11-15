# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False


# =============================================================================
# Imports
# =============================================================================

# Third party
from libc.math cimport isnan
from libc.stdlib cimport const_void, free, qsort, malloc


# =============================================================================
# Methods
# =============================================================================

cdef int _compare(const_void *a, const_void *b) nogil:
    """Compare function for sorting the indexes."""
    # Define some values to be employed
    cdef DTYPE_t v

    # Compare the values, taking into account that the NaN values
    # must be always put in the last positions and, when both values
    # to be compared are the same (including the case where both
    # values are NaN), the indexes must be used for sorting. By this
    # way, it is possible to ensure the reproducibility of the experiments
    if (isnan((<IndexedElement*> a).value) and
            not isnan((<IndexedElement*> b).value)):
        v = 1
    elif (not isnan((<IndexedElement*> a).value) and
            isnan((<IndexedElement*> b).value)):
        v = -1
    elif (isnan((<IndexedElement*> a).value) and
            isnan((<IndexedElement*> b).value)):
        v = (<IndexedElement*> a).index - (<IndexedElement*> b).index
    else:
        if (<IndexedElement*> a).value == (<IndexedElement*> b).value:
            v = (<IndexedElement*> a).index - (<IndexedElement*> b).index
        else:
            v = (<IndexedElement*> a).value - (<IndexedElement*> b).value

    # Return the result of the comparison
    return -1 if v <= 0 else 1


cdef void argsort(DTYPE_t *data, SIZE_t **order, INT64_t n_values) nogil:
    """Compute the indexes that would sort the array."""
    # Define the indexes
    cdef SIZE_t index

    # Allocate memory for the index tracking array, that
    # will store the values and their corresponding indexes
    cdef IndexedElement *order_struct = <IndexedElement*> malloc(
        n_values * sizeof(IndexedElement))

    # Copy the values and their indexes from
    # the input data to the index tracking array
    for index in range(n_values):
        order_struct[index].index = index
        order_struct[index].value = data[index]

    # Sort the index tracking array
    qsort(<void*> order_struct, n_values, sizeof(IndexedElement), _compare)

    # Once the index tracking array has been
    # sorted, copy the indexes to the output array
    for index in range(n_values):
        order[0][index] = order_struct[index].index

    # The index tracking array is not needed
    # anymore, free the allocated memory
    free(order_struct)
