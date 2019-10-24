# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False


# =============================================================================
# Imports
# =============================================================================

# Third party
from cython.operator cimport dereference as deref
from libc.math cimport isnan
from libc.stdlib cimport const_void, free, qsort, malloc


# =============================================================================
# Methods
# =============================================================================

cdef int _compare(const_void *a, const_void *b) nogil:
    """Compare function for sorting the indexes."""
    # Define some values to be employed
    cdef DTYPE_t v

    # Compare the values (fixing NaN values,
    # just to sort in the same way than NumPy)
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


cdef void argsort_view(DTYPE_t_1D data, SIZE_t **order) nogil:
    """Compute the indexes that would sort the array."""
    # Initialize some values from the input arrays
    cdef INT64_t n_values = data.shape[0]

    # Define the indexes
    cdef SIZE_t index

    # Allocate index tracking array
    cdef IndexedElement *order_struct = <IndexedElement*> malloc(
        n_values * sizeof(IndexedElement))

    # Copy data into index tracking array
    for index in range(n_values):
        order_struct[index].index = index
        order_struct[index].value = data[index]

    # Sort index tracking array
    qsort(<void*> order_struct, n_values, sizeof(IndexedElement), _compare)

    # Copy indices from index tracking array to output array
    for index in range(n_values):
        deref(order)[index] = order_struct[index].index

    # Free index tracking array
    free(order_struct)


cdef void argsort_pointer(DTYPE_t *data, SIZE_t **order,
                          INT64_t n_values) nogil:
    """Compute the indexes that would sort the array."""
    # Define the indexes
    cdef SIZE_t index

    # Allocate index tracking array
    cdef IndexedElement *order_struct = <IndexedElement*> malloc(
        n_values * sizeof(IndexedElement))

    # Copy data into index tracking array
    for index in range(n_values):
        order_struct[index].index = index
        order_struct[index].value = data[index]

    # Sort index tracking array
    qsort(<void*> order_struct, n_values, sizeof(IndexedElement), _compare)

    # Copy indices from index tracking array to output array
    for index in range(n_values):
        deref(order)[index] = order_struct[index].index

    # Free index tracking array
    free(order_struct)
