# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False


# =============================================================================
# Imports
# =============================================================================

# Third party
from libc.stdlib cimport free, malloc
import numpy as np

# Local application
from ._argsort cimport argsort_view, argsort_pointer


# =============================================================================
# Constants
# =============================================================================

# Rank methods
RANK_METHODS = {
    "ordinal": ORDINAL,
    "dense": DENSE
}

# Minimum and maximum integers
cdef INT64_t MIN_INT = np.iinfo(np.int32).min + 1
cdef INT64_t MAX_INT = np.iinfo(np.int32).max - 1


# =============================================================================
# Methods
# =============================================================================

cpdef BOOL_t is_ranking_without_ties_fast(INT64_t_1D y) nogil:
    """Fast version to check whether y is a ranking without ties."""
    # Initialize some values from the input arrays
    cdef INT64_t n_classes = y.shape[0]

    # Define some values to be employed
    cdef INT64_t min_class
    cdef INT64_t max_class

    cdef INT64_t n_finite_classes
    cdef INT64_t n_unique_classes

    cdef UNIQUE_CLASSES unique_classes

    # Define the indexes
    cdef SIZE_t label

    # Initialize the minimum class to the highest possible value
    # and the maximum class to the lowest possible value
    min_class = MAX_INT
    max_class = MIN_INT

    # Initialize the number of finite classes to the number of classes
    n_finite_classes = n_classes

    # Obtain the minimum class, the maximum class and the unique classes
    for label in range(n_classes):
        if y[label] == RANK_TYPE.RANDOM or y[label] == RANK_TYPE.TOP:
            n_finite_classes -= 1
        else:
            if y[label] < min_class:
                min_class = y[label]
            if y[label] > max_class:
                max_class = y[label]
            unique_classes.insert(y[label])

    # Obtain the number of unique classes
    n_unique_classes = unique_classes.size()

    # It is a ranking without ties if the minimum position
    # in the ranking is one, the maximum position is equal
    # to the number of classes in the ranking and the
    # position of all the classes is different
    return ((min_class == 1 and
             max_class == n_unique_classes == n_finite_classes) or
            n_finite_classes == 0)


cpdef BOOL_t is_ranking_with_ties_fast(INT64_t_1D y) nogil:
    """Fast version to check whether y is a ranking with ties."""
    # Initialize some values from the input arrays
    cdef INT64_t n_classes = y.shape[0]

    # Define some values to be employed
    cdef INT64_t min_class
    cdef INT64_t max_class

    cdef INT64_t n_finite_classes
    cdef INT64_t n_unique_classes

    cdef UNIQUE_CLASSES unique_classes

    # Define the indexes
    cdef SIZE_t label

    # Initialize the minimum class to the highest possible value
    # and the maximum class to the lowest possible value
    min_class = MAX_INT
    max_class = MIN_INT

    # Initialize the number of finite classes to the number of classes
    n_finite_classes = n_classes

    # Obtain the minimum class, the maximum class and the unique classes
    for label in range(n_classes):
        if y[label] == RANK_TYPE.RANDOM or y[label] == RANK_TYPE.TOP:
            n_finite_classes -= 1
        else:
            if y[label] < min_class:
                min_class = y[label]
            if y[label] > max_class:
                max_class = y[label]
            unique_classes.insert(y[label])

    # Obtain the number of unique classes
    n_unique_classes = unique_classes.size()

    # It is a ranking without ties if the minimum position
    # in the ranking is one, the maximum position in the ranking
    # is equal to the number of the unique classes and also
    # less or equal than the number of classes of the ranking
    return ((min_class == 1 and
             max_class == n_unique_classes and
             n_unique_classes <= n_finite_classes) or
            n_finite_classes == 0)


cpdef void rank_view_fast(DTYPE_t_1D data, INT64_t_1D y,
                          RANK_METHOD method) nogil:
    """Fast version to rank data according to the given method."""
    # Initialize some values from the input arrays
    cdef INT64_t n_classes = y.shape[0]

    # Define some values to be employed
    cdef SIZE_t *data_idx

    # Define the indexes
    cdef SIZE_t label

    # Allocate memory for the indexes
    data_idx = <SIZE_t*> malloc(n_classes * sizeof(SIZE_t))

    # Sort the data to obtain the indexes
    argsort_view(data, &data_idx)

    # Rank the data according to the given method
    for label in range(n_classes):
        # For ordinal method, just use the index of the sorted data
        if method == ORDINAL:
            y[data_idx[label]] = label + 1
        # For the dense method, use the ranking of
        # the previous label and increase the ranking
        # if its data value is less than the data value
        # of the previous ranked item
        elif method == DENSE:
            # If it is the first label,
            # just assign the first ranking
            if label == 0:
                y[data_idx[label]] = 1
            # Otherwise, use the standard procedure
            else:
                # Assign the ranking of the previous label
                y[data_idx[label]] = y[data_idx[label - 1]]
                # Increase if the data value of the previous label
                # is different than the data value of the current one
                if data[data_idx[label - 1]] != data[data_idx[label]]:
                    y[data_idx[label]] += 1

    # Free indexes array
    free(data_idx)


cdef void rank_pointer_fast(DTYPE_t *data, INT64_t_1D y,
                            RANK_METHOD method) nogil:
    """Fast version to rank data according to the given method."""
    # Initialize some values from the input arrays
    cdef INT64_t n_classes = y.shape[0]

    # Define some values to be employed
    cdef SIZE_t *data_idx

    # Define the indexes
    cdef SIZE_t label

    # Allocate memory for the indexes
    data_idx = <SIZE_t*> malloc(n_classes * sizeof(SIZE_t))

    # Sort the data to obtain the indexes
    argsort_pointer(data, &data_idx, n_classes)

    # Rank the data according to the given method
    for label in range(n_classes):
        # For ordinal method, just use the index of the sorted data
        if method == ORDINAL:
            y[data_idx[label]] = label + 1
        # For the dense method, use the ranking of
        # the previous label and increase the ranking
        # if its data value is less than the data value
        # of the previous ranked item
        elif method == DENSE:
            # If it is the first label,
            # just assign the first ranking
            if label == 0:
                y[data_idx[label]] = 1
            # Otherwise, use the standard procedure
            else:
                # Assign the ranking of the previous label
                y[data_idx[label]] = y[data_idx[label - 1]]
                # Increase if the data value of the previous label
                # is different than the data value of the current one
                if data[data_idx[label - 1]] != data[data_idx[label]]:
                    y[data_idx[label]] += 1

    # Free indexes array
    free(data_idx)
