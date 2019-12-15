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
from ._argsort cimport argsort


# =============================================================================
# Constants
# =============================================================================

# Type of rank methods
RANK_METHODS = {
    "ordinal": ORDINAL,
    "dense": DENSE
}

# Minimum and maximum integers. They will be employed to
# check whether a given ranking is without or with ties
cdef INT64_t MIN_INT = np.iinfo(np.int32).min + 1
cdef INT64_t MAX_INT = np.iinfo(np.int32).max - 1


# =============================================================================
# Methods
# =============================================================================

cpdef BOOL_t are_label_ranking_targets(INT64_t_2D Y) nogil:
    """Check if the rankings in Y are Label Ranking targets,
    that is, all of them are rankings without ties."""
    # Initialize some values from the input arrays
    cdef INT64_t n_samples = Y.shape[0]

    # Define some values to be employed
    cdef BOOL_t out

    # Define the indexes
    cdef INT64_t sample

    # Initialize the output boolean value as if all the rankings were
    # Label Ranking targets. By this way, it is possible to apply
    # short-circuit as far as a ranking that is not without ties is found
    out = True

    # Check whether Y contains all rankings without ties
    for sample in range(n_samples):
        # If this ranking is not a ranking without ties, apply
        # short-circuit (that is, directly break), since all of
        # them must hold this condition
        out = is_ranking_without_ties_fast(Y[sample])
        if not out:
            break

    # Return whether the rankings
    # are Label Ranking targets
    return out


cpdef BOOL_t are_partial_label_ranking_targets(INT64_t_2D Y) nogil:
    """Check if the rankings in Y are Partial Label Ranking
    targets, that is, all of them are rankings with ties."""
    # Initialize some values from the input arrays
    cdef INT64_t n_samples = Y.shape[0]

    # Define some values to be employed
    cdef BOOL_t out

    # Define the indexes
    cdef INT64_t sample

    # Initialize the output boolean value as if all the rankings were
    # Partial Label Ranking targets. By this way, it is possible to apply
    # short-circuit as far as a ranking that is not with ties is found
    out = True

    # Check whether all the rankings in Y are rankings with ties
    for sample in range(n_samples):
        # If this ranking is not a ranking with ties, apply
        # short-circuit (that is, directly break), since all of
        # them must hold this condition
        out = is_ranking_with_ties_fast(Y[sample])
        if not out:
            break

    # Return whether the rankings
    # are Partial Label Ranking targets
    return out


cdef void _extract(INT64_t_1D y, INT64_t *min_class, INT64_t *max_class,
                   INT64_t *n_finite_classes, INT64_t *n_unique_classes) nogil:
    """Extract the minimum class, the maximum class the number of finite
    classes and the number of unique classes from the ranking in y."""
    # Initialize some values from the input arrays
    cdef INT64_t n_classes = y.shape[0]

    # Define some values to be employed
    cdef UNIQUE_CLASSES unique_classes

    # Define the indexes
    cdef SIZE_t label

    # Initialize the minimum class to the maximum
    # value and the maximum class to the minimum one
    min_class[0] = MAX_INT
    max_class[0] = MIN_INT

    # Initialize the number of finite classes to zero
    # (it will be updated by looping through the classes)
    n_finite_classes[0] = 0

    # Extract the minimum class, the maximum class and the unique
    # classes from the information provided by the input ranking
    for label in range(n_classes):
        if y[label] != RANK_TYPE.RANDOM and y[label] != RANK_TYPE.TOP:
            # Update the minimum class if this class
            # is less than the previously found
            if y[label] < min_class[0]:
                min_class[0] = y[label]
            # Update the maximum class if this class
            # is greater than the previously found
            if y[label] > max_class[0]:
                max_class[0] = y[label]
            # Insert this class in the set with the unique classes
            unique_classes.insert(y[label])
            # Update the number of finite classes
            n_finite_classes[0] += 1

    # Once the set with the unique classes has been
    # obtained, extract the number of unique classes
    # just checking the size of this structure
    n_unique_classes[0] = unique_classes.size()


cpdef BOOL_t is_ranking_without_ties_fast(INT64_t_1D y) nogil:
    """Fast version to check whether y is a ranking without ties."""
    # Define some values to be employed
    cdef INT64_t min_class
    cdef INT64_t max_class
    cdef INT64_t n_finite_classes
    cdef INT64_t n_unique_classes

    # Extract the information provided by the
    # input ranking to know if it is without ties
    _extract(y, &min_class, &max_class, &n_finite_classes, &n_unique_classes)

    # The provided ranking is a without ties if the minimum position in the
    # ranking is one, the maximum position is equal to the number of classes
    # in the ranking and the position of all the classes is different. Also,
    # a ranking with all the classes missed is a without ties
    return ((min_class == 1 and
             max_class == n_unique_classes == n_finite_classes) or
            n_finite_classes == 0)


cpdef BOOL_t is_ranking_with_ties_fast(INT64_t_1D y) nogil:
    """Fast version to check whether y is a ranking with ties."""
    # Define some values to be employed
    cdef INT64_t min_class
    cdef INT64_t max_class
    cdef INT64_t n_finite_classes
    cdef INT64_t n_unique_classes

    # Extract the information provided by the
    # input ranking to know if it is without ties
    _extract(y, &min_class, &max_class, &n_finite_classes, &n_unique_classes)

    # The provided ranking is with ties if the minimum position
    # in the ranking is one, the maximum position in the ranking
    # is equal to the number of the unique classes and also
    # less or equal than the number of classes of the ranking. Also,
    # a ranking with all the classes missed is a without ties
    return ((min_class == 1 and
             max_class == n_unique_classes and
             n_unique_classes <= n_finite_classes) or
            n_finite_classes == 0)


cpdef void rank_data_view(DTYPE_t_1D data, INT64_t_1D y,
                          RANK_METHOD method) nogil:
    """Rank the data provided in a memory
    view according to the given method."""
    # Call to the method that rank the data from a pointer
    # by accessing to the proper element of the view structure
    rank_data_pointer(&data[0], y, method)


cdef void rank_data_pointer(DTYPE_t *data, INT64_t_1D y,
                            RANK_METHOD method) nogil:
    """Rank the data provided in a pointer
    view according to the given method."""
    # Initialize some values from the input arrays
    cdef INT64_t n_classes = y.shape[0]

    # Define some values to be employed
    cdef SIZE_t *data_idx

    # Define the indexes
    cdef SIZE_t label

    # Allocate memory for the indexes that
    # will sort the data and obtain them
    data_idx = <SIZE_t*> malloc(n_classes * sizeof(SIZE_t))
    argsort(data, &data_idx, n_classes)

    # Rank the data according to the given method
    for label in range(n_classes):
        # For ordinal method, just use
        # the index of the sorted data
        if method == ORDINAL:
            y[data_idx[label]] = label + 1
        # For the dense method, use the ranking of the
        # previous label and increase the ranking if its
        # data value is less than the data value of the
        # previous ranked item
        else:
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

    # The array with the indexes that sorts the data is
    # not needed anymore, free the allocated memory
    free(data_idx)
