# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False


# =============================================================================
# Imports
# =============================================================================

# Third party
from libc.limits cimport INT_MIN, INT_MAX
from libc.stdlib cimport free

# Local application
from ._argsort cimport argsort


# =============================================================================
# Methods
# =============================================================================

cdef (INT64_t, INT64_t, INT64_t, INT64_t) _extract(INT64_t_1D y) nogil:
    """Extract the relevant information from the input ranking."""
    cdef INT64_t n_classes = y.shape[0]

    cdef INT64_t min_position = INT_MAX
    cdef INT64_t max_position = INT_MIN
    cdef INT64_t n_ranked_classes = 0
    cdef INT64_t n_unique_positions = 0

    # Use an unordered set for efficiently keep
    # track of the unique positions in the ranking
    cdef UNIQUE_POSITIONS unique_positions

    cdef SIZE_t label

    for label in range(n_classes):
        # Avoid the missed classes since only the ranked
        # ones contain relevant information of the ranking
        if (y[label] != RANK_TYPE.TOP and
                y[label] != RANK_TYPE.RANDOM):
            # Update the number of ranked classes and the
            # minimum and maximum position in the ranking
            n_ranked_classes += 1
            min_position = min(min_position, y[label])
            max_position = max(max_position, y[label])

            # Check whether the position exists in the ranking
            # (since unordered set does not allow duplicate values,
            # count is the same that checking if the value exists)
            if not unique_positions.count(y[label]):
                n_unique_positions += 1
                unique_positions.insert(y[label])

    return (min_position, max_position, n_ranked_classes, n_unique_positions)


cpdef BOOL_t is_label_ranking(INT64_t_2D Y) nogil:
    """Check if Y is in Label Ranking format."""
    cdef INT64_t n_samples = Y.shape[0]
    cdef BOOL_t out = True

    cdef INT64_t min_position
    cdef INT64_t max_position
    cdef INT64_t n_ranked_classes
    cdef INT64_t n_unique_positions

    cdef INT64_t sample

    for sample in range(n_samples):
        (min_position,
         max_position,
         n_ranked_classes,
         n_unique_positions) = _extract(Y[sample])

        # The ranking is in Label Ranking format if the minimum
        # position is one, the maximum position is equal to the
        # number of ranked classes and, at the same time, equal
        # to the number of unique positions. Note that rankings
        # with all the classes missed are also considered
        out = (n_ranked_classes == 0 or
               min_position == 1 and
               max_position == n_ranked_classes and
               n_ranked_classes == n_unique_positions)

        if not out:
            break

    return out


cpdef BOOL_t is_partial_label_ranking(INT64_t_2D Y) nogil:
    """Check if Y is in Partial Label Ranking format."""
    cdef INT64_t n_samples = Y.shape[0]
    cdef BOOL_t out = True

    cdef INT64_t min_position
    cdef INT64_t max_position
    cdef INT64_t n_ranked_classes
    cdef INT64_t n_unique_positions

    cdef INT64_t sample

    for sample in range(n_samples):
        (min_position,
         max_position,
         n_ranked_classes,
         n_unique_positions) = _extract(Y[sample])

        # The ranking is in Partial Label Ranking format if the
        # minimum position is one, the maximum position is equal
        # to the number of unique positions and, at the same time,
        # is less than or equal to the number of ranked classes.
        # Note that rankings with all the classes missed are also
        # considered
        out = (n_ranked_classes == 0 or
               min_position == 1 and
               max_position == n_unique_positions and
               n_unique_positions <= n_ranked_classes)

        if not out:
            break

    return out


cdef void rank_data_view(DTYPE_t_1D data,
                         INT64_t_1D y,
                         RANK_METHOD method) nogil:
    """Assign ranks to a memory view with data."""
    cdef INT64_t n_classes = y.shape[0]

    rank_data_pointer(&data[0], &y[0], method, n_classes)


cdef void rank_data_pointer(DTYPE_t *data,
                            INT64_t *y,
                            RANK_METHOD method,
                            INT64_t n_classes) nogil:
    """Assign ranks to a pointer with data."""
    # Computing the indexes that would sort the data
    # allow to rank the data in an efficient manner
    cdef SIZE_t *sorter = argsort(data, n_classes)

    cdef SIZE_t label
    cdef SIZE_t prev_ranked
    cdef SIZE_t curr_ranked

    for label in range(n_classes):
        curr_ranked = sorter[label]

        # The ordinal method assigns to all the
        # values a distinct rank, according with
        # the order that they occur in the data
        if method == ORDINAL:
            y[curr_ranked] = label + 1

        # The dense method assigns to the next highest
        # value in the data, the rank inmediately after
        # those assigned to the tied values
        else:
            if label == 0:
                y[curr_ranked] = 1
            else:
                # Assign the rank of the previous class and,
                # in case that the values in data are different,
                # increase the rank to the one inmediately after
                y[curr_ranked] = y[prev_ranked]
                
                if data[curr_ranked] != data[prev_ranked]:
                    y[curr_ranked] += 1

        prev_ranked = curr_ranked

    free(sorter)
