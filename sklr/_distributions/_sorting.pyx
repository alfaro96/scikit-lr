"""Sorting of the labels by their scores."""

cimport cython

from sklearn.utils._typedefs cimport float64_t, intp_t


# The indices of the loops are within the bounds of the arrays, which the callers
# create with the right sizes, so checking them would only slow down the loops
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void rank_by_scores(
    const float64_t[::1] scores,
    intp_t[::1] ranking,
) noexcept nogil:
    """Rank the labels by decreasing score, the smaller index first on ties.

    The position of each label is counted instead of sorting the labels, which takes
    quadratic time in their number but no extra memory.
    """
    cdef intp_t n_labels = scores.shape[0]
    cdef intp_t label, other

    for label in range(n_labels):
        ranking[label] = 1
        for other in range(n_labels):
            if scores[other] > scores[label] or (
                scores[other] == scores[label] and other < label
            ):
                ranking[label] += 1
