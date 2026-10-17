"""Most probable extensions of incomplete label rankings."""

import numpy as np

cimport cython
from libc.math cimport isnan

from sklearn.utils._typedefs cimport float64_t, intp_t


# The rankings have no ties, so the positions of their ranked labels are the
# integers from one to their number, which index ranked_at, and the other indices
# are within the bounds of arrays of n_labels elements
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void complete_ranking(
    const float64_t[::1] y,
    const intp_t[::1] center,
    float64_t[::1] completed,
    intp_t[::1] ranked_at,
    intp_t[::1] gaps,
) noexcept nogil:
    """Complete a ranking with its closest extension in Kendall distance to a center.

    The ranking `y` must not have ties. Each unranked label goes to the gap between
    the ranked labels that minimizes its disagreements with the center, the first
    one if several do, and the labels of a gap follow the order of the center
    (Proposition 1 of [cheng_decision_2009]). `ranked_at` and `gaps` are buffers of
    ``n_labels`` elements.
    """
    cdef intp_t n_labels = y.shape[0]
    cdef intp_t n_ranked = 0, n_unranked = 0
    cdef intp_t label, other, position, i, j, cost, best_cost, n_before

    # The unranked labels fill ranked_at from the end, in any order, so that the
    # loops below go through them alone instead of checking every label
    for label in range(n_labels):
        if isnan(y[label]):
            n_unranked += 1
            ranked_at[n_labels - n_unranked] = label
        else:
            ranked_at[<intp_t> y[label] - 1] = label
            n_ranked += 1

    for i in range(n_ranked, n_labels):
        label = ranked_at[i]
        # Moving the label past the next ranked label adds a disagreement with the
        # center if the center puts that label behind it and removes one
        # otherwise. The cost of each gap is counted from that of the first one,
        # ahead of every ranked label, which is enough to find the best gap
        cost = 0
        best_cost = 0
        gaps[label] = 0
        for position in range(n_ranked):
            if center[ranked_at[position]] > center[label]:
                cost += 1
            else:
                cost -= 1
            if cost < best_cost:
                best_cost = cost
                gaps[label] = position + 1

    # The gap j is after the ranked label at position j, so a ranked label comes
    # after the unranked labels of the gaps before its position, and an unranked
    # label in the gap j after the first j ranked labels, the unranked labels of
    # the earlier gaps and those of its own gap that the center puts ahead of it
    for position in range(n_ranked):
        label = ranked_at[position]
        n_before = position
        for j in range(n_ranked, n_labels):
            if gaps[ranked_at[j]] <= position:
                n_before += 1
        completed[label] = n_before + 1
    for i in range(n_ranked, n_labels):
        label = ranked_at[i]
        n_before = gaps[label]
        for j in range(n_ranked, n_labels):
            other = ranked_at[j]
            if gaps[other] < gaps[label] or (
                gaps[other] == gaps[label] and center[other] < center[label]
            ):
                n_before += 1
        completed[label] = n_before + 1


# The callers pass a center with the labels of the rankings, and the other arrays
# are created with the sizes of the inputs
@cython.boundscheck(False)
@cython.wraparound(False)
def complete_rankings(const float64_t[:, ::1] y, const intp_t[::1] center):
    """Complete each ranking, without ties, with its closest extension to the center."""
    cdef intp_t n_samples = y.shape[0], n_labels = y.shape[1]
    cdef intp_t sample

    completed = np.empty((n_samples, n_labels), dtype=np.float64)
    cdef float64_t[:, ::1] completed_view = completed
    cdef intp_t[::1] ranked_at = np.empty(n_labels, dtype=np.intp)
    cdef intp_t[::1] gaps = np.empty(n_labels, dtype=np.intp)

    with nogil:
        for sample in range(n_samples):
            complete_ranking(
                y[sample], center, completed_view[sample], ranked_at, gaps
            )
    return completed
