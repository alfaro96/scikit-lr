"""Generalized Borda count of label rankings, possibly incomplete."""

cimport cython
from libc.math cimport isnan

from sklearn.utils._typedefs cimport float64_t, intp_t


# The indices of the loops are within the bounds of the arrays, which the callers
# create with the right sizes, so checking them would only slow down the loops.
# The divisors, two and the number of ranked labels plus one, are never zero
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void borda_scores(
    const float64_t[:, ::1] y,
    const float64_t[::1] sample_weight,
    float64_t[::1] scores,
) noexcept nogil:
    """Add up the weighted votes of the rankings for each label.

    A label at position :math:`i` among the :math:`m` ranked labels of a ranking of
    :math:`n` labels receives :math:`(m - i + 1)(n + 1) / (m + 1)` votes, and an
    unranked one :math:`(n + 1) / 2`, as in the generalized Borda count of
    Proposition 2 of [cheng_decision_2009]. These are the expected votes when the
    ranking is extended uniformly at random. The votes of a complete ranking are
    integers, and so are their totals when the weights are integers too.
    """
    cdef intp_t n_samples = y.shape[0], n_labels = y.shape[1]
    cdef intp_t sample, label, n_ranked
    cdef float64_t scale

    for label in range(n_labels):
        scores[label] = 0
    for sample in range(n_samples):
        n_ranked = 0
        for label in range(n_labels):
            if not isnan(y[sample, label]):
                n_ranked += 1
        scale = (n_labels + 1) / (n_ranked + 1.0)
        for label in range(n_labels):
            if isnan(y[sample, label]):
                scores[label] += sample_weight[sample] * (n_labels + 1) / 2
            else:
                scores[label] += (
                    sample_weight[sample] * (n_ranked - y[sample, label] + 1) * scale
                )
