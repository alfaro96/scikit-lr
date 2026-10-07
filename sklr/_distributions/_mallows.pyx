"""Center ranking of the Mallows model of label rankings, possibly incomplete."""

import numpy as np

cimport cython
from cython.parallel cimport prange, threadid
from libc.math cimport isnan

from sklearn.utils._typedefs cimport float64_t, intp_t, uint8_t

from sklr._distributions._borda cimport borda_scores
from sklr._distributions._extension cimport complete_ranking
from sklr._distributions._sorting cimport rank_by_scores


# The indices of the loops are within the bounds of the arrays, which the callers
# create with the right sizes, so checking them would only slow down the loops
@cython.boundscheck(False)
@cython.wraparound(False)
cdef bint _is_equal(const intp_t[::1] a, const intp_t[::1] b) noexcept nogil:
    """Tell whether two rankings are equal."""
    cdef intp_t label
    for label in range(a.shape[0]):
        if a[label] != b[label]:
            return False
    return True


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _copy(const intp_t[::1] source, intp_t[::1] target) noexcept nogil:
    """Copy a ranking into another one."""
    cdef intp_t label
    for label in range(source.shape[0]):
        target[label] = source[label]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef bint _has_unranked(const float64_t[:, ::1] y) noexcept nogil:
    """Tell whether some ranking has unranked labels."""
    cdef intp_t sample, label
    for sample in range(y.shape[0]):
        for label in range(y.shape[1]):
            if isnan(y[sample, label]):
                return True
    return False


@cython.boundscheck(False)
@cython.wraparound(False)
cdef bint estimate_center(
    const float64_t[:, ::1] y,
    const float64_t[::1] sample_weight,
    intp_t max_iter,
    intp_t[::1] center,
    intp_t* n_iter,
    float64_t[:, ::1] completed,
    float64_t[::1] scores,
    intp_t[:, ::1] work,
) noexcept nogil:
    """Estimate the center ranking of the Mallows model by the Borda count.

    The rankings `y` must not have ties. The center of incomplete rankings follows
    Algorithm 1 of [cheng_decision_2009]: the rankings are completed with their
    closest extensions to the center, which is then found again from them by the
    Borda count, until it does not change, it comes back to an earlier center, or
    `max_iter` iterations are done. Returns whether the center converged, and sets
    ``n_iter[0]`` to the number of iterations, ``0`` for complete rankings.
    `completed` has the shape of `y`, `scores` has ``n_labels`` elements and `work`
    has shape ``(4, n_labels)``.
    """
    cdef intp_t n_samples = y.shape[0]
    cdef intp_t[::1] previous = work[0], saved = work[1]
    cdef intp_t[::1] ranked_at = work[2], gaps = work[3]
    cdef intp_t sample
    cdef intp_t power = 1, n_since_saved = 0

    borda_scores(y, sample_weight, scores)
    rank_by_scores(scores, center)
    n_iter[0] = 0
    if not _has_unranked(y):
        return True

    # A center that comes back after the center has changed means that the
    # iterations cycle and will never converge. Brent's method finds it with a
    # single saved center, which is replaced whenever the number of iterations
    # since it was saved reaches a power of two that doubles each time
    _copy(center, saved)
    while n_iter[0] < max_iter:
        n_iter[0] += 1
        _copy(center, previous)
        for sample in range(n_samples):
            complete_ranking(y[sample], previous, completed[sample], ranked_at, gaps)
        borda_scores(completed, sample_weight, scores)
        rank_by_scores(scores, center)
        if _is_equal(center, previous):
            return True
        if _is_equal(center, saved):
            return False
        n_since_saved += 1
        if n_since_saved == power:
            _copy(center, saved)
            power *= 2
            n_since_saved = 0
    return False


# The shapes of the inputs and the number of threads are checked first, and the
# other arrays are created with them
@cython.boundscheck(False)
@cython.wraparound(False)
def estimate_center_batch(
    const float64_t[:, :, ::1] y,
    const float64_t[:, ::1] sample_weight,
    intp_t max_iter,
    int n_threads,
):
    """Estimate the center ranking of the Mallows model of each group of rankings.

    The rankings `y` must not have ties. Returns the center, the number of
    iterations and whether it converged, for each group.
    """
    cdef intp_t n_groups = y.shape[0], n_samples = y.shape[1], n_labels = y.shape[2]
    cdef intp_t group
    cdef int thread
    if sample_weight.shape[0] != n_groups or sample_weight.shape[1] != n_samples:
        raise ValueError(
            f"Expected sample weights of shape ({n_groups}, {n_samples}), got "
            f"({sample_weight.shape[0]}, {sample_weight.shape[1]}) instead."
        )
    if n_threads < 1:
        raise ValueError(f"Expected at least 1 thread, got n_threads={n_threads}.")

    center = np.empty((n_groups, n_labels), dtype=np.intp)
    n_iter = np.empty(n_groups, dtype=np.intp)
    converged = np.empty(n_groups, dtype=np.uint8)
    cdef intp_t[:, ::1] center_view = center
    cdef intp_t[::1] n_iter_view = n_iter
    cdef uint8_t[::1] converged_view = converged

    # threadid() is below n_threads, so each thread works on its own row of the
    # buffers, which are created once per thread instead of once per group
    cdef float64_t[:, :, ::1] completed = np.empty(
        (n_threads, n_samples, n_labels), dtype=np.float64
    )
    cdef float64_t[:, ::1] scores = np.empty((n_threads, n_labels), dtype=np.float64)
    cdef intp_t[:, :, ::1] work = np.empty((n_threads, 4, n_labels), dtype=np.intp)

    for group in prange(n_groups, nogil=True, schedule="static", num_threads=n_threads):
        thread = threadid()
        converged_view[group] = estimate_center(
            y[group],
            sample_weight[group],
            max_iter,
            center_view[group],
            &n_iter_view[group],
            completed[thread],
            scores[thread],
            work[thread],
        )
    return center, n_iter, converged.astype(bool)
