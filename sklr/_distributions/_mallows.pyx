"""Center ranking and spread of the Mallows model of possibly incomplete rankings."""

import numpy as np

cimport cython
from cython.parallel cimport prange, threadid
from libc.float cimport DBL_EPSILON
from libc.math cimport INFINITY, exp, isnan
from scipy.optimize.cython_optimize cimport brentq

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


cdef struct _SpreadEquation:
    intp_t n_labels
    float64_t mean_distance


@cython.boundscheck(False)
@cython.wraparound(False)
cdef intp_t _kendall_distance(
    const float64_t[::1] y, const intp_t[::1] center
) noexcept nogil:
    """Count the pairs of labels that a complete ranking and the center disagree on."""
    cdef intp_t n_labels = y.shape[0]
    cdef intp_t label, other, distance = 0
    for label in range(n_labels):
        for other in range(label + 1, n_labels):
            if (y[label] < y[other]) != (center[label] < center[other]):
                distance += 1
    return distance


# The total of the probabilities, a divisor, starts at one and only grows
@cython.cdivision(True)
cdef float64_t _expected_distance(float64_t theta, intp_t n_labels) noexcept nogil:
    """Compute the expected Kendall distance to the center for a spread `theta`.

    The expected distance of [cheng_decision_2009] is the sum, over :math:`j` from
    :math:`2` to :math:`n`, of the mean of a variable from :math:`0` to
    :math:`j - 1` with probabilities proportional to :math:`q^r`, where :math:`q`
    is the exponential of minus `theta`. Its closed form subtracts terms that grow
    as the inverse of `theta` and cancel out near zero, while these means only add
    up positive terms.
    """
    cdef float64_t q = exp(-theta)
    cdef float64_t power = 1, total = 1, weighted = 0, distance = 0
    cdef intp_t j
    for j in range(1, n_labels):
        power *= q
        total += power
        weighted += j * power
        distance += weighted / total
    return distance


cdef float64_t _spread_equation(float64_t theta, void* args) noexcept nogil:
    """Compute the expected distance for `theta` minus the mean observed distance."""
    cdef _SpreadEquation* equation = <_SpreadEquation*> args
    return _expected_distance(theta, equation.n_labels) - equation.mean_distance


# A mean distance of zero has no finite spread, and from n_labels(n_labels - 1)/4,
# the expected distance of the uniform distribution, the spread would be negative
cdef float64_t _solve_spread(float64_t mean_distance, intp_t n_labels) noexcept nogil:
    """Find the spread whose expected distance is the mean observed distance."""
    cdef _SpreadEquation equation
    cdef float64_t lower = 0, upper = 1
    if mean_distance <= 0:
        return INFINITY
    if mean_distance >= n_labels * (n_labels - 1) / 4.0:
        return 0
    # The expected distance decreases to zero, so doubling the upper bound
    # brackets the root between its last two values
    while _expected_distance(upper, n_labels) >= mean_distance:
        lower = upper
        upper *= 2
    equation.n_labels = n_labels
    equation.mean_distance = mean_distance
    # The root is positive, so the relative tolerance alone, the smallest that
    # scipy.optimize.brentq accepts, gives it to nearly full precision. The C
    # routine, unlike that function, accepts a zero absolute tolerance. Mean
    # distances far below one, from very uneven weights, take close to a hundred
    # iterations, so the cap leaves room for them
    return brentq(
        _spread_equation, lower, upper, &equation, 0, 4 * DBL_EPSILON, 200, NULL
    )


# The callers do not let the weights add up to zero, the divisor of the mean
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef float64_t estimate_spread(
    const float64_t[:, ::1] y,
    const float64_t[::1] sample_weight,
    const intp_t[::1] center,
    float64_t[::1] completed,
    intp_t[:, ::1] work,
) noexcept nogil:
    """Estimate the spread of the Mallows model given its center.

    The rankings `y` must not have ties and the weights must not all be zero. As in
    Algorithm 1 of [cheng_decision_2009], the rankings are completed with their
    closest extensions to the center, and the spread is the maximum likelihood
    estimate, whose expected distance to the center is the weighted mean of their
    Kendall distances to it: ``0`` if it is not below that of the uniform
    distribution, and infinite if it is zero. `completed` has ``n_labels``
    elements and `work` has shape ``(2, n_labels)``.
    """
    cdef intp_t sample
    cdef float64_t distance = 0, total_weight = 0
    for sample in range(y.shape[0]):
        complete_ranking(y[sample], center, completed, work[0], work[1])
        distance += sample_weight[sample] * _kendall_distance(completed, center)
        total_weight += sample_weight[sample]
    return _solve_spread(distance / total_weight, y.shape[1])


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


@cython.boundscheck(False)
@cython.wraparound(False)
def estimate_spread_batch(
    const float64_t[:, :, ::1] y,
    const float64_t[:, ::1] sample_weight,
    const intp_t[:, ::1] center,
    int n_threads,
):
    """Estimate the spread of the Mallows model of each group given its center.

    The rankings `y` must not have ties and the weights of a group must not all be
    zero.
    """
    cdef intp_t n_groups = y.shape[0], n_samples = y.shape[1], n_labels = y.shape[2]
    cdef intp_t group
    cdef int thread
    if sample_weight.shape[0] != n_groups or sample_weight.shape[1] != n_samples:
        raise ValueError(
            f"Expected sample weights of shape ({n_groups}, {n_samples}), got "
            f"({sample_weight.shape[0]}, {sample_weight.shape[1]}) instead."
        )
    if center.shape[0] != n_groups or center.shape[1] != n_labels:
        raise ValueError(
            f"Expected centers of shape ({n_groups}, {n_labels}), got "
            f"({center.shape[0]}, {center.shape[1]}) instead."
        )
    if n_threads < 1:
        raise ValueError(f"Expected at least 1 thread, got n_threads={n_threads}.")

    spread = np.empty(n_groups, dtype=np.float64)
    cdef float64_t[::1] spread_view = spread
    cdef float64_t[:, ::1] completed = np.empty((n_threads, n_labels), dtype=np.float64)
    cdef intp_t[:, :, ::1] work = np.empty((n_threads, 2, n_labels), dtype=np.intp)

    for group in prange(n_groups, nogil=True, schedule="static", num_threads=n_threads):
        thread = threadid()
        spread_view[group] = estimate_spread(
            y[group],
            sample_weight[group],
            center[group],
            completed[thread],
            work[thread],
        )
    return spread
