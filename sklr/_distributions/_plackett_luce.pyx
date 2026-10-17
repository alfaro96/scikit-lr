"""Parameters of the Plackett-Luce model of possibly incomplete rankings."""

import numpy as np

cimport cython
from cython.parallel cimport prange, threadid
from libc.math cimport fabs, fmax, isnan

from sklearn.utils._typedefs cimport float64_t, intp_t, uint8_t


# The indices of the loops are within the bounds of the arrays, which the callers
# create with the right sizes, so checking them would only slow down the loops. The
# columns of the order come from the positions of the rankings, which the callers
# also pass dense from one and without ties
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _order_labels(
    const float64_t[:, ::1] y, intp_t[:, ::1] order, intp_t[::1] n_ranked
) noexcept nogil:
    """List the ranked labels of each ranking from the first to the last.

    The positions of the ranked labels are dense from one and without ties, so each
    label goes straight to its place.
    """
    cdef intp_t sample, label
    for sample in range(y.shape[0]):
        n_ranked[sample] = 0
        for label in range(y.shape[1]):
            if not isnan(y[sample, label]):
                order[sample, <intp_t> y[sample, label] - 1] = label
                n_ranked[sample] += 1


# A sum of parameters includes the label at its position, which has wins and so a
# positive parameter unless it underflows. Then the division gives infinity, and
# the labels whose denominators include it get a zero parameter. The denominator of
# a label with wins and the total of the updated parameters are always positive,
# and so is the largest weight once there are wins
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef bint estimate_parameters(
    const float64_t[:, ::1] y,
    const float64_t[::1] sample_weight,
    float64_t tol,
    intp_t max_iter,
    float64_t[::1] parameters,
    intp_t* n_iter,
    intp_t[:, ::1] order,
    intp_t[::1] n_ranked,
    float64_t[:, ::1] work,
) noexcept nogil:
    """Estimate the parameters of the Plackett-Luce model by the MM algorithm.

    The rankings `y` must not have ties. Starting from equal parameters, equation
    (30) of [hunter_mm_2004] updates them, with the rankings weighted, and they are
    scaled to add up to one, until none changes by more than `tol` or `max_iter`
    iterations are done. The labels never ranked above another one in a ranking
    with a positive weight get a zero parameter, and all the labels get the same one
    if there is no such ranking. Returns whether the parameters converged, and sets
    ``n_iter[0]`` to the number of iterations. `order` has the shape of `y`,
    `n_ranked` has ``n_samples`` elements and `work` has shape ``(4, n_labels)``.
    """
    cdef intp_t n_samples = y.shape[0], n_labels = y.shape[1]
    cdef float64_t[::1] wins = work[0], denominators = work[1]
    cdef float64_t[::1] updated = work[2], tail_sums = work[3]
    cdef intp_t sample, label, position, last
    cdef float64_t weight, max_weight = 0, total, change, inverse_sums

    _order_labels(y, order, n_ranked)
    for label in range(n_labels):
        parameters[label] = 1.0 / n_labels
    n_iter[0] = 0
    for sample in range(n_samples):
        if n_ranked[sample] > 1:
            max_weight = fmax(max_weight, sample_weight[sample])
    if max_weight == 0:
        return True

    # Equation (30) does not change when all the weights are scaled, so they are
    # divided by the largest one, which keeps the sums of very large weights finite
    # and those of very small ones precise. The wins of a label, the numerator of
    # the update, add up the weights of the rankings where it is above the last label
    for label in range(n_labels):
        wins[label] = 0
    for sample in range(n_samples):
        for position in range(n_ranked[sample] - 1):
            wins[order[sample, position]] += sample_weight[sample] / max_weight

    while n_iter[0] < max_iter:
        n_iter[0] += 1
        for label in range(n_labels):
            denominators[label] = 0
        for sample in range(n_samples):
            weight = sample_weight[sample] / max_weight
            last = n_ranked[sample] - 1
            if weight == 0 or last < 1:
                continue
            # The denominator of the label at a position adds, over each position
            # up to its own one but the last, which is not a choice, the inverse
            # of the sum of the parameters from there to the end. The sums are
            # found back from the last position and their inverses added up
            # forward, so each ranking takes linear time
            tail_sums[last] = parameters[order[sample, last]]
            for position in range(last - 1, -1, -1):
                tail_sums[position] = (
                    tail_sums[position + 1] + parameters[order[sample, position]]
                )
            inverse_sums = 0
            for position in range(last + 1):
                if position < last:
                    inverse_sums += weight / tail_sums[position]
                denominators[order[sample, position]] += inverse_sums

        total = 0
        for label in range(n_labels):
            if wins[label] > 0:
                updated[label] = wins[label] / denominators[label]
            else:
                updated[label] = 0
            total += updated[label]
        change = 0
        for label in range(n_labels):
            updated[label] /= total
            change = fmax(change, fabs(updated[label] - parameters[label]))
            parameters[label] = updated[label]
        if change <= tol:
            return True
    return False


# The callers pass inputs of matching shapes and at least one thread, and the other
# arrays are created with those shapes
@cython.boundscheck(False)
@cython.wraparound(False)
def estimate_parameters_batch(
    const float64_t[:, :, ::1] y,
    const float64_t[:, ::1] sample_weight,
    float64_t tol,
    intp_t max_iter,
    int n_threads,
):
    """Estimate the parameters of the Plackett-Luce model of each group of rankings.

    The rankings `y` must not have ties. Returns the parameters, the number of
    iterations and whether they converged, for each group.
    """
    cdef intp_t n_groups = y.shape[0], n_samples = y.shape[1], n_labels = y.shape[2]
    cdef intp_t group
    cdef int thread

    parameters = np.empty((n_groups, n_labels), dtype=np.float64)
    n_iter = np.empty(n_groups, dtype=np.intp)
    converged = np.empty(n_groups, dtype=np.uint8)
    cdef float64_t[:, ::1] parameters_view = parameters
    cdef intp_t[::1] n_iter_view = n_iter
    cdef uint8_t[::1] converged_view = converged

    # threadid() is below n_threads, so each thread works on its own row of the
    # buffers, which are created once per thread instead of once per group
    cdef intp_t[:, :, ::1] order = np.empty(
        (n_threads, n_samples, n_labels), dtype=np.intp
    )
    cdef intp_t[:, ::1] n_ranked = np.empty((n_threads, n_samples), dtype=np.intp)
    cdef float64_t[:, :, ::1] work = np.empty(
        (n_threads, 4, n_labels), dtype=np.float64
    )

    for group in prange(n_groups, nogil=True, schedule="static", num_threads=n_threads):
        thread = threadid()
        converged_view[group] = estimate_parameters(
            y[group],
            sample_weight[group],
            tol,
            max_iter,
            parameters_view[group],
            &n_iter_view[group],
            order[thread],
            n_ranked[thread],
            work[thread],
        )
    return parameters, n_iter, converged.astype(bool)
