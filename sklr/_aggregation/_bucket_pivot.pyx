"""Consensus bucket order of partial rankings by the Bucket Pivot Algorithm."""

import numpy as np

cimport cython
from cython.parallel cimport prange, threadid
from libc.math cimport fabs, isnan

from sklearn.utils._typedefs cimport float64_t, intp_t

# The threshold of [aledo_utopia_2017] that decides whether an item precedes, is
# tied with or follows the pivots
cdef float64_t _BETA = 0.25

# The entries of the pair order matrix are ratios of sums of weights, so values
# that are equal in exact arithmetic, such as the DANI indices of two labels that
# are only uncertain about each other, or an entry of 3/4, may differ in the last
# bits. Differences below this tolerance are taken as ties, which are then decided
# as the rules say instead of by rounding
cdef float64_t _TOLERANCE = 1e-9

cdef enum:
    _LEFT
    _CENTRAL
    _RIGHT


# The indices of the loops are within the bounds of the arrays, which the callers
# create with the right sizes, so checking them would only slow down the loops, and
# the only division is by a positive total
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void pair_order_matrix(
    const float64_t[:, ::1] precedes,
    const float64_t[:, ::1] ties,
    float64_t[:, ::1] matrix,
) noexcept nogil:
    """Compute the pair order matrix of the precedences and ties of some rankings.

    ``precedes[i, j]`` is the weight of the rankings that put the label of index
    ``i`` ahead of that of index ``j``, and ``ties[i, j]`` that of the rankings that
    tie them. Each entry of `matrix` is the weighted fraction of the rankings
    ranking both labels that put the first one ahead, with ties counting one half,
    that is, :math:`C(y_i, y_j)` in equation (3) of [alfaro_learning_2021] for the
    labels :math:`y_i` and :math:`y_j`, and ``0.5`` when no ranking ranks both.
    """
    cdef intp_t n_labels = matrix.shape[0]
    cdef intp_t label, other
    cdef float64_t total

    for label in range(n_labels):
        matrix[label, label] = 0.5
        for other in range(label + 1, n_labels):
            total = precedes[label, other] + ties[label, other] + precedes[other, label]
            if total > 0:
                matrix[label, other] = (
                    precedes[label, other] + 0.5 * ties[label, other]
                ) / total
                matrix[other, label] = (
                    precedes[other, label] + 0.5 * ties[label, other]
                ) / total
            else:
                matrix[label, other] = 0.5
                matrix[other, label] = 0.5


cdef inline float64_t _utopian(float64_t value) noexcept nogil:
    """Round an entry of a pair order matrix to that of its utopian matrix.

    The entry of the labels :math:`y_i` and :math:`y_j` is rounded from
    :math:`C(y_i, y_j)`, as in Example 1 of [aledo_utopia_2017], and not from
    :math:`C(y_j, y_i)` as its Definition 1 writes.
    """
    if value > 0.75 + _TOLERANCE:
        return 1.0
    if value < 0.25 - _TOLERANCE:
        return 0.0
    return 0.5


cdef intp_t _place(float64_t total, intp_t n_pivots) noexcept nogil:
    """Place an item from the sum of the entries of the pivots in its column."""
    # The mean is compared through the sum, and the bounds of the central bucket
    # are closed
    if total < (0.5 - _BETA) * n_pivots - _TOLERANCE:
        return _LEFT
    if total > (0.5 + _BETA) * n_pivots + _TOLERANCE:
        return _RIGHT
    return _CENTRAL


# From here on the arrays are not only indexed by the loops: dani, totals, side and
# ranking are indexed by the labels in items, which the callers keep below the size
# of those arrays, and sorted_items within the segment of items being split
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _split(
    const float64_t[:, ::1] matrix,
    const float64_t[::1] dani,
    intp_t[::1] items,
    intp_t start,
    intp_t end,
    intp_t* n_left,
    intp_t* n_central,
    float64_t[::1] totals,
    intp_t[::1] side,
    intp_t[::1] sorted_items,
) noexcept nogil:
    """Split some items into those ahead of, tied with and behind the pivots.

    The items are ``items[start:end]``. This is one call of the Bucket Pivot
    Algorithm with the least indecision assumption, global DANI indices and
    two-stage multiple pivots, :math:`\\mathrm{LIA}_G^{\\mathrm{MP2}}` in
    [aledo_utopia_2017]. The pivot is the item with the smallest index among those
    with the least DANI, and the items, sorted by index, are placed in that order.
    They are rearranged in the three parts, each still sorted by index, whose sizes
    go to ``n_left[0]`` and ``n_central[0]``.
    """
    cdef intp_t pivot, item, other, i, j, n_pivots, part
    cdef float64_t least

    # The least DANI is found first, so that the ties within the tolerance go to
    # the smallest index whatever the order of the comparisons
    least = dani[items[start]]
    for i in range(start + 1, end):
        if dani[items[i]] < least:
            least = dani[items[i]]
    pivot = items[start]
    for i in range(start, end):
        if dani[items[i]] <= least + _TOLERANCE:
            pivot = items[i]
            break

    # The first stage adds each item tied with the pivots to them, and keeps the
    # sum of the entries of the pivots for every item
    for i in range(start, end):
        item = items[i]
        totals[item] = matrix[pivot, item]
    side[pivot] = _CENTRAL
    n_pivots = 1
    for i in range(start, end):
        item = items[i]
        if item == pivot:
            continue
        side[item] = _place(totals[item], n_pivots)
        if side[item] == _CENTRAL:
            n_pivots += 1
            for j in range(start, end):
                other = items[j]
                totals[other] += matrix[item, other]

    # The second stage places again the items left out against the pivots of the
    # first one, which do not change as items join them
    for i in range(start, end):
        item = items[i]
        if side[item] != _CENTRAL:
            side[item] = _place(totals[item], n_pivots)

    n_left[0] = 0
    n_central[0] = 0
    j = start
    # The values of the sides follow the order of the parts
    for part in range(_RIGHT + 1):
        for i in range(start, end):
            item = items[i]
            if side[item] == part:
                sorted_items[j] = item
                j += 1
        if part == _LEFT:
            n_left[0] = j - start
        elif part == _CENTRAL:
            n_central[0] = j - start - n_left[0]
    for i in range(start, end):
        items[i] = sorted_items[i]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef intp_t bucket_order(
    const float64_t[:, ::1] matrix,
    intp_t[::1] items,
    intp_t[::1] ranking,
    float64_t[:, ::1] work,
    intp_t[:, ::1] index_work,
) noexcept nogil:
    """Find a consensus bucket order of some labels from their pair order matrix.

    The labels `items`, distinct and sorted by index, are ordered by the Bucket
    Pivot Algorithm with the least indecision assumption, global DANI indices and
    two-stage multiple pivots, :math:`\\mathrm{LIA}_G^{\\mathrm{MP2}}` in
    [aledo_utopia_2017], with the DANI indices computed once among them. Their
    positions, dense from one, go to `ranking`, whose other entries are left
    unchanged, and `items` is reordered. Returns the number of buckets. The labels
    must be below ``n_labels``, the number of rows and columns of `matrix` and the
    length of `ranking`. `work` has shape ``(2, n_labels)`` and `index_work` shape
    ``(4, n_labels)``, and `items` must not share memory with them.
    """
    cdef intp_t n_items = items.shape[0]
    cdef float64_t[::1] dani = work[0], totals = work[1]
    cdef intp_t[::1] side = index_work[0], sorted_items = index_work[1]
    cdef intp_t[::1] stack_start = index_work[2], stack_end = index_work[3]
    cdef intp_t i, j, item, other, start, end, n_left, n_central
    cdef intp_t n_segments = 0, n_buckets = 0

    # Every DANI is divided by the same number of other items, so the sums are
    # compared instead
    for i in range(n_items):
        item = items[i]
        dani[item] = 0.0
        for j in range(n_items):
            other = items[j]
            if other != item:
                dani[item] += fabs(
                    matrix[item, other] - _utopian(matrix[item, other])
                )

    # The recursion of the algorithm on the items ahead of and behind the pivots
    # can be as deep as the number of items, which could overflow the stack of
    # the threads, so the segments still to split are kept in an explicit stack.
    # It holds disjoint segments that are not empty, at most n_items of them. Each
    # split leaves its tied items in their final place within items, so the
    # buckets follow the order of items once all the segments are split, and the
    # first item of each bucket is marked in ranking meanwhile
    if n_items > 0:
        stack_start[0] = 0
        stack_end[0] = n_items
        n_segments = 1
    while n_segments > 0:
        n_segments -= 1
        start = stack_start[n_segments]
        end = stack_end[n_segments]
        _split(
            matrix,
            dani,
            items,
            start,
            end,
            &n_left,
            &n_central,
            totals,
            side,
            sorted_items,
        )
        ranking[items[start + n_left]] = 1
        for i in range(start + n_left + 1, start + n_left + n_central):
            ranking[items[i]] = 0
        if n_left > 0:
            stack_start[n_segments] = start
            stack_end[n_segments] = start + n_left
            n_segments += 1
        if start + n_left + n_central < end:
            stack_start[n_segments] = start + n_left + n_central
            stack_end[n_segments] = end
            n_segments += 1
    for i in range(n_items):
        n_buckets += ranking[items[i]]
        ranking[items[i]] = n_buckets
    return n_buckets


@cython.boundscheck(False)
@cython.wraparound(False)
cdef intp_t aggregate_pairs(
    const float64_t[:, ::1] precedes,
    const float64_t[:, ::1] ties,
    intp_t[::1] ranking,
    float64_t[:, ::1] matrix,
    float64_t[:, ::1] work,
    intp_t[:, ::1] index_work,
) noexcept nogil:
    """Find a consensus bucket order of partial rankings from their pairs of labels.

    `precedes` and `ties` are as in ``pair_order_matrix``. The labels that some
    ranking of positive weight ranks together with another label are ordered by
    ``bucket_order``, and the others go together to a last bucket. Returns the
    number of buckets. `ranking` has length ``n_labels``, `matrix` shape
    ``(n_labels, n_labels)``, `work` shape ``(2, n_labels)`` and `index_work` shape
    ``(5, n_labels)``.
    """
    cdef intp_t n_labels = matrix.shape[0]
    cdef intp_t[::1] items = index_work[0]
    cdef intp_t label, other, n_items = 0, n_buckets
    cdef bint has_information

    pair_order_matrix(precedes, ties, matrix)
    for label in range(n_labels):
        # The labels without information keep this mark
        ranking[label] = 0
        has_information = False
        for other in range(n_labels):
            if (
                precedes[label, other] > 0
                or precedes[other, label] > 0
                or ties[label, other] > 0
            ):
                has_information = True
                break
        if has_information:
            items[n_items] = label
            n_items += 1
    n_buckets = bucket_order(
        matrix, items[:n_items], ranking, work, index_work[1:]
    )
    if n_items < n_labels:
        n_buckets += 1
        for label in range(n_labels):
            if ranking[label] == 0:
                ranking[label] = n_buckets
    return n_buckets


@cython.boundscheck(False)
@cython.wraparound(False)
cdef intp_t aggregate(
    const float64_t[:, ::1] y,
    const float64_t[::1] sample_weight,
    intp_t[::1] ranking,
    float64_t[:, :, ::1] matrices,
    float64_t[:, ::1] work,
    intp_t[:, ::1] index_work,
) noexcept nogil:
    """Find a consensus bucket order of possibly incomplete partial rankings.

    The weights of the pairs of labels of the rankings are aggregated by
    ``aggregate_pairs``. Returns the number of buckets. `matrices` has shape
    ``(3, n_labels, n_labels)``, `work` shape ``(2, n_labels)`` and `index_work`
    shape ``(5, n_labels)``.
    """
    cdef intp_t n_samples = y.shape[0], n_labels = y.shape[1]
    cdef float64_t[:, ::1] precedes = matrices[0], ties = matrices[1]
    cdef intp_t sample, label, other
    cdef float64_t weight

    for label in range(n_labels):
        for other in range(n_labels):
            precedes[label, other] = 0.0
            ties[label, other] = 0.0
    for sample in range(n_samples):
        weight = sample_weight[sample]
        if weight == 0:
            continue
        for label in range(n_labels):
            if isnan(y[sample, label]):
                continue
            for other in range(label + 1, n_labels):
                if isnan(y[sample, other]):
                    continue
                if y[sample, label] < y[sample, other]:
                    precedes[label, other] += weight
                elif y[sample, label] > y[sample, other]:
                    precedes[other, label] += weight
                else:
                    ties[label, other] += weight
                    ties[other, label] += weight
    return aggregate_pairs(
        precedes, ties, ranking, matrices[2], work, index_work
    )


# The callers pass inputs of matching shapes and at least one thread, and the other
# arrays are created with those shapes
@cython.boundscheck(False)
@cython.wraparound(False)
def aggregate_batch(
    const float64_t[:, :, ::1] y,
    const float64_t[:, ::1] sample_weight,
    int n_threads,
):
    """Find a consensus bucket order of each group of partial rankings."""
    cdef intp_t n_groups = y.shape[0], n_labels = y.shape[2]
    cdef intp_t group
    cdef int thread

    ranking = np.empty((n_groups, n_labels), dtype=np.intp)
    cdef intp_t[:, ::1] ranking_view = ranking

    # threadid() is below n_threads, so each thread works on its own row of the
    # buffers, which are created once per thread instead of once per group
    cdef float64_t[:, :, :, ::1] matrices = np.empty(
        (n_threads, 3, n_labels, n_labels), dtype=np.float64
    )
    cdef float64_t[:, :, ::1] work = np.empty(
        (n_threads, 2, n_labels), dtype=np.float64
    )
    cdef intp_t[:, :, ::1] index_work = np.empty(
        (n_threads, 5, n_labels), dtype=np.intp
    )

    for group in prange(n_groups, nogil=True, schedule="static", num_threads=n_threads):
        thread = threadid()
        aggregate(
            y[group],
            sample_weight[group],
            ranking_view[group],
            matrices[thread],
            work[thread],
            index_work[thread],
        )
    return ranking
