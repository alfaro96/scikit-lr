"""Split criterion of the decision tree label ranker, based on the Mallows model."""

import numpy as np

cimport cython
from libc.math cimport NAN, isnan

from sklearn.tree._criterion cimport Criterion
from sklearn.utils._typedefs cimport float64_t, intp_t

from sklr._distributions._mallows cimport (
    estimate_center,
    estimate_spread,
    solve_spread,
)
from sklr._distributions._sorting cimport rank_by_scores

# The parts of a node whose rankings the impurity is computed from
cdef enum:
    _NODE
    _LEFT
    _RIGHT


# The caller creates the criterion with the number of samples and labels of the
# training set and its distinct rankings, so the indices of the loops, which are
# non-negative, are within the bounds of the arrays: the node has at most
# n_samples samples, whose indices from the splitter are below n_samples, with at
# most n_patterns distinct rankings of at most n_labels labels. The arrays are
# created in __cinit__, and those of the node are set by init, which the splitter
# calls before any other method
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef class MallowsCriterion(Criterion):
    """Impurity of the rankings of a node from the spread of their Mallows model.

    As in [cheng_decision_2009], the Mallows model of a node is estimated only on the
    labels that some of its rankings rank. Its impurity is
    :math:`1 / (1 + \\hat{\\theta})`, where :math:`\\hat{\\theta}` is the estimated
    spread of that model, so that it decreases with the spread, is zero when the spread
    is infinite and one when it is zero. It is also zero when the rankings of the node
    are consistent, that is, when no two of them order a pair of labels differently,
    which [cheng_decision_2009] takes as a pure node. The value of a node is its center
    ranking on those labels, with ``nan`` for the others.

    The model is estimated from the distinct rankings `patterns` of the training
    set, each weighted by the total weight of its samples in the node, which gives
    the same estimate as the samples themselves. `pattern_of` holds the index of the
    ranking of each sample.
    """

    cdef intp_t max_iter
    cdef const float64_t[:, ::1] patterns
    cdef const intp_t[::1] pattern_of

    # The number of node values computed and of their centers that did not converge
    cdef public intp_t n_values
    cdef public intp_t n_not_converged

    # The distinct rankings of the node, and the number and weight of their samples
    # in the node and in the left child, updated as the samples move between the
    # children. The numbers tell exactly which rankings each child has, while the
    # weights of the right child, found by subtraction, may be rounded
    cdef intp_t[::1] node_patterns
    cdef intp_t n_node_patterns
    cdef intp_t[::1] pattern_count_total
    cdef intp_t[::1] pattern_count_left
    cdef float64_t[::1] pattern_weight_total
    cdef float64_t[::1] pattern_weight_left

    # The distinct rankings of a node or a child on their ranked labels, their
    # weights, and the buffers of the estimation of the Mallows model
    cdef float64_t[:, ::1] node_y
    cdef float64_t[::1] node_weight
    cdef intp_t[::1] labels
    cdef intp_t[::1] center
    cdef float64_t[:, ::1] completed
    cdef float64_t[::1] scores
    cdef intp_t[:, ::1] work

    # Whether the rankings of the node are complete, and then the Borda scores of
    # the labels, and the number and weight of the rankings that put each label
    # ahead of each other one, in the node and in each child, which are updated as
    # the samples move between the children. The numbers decide the consistency
    # exactly, as those of the distinct rankings above
    cdef bint is_complete
    cdef float64_t[::1] scores_total
    cdef float64_t[::1] scores_left
    cdef float64_t[::1] scores_right
    cdef float64_t[:, ::1] ahead_total
    cdef float64_t[:, ::1] ahead_left
    cdef float64_t[:, ::1] ahead_right
    cdef intp_t[:, ::1] count_total
    cdef intp_t[:, ::1] count_left
    cdef intp_t[:, ::1] count_right

    def __cinit__(
        self,
        intp_t n_labels,
        intp_t n_samples,
        intp_t max_iter,
        const float64_t[:, ::1] patterns,
        const intp_t[::1] pattern_of,
    ):
        cdef intp_t n_patterns = patterns.shape[0]

        self.start = 0
        self.pos = 0
        self.end = 0

        self.n_outputs = n_labels
        self.n_samples = n_samples
        self.n_node_samples = 0
        self.weighted_n_node_samples = 0.0
        self.weighted_n_left = 0.0
        self.weighted_n_right = 0.0

        self.max_iter = max_iter
        self.n_values = 0
        self.n_not_converged = 0
        self.patterns = patterns
        self.pattern_of = pattern_of

        self.node_patterns = np.empty(n_patterns, dtype=np.intp)
        self.n_node_patterns = 0
        self.pattern_count_total = np.empty(n_patterns, dtype=np.intp)
        self.pattern_count_left = np.empty(n_patterns, dtype=np.intp)
        self.pattern_weight_total = np.empty(n_patterns, dtype=np.float64)
        self.pattern_weight_left = np.empty(n_patterns, dtype=np.float64)

        self.node_y = np.empty((n_patterns, n_labels), dtype=np.float64)
        self.node_weight = np.empty(n_patterns, dtype=np.float64)
        self.labels = np.empty(n_labels, dtype=np.intp)
        self.center = np.empty(n_labels, dtype=np.intp)
        self.completed = np.empty((n_patterns, n_labels), dtype=np.float64)
        self.scores = np.empty(n_labels, dtype=np.float64)
        self.work = np.empty((4, n_labels), dtype=np.intp)

        self.is_complete = False
        self.scores_total = np.empty(n_labels, dtype=np.float64)
        self.scores_left = np.empty(n_labels, dtype=np.float64)
        self.scores_right = np.empty(n_labels, dtype=np.float64)
        self.ahead_total = np.empty((n_labels, n_labels), dtype=np.float64)
        self.ahead_left = np.empty((n_labels, n_labels), dtype=np.float64)
        self.ahead_right = np.empty((n_labels, n_labels), dtype=np.float64)
        self.count_total = np.empty((n_labels, n_labels), dtype=np.intp)
        self.count_left = np.empty((n_labels, n_labels), dtype=np.intp)
        self.count_right = np.empty((n_labels, n_labels), dtype=np.intp)

    def __reduce__(self):
        return (
            type(self),
            (
                self.n_outputs,
                self.n_samples,
                self.max_iter,
                np.asarray(self.patterns),
                np.asarray(self.pattern_of),
            ),
        )

    def __deepcopy__(self, memo):
        # BaseDecisionTree._fit copies the criterion that it receives, which would
        # lose the convergence counts. Each fit creates its own criterion, so it
        # does not need to be protected from changes
        return self

    cdef int init(
        self,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
        float64_t weighted_n_samples,
        const intp_t[:] sample_indices,
        intp_t start,
        intp_t end,
    ) except -1 nogil:
        """Initialize the criterion at the node ``sample_indices[start:end]``."""
        cdef intp_t p, label, other, pattern
        cdef float64_t weight

        self.y = y
        self.sample_weight = sample_weight
        self.sample_indices = sample_indices
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_samples = weighted_n_samples
        self.weighted_n_node_samples = 0.0
        for p in range(start, end):
            self.pattern_count_total[self.pattern_of[sample_indices[p]]] = 0
            self.pattern_weight_total[self.pattern_of[sample_indices[p]]] = 0.0
        self.n_node_patterns = 0
        for p in range(start, end):
            pattern = self.pattern_of[sample_indices[p]]
            weight = self._weight(sample_indices[p])
            self.weighted_n_node_samples += weight
            if self.pattern_count_total[pattern] == 0:
                self.node_patterns[self.n_node_patterns] = pattern
                self.n_node_patterns += 1
            self.pattern_count_total[pattern] += 1
            self.pattern_weight_total[pattern] += weight

        self.is_complete = True
        for p in range(start, end):
            for label in range(self.n_outputs):
                if isnan(y[sample_indices[p], label]):
                    self.is_complete = False
        if self.is_complete:
            for label in range(self.n_outputs):
                self.scores_total[label] = 0.0
                for other in range(self.n_outputs):
                    self.ahead_total[label, other] = 0.0
                    self.count_total[label, other] = 0
            for p in range(start, end):
                self._add_ranking(
                    sample_indices[p],
                    self.scores_total,
                    self.ahead_total,
                    self.count_total,
                )
        self.reset()
        return 0

    cdef int reset(self) except -1 nogil:
        """Reset the criterion at ``pos=start``."""
        cdef intp_t label, other, i
        self.pos = self.start
        self.weighted_n_left = 0.0
        self.weighted_n_right = self.weighted_n_node_samples
        for i in range(self.n_node_patterns):
            self.pattern_count_left[self.node_patterns[i]] = 0
            self.pattern_weight_left[self.node_patterns[i]] = 0.0
        if self.is_complete:
            for label in range(self.n_outputs):
                self.scores_left[label] = 0.0
                for other in range(self.n_outputs):
                    self.ahead_left[label, other] = 0.0
                    self.count_left[label, other] = 0
        return 0

    cdef int reverse_reset(self) except -1 nogil:
        """Reset the criterion at ``pos=end``."""
        cdef intp_t label, other, i, pattern
        self.pos = self.end
        self.weighted_n_left = self.weighted_n_node_samples
        self.weighted_n_right = 0.0
        for i in range(self.n_node_patterns):
            pattern = self.node_patterns[i]
            self.pattern_count_left[pattern] = self.pattern_count_total[pattern]
            self.pattern_weight_left[pattern] = self.pattern_weight_total[pattern]
        if self.is_complete:
            for label in range(self.n_outputs):
                self.scores_left[label] = self.scores_total[label]
                for other in range(self.n_outputs):
                    self.ahead_left[label, other] = self.ahead_total[label, other]
                    self.count_left[label, other] = self.count_total[label, other]
        return 0

    cdef int update(self, intp_t new_pos) except -1 nogil:
        """Move the samples ``sample_indices[pos:new_pos]`` to the left child."""
        cdef intp_t p, sample
        cdef float64_t weight
        if new_pos < self.pos:
            self.reset()
        for p in range(self.pos, new_pos):
            sample = self.sample_indices[p]
            weight = self._weight(sample)
            self.weighted_n_left += weight
            self.pattern_count_left[self.pattern_of[sample]] += 1
            self.pattern_weight_left[self.pattern_of[sample]] += weight
            if self.is_complete:
                self._add_ranking(
                    self.sample_indices[p],
                    self.scores_left,
                    self.ahead_left,
                    self.count_left,
                )
        self.weighted_n_right = self.weighted_n_node_samples - self.weighted_n_left
        self.pos = new_pos
        return 0

    cdef float64_t node_impurity(self) noexcept nogil:
        """Compute the impurity of the node."""
        if self.is_complete:
            return self._complete_impurity(
                self.scores_total,
                self.ahead_total,
                self.count_total,
                self.weighted_n_node_samples,
            )
        return self._impurity(_NODE)

    cdef void children_impurity(
        self, float64_t* impurity_left, float64_t* impurity_right
    ) noexcept nogil:
        """Compute the impurities of the left and right children."""
        cdef intp_t label, other
        if self.is_complete:
            for label in range(self.n_outputs):
                self.scores_right[label] = (
                    self.scores_total[label] - self.scores_left[label]
                )
                for other in range(self.n_outputs):
                    self.ahead_right[label, other] = (
                        self.ahead_total[label, other] - self.ahead_left[label, other]
                    )
                    self.count_right[label, other] = (
                        self.count_total[label, other] - self.count_left[label, other]
                    )
            impurity_left[0] = self._complete_impurity(
                self.scores_left, self.ahead_left, self.count_left, self.weighted_n_left
            )
            impurity_right[0] = self._complete_impurity(
                self.scores_right,
                self.ahead_right,
                self.count_right,
                self.weighted_n_right,
            )
            return
        impurity_left[0] = self._impurity(_LEFT)
        impurity_right[0] = self._impurity(_RIGHT)

    cdef void node_value(self, float64_t* dest) noexcept nogil:
        """Write the center ranking of the node, ``nan`` for its unranked labels."""
        cdef intp_t n_rankings, n_labels, label
        cdef bint converged = True

        n_labels = self._gather(_NODE, &n_rankings)
        if n_labels > 1:
            converged = self._estimate_center(n_rankings, n_labels)
        elif n_labels == 1:
            self.center[0] = 1
        for label in range(self.n_outputs):
            dest[label] = NAN
        for label in range(n_labels):
            dest[self.labels[label]] = self.center[label]
        self.n_values += 1
        if not converged:
            self.n_not_converged += 1

    cdef inline float64_t _weight(self, intp_t sample) noexcept nogil:
        """Return the weight of a sample, ``1`` without weights."""
        if self.sample_weight is None:
            return 1.0
        return self.sample_weight[sample]

    cdef void _add_ranking(
        self,
        intp_t sample,
        float64_t[::1] scores,
        float64_t[:, ::1] ahead,
        intp_t[:, ::1] count,
    ) noexcept nogil:
        """Add a complete ranking to the Borda scores and the pairwise counts."""
        cdef intp_t label, other
        cdef float64_t weight = self._weight(sample)
        for label in range(self.n_outputs):
            scores[label] += weight * (self.n_outputs - self.y[sample, label] + 1)
            for other in range(self.n_outputs):
                if self.y[sample, label] < self.y[sample, other]:
                    ahead[label, other] += weight
                    count[label, other] += 1

    # The children that the splitter evaluates are not empty, so their weight, the
    # divisor of the mean distance, is positive
    @cython.cdivision(True)
    cdef float64_t _complete_impurity(
        self,
        const float64_t[::1] scores,
        const float64_t[:, ::1] ahead,
        const intp_t[:, ::1] count,
        float64_t weight,
    ) noexcept nogil:
        """Compute the impurity of complete rankings from their summaries.

        The rankings are consistent when no pair of labels is ordered both ways. The
        center ranking is that of the Borda count, which Algorithm 1 of
        [cheng_decision_2009] does not iterate for complete rankings, and the
        weighted sum of the Kendall distances to it adds up, over each pair of
        labels, the weight of the rankings that order it unlike the center.
        """
        cdef intp_t n_labels = self.n_outputs
        cdef intp_t label, other
        cdef bint consistent = True
        cdef float64_t distance = 0.0

        for label in range(n_labels):
            for other in range(label + 1, n_labels):
                if count[label, other] > 0 and count[other, label] > 0:
                    consistent = False
        if consistent:
            return 0.0
        rank_by_scores(scores, self.center)
        for label in range(n_labels):
            for other in range(n_labels):
                if self.center[label] < self.center[other]:
                    distance += ahead[other, label]
        return 1.0 / (1.0 + solve_spread(distance / weight, n_labels))

    cdef void _pattern_count(
        self, int part, intp_t pattern, intp_t* count, float64_t* weight
    ) noexcept nogil:
        """Get the number and weight of the samples of a ranking in a part."""
        if part == _NODE:
            count[0] = self.pattern_count_total[pattern]
            weight[0] = self.pattern_weight_total[pattern]
        elif part == _LEFT:
            count[0] = self.pattern_count_left[pattern]
            weight[0] = self.pattern_weight_left[pattern]
        else:
            count[0] = (
                self.pattern_count_total[pattern] - self.pattern_count_left[pattern]
            )
            weight[0] = (
                self.pattern_weight_total[pattern] - self.pattern_weight_left[pattern]
            )

    cdef intp_t _gather(self, int part, intp_t* n_rankings) noexcept nogil:
        """Copy the distinct rankings of the node or of a child on their ranked labels.

        Returns the number of labels that some of the rankings rank, whose indices
        go to the first elements of `labels`, and sets ``n_rankings[0]`` to the
        number of distinct rankings, whose positions of those labels go to the first
        rows and columns of `node_y` and whose total weights go to `node_weight`.
        """
        cdef intp_t n_labels = 0
        cdef intp_t i, pattern, label, column, count
        cdef float64_t weight
        cdef bint is_ranked

        for label in range(self.n_outputs):
            is_ranked = False
            for i in range(self.n_node_patterns):
                pattern = self.node_patterns[i]
                self._pattern_count(part, pattern, &count, &weight)
                if count > 0 and not isnan(self.patterns[pattern, label]):
                    is_ranked = True
                    break
            if is_ranked:
                self.labels[n_labels] = label
                n_labels += 1

        n_rankings[0] = 0
        for i in range(self.n_node_patterns):
            pattern = self.node_patterns[i]
            self._pattern_count(part, pattern, &count, &weight)
            if count == 0:
                continue
            self.node_weight[n_rankings[0]] = weight
            for column in range(n_labels):
                self.node_y[n_rankings[0], column] = self.patterns[
                    pattern, self.labels[column]
                ]
            n_rankings[0] += 1
        return n_labels

    cdef bint _is_consistent(self, intp_t n_rankings, intp_t n_labels) noexcept nogil:
        """Tell whether no two gathered rankings order a pair of labels differently."""
        cdef intp_t label, other, ranking
        cdef bint ahead, behind

        for label in range(n_labels):
            for other in range(label + 1, n_labels):
                ahead = False
                behind = False
                for ranking in range(n_rankings):
                    if isnan(self.node_y[ranking, label]) or isnan(
                        self.node_y[ranking, other]
                    ):
                        continue
                    if self.node_y[ranking, label] < self.node_y[ranking, other]:
                        ahead = True
                    else:
                        behind = True
                    if ahead and behind:
                        return False
        return True

    cdef bint _estimate_center(self, intp_t n_rankings, intp_t n_labels) noexcept nogil:
        """Estimate the center ranking of the gathered rankings into `center`."""
        cdef intp_t n_iter
        # The slices keep the row stride of the whole buffers. Cython takes them as
        # C-contiguous without checking or copying them, since only their last
        # axis has to be contiguous, and the estimation indexes them by rows
        return estimate_center(
            self.node_y[:n_rankings, :n_labels],
            self.node_weight[:n_rankings],
            self.max_iter,
            self.center[:n_labels],
            &n_iter,
            self.completed[:n_rankings, :n_labels],
            self.scores[:n_labels],
            self.work[:, :n_labels],
        )

    # The spread is not negative, so the divisor is at least one
    @cython.cdivision(True)
    cdef float64_t _impurity(self, int part) noexcept nogil:
        """Compute the impurity of the node or of one of its children."""
        cdef intp_t n_rankings, n_labels
        cdef float64_t spread

        n_labels = self._gather(part, &n_rankings)
        if n_labels <= 1 or self._is_consistent(n_rankings, n_labels):
            return 0.0
        # Only the centers of the nodes, in node_value, count for the convergence
        # warning, not those of the candidate splits
        self._estimate_center(n_rankings, n_labels)
        spread = estimate_spread(
            self.node_y[:n_rankings, :n_labels],
            self.node_weight[:n_rankings],
            self.center[:n_labels],
            self.completed[0, :n_labels],
            self.work[:2, :n_labels],
        )
        # An infinite spread gives exactly zero
        return 1.0 / (1.0 + spread)
