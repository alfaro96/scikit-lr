"""Split criteria of the decision tree rankers."""

import numpy as np

cimport cython
from libc.math cimport NAN, isnan

from sklr._distributions._mallows cimport (
    estimate_center,
    estimate_spread,
    solve_spread,
)
from sklr._distributions._sorting cimport rank_by_scores


# The caller creates the criterion with the number of samples and labels of the
# training set, so the indices of the loops, which are non-negative, are within the
# bounds of the arrays: the indices of the samples from the splitter are below
# n_samples, those of the labels below n_labels, and the three parts of the node
# index the first axis of the pairwise statistics. The arrays are created in
# __cinit__, and those of the node are set by init, which the splitter calls before
# any other method
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef class RankingCriterion(Criterion):
    """Base of the impurities of rankings computed from their pairwise statistics.

    The criterion keeps, for the node and each child, the weight and the number of
    the rankings that put each label ahead of each other one and that tie them,
    counting a pair only in the rankings that rank both labels. Those of the node
    are computed in `init`, those of the left child as its samples move in `update`,
    and those of the right child by subtraction. A part is consistent, and its
    impurity zero, when no two of its rankings order a pair of labels differently;
    the subclasses compute the impurity of the other parts in `_impurity`, and can
    keep their own statistics through the methods that `init`, `reset`,
    `reverse_reset` and `update` call.
    """

    # Cython also calls this method with the arguments of the subclasses, so it takes
    # and ignores the extra ones
    def __cinit__(self, intp_t n_labels, intp_t n_samples, *args, **kwargs):
        self.start = 0
        self.pos = 0
        self.end = 0

        self.n_outputs = n_labels
        self.n_samples = n_samples
        self.n_node_samples = 0
        self.weighted_n_node_samples = 0.0
        self.weighted_n_left = 0.0
        self.weighted_n_right = 0.0

        self.precedence_weight = np.empty((3, n_labels, n_labels), dtype=np.float64)
        self.precedence_count = np.empty((3, n_labels, n_labels), dtype=np.intp)
        self.tie_weight = np.empty((3, n_labels, n_labels), dtype=np.float64)
        self.tie_count = np.empty((3, n_labels, n_labels), dtype=np.intp)

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
        cdef intp_t p
        cdef float64_t weight

        self.y = y
        self.sample_weight = sample_weight
        self.sample_indices = sample_indices
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_samples = weighted_n_samples
        self.weighted_n_node_samples = 0.0
        self._clear(_NODE)
        for p in range(start, end):
            weight = self._weight(sample_indices[p])
            self.weighted_n_node_samples += weight
            self._add_pairs(sample_indices[p], weight, _NODE)
        self._init_node()
        self.reset()
        return 0

    cdef int reset(self) except -1 nogil:
        """Reset the criterion at ``pos=start``."""
        self.pos = self.start
        self.weighted_n_left = 0.0
        self.weighted_n_right = self.weighted_n_node_samples
        self._clear(_LEFT)
        self._reset_left()
        return 0

    cdef int reverse_reset(self) except -1 nogil:
        """Reset the criterion at ``pos=end``."""
        cdef intp_t label, other
        self.pos = self.end
        self.weighted_n_left = self.weighted_n_node_samples
        self.weighted_n_right = 0.0
        for label in range(self.n_outputs):
            for other in range(self.n_outputs):
                self.precedence_weight[_LEFT, label, other] = self.precedence_weight[
                    _NODE, label, other
                ]
                self.precedence_count[_LEFT, label, other] = self.precedence_count[
                    _NODE, label, other
                ]
                self.tie_weight[_LEFT, label, other] = self.tie_weight[
                    _NODE, label, other
                ]
                self.tie_count[_LEFT, label, other] = self.tie_count[
                    _NODE, label, other
                ]
        self._reverse_reset_left()
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
            self._add_pairs(sample, weight, _LEFT)
            self._add_left(sample, weight)
        self.weighted_n_right = self.weighted_n_node_samples - self.weighted_n_left
        self.pos = new_pos
        return 0

    cdef float64_t node_impurity(self) noexcept nogil:
        """Compute the impurity of the node."""
        return self._part_impurity(_NODE)

    cdef void children_impurity(
        self, float64_t* impurity_left, float64_t* impurity_right
    ) noexcept nogil:
        """Compute the impurities of the left and right children."""
        cdef intp_t label, other
        for label in range(self.n_outputs):
            for other in range(self.n_outputs):
                self.precedence_weight[_RIGHT, label, other] = (
                    self.precedence_weight[_NODE, label, other]
                    - self.precedence_weight[_LEFT, label, other]
                )
                self.precedence_count[_RIGHT, label, other] = (
                    self.precedence_count[_NODE, label, other]
                    - self.precedence_count[_LEFT, label, other]
                )
                self.tie_weight[_RIGHT, label, other] = (
                    self.tie_weight[_NODE, label, other]
                    - self.tie_weight[_LEFT, label, other]
                )
                self.tie_count[_RIGHT, label, other] = (
                    self.tie_count[_NODE, label, other]
                    - self.tie_count[_LEFT, label, other]
                )
        impurity_left[0] = self._part_impurity(_LEFT)
        impurity_right[0] = self._part_impurity(_RIGHT)

    @cython.final
    cdef float64_t _part_impurity(self, int part) noexcept nogil:
        """Compute the impurity of a part, zero if its rankings are consistent."""
        if self._is_consistent(part):
            return 0.0
        return self._impurity(part)

    @cython.final
    cdef float64_t _weight(self, intp_t sample) noexcept nogil:
        """Return the weight of a sample, ``1`` without weights."""
        if self.sample_weight is None:
            return 1.0
        return self.sample_weight[sample]

    @cython.final
    cdef void _clear(self, int part) noexcept nogil:
        """Set the pairwise statistics of a part to zero."""
        cdef intp_t label, other
        for label in range(self.n_outputs):
            for other in range(self.n_outputs):
                self.precedence_weight[part, label, other] = 0.0
                self.precedence_count[part, label, other] = 0
                self.tie_weight[part, label, other] = 0.0
                self.tie_count[part, label, other] = 0

    @cython.final
    cdef void _add_pairs(
        self, intp_t sample, float64_t weight, int part
    ) noexcept nogil:
        """Add the pairs of labels ordered by the ranking of a sample to a part."""
        cdef intp_t label, other
        cdef float64_t position, other_position
        # The comparisons with nan are false, so the pairs with an unranked label
        # are left out
        for label in range(self.n_outputs):
            position = self.y[sample, label]
            for other in range(label + 1, self.n_outputs):
                other_position = self.y[sample, other]
                if position < other_position:
                    self.precedence_weight[part, label, other] += weight
                    self.precedence_count[part, label, other] += 1
                elif position > other_position:
                    self.precedence_weight[part, other, label] += weight
                    self.precedence_count[part, other, label] += 1
                elif position == other_position:
                    self.tie_weight[part, label, other] += weight
                    self.tie_weight[part, other, label] += weight
                    self.tie_count[part, label, other] += 1
                    self.tie_count[part, other, label] += 1

    @cython.final
    cdef bint _is_consistent(self, int part) noexcept nogil:
        """Tell whether no two rankings of a part order a pair of labels differently."""
        cdef intp_t label, other
        for label in range(self.n_outputs):
            for other in range(label + 1, self.n_outputs):
                if (
                    (self.precedence_count[part, label, other] > 0)
                    + (self.tie_count[part, label, other] > 0)
                    + (self.precedence_count[part, other, label] > 0)
                ) > 1:
                    return False
        return True

    @cython.final
    cdef float64_t _part_weight(self, int part) noexcept nogil:
        """Return the total weight of the samples of a part."""
        if part == _NODE:
            return self.weighted_n_node_samples
        if part == _LEFT:
            return self.weighted_n_left
        return self.weighted_n_right

    cdef void _init_node(self) noexcept nogil:
        """Initialize the statistics of a subclass once the node pairs are counted."""

    cdef void _reset_left(self) noexcept nogil:
        """Reset the statistics of a subclass for an empty left child."""

    cdef void _reverse_reset_left(self) noexcept nogil:
        """Reset the statistics of a subclass for a left child with the whole node."""

    cdef void _add_left(self, intp_t sample, float64_t weight) noexcept nogil:
        """Add a sample to the statistics of a subclass for the left child."""

    cdef float64_t _impurity(self, int part) noexcept nogil:
        """Compute the impurity of a part whose rankings are not consistent."""
        return 0.0


# The arrays of the distinct rankings have n_patterns rows, and the node has at most
# that many distinct rankings, of at most n_labels labels
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef class MallowsCriterion(RankingCriterion):
    """Impurity of the rankings of a node from the spread of their Mallows model.

    As in [cheng_decision_2009], the Mallows model of a node is estimated only on the
    labels that some of its rankings rank. Its impurity is
    :math:`1 / (1 + \\hat{\\theta})`, where :math:`\\hat{\\theta}` is the estimated
    spread of that model, so that it decreases with the spread, is zero when the spread
    is infinite and one when it is zero. It is also zero when the rankings of the node
    are consistent, which [cheng_decision_2009] takes as a pure node. The value of a
    node is its center ranking on those labels, with ``nan`` for the others.

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
    # children, as the pairwise statistics
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

    # Whether the rankings of the node are complete, so that the impurity of each
    # part follows from its pairwise statistics
    cdef bint is_complete

    def __cinit__(
        self,
        intp_t n_labels,
        intp_t n_samples,
        intp_t max_iter,
        const float64_t[:, ::1] patterns,
        const intp_t[::1] pattern_of,
    ):
        cdef intp_t n_patterns = patterns.shape[0]

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

    cdef void _init_node(self) noexcept nogil:
        """Count the distinct rankings of the node and whether they are complete."""
        cdef intp_t p, label, pattern, sample

        for p in range(self.start, self.end):
            self.pattern_count_total[self.pattern_of[self.sample_indices[p]]] = 0
            self.pattern_weight_total[self.pattern_of[self.sample_indices[p]]] = 0.0
        self.n_node_patterns = 0
        self.is_complete = True
        for p in range(self.start, self.end):
            sample = self.sample_indices[p]
            pattern = self.pattern_of[sample]
            if self.pattern_count_total[pattern] == 0:
                self.node_patterns[self.n_node_patterns] = pattern
                self.n_node_patterns += 1
                for label in range(self.n_outputs):
                    if isnan(self.patterns[pattern, label]):
                        self.is_complete = False
            self.pattern_count_total[pattern] += 1
            self.pattern_weight_total[pattern] += self._weight(sample)

    cdef void _reset_left(self) noexcept nogil:
        """Remove the distinct rankings of the node from the left child."""
        cdef intp_t i
        for i in range(self.n_node_patterns):
            self.pattern_count_left[self.node_patterns[i]] = 0
            self.pattern_weight_left[self.node_patterns[i]] = 0.0

    cdef void _reverse_reset_left(self) noexcept nogil:
        """Put the distinct rankings of the node in the left child."""
        cdef intp_t i, pattern
        for i in range(self.n_node_patterns):
            pattern = self.node_patterns[i]
            self.pattern_count_left[pattern] = self.pattern_count_total[pattern]
            self.pattern_weight_left[pattern] = self.pattern_weight_total[pattern]

    cdef void _add_left(self, intp_t sample, float64_t weight) noexcept nogil:
        """Add the ranking of a sample to the left child."""
        self.pattern_count_left[self.pattern_of[sample]] += 1
        self.pattern_weight_left[self.pattern_of[sample]] += weight

    cdef float64_t _impurity(self, int part) noexcept nogil:
        """Compute the impurity of a part whose rankings are not consistent."""
        if self.is_complete:
            return self._complete_impurity(part)
        return self._incomplete_impurity(part)

    # The parts that are not consistent have rankings, whose samples have positive
    # weights because the splitter discards the others, so the weight of the part,
    # the divisor of the mean distance, is positive, up to the rounding of the
    # subtraction for the right child, as in the criteria of scikit-learn
    @cython.cdivision(True)
    cdef float64_t _complete_impurity(self, int part) noexcept nogil:
        """Compute the impurity of complete rankings from their pairwise statistics.

        The center ranking is that of the Borda count, which Algorithm 1 of
        [cheng_decision_2009] does not iterate for complete rankings. The Borda
        score of a label is the weight of the part plus, for each other label, the
        weight of the rankings that put it ahead of that one, so the labels are
        ranked by the latter sum.
        The weighted sum of the Kendall distances to the center adds up, over each
        pair of labels, the weight of the rankings that order it unlike the center.
        """
        cdef intp_t n_labels = self.n_outputs
        cdef intp_t label, other
        cdef float64_t distance = 0.0

        for label in range(n_labels):
            self.scores[label] = 0.0
            for other in range(n_labels):
                self.scores[label] += self.precedence_weight[part, label, other]
        rank_by_scores(self.scores, self.center)
        for label in range(n_labels):
            for other in range(n_labels):
                if self.center[label] < self.center[other]:
                    distance += self.precedence_weight[part, other, label]
        return 1.0 / (1.0 + solve_spread(distance / self._part_weight(part), n_labels))

    # The spread is not negative, so the divisor is at least one
    @cython.cdivision(True)
    cdef float64_t _incomplete_impurity(self, int part) noexcept nogil:
        """Compute the impurity of incomplete rankings from their Mallows model."""
        cdef intp_t n_rankings, n_labels
        cdef float64_t spread

        n_labels = self._gather(part, &n_rankings)
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
