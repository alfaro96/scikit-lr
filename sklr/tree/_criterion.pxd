cimport cython
from sklearn.tree._criterion cimport Criterion
from sklearn.utils._typedefs cimport float64_t, intp_t

# The parts of a node whose rankings the impurity is computed from, which index the
# first axis of the pairwise statistics
cdef enum:
    _NODE
    _LEFT
    _RIGHT


cdef class RankingCriterion(Criterion):
    # The weight and the number of the rankings of each part that put each label ahead
    # of each other one and that tie them. The numbers tell exactly whether a part is
    # consistent, while the weights of the right child, found by subtraction, may be
    # rounded, except that they are exactly zero where their numbers are zero
    cdef float64_t[:, :, ::1] precedence_weight
    cdef intp_t[:, :, ::1] precedence_count
    cdef float64_t[:, :, ::1] tie_weight
    cdef intp_t[:, :, ::1] tie_count

    # Not meant to be overridden, so that they are called directly
    @cython.final
    cdef float64_t _part_impurity(self, int part) noexcept nogil

    @cython.final
    cdef float64_t _weight(self, intp_t sample) noexcept nogil

    @cython.final
    cdef void _clear(self, int part) noexcept nogil

    @cython.final
    cdef void _add_pairs(
        self, intp_t sample, float64_t weight, int part
    ) noexcept nogil

    @cython.final
    cdef bint _is_consistent(self, int part) noexcept nogil

    @cython.final
    cdef float64_t _part_weight(self, int part) noexcept nogil

    cdef void _init_node(self) noexcept nogil
    cdef void _reset_left(self) noexcept nogil
    cdef void _reverse_reset_left(self) noexcept nogil
    cdef void _add_left(self, intp_t sample, float64_t weight) noexcept nogil
    cdef float64_t _impurity(self, int part) noexcept nogil
