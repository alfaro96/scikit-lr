# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False


# =============================================================================
# Imports
# =============================================================================

# Local application
from ..consensus cimport RankAggregationAlgorithm
from .._types cimport (
    BOOL_t,
    DTYPE_t, DTYPE_t_1D, DTYPE_t_2D, DTYPE_t_3D, DTYPE_t_4D,
    INT64_t, INT64_t_1D, INT64_t_2D,
    SIZE_t, SIZE_t_1D)


# =============================================================================
# Enums
# =============================================================================

# Distance measures for the Mallows criterion
ctypedef enum DISTANCE:
    KENDALL


# =============================================================================
# Classes
# =============================================================================

# =============================================================================
# Criterion
# =============================================================================
cdef class Criterion:
    """Interface for impurity criteria."""

    # Hyperparameters
    cdef RankAggregationAlgorithm rank_algorithm  # Rank Aggregation algorithm
    cdef DISTANCE distance  # Distance function

    # Attributes
    cdef INT64_t_2D Y  # Rankings
    cdef DTYPE_t_1D sample_weight  # Sample weights

    cdef INT64_t n_classes  # Number of classes
    cdef INT64_t n_children  # Number of children

    cdef SIZE_t_1D samples  # Sample indexes
    cdef SIZE_t_1D pos  # Position of the samples in the children

    cdef INT64_t n_samples  # Number of samples in total
    cdef INT64_t n_node_samples  # Number of samples in the node
    cdef INT64_t_1D n_children_samples  # Number of samples in the children

    cdef DTYPE_t weighted_n_samples  # Weighted number of samples in total
    cdef DTYPE_t weighted_n_node_samples  # Weighted number of samples in the node  # nopep8
    cdef DTYPE_t_1D weighted_n_children_samples  # Weighted number of samples in the children  # nopep8

    cdef DTYPE_t impurity_node  # Impurity in the node
    cdef DTYPE_t_1D impurity_children  # Impurity in the children

    cdef INT64_t_1D consensus_node  # Consensus ranking in the node
    cdef INT64_t_2D consensus_children  # Consensus ranking in the children

    # The following attributes are required for a few of the subclasses.
    # We must define it here so that Cython's limited polymorphism will work.
    # Because we do not expect to instantiate a lot of these objects (just
    # once), the extra memory overhead of this setup should not be an issue.

    cdef INT64_t_2D Y_node  # Rankings in the node
    cdef INT64_t_2D Y_node_copy  # Copy of the rankings in the node
    cdef DTYPE_t_1D sample_weight_node  # Sample weights in the node

    cdef DTYPE_t_1D count_node  # Count in the node
    cdef DTYPE_t_2D count_children  # Count in the children

    cdef DTYPE_t_3D precedences_matrix_node  # Precedences matrix in the node
    cdef DTYPE_t_4D precedences_matrix_children  # Precedences matrix in the children  # nopep8

    # Methods
    cdef void init_from_rankings(self, INT64_t_2D Y, DTYPE_t_1D sample_weight,
                                 INT64_t max_children)

    cdef void init_from_record(self, INT64_t n_samples,
                               DTYPE_t weighted_n_samples,
                               DTYPE_t impurity, INT64_t_1D consensus,
                               DTYPE_t_1D count,
                               DTYPE_t_3D precedences_matrix)

    cdef void reset(self, SIZE_t_1D samples) nogil

    cdef void update(self, SIZE_t new_pos) nogil

    cdef void node_children(self) nogil

    cdef void node_impurity(self) nogil

    cdef void children_impurity(self) nogil

    cdef DTYPE_t impurity_improvement(self) nogil


# =============================================================================
# Label Ranking criterion
# =============================================================================
cdef class LabelRankingCriterion(Criterion):
    """Abstract criterion for Label Ranking."""


# =============================================================================
# Partial Label Ranking criterion
# =============================================================================
cdef class PartialLabelRankingCriterion(Criterion):
    """Abstract criterion for Partial Label Ranking."""

    # Methods
    cdef DTYPE_t impurity(self, DTYPE_t_3D precedences_matrix,
                          INT64_t_1D consensus) nogil
