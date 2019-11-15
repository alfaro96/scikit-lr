# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False


# =============================================================================
# Imports
# =============================================================================

# Third party
from libcpp.vector cimport vector

# Local application
from ._types cimport (
    BOOL_t,
    DTYPE_t, DTYPE_t_1D, DTYPE_t_2D, DTYPE_t_3D,
    INT64_t, INT64_t_1D, INT64_t_2D, SIZE_t)
from ._types cimport RANK_TYPE


# =============================================================================
# Types
# =============================================================================

# =============================================================================
# Generic
# =============================================================================
ctypedef vector[INT64_t] BUCKET_t
ctypedef vector[BUCKET_t] BUCKETS_t


# =============================================================================
# Classes
# =============================================================================

# =============================================================================
# Rank aggregation algorithm
# =============================================================================
cdef class RankAggregationAlgorithm:
    """Provides a uniform interface to fast Rank Aggregation algorithms."""

    # The following hyperparameter and attributes are required for a few
    # of the subclasses. We must define it here so that Cython's limited
    # polymorphism will work. Because we do not expect to instantiate a
    # lot of these objects (just once), the extra memory overhead
    # of this setup should not be an issue.

    # Hyperparameters
    cdef DTYPE_t beta  # Beta

    # Attributes
    cdef CountBuilder count_builder  # Count builder
    cdef PairOrderMatrixBuilder pair_order_matrix_builder  # Pair order matrix builder  # nopep8
    cdef UtopianMatrixBuilder utopian_matrix_builder  # Utopian matrix builder
    cdef DANIIndexesBuilder dani_indexes_builder   # DANI indexes builder

    # Methods
    cdef void init(self, INT64_t n_classes)

    cdef void reset(self, DTYPE_t_1D count=*,
                    DTYPE_t_3D precedences_matrix=*) nogil

    cdef void update_params(self, INT64_t_1D y, DTYPE_t weight,
                            DTYPE_t_1D count=*,
                            DTYPE_t_3D precedences_matrix=*) nogil

    cdef void build_params(self, INT64_t_2D Y, DTYPE_t_1D sample_weight,
                           DTYPE_t_1D count=*,
                           DTYPE_t_3D precedences_matrix=*) nogil

    cdef void aggregate_params(self, INT64_t_1D consensus,
                               DTYPE_t_1D count=*,
                               DTYPE_t_3D precedences_matrix=*) nogil

    cdef void aggregate_rankings(self, INT64_t_2D Y, DTYPE_t_1D sample_weight,
                                 INT64_t_1D consensus) nogil

    cdef void complete_rankings(self, INT64_t_2D Y,
                                INT64_t_1D consensus) nogil

    cdef void aggregate_mle(self, INT64_t_2D Y, DTYPE_t_1D sample_weight,
                            INT64_t_1D consensus,
                            DTYPE_t_1D count=*,
                            DTYPE_t_3D precedences_matrix=*,
                            BOOL_t replace=*) nogil


# =============================================================================
# Count builder
# =============================================================================
cdef class CountBuilder:
    """Count builder."""

    # Attributes
    cdef DTYPE_t_1D count  # Count

    # Methods
    cdef void init(self, INT64_t n_classes)

    cdef void reset(self, DTYPE_t_1D count=*) nogil

    cdef void update(self, INT64_t_1D y, DTYPE_t weight,
                     DTYPE_t_1D count=*) nogil

    cdef void build(self, INT64_t_2D Y, DTYPE_t_1D sample_weight,
                    DTYPE_t_1D count=*) nogil


# =============================================================================
# Pair order matrix builder
# =============================================================================
cdef class PairOrderMatrixBuilder:
    """Pair order matrix builder."""

    # Attributes
    cdef DTYPE_t_3D precedences_matrix  # Precedences matrix
    cdef DTYPE_t_2D pair_order_matrix  # Pair order matrix

    # Methods
    cdef void init(self, INT64_t n_classes)

    cdef void reset(self, DTYPE_t_3D precedences_matrix=*) nogil

    cdef void update_precedences_matrix(
            self, INT64_t_1D y, DTYPE_t weight,
            DTYPE_t_3D precedences_matrix=*) nogil

    cdef void build_precedences_matrix(
            self, INT64_t_2D Y,
            DTYPE_t_1D sample_weight,
            DTYPE_t_3D precedences_matrix=*) nogil

    cdef void build_pair_order_matrix(
            self,
            DTYPE_t_3D precedences_matrix=*) nogil


# =============================================================================
# Utopian matrix builder
# =============================================================================
cdef class UtopianMatrixBuilder:
    """Utopian matrix builder."""

    # Attributes
    cdef DTYPE_t_2D utopian_matrix  # Utopian matrix

    # Methods
    cdef void init(self, INT64_t n_classes)

    cdef void reset(self) nogil

    cdef void build(self, DTYPE_t_2D pair_order_matrix) nogil


# =============================================================================
# DANI indexes builder
# =============================================================================
cdef class DANIIndexesBuilder:
    """DANI indexes builder."""

    # Attributes
    cdef DTYPE_t_1D dani_indexes  # DANI indexes

    # Methods
    cdef void init(self, INT64_t n_classes)

    cdef void reset(self) nogil

    cdef void build(self, DTYPE_t_2D pair_order_matrix,
                    DTYPE_t_2D utopian_matrix) nogil
