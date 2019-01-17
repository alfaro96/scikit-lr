# cython: language_level = 3
# cython: cdivision      = True
# cython: boundscheck    = False
# cython: wraparound     = False

# =============================================================================
# Imports
# =============================================================================

# PLR
from ..bucket._builder cimport OBOP_ALGORITHM, UtopianMatrixBuilder, DANIIndexesBuilder, OptimalBucketOrderProblemBuilder
from ..bucket._types   cimport BUCKET_t
from .._types          cimport BOOL_t, DTYPE_t, SIZE_t

# =============================================================================
# Base criterion
# =============================================================================
cdef class Criterion:
    """
        Base class for computing the impurity of a node.
    """

    # Hyperparameters
    cdef OBOP_ALGORITHM bucket
    cdef DTYPE_t        beta
    cdef BOOL_t         require_obop
    cdef object         random_state

    # Attributes
    cdef BUCKET_t                         items
    cdef DTYPE_t[:, :]                    pair_order_matrix
    cdef DTYPE_t[:, :]                    utopian_matrix
    cdef UtopianMatrixBuilder             utopian_matrix_builder
    cdef DTYPE_t[:]                       dani_indexes
    cdef DANIIndexesBuilder               dani_indexes_builder
    cdef OptimalBucketOrderProblemBuilder obop_builder
    cdef DTYPE_t[:, :]                    consensus_pair_order_matrix
    
    # Methods
    cpdef Criterion init(self,
                         SIZE_t n_classes)

    cdef DTYPE_t node_impurity_consensus(self,
                                         DTYPE_t[:, :, :] precedences,
                                         DTYPE_t[:, :, :] pair_order_matrices,
                                         SIZE_t[:]        consensus,
                                         DTYPE_t[:]       sample_weight,
                                         SIZE_t[:]        sorted_indexes,
                                         SIZE_t           start_index,
                                         SIZE_t           end_index) nogil

    cdef DTYPE_t node_impurity(self,
                               DTYPE_t[:, :, :] precedences,
                               DTYPE_t[:, :, :] pair_order_matrices,
                               SIZE_t[:]        consensus,
                               DTYPE_t[:]       sample_weight,
                               SIZE_t[:]        sorted_indexes,
                               SIZE_t           start_index,
                               SIZE_t           end_index) nogil

    cdef void node_consensus(self,
                             DTYPE_t[:, :, :] precedences,
                             SIZE_t[:]        consensus) nogil

# =============================================================================
# Disagreements criterion
# =============================================================================
cdef class DisagreementsCriterion(Criterion):
    """
        Class for computing the impurity of a node using the disagreements criterion.
    """
    pass

# =============================================================================
# Distance criterion
# =============================================================================
cdef class DistanceCriterion(Criterion):
    """
        Class for computing the impurity of a node using the distance criterion.
    """
    pass

# =============================================================================
# Entropy criterion
# =============================================================================
cdef class EntropyCriterion(Criterion):
    """
        Class for computing the impurity of a node using the entropy criterion.
    """
    pass
