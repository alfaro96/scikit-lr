# cython: language_level = 3
# cython: cdivision      = True
# cython: boundscheck    = False
# cython: wraparound     = False

# =============================================================================
# Imports
# =============================================================================

# PLR
from ..bucket._builder cimport OBOP_ALGORITHM, PairOrderMatrixBuilder, UtopianMatrixBuilder, DANIIndexesBuilder, OptimalBucketOrderProblemBuilder
from ..bucket._types   cimport BUCKET_t
from .._types          cimport DTYPE_t, SIZE_t

# =============================================================================
# Base builder for the nearest-neighbors paradigm
# =============================================================================
cdef class BaseNeighborsBuilder:
    """
        Base builder for the nearest-neighbors algorithm.
    """

    # Hyperparameters
    cdef OBOP_ALGORITHM bucket
    cdef DTYPE_t        beta
    cdef object         random_state

    # Attributes
    cdef BUCKET_t                         items
    cdef DTYPE_t[:, :, :]                 precedences
    cdef DTYPE_t[:, :]                    pair_order_matrix
    cdef PairOrderMatrixBuilder           pair_order_matrix_builder
    cdef DTYPE_t[:, :]                    utopian_matrix
    cdef UtopianMatrixBuilder             utopian_matrix_builder
    cdef DTYPE_t[:]                       dani_indexes
    cdef DANIIndexesBuilder               dani_indexes_builder
    cdef OptimalBucketOrderProblemBuilder obop_builder

    # Methods
    cpdef BaseNeighborsBuilder init(self,
                                    SIZE_t n_classes)

# =============================================================================
# Builder for the k-nearest-neighbors paradigm
# =============================================================================
cdef class KNeighborsBuilder(BaseNeighborsBuilder):
    """
        Builder for the k-nearest-neighbors algorithm.
    """

    # Methods
    cpdef void predict(self,
                       DTYPE_t[:, :, :] Y,
                       DTYPE_t[:, :]    sample_weight,
                       SIZE_t[:, :]     predictions) nogil
