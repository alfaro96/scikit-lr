# cython: language_level = 3
# cython: cdivision      = True
# cython: boundscheck    = False
# cython: wraparound     = False

# =============================================================================
# Imports
# =============================================================================

# PLR
from ._types  cimport BUCKET_t, BUCKETS_t, OBOP_ALGORITHM
from .._types cimport DTYPE_t, SIZE_t, DTYPE_OR_SIZE_t

# =============================================================================
# Pair order matrix builder
# =============================================================================
cdef class PairOrderMatrixBuilder:
    """
        Builder for a pair order matrix.
    """

    # Methods
    cpdef void build(self,
                     DTYPE_t[:, :]    Y,
                     DTYPE_t[:]       sample_weight,
                     DTYPE_t[:, :, :] precedences,
                     DTYPE_t[:, :]    pair_order_matrix) nogil

# =============================================================================
# Utopian matrix builder
# =============================================================================
cdef class UtopianMatrixBuilder:
    """
        Builder for an utopian matrix.
    """

    # Methods
    cpdef void build(self,
                     DTYPE_t[:, :] pair_order_matrix,
                     DTYPE_t[:, :] utopian_matrix) nogil

# =============================================================================
# Anti-utopian matrix builder
# =============================================================================
cdef class AntiUtopianMatrixBuilder:
    """
        Builder for an anti-utopian matrix.
    """

    # Methods
    cpdef void build(self,
                     DTYPE_t[:, :] pair_order_matrix,
                     DTYPE_t[:, :] anti_utopian_matrix) nogil

# =============================================================================
# DANI indexes builder
# =============================================================================
cdef class DANIIndexesBuilder:
    """
        Builder for the DANI indexes.
    """

    # Methods
    cpdef void build(self,
                     DTYPE_t[:, :] pair_order_matrix,
                     DTYPE_t[:, :] utopian_matrix,
                     DTYPE_t[:]    dani_indexes) nogil

# =============================================================================
# Optimal Bucket Order Problem builder
# =============================================================================
cdef class OptimalBucketOrderProblemBuilder:
    """
        Builder for the Optimal Bucket Order Problem.
    """

    # Hyperparameters
    cdef OBOP_ALGORITHM algorithm
    cdef DTYPE_t        beta
    cdef object         random_state

    # Methods
    cpdef void build(self,
                     BUCKET_t      items,
                     SIZE_t[:]     y,
                     DTYPE_t[:, :] pair_order_matrix,
                     DTYPE_t[:, :] utopian_matrix,
                     DTYPE_t[:]    dani_indexes) nogil

    cdef BUCKETS_t _build(self,
                          BUCKET_t      items,
                          DTYPE_t[:, :] pair_order_matrix,
                          DTYPE_t[:, :] utopian_matrix,
                          DTYPE_t[:]    dani_indexes) nogil

# =============================================================================
# Global methods
# =============================================================================

cpdef DTYPE_t distance(DTYPE_t[:, :] matrix_1,
                       DTYPE_t[:, :] matrix_2) nogil

cpdef void normalize_matrix(DTYPE_t[:, :, :] precedences,
                            DTYPE_t[:, :]    pair_order_matrix) nogil

cpdef void normalize_sample_weight(DTYPE_t[:] sample_weight) nogil

cpdef void set_matrix(DTYPE_OR_SIZE_t[:] y,
                      DTYPE_t[:, :]      pair_order_matrix) nogil

cpdef void set_precedences(DTYPE_OR_SIZE_t[:] y,
                           DTYPE_t            weight,
                           DTYPE_t[:, :, :]   precedences) nogil
