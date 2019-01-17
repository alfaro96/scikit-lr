# cython: language_level = 3
# cython: cdivision      = True
# cython: boundscheck    = False
# cython: wraparound     = False

# =============================================================================
# Imports
# =============================================================================

# PLR
from .._types cimport BOOL_t, DTYPE_t, SIZE_t, UINT8_t

# C++
from libcpp.vector        cimport vector
from libcpp.unordered_set cimport unordered_set

# =============================================================================
# Cluster probability transformer
# =============================================================================
cdef class ClusterProbabilityTransformer:
    """
        Transformer for probability distributions to bucket orders.
    """

    # Hyperparameters
    cdef DTYPE_t threshold
    cdef object  metric

    # Methods
    cpdef void transform(self,
                         DTYPE_t[:, :]    Y,
                         DTYPE_t[:, :]    prev_prob_dists,
                         DTYPE_t[:, :]    new_prob_dists,
                         UINT8_t[:, :, :] matrices) nogil

    cdef void _transform(self,
                         DTYPE_t[:]    ranking,
                         DTYPE_t[:]    prev_prob_dist,
                         DTYPE_t[:]    new_prob_dist,
                         UINT8_t[:, :] matrix) nogil

    cdef BOOL_t _all_same_bucket(self,
                                 UINT8_t[:, :] matrix) nogil

# =============================================================================
# Top-K transformer
# =============================================================================
cdef class TopKTransformer:
    """
        Transformer for bucket orders to new ones using top-k process.
    """

    # Hyperparameters
    cdef DTYPE_t perc

    # Methods
    cpdef void transform_from_probs(self,
                                    DTYPE_t[:, :] Y,
                                    DTYPE_t[:, :] prob_dists,
                                    SIZE_t[:, :]  sorted_prob_dists) nogil

    cpdef void transform_from_bucket_orders(self,
                                            DTYPE_t[:, :] Y,
                                            SIZE_t[:, :]  sorted_Y) nogil

# =============================================================================
# MissRandom transformer
# =============================================================================
cdef class MissRandomTransformer:
    """
        Transformer for bucket orders missing labels in a random way.
    """

    # Hyperparameters
    cdef DTYPE_t perc
    cdef object  random_state

    # Methods
    cpdef void transform(self,
                         DTYPE_t[:, :] Y) nogil

# =============================================================================
# Global methods
# =============================================================================

cpdef void rank_data(DTYPE_t[:] ranking,
                     DTYPE_t[:] data,
                     BOOL_t     inverse) nogil
