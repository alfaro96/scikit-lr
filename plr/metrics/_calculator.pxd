# cython: language_level = 3
# cython: cdivision      = True
# cython: boundscheck    = False
# cython: wraparound     = False

# =============================================================================
# Imports
# =============================================================================

# PLR
from .._types cimport DTYPE_t, SIZE_t, BOOL_t

# =============================================================================
# Global methods
# =============================================================================

cpdef DTYPE_t bhattacharyya_distance_calculator(DTYPE_t[:, :] probs_true,
                                                DTYPE_t[:, :] probs_pred,
                                                DTYPE_t[:]    sample_weight,
                                                DTYPE_t[:]    fake_sample_weight) nogil

cpdef DTYPE_t bhattacharyya_score_calculator(DTYPE_t[:, :] probs_true,
                                             DTYPE_t[:, :] probs_pred,
                                             DTYPE_t[:]    sample_weight) nogil

cpdef DTYPE_t kendall_distance_calculator(SIZE_t[:, :] Y_true,
                                          SIZE_t[:, :] Y_pred,
                                          DTYPE_t[:]   sample_weight,
                                          BOOL_t       normalize) nogil                                            

cpdef DTYPE_t tau_x_score_calculator(SIZE_t[:, :] Y_true,
                                     SIZE_t[:, :] Y_pred,
                                     DTYPE_t[:]   sample_weight) nogil

cpdef DTYPE_t minkowski_calculator(DTYPE_t[:] u,
                                   DTYPE_t[:] v,
                                   SIZE_t     p) nogil

cdef void normalize_sample_weight(DTYPE_t[:] sample_weight) nogil
