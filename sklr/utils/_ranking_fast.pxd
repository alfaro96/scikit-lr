# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False


# =============================================================================
# Imports
# =============================================================================

# Third party
from libcpp.unordered_set cimport unordered_set

# Local application
from .._types cimport (
    BOOL_t,
    DTYPE_t, DTYPE_t_1D,
    INT64_t, INT64_t_1D, INT64_t_2D,
    SIZE_t)
from .._types cimport RANK_TYPE


# =============================================================================
# Types
# =============================================================================

# Unique classes
ctypedef unordered_set[INT64_t] UNIQUE_CLASSES


# =============================================================================
# Enums
# =============================================================================

# Rank method
ctypedef enum RANK_METHOD:
    ORDINAL
    DENSE


# =============================================================================
# Methods
# =============================================================================

cpdef void rank_data_view(DTYPE_t_1D data, INT64_t_1D y,
                          RANK_METHOD method) nogil


cdef void rank_data_pointer(DTYPE_t *data, INT64_t_1D y,
                            RANK_METHOD method) nogil
