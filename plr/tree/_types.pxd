# cython:    language_level = 3
# cython:    cdivision      = True
# cython:    boundscheck    = False
# cython:    wraparound     = False
# distutils: language       = c++

# =============================================================================
# Imports
# =============================================================================

# PLR
from .._types cimport DTYPE_t, SIZE_t

# C++
from libcpp.unordered_set cimport unordered_set
from libcpp.vector        cimport vector

# =============================================================================
# Types
# =============================================================================
ctypedef unordered_set[SIZE_t] REMOVED_ATTS_t # Type for the removed attributes
ctypedef unordered_set[SIZE_t] INDEXES_t      # Type for the kept indexes
ctypedef vector[SIZE_t]        USEFUL_ATTS_t  # Type for the useful attributes

# =============================================================================
# Parameters for a partition
# =============================================================================
cdef class Parameters:
    """
        Class for storing the parameters for a partition.
        Use a class instead of a struct because C hates memory views
        and Python objects (like the employed list).
    """
    
    # Attributes
    cdef SIZE_t              n_splits
    cdef DTYPE_t[:]          impurities
    cdef DTYPE_t[:, :, :, :] precedences
    cdef SIZE_t[:, :]        consensus
    cdef list                sorted_indexes
