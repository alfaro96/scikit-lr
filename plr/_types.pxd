# cython:    language_level = 3
# cython:    cdivision      = True
# cython:    boundscheck    = False
# cython:    wraparound     = False

# =============================================================================
# Imports
# =============================================================================

# NumPy
import  numpy as np
cimport numpy as np

# Always include this statement after cimporting NumPy, to avoid
# segmentation faults
np.import_array()

# =============================================================================
# Types
# =============================================================================
ctypedef bint           BOOL_t  # Type for booleans
ctypedef np.npy_float64 DTYPE_t # Type for doubles
ctypedef np.npy_int64   INT64_t # Type for long integers
ctypedef np.npy_uint8   UINT8_t # Type for boolean arrays
ctypedef np.npy_intp    SIZE_t  # Type for indexes and positive integers
ctypedef fused DTYPE_OR_SIZE_t: # Fused type
    DTYPE_t
    SIZE_t
