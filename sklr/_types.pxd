# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False


# =============================================================================
# Imports
# =============================================================================

# Third party
import numpy as np
cimport numpy as np

# Always include this statement after cimporting
# NumPy, to avoid segmentation faults
np.import_array()


# =============================================================================
# Enums
# =============================================================================

# Type of rankings
cdef enum RANK_TYPE:
    RANDOM = 2147483646


# =============================================================================
# Types
# =============================================================================

# =============================================================================
# Generic
# =============================================================================
ctypedef bint BOOL_t  # Boolean (values)
ctypedef np.npy_uint8 UINT8_t  # Boolean (arrays)
ctypedef np.npy_int64 INT64_t  # Integer
ctypedef np.npy_float64 DTYPE_t  # Double
ctypedef np.npy_intp SIZE_t  # Indexes


# =============================================================================
# Memory views
# =============================================================================

# Boolean
ctypedef UINT8_t[:] UINT8_t_1D
ctypedef UINT8_t[:, :] UINT8_t_2D
ctypedef UINT8_t[:, :, :] UINT8_t_3D
ctypedef UINT8_t[:, :, :, :] UINT8_t_4D

# Integer
ctypedef INT64_t[:] INT64_t_1D
ctypedef INT64_t[:, :] INT64_t_2D
ctypedef INT64_t[:, :, :] INT64_t_3D
ctypedef INT64_t[:, :, :, :] INT64_t_4D

# Double
ctypedef DTYPE_t[:] DTYPE_t_1D
ctypedef DTYPE_t[:, :] DTYPE_t_2D
ctypedef DTYPE_t[:, :, :] DTYPE_t_3D
ctypedef DTYPE_t[:, :, :, :] DTYPE_t_4D

# Indexes
ctypedef SIZE_t[:] SIZE_t_1D
ctypedef SIZE_t[:, :] SIZE_t_2D
ctypedef SIZE_t[:, :, :] SIZE_t_3D
ctypedef SIZE_t[:, :, :, :] SIZE_t_4D
