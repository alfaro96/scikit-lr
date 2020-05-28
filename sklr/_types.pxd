# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False


# =============================================================================
# Imports
# =============================================================================

# Third party
from libc.float cimport DBL_EPSILON, DBL_MIN, DBL_MAX
from libc.stdint cimport INT64_MIN, INT64_MAX
cimport numpy as np


# =============================================================================
# Types
# =============================================================================

# Use fixed-size aliases to avoid platform-dependent definitions
ctypedef np.npy_bool BOOL_t
ctypedef np.npy_int64 INT64_t
ctypedef np.npy_float64 DTYPE_t
ctypedef np.npy_intp SIZE_t

ctypedef BOOL_t[:] BOOL_t_1D
ctypedef BOOL_t[:, :] BOOL_t_2D
ctypedef BOOL_t[:, :, :] BOOL_t_3D
ctypedef BOOL_t[:, :, :, :] BOOL_t_4D

ctypedef INT64_t[:] INT64_t_1D
ctypedef INT64_t[:, :] INT64_t_2D
ctypedef INT64_t[:, :, :] INT64_t_3D
ctypedef INT64_t[:, :, :, :] INT64_t_4D

ctypedef DTYPE_t[:] DTYPE_t_1D
ctypedef DTYPE_t[:, :] DTYPE_t_2D
ctypedef DTYPE_t[:, :, :] DTYPE_t_3D
ctypedef DTYPE_t[:, :, :, :] DTYPE_t_4D

ctypedef SIZE_t[:] SIZE_t_1D
ctypedef SIZE_t[:, :] SIZE_t_2D
ctypedef SIZE_t[:, :, :] SIZE_t_3D
ctypedef SIZE_t[:, :, :, :] SIZE_t_4D


# =============================================================================
# Constants
# =============================================================================

# Map to the used data type name to help with maintenance
cdef DTYPE_t DTYPE_MIN = DBL_MIN
cdef DTYPE_t DTYPE_MAX = DBL_MAX
cdef DTYPE_t DTYPE_EPSILON = DBL_EPSILON


# =============================================================================
# Enums
# =============================================================================

# Use the maximum values to avoid collisions
cpdef enum RANK_TYPE:
    TOP = 0x7ffffffffffffffe
    RANDOM = 0x7fffffffffffffff
