# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False


# =============================================================================
# Imports
# =============================================================================

# Third party
cimport numpy as np

# Local application
from .._types cimport DTYPE_t_2D, INT64_t, INT64_t_2D, SIZE_t


# =============================================================================
# Types
# =============================================================================

ctypedef np.npy_uint8[:, :] UINT8_t_2D


# =============================================================================
# Enums
# =============================================================================

ctypedef enum STRATEGY:
    RANDOM
    TOP
