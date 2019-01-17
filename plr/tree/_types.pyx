# cython: language_level = 3
# cython: cdivision      = True
# cython: boundscheck    = False
# cython: wraparound     = False

# =============================================================================
# Imports
# =============================================================================

# Cython
import cython

# =============================================================================
# Parameters for a partition
# =============================================================================
@cython.final
cdef class Parameters:
    """
        Class for storing the parameters for a partition.
        Use a class instead of a struct because C hates memory views
        and Python objects (like the employed list).
    """
    pass
