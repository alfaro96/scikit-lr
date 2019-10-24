# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False


# =============================================================================
# Imports
# =============================================================================

# Local application
from .._types cimport DTYPE_t, DTYPE_t_1D, DTYPE_t_2D, INT64_t, SIZE_t


# =============================================================================
# Classes
# =============================================================================

# =============================================================================
# Distance metric
# =============================================================================
cdef class DistanceMetric:
    """DistanceMetric class."""

    # Hyperparameters

    # The following hyperparameter is required for a few of the subclasses.
    # We must define it here so that Cython's limited polymorphism will work.
    # Because we do not expect to instantiate a lot of this object (just once),
    # the extra memory overhead of this setup should not be an issue.
    cdef DTYPE_t p  # Power

    # Methods
    cdef DTYPE_t dist(self, DTYPE_t_1D x1, DTYPE_t_1D x2) nogil

    cdef void pdist(self, DTYPE_t_2D X, DTYPE_t_2D dist) nogil

    cdef void cdist(self, DTYPE_t_2D X1, DTYPE_t_2D X2, DTYPE_t_2D dist) nogil
