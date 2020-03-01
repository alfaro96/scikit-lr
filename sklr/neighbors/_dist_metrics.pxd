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

cdef class DistanceMetric:
    """A distance metric wrapper."""

    # The following hyperparameters are required for a few of the
    # subclasses to work, due to the Cython's limited polymorphism.
    # Because it is not expected to instantiate a lot of this object
    # (just once), the extra memory overhead should not be an issue.

    # Hyperparameters
    cdef DTYPE_t p

    # Methods
    cdef DTYPE_t dist(self, DTYPE_t_1D x1, DTYPE_t_1D x2) nogil

    cdef void pdist(self, DTYPE_t_2D X, DTYPE_t_2D dist) nogil

    cdef void cdist(self, DTYPE_t_2D X1, DTYPE_t_2D X2, DTYPE_t_2D dist) nogil
