# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False


# =============================================================================
# Imports
# =============================================================================

# Third party
from libc.math cimport NAN, INFINITY

# Local application
from ..utils._ranking_fast cimport rank_data_view
from ..utils._ranking_fast cimport RANK_METHOD


# =============================================================================
# Constants
# =============================================================================

# Strategies used to miss
# the classes from the rankings
STRATEGIES = {
    "random": 0,
    "top": 1
}


# =============================================================================
# Methods
# =============================================================================

cpdef void miss_classes(INT64_t_2D Y, DTYPE_t_2D Yt,
                        UINT8_t_2D masks, STRATEGY strategy) nogil:
    """Miss the classes according to the provided masks."""
    # Initialize some values from the input arrays
    cdef INT64_t n_samples = Y.shape[0]
    cdef INT64_t n_classes = Y.shape[1]

    # Define the indexes
    cdef SIZE_t sample
    cdef SIZE_t label

    # Miss the classes from the
    # rankings using the provided masks
    for sample in range(n_samples):
        for label in range(n_classes):
            if masks[sample, label]:
                if strategy == RANDOM:
                    Yt[sample, label] = NAN
                else:
                    Yt[sample, label] = INFINITY

    # Once the classes have been missed from the rankings, rank
    # the data again to ensure that they are properly formatted
    # (only needed for randomly missed classes)
    if strategy == RANDOM:
        for sample in range(n_samples):
            # Rank the classes of this ranking using the dense method
            # (since, in this case, dense returns the same results than ordinal)
            rank_data_view(data=Yt[sample], y=Y[sample], method=RANK_METHOD.DENSE)
            # Set the position in this ranking to
            # ensure that it is properly formatted
            for label in range(n_classes):
                if not masks[sample, label]:
                    Yt[sample, label] = Y[sample, label]
