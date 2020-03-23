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
from ..utils._ranking cimport RANK_METHOD
from ..utils._ranking cimport rank_data_view


# =============================================================================
# Constants
# =============================================================================

# Map the strategy string identifier
# to the strategy integer identifier
STRATEGY_MAPPING = {
    "random": RANDOM,
    "top": TOP
}


# =============================================================================
# Methods
# =============================================================================

cpdef void miss_classes(INT64_t_2D Y, DTYPE_t_2D Yt,
                        UINT8_t_2D masks, STRATEGY strategy) nogil:
    """Miss the classes from the rankings using the given masks and
    taking into account the strategy for the deletion of classes."""
    cdef INT64_t n_samples = Y.shape[0]
    cdef INT64_t n_classes = Y.shape[1]

    cdef SIZE_t sample
    cdef SIZE_t label

    for sample in range(n_samples):
        for label in range(n_classes):
            if masks[sample, label]:
                if strategy == RANDOM:
                    Yt[sample, label] = NAN
                else:
                    Yt[sample, label] = INFINITY

    # Rank the randomly missed classes to ensure that the
    # rankings are properly formatted for the estimators
    if strategy == RANDOM:
        for sample in range(n_samples):
            # Use the dense method to rank the classes
            # since it works for all type of targets
            rank_data_view(Yt[sample], Y[sample],
                           method=RANK_METHOD.DENSE)

            for label in range(n_classes):
                if not masks[sample, label]:
                    Yt[sample, label] = Y[sample, label]
