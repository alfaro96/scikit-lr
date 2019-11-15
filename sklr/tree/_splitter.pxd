# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False


# =============================================================================
# Imports
# =============================================================================

# Third party
from libcpp.unordered_set cimport unordered_set
from libcpp.vector cimport vector

# Local application
from ._criterion cimport Criterion
from .._types cimport (
    BOOL_t,
    DTYPE_t, DTYPE_t_1D, DTYPE_t_2D, DTYPE_t_3D, DTYPE_t_4D,
    INT64_t, INT64_t_1D, INT64_t_2D,
    SIZE_t, SIZE_t_1D, SIZE_t_2D,
    UINT8_t)


# =============================================================================
# Types
# =============================================================================

# Useful attributes
ctypedef vector[SIZE_t] USEFUL_FEATURES
ctypedef unordered_set[SIZE_t] INDEXES_t


# =============================================================================
# Classes
# =============================================================================

# =============================================================================
# Splitter
# =============================================================================
cdef class Splitter:
    """Abstract splitter class."""

    # Hyperparameters
    cdef Criterion criterion  # Impurity criterion
    cdef INT64_t max_features  # Number of features to test
    cdef INT64_t max_splits  # Number of splits to test
    cdef object random_state  # Random state

    # Attributes
    cdef DTYPE_t_2D X  # Data
    cdef DTYPE_t_1D sample_weight  # Sample weights
    cdef SIZE_t_2D X_idx_sorted  # Sorted indexes for the data

    cdef SIZE_t n_features  # Number of features
    cdef SIZE_t_1D features  # Features indexes
    cdef USEFUL_FEATURES useful_features  # Useful features

    # Methods
    cdef void init(self, DTYPE_t_2D X, INT64_t_2D Y,
                   DTYPE_t_1D sample_weight, SIZE_t_2D X_idx_sorted)

    cdef void reset(self, SIZE_t_2D X_idx_sorted,
                    SplitRecord record, INT64_t child)

    cdef void drawn(self)

    cdef void init_params(self, SIZE_t feature) nogil

    cdef void update_params(self, SIZE_t feature, SIZE_t sample,
                            BOOL_t add_split,
                            BOOL_t update_criterion,
                            BOOL_t check_improvement) nogil

    cdef BOOL_t add_split(self, SIZE_t feature, SIZE_t sample) nogil

    cdef BOOL_t update_criterion(self, SIZE_t feature, SIZE_t sample,
                                 BOOL_t add_split) nogil

    cdef BOOL_t check_improvement(self, SIZE_t feature, SIZE_t sample,
                                  BOOL_t add_split,
                                  BOOL_t update_criterion) nogil

    cdef void node_split(self, SplitRecord split) nogil

    cdef list node_indexes(self, SplitRecord record)


# =============================================================================
# Binary splitter
# =============================================================================
cdef class BinarySplitter(Splitter):
    """Splitter for finding the best binary split."""


# =============================================================================
# Frequency splitter
# =============================================================================
cdef class FrequencySplitter(Splitter):
    """Splitter for finding the best equal-frequency split."""

    # Attributes
    cdef DTYPE_t frequency


# =============================================================================
# Width splitter
# =============================================================================
cdef class WidthSplitter(Splitter):
    """Splitter for finding the best equal-width split."""

    # Attributes
    cdef DTYPE_t width


# =============================================================================
# Split record
# =============================================================================
cdef class SplitRecord:
    """Split record."""

    # Attributes
    cdef INT64_t n_splits  # Number of splits

    cdef SIZE_t feature  # Feature used for splitting
    cdef DTYPE_t_1D thresholds  # Threshold value on each split

    cdef SIZE_t_1D pos  # Sample positions on each split

    cdef INT64_t_1D samples  # Number of samples on each split
    cdef DTYPE_t_1D weights  # Weighted number of samples on each split

    cdef DTYPE_t_1D impurities  # Impurity on each split
    cdef INT64_t_2D consensus  # Consensus ranking on each split

    cdef DTYPE_t_2D counts  # Count on each split
    cdef DTYPE_t_4D precedences_matrices  # Precedences matrix on each split

    # Methods
    cdef void record(self, Criterion criterion, SIZE_t feature,
                     DTYPE_t_2D X, SIZE_t_2D X_idx_sorted) nogil
