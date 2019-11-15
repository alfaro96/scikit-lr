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

# Local application
from ._splitter cimport Splitter, SplitRecord
from ._splitter cimport USEFUL_FEATURES
from .._types cimport (
    BOOL_t,
    DTYPE_t, DTYPE_t_1D, DTYPE_t_2D, DTYPE_t_3D,
    INT64_t, INT64_t_1D, INT64_t_2D,
    SIZE_t, SIZE_t_1D, SIZE_t_2D,
    UINT8_t_1D)
from .._types cimport RANK_TYPE


# =============================================================================
# Enums
# =============================================================================
ctypedef enum NODE:
    INTERNAL,
    LEAF


# =============================================================================
# Classes
# =============================================================================

# =============================================================================
# Tree builder
# =============================================================================
cdef class TreeBuilder:
    """Build a decision tree."""

    # Attributes
    cdef Splitter splitter  # Splitting algorithm

    cdef INT64_t min_samples_split  # Minimum number of samples in a node
    cdef INT64_t max_depth  # Maximal tree depth

    # Methods
    cdef void _build_leaf(self, Tree tree)

    cdef void _build_internal(self, Tree tree, INT64_t depth)

    cdef void _build(self, Tree tree, INT64_t depth=*)

    cpdef void build(self, Tree tree, DTYPE_t_2D X, INT64_t_2D Y,
                     DTYPE_t_1D sample_weight, SIZE_t_2D X_idx_sorted)


# =============================================================================
# Tree
# =============================================================================
cdef class Tree:
    """Representation of a decision tree."""

    # Attributes
    cdef INT64_t n_features  # Number of features in X
    cdef INT64_t n_classes  # Number of classes in Y

    cdef public INT64_t internal_count  # Counter for internal node IDs
    cdef public INT64_t leaf_count  # Counter for leaf node IDs
    cdef public INT64_t max_depth  # Maximal tree depth

    cdef public Node root  # Root of the tree
    cdef public list children  # List of children of the tree

    # Methods
    cdef void _add_root(self, Node node)

    cdef void _add_child(self, Tree tree)

    cpdef np.ndarray[INT64_t, ndim=2] predict(self, DTYPE_t_2D X)

    cpdef void _predict(self, DTYPE_t_1D x, INT64_t_1D prediction)


# =============================================================================
# Node
# =============================================================================
cdef class Node:
    """Abstract interface of a node of a decision tree."""

    # Attributes
    cdef public SIZE_t feature  # Feature used for splitting
    cdef public DTYPE_t_1D thresholds  # Threshold values

    cdef public INT64_t n_samples  # Number of samples
    cdef public DTYPE_t weighted_n_samples  # Weighted number of samples

    cdef public DTYPE_t impurity  # Impurity
    cdef public INT64_t_1D consensus  # Consensus ranking

    cdef public DTYPE_t_1D count  # Count
    cdef public DTYPE_t_3D precedences_matrix  # Precedences matrix

    # This attribute is required for fast inference, just to know
    # the kind of node, instead of calling to the "isinstance" method
    # everytime a prediction is required to be obtained
    cdef public NODE node
