# cython: language_level = 3
# cython: cdivision      = True
# cython: boundscheck    = False
# cython: wraparound     = False

# =============================================================================
# Imports
# =============================================================================

# PLR
from .._types cimport BOOL_t, DTYPE_t, SIZE_t

# Cython
from cpython.list cimport list

# =============================================================================
# Tree
# =============================================================================
cdef class Tree:
    """
        Class for holding the underlying decision tree structure.
    """

    # Attributes
    cdef        Node root
    cdef        list childs
    cdef public SIZE_t depth_
    cdef public SIZE_t n_inner_nodes_
    cdef public SIZE_t n_leaf_nodes_

    # Methods
    cpdef void predict(self,
                       DTYPE_t[:, :] X,
                       SIZE_t[:, :]  predictions)

    cpdef void _predict(self,
                        DTYPE_t[:] x,
                        SIZE_t[:]  prediction)

# =============================================================================
#  Node
# =============================================================================
cdef class Node:
    """
        Interface for a node of a decision tree.
    """

    # Attributes
    cdef DTYPE_t   impurity
    cdef SIZE_t    n_samples
    cdef SIZE_t[:] y

# =============================================================================
# Inner node
# =============================================================================
cdef class InnerNode(Node):
    """
        Class for holding an inner node of a decision tree.
    """

    # Attributes
    cdef public SIZE_t     att
    cdef public DTYPE_t[:] thresholds

    # Methods
    cpdef InnerNode init(self,
                         SIZE_t     att,
                         DTYPE_t[:] thresholds)

# =============================================================================
# Leaf node
# =============================================================================
cdef class LeafNode(Node):
    """
        Class for holding a leaf node of a decision tree.
    """
    pass
