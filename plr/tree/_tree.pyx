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
# Tree
# =============================================================================
@cython.final
cdef class Tree:
    """
        Class for holding the underlying tree structure.
    """
    
    def __cinit__(self,
                  Node root,
                  list childs):
        """
            Constructor.
        """
        # Initialize some values from the input arrays
        cdef SIZE_t n_childs = len(childs)

        # Initialize some attributes of the current object
        self.root           = root
        self.childs         = childs
        self.depth_         = 0
        self.n_inner_nodes_ = 0
        self.n_leaf_nodes_  = 0

        # Initialize some values to be employed
        cdef SIZE_t max_child_depth = 0

        # Define the indexes
        cdef SIZE_t child

        # If the root of the tree is a inner node, add one
        if isinstance(self.root, InnerNode):
            self.n_inner_nodes_ += 1
        # Otherwise, if it is a leaf node, proceed as stated
        elif isinstance(self.root, LeafNode):
            self.n_leaf_nodes_ += 1

        # Obtain the maximum depth of the childs and the number of inner and leaf nodes
        for child in range(n_childs):
            # The depth of the tree is bounded by the maximum of the childs + 1
            if (childs[child].depth_ + 1) > self.depth_:
                self.depth_ = childs[child].depth_ + 1
            # Increment the number of inner nodes and leaf nodes
            self.n_inner_nodes_ += childs[child].n_inner_nodes_
            self.n_leaf_nodes_  += childs[child].n_leaf_nodes_

    cpdef void predict(self,
                       DTYPE_t[:, :] X,
                       SIZE_t[:, :]  predictions):
        """
            Predict the input test dataset.
        """
        # Initialize some values from the input arrays
        cdef SIZE_t n_samples = X.shape[0]

        # Define the indexes
        cdef SIZE_t sample

        # Obtain the prediction for each input instance
        for sample in range(n_samples):
            self._predict(x          = X[sample],
                          prediction = predictions[sample])

    cpdef void _predict(self,
                        DTYPE_t[:] x,
                        SIZE_t[:]  prediction):
        """
            Predict the input instance recursively.
        """
        # Define and initialize some values to be employed
        cdef SIZE_t att
        cdef SIZE_t n_childs

        # Define the indexes
        cdef SIZE_t child

        # If the root of the tree is a leaf node, just copy the bucket order
        if isinstance(self.root, LeafNode):
            prediction[:] = self.root.y
        # Otherwise, if it is an inner node, apply the algorithm recursively
        # looking for the child that satisfy the threshold condition
        elif isinstance(self.root, InnerNode):  
            # Initialize the attribute of the current inner node and the number of childs
            att      = self.root.att
            n_childs = len(self.childs)
            # Look for the child satisfying the condition          
            for child in range(n_childs):
                if x[att] > self.root.thresholds[child] and x[att] <= self.root.thresholds[child + 1]:
                    self.childs[child]._predict(x, prediction)

# =============================================================================
# Node
# =============================================================================
cdef class Node:
    """
        Interface for a node of a tree.
    """

    def __cinit__(self,
                  DTYPE_t   impurity,
                  SIZE_t    n_samples,
                  SIZE_t[:] y):
        """
            Constructor.
        """
        self.impurity  = impurity
        self.n_samples = n_samples
        self.y         = y

# =============================================================================
# Inner node
# =============================================================================
@cython.final
cdef class InnerNode(Node):
    """
        Class for holding an inner node of a decision tree.
    """

    cpdef InnerNode init(self,
                         SIZE_t     att,
                         DTYPE_t[:] thresholds):
        """
            Extra constructor for the class InnerNode.
        """
        # Initialize other parameters of the inner node
        self.att        = att
        self.thresholds = thresholds

        # Return the current object
        return self

# =============================================================================
# Leaf node
# =============================================================================
@cython.final          
cdef class LeafNode(Node):
    """
        Class for holding a leaf node of a tree.
    """
    pass
