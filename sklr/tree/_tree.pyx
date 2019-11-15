# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False


# =============================================================================
# Imports
# =============================================================================

# Third party
from libc.stdlib cimport free, malloc
import numpy as np
cimport numpy as np

# Always include this statement after cimporting
# NumPy, to avoid segmentation faults
np.import_array()


# =============================================================================
# Classes
# =============================================================================

# =============================================================================
# Tree builder
# =============================================================================
cdef class TreeBuilder:
    """Build a decision tree."""

    def __cinit__(self, Splitter splitter, INT64_t min_samples_split,
                  INT64_t max_depth):
        """Constructor."""
        # Initialize the hyperparameters
        self.splitter = splitter
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    cdef void _build_leaf(self, Tree tree):
        """Build a leaf node."""
        # Define some values to be employed
        cdef INT64_t n_classes
        cdef INT64_t n_samples
        cdef DTYPE_t weighted_n_samples
        cdef DTYPE_t impurity
        cdef INT64_t_1D consensus
        cdef DTYPE_t_1D count
        cdef DTYPE_t_3D precedences_matrix

        # Initialize the number of classes (from
        # the information provided by the criterion)
        n_classes = self.splitter.criterion.n_classes

        # Initialize the number of samples and the weighted number of
        # samples (from the information provided by the criterion)
        n_samples = self.splitter.criterion.n_node_samples
        weighted_n_samples = self.splitter.criterion.weighted_n_node_samples

        # Initialize the impurity and the consensus ranking
        # (from the information provided by the criterion)
        impurity = self.splitter.criterion.impurity_node
        consensus = self.splitter.criterion.consensus_node

        # Initialize the count and the precedences matrix
        # (from the information provided by the criterion)
        count = self.splitter.criterion.count_node
        precedences_matrix = self.splitter.criterion.precedences_matrix_node

        # Build the leaf node
        tree._add_root(
            LeafNode(n_classes,
                     n_samples, weighted_n_samples,
                     impurity, consensus,
                     count, precedences_matrix))

    cdef void _build_internal(self, Tree tree, INT64_t depth):
        """Build an internal node."""
        # Initialize the number of features and the number of
        # maximum splits from the stored values in the
        # corresponding structures of the splitter
        cdef INT64_t n_features = self.splitter.n_features
        cdef INT64_t max_splits = self.splitter.max_splits

        # Define some values to be employed
        cdef INT64_t n_classes
        cdef SIZE_t feature
        cdef DTYPE_t_1D thresholds
        cdef INT64_t n_samples
        cdef DTYPE_t weighted_n_samples
        cdef DTYPE_t impurity
        cdef INT64_t_1D consensus
        cdef DTYPE_t_1D count
        cdef DTYPE_t_3D precedences_matrix

        cdef SplitRecord split
        cdef list new_X_idx_sorted

        cdef Tree tree_child

        # Define the indexes
        cdef SIZE_t child

        # Initialize the number of classes (from
        # the information provided by the criterion)
        n_classes = self.splitter.criterion.n_classes

        # Initialize the number of samples and the weighted number of
        # samples (from the information provided by the criterion)
        n_samples = self.splitter.criterion.n_node_samples
        weighted_n_samples = self.splitter.criterion.weighted_n_node_samples

        # Initialize the impurity and the consensus ranking
        # (from the information provided by the criterion)
        impurity = self.splitter.criterion.impurity_node
        consensus = self.splitter.criterion.consensus_node

        # Initialize the count and the precedences matrix
        # (from the information provided by the criterion)
        count = self.splitter.criterion.count_node
        precedences_matrix = self.splitter.criterion.precedences_matrix_node

        # Initialize the split record which will hold
        # the information for splitting this node
        split = SplitRecord(self.splitter.criterion, max_splits, n_classes)

        # Split the node previously
        # drawing random features
        self.splitter.drawn()
        self.splitter.node_split(split)

        # Build the internal node
        tree._add_root(
            InternalNode(split.n_splits, n_classes,
                         split.feature, split.thresholds,
                         n_samples, weighted_n_samples,
                         impurity, consensus,
                         count, precedences_matrix))

        # Compute the sorted indexes for
        # each split that has been produced
        new_X_idx_sorted = self.splitter.node_indexes(split)

        # Recursively build each tree
        for child in range(split.n_splits):
            # Reset the splitter
            self.splitter.reset(new_X_idx_sorted[child], split, child)
            # Initialize a new tree for the child
            tree_child = Tree(n_features, n_classes)
            # Recursively build the tree
            self._build(tree_child, depth + 1)
            # Add the child tree to the tree
            tree._add_child(tree_child)

    cdef void _build(self, Tree tree, INT64_t depth=0):
        """Recursively build a decision tree."""
        # Initialize the number of samples from
        # the information provided by the criterion
        cdef INT64_t n_samples = self.splitter.criterion.n_node_samples

        # Define some values to be employed
        cdef DTYPE_t_2D X
        cdef INT64_t_2D Y
        cdef SIZE_t_1D samples
        cdef USEFUL_FEATURES *useful_features
        cdef BOOL_t is_leaf

        # Initialize the memory view with the data,
        # the rankings and the samples to consider
        # (to save some lines of code)
        X = self.splitter.X
        Y = self.splitter.criterion.Y
        samples = self.splitter.X_idx_sorted[0]

        # Initialize the pointer to the useful
        # features (since they must be modified
        # when checking if all features are equal)
        useful_features = &self.splitter.useful_features

        # Check whether to create a leaf node. In this case,
        # a leaf node is created if the maximum depth has been
        # achieved, the number of samples in less than the
        # minimum number of samples to split an internal node,
        # all the values for each features are equal or all
        # the rankings are equal (properly managed missed classes)
        is_leaf = (depth == self.max_depth or
                   n_samples < self.min_samples_split or
                   are_features_equal(X, samples, useful_features) or
                   are_rankings_equal(Y, samples))

        # Build the proper node (either a
        # leaf node or an internal node)
        if is_leaf:
            self._build_leaf(tree)
        else:
            self._build_internal(tree, depth)

    cpdef void build(self, Tree tree, DTYPE_t_2D X, INT64_t_2D Y,
                     DTYPE_t_1D sample_weight, SIZE_t_2D X_idx_sorted):
        """Build a decision tree."""
        # Initialize the splitter
        self.splitter.init(X, Y, sample_weight, X_idx_sorted)

        # Recursively build
        # the decision tree
        self._build(tree)


# =============================================================================
# Tree
# =============================================================================
cdef class Tree:
    """Representation of a decision tree.

    Attributes
    ----------
    internal_count : int
        The number of internal nodes in the tree.

    leaf_count : int
        The number of leaf nodes in the tree.

    max_depth : int
        The depth of the tree, i.e. the maximum depth of its leaves.

    root : Node
        The root of the tree.

    children : list of Tree
        The children of the tree.
    """

    def __cinit__(self, INT64_t n_features, INT64_t n_classes):
        """Constructor."""
        # Initialize the number of features
        # and the number of classes
        self.n_features = n_features
        self.n_classes = n_classes

        # Initialize the counter for internal node IDs, the
        # counter for leaf node IDs and the maximal tree depth
        self.internal_count = 0
        self.leaf_count = 0
        self.max_depth = 0

        # Initialize an empty list with
        # the children for this tree
        self.children = list()

    cdef void _add_root(self, Node node):
        """Add a root to the tree."""
        # Add the root to the tree (directly copy
        # the reference and not the contents)
        self.root = node

        # Increase the number of
        # internal nodes or leaves
        if node.node == INTERNAL:
            self.internal_count += 1
        else:
            self.leaf_count += 1

    cdef void _add_child(self, Tree tree):
        """Add a child to the tree."""
        # Increase the total number
        # of internal nodes and leaves
        self.internal_count += tree.internal_count
        self.leaf_count += tree.leaf_count

        # Update the maximum depth of the tree (the maximum depth
        # of the tree is the maximum depth of their leaves)
        self.max_depth = max(self.max_depth, tree.max_depth + 1)

        # Add the child to the list
        self.children.append(tree)

    cpdef np.ndarray[INT64_t, ndim=2] predict(self, DTYPE_t_2D X):
        """Predict target for X."""
        # Initialize some values from the input arrays
        cdef INT64_t n_samples = X.shape[0]

        # Define the indexes
        cdef SIZE_t sample

        # Initialize the predictions
        cdef np.ndarray[INT64_t, ndim=2] predictions = np.zeros(
            (n_samples, self.n_classes), dtype=np.int64)

        # Predict each sample on X
        for sample in range(n_samples):
            self._predict(x=X[sample], prediction=predictions[sample])

        # Return the predictions
        return predictions

    cpdef void _predict(self, DTYPE_t_1D x, INT64_t_1D prediction):
        """Predict target for x."""
        # Define the indexes
        cdef SIZE_t feature
        cdef SIZE_t child

        # If the root of the tree is a leaf
        # node, predict the consensus ranking
        if self.root.node == LEAF:
            prediction[:] = self.root.consensus
        # Otherwise, apply the decision path and recursively
        # predict the sample following the corresponding path
        else:
            # Initialize the feature on the internal node
            feature = (<InternalNode> self.root).feature
            # Iterate all the children to recursively obtain the prediction
            for child in range(len(self.children)):
                if (x[feature] > self.root.thresholds[child] and
                        x[feature] <= self.root.thresholds[child + 1]):
                    self.children[child]._predict(x, prediction)


# =============================================================================
# Node
# =============================================================================
cdef class Node:
    """Abstract interface of a node of a decision tree."""


# =============================================================================
# Internal node
# =============================================================================
cdef class InternalNode(Node):
    """Representation of an internal node of a decision tree.

    Attributes
    ----------
    feature : int
       Feature to split on the internal node.

    thresholds : np.ndarray of shape (n_splits,)
        Thresholds for the internal node.

    n_samples : int
        Number of training samples reaching the internal node.

    weighted_n_samples : float
        Weighted number of training samples reaching the internal node.

    impurity : float
        Value of the splitting criterion at the internal node.

    consensus : np.ndarray of shape (n_classes,)
        Contains the constant prediction ranking of the internal node.

    count : {None, np.ndarray} of shape (n_classes,)
        Count at the internal node.

    precedences_matrix : {None, np.ndarray} of shape (n_classes, n_classes, 2)
        Precedences matrix at the internal node.
    """

    def __cinit__(self, INT64_t n_splits, INT64_t n_classes,
                  SIZE_t feature, DTYPE_t_1D thresholds,
                  INT64_t n_samples, DTYPE_t weighted_n_samples,
                  DTYPE_t impurity, INT64_t_1D consensus,
                  DTYPE_t_1D count, DTYPE_t_3D precedences_matrix):
        """Constructor."""
        # Initialize the feature used for splitting and the threshold values
        self.feature = feature
        self.thresholds = np.zeros(n_splits + 1, dtype=np.float64)

        # Initialize the number of samples and
        # the weighted number of samples
        self.n_samples = n_samples
        self.weighted_n_samples = weighted_n_samples

        # Initialize the impurity and
        # the consensus ranking
        self.impurity = impurity
        self.consensus = np.zeros(n_classes, dtype=np.int64)

        # Initialize the count and the precedences matrix
        # to None and, afterwards, they will be properly set
        self.count = None
        self.precedences_matrix = None

        # Initialize the count or the precedences matrix
        # (taking into account the one needed by the criterion)
        if count is not None:
            self.count = np.zeros(
                n_classes, dtype=np.float64)
        else:
            self.precedences_matrix = np.zeros(
                (n_classes, n_classes, 2), dtype=np.float64)

        # Initialize the internal node
        self.node = INTERNAL

        # Fix the threshold values (only using the number
        # of splits that has been found to save memory)
        self.thresholds[:] = thresholds[:n_splits + 1]

        # Fix the consensus ranking
        self.consensus[:] = consensus

        # Fix the count or the precedences matrix
        # (taking into account the one needed by the criterion)
        if count is not None:
            self.count[:] = count
        else:
            self.precedences_matrix[:] = precedences_matrix


# =============================================================================
# Leaf node
# =============================================================================
cdef class LeafNode(Node):
    """Representation of a leaf node of a decision tree.

    Attributes
    ----------
    n_samples : int
        Number of training samples reaching the internal node.

    weighted_n_samples : float
        Weighted number of training samples reaching the internal node.

    impurity : float
        Value of the splitting criterion at the internal node.

    consensus : np.ndarray of shape (n_classes,)
        Contains the constant prediction ranking of the internal node.

    count : {None, np.ndarray} of shape (n_classes,)
        Count at the internal node.

    precedences_matrix : {None, np.ndarray} of shape (n_classes, n_classes, 2)
        Precedences matrix at the internal node.
    """

    def __cinit__(self, INT64_t n_classes,
                  INT64_t n_samples, DTYPE_t weighted_n_samples,
                  DTYPE_t impurity, INT64_t_1D consensus,
                  DTYPE_t_1D count, DTYPE_t_3D precedences_matrix):
        """Constructor."""
        # Initialize the number of samples and
        # the weighted number of samples
        self.n_samples = n_samples
        self.weighted_n_samples = weighted_n_samples

        # Initialize the impurity
        # and the consensus ranking
        self.impurity = impurity
        self.consensus = np.zeros(n_classes, dtype=np.int64)

        # Initialize the count and the precedences matrix
        # to None and, afterwards, they will be properly set
        self.count = None
        self.precedences_matrix = None

        # Initialize the count or the precedences matrix
        # (taking into account the one needed by the criterion)
        if count is not None:
            self.count = np.zeros(
                n_classes, dtype=np.float64)
        else:
            self.precedences_matrix = np.zeros(
                (n_classes, n_classes, 2), dtype=np.float64)

        # Initialize the leaf node
        self.node = LEAF

        # Fix the consensus ranking
        self.consensus[:] = consensus

        # Fix the count or the precedences matrix
        # (taking into account the one needed by the criterion)
        if count is not None:
            self.count[:] = count
        else:
            self.precedences_matrix[:] = precedences_matrix


# =============================================================================
# Methods
# =============================================================================

cdef BOOL_t are_features_equal(DTYPE_t_2D X, SIZE_t_1D samples,
                               USEFUL_FEATURES *useful_features) nogil:
    """Check whether all the samples in X are equal
    (only considering the useful features)."""
    # Initialize some values from the input arrays
    cdef INT64_t n_samples = samples.shape[0]
    cdef INT64_t n_features = X.shape[1]

    # Define some values to be employed
    cdef BOOL_t equal

    # Define the indexes
    cdef SIZE_t sample
    cdef SIZE_t feature

    # IMPORTANT: BEFORE OBTAINING THE USEFUL
    # FEATURES, CLEAR THEM FOR SANITY
    useful_features[0].clear()

    # Check whether all the features are equal,
    # erasing, at the same time the useless ones
    for feature in range(n_features):
        # Initialize this feature as
        # if all the samples were equal
        equal = True
        # Check whether all the samples
        # for this feature are equal
        for sample in range(n_samples):
            equal = (X[samples[0], feature] ==
                     X[samples[sample], feature])
            # Short-circuit: Break as far as a
            # different value has been found
            if not equal:
                break
        if not equal:
            useful_features[0].push_back(feature)

    # All the features are equal if the
    # size of the useful ones is zero
    equal = useful_features[0].size() == 0

    # Return whether all the
    # features are equal
    return equal


cdef BOOL_t are_rankings_equal(INT64_t_2D Y, SIZE_t_1D samples) nogil:
    """Check whether all the samples in Y are equal."""
    # Initialize some values from the input arrays
    cdef INT64_t n_samples = samples.shape[0]
    cdef INT64_t n_classes = Y.shape[1]

    # Define some values to be employed
    cdef BOOL_t equal
    cdef INT64_t relation

    # Define the indexes
    cdef SIZE_t sample
    cdef SIZE_t f_class
    cdef SIZE_t s_class

    # The rankings are initialize as
    # if all the classes were equal
    equal = True

    # Check whether all the rankings are equal, properly
    # handling the cases where missed classes are found
    for f_class in range(n_classes - 1):
        for s_class in range(f_class + 1, n_classes):
            # Initialize the precedence relation between this
            # pair of classes being tested for consistency
            relation = RANK_TYPE.RANDOM
            # Check the precedence relation between
            # this pair of classes for all the samples
            for sample in range(n_samples):
                # Found one sample where both pair of classes occur
                if (Y[samples[sample], f_class] != RANK_TYPE.RANDOM and
                        Y[samples[sample], s_class] != RANK_TYPE.RANDOM):
                    # Check whether this sample is the first one
                    # where both pair of classes ocurr. If it is,
                    # store the precedence relation for checking
                    # it in the next samples
                    if relation == RANK_TYPE.RANDOM:
                        if (Y[samples[sample], f_class] <
                                Y[samples[sample], s_class]):
                            relation = -1
                        elif (Y[samples[sample], f_class] ==
                                Y[samples[sample], s_class]):
                            relation = 0
                        else:
                            relation = 1
                    # Otherwise, check the precedence relation
                    # of this pair of classes for this sample
                    else:
                        if relation == -1:
                            equal = (Y[samples[sample], f_class] <
                                     Y[samples[sample], s_class])
                        elif relation == 0:
                            equal = (Y[samples[sample], f_class] ==
                                     Y[samples[sample], s_class])
                        else:
                            equal = (Y[samples[sample], f_class] >
                                     Y[samples[sample], s_class])
                if not equal:
                    break
            if not equal:
                break
        if not equal:
            break

    # Return whether all
    # the rankings are equal
    return equal
