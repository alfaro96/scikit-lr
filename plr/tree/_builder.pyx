# cython:    language_level = 3
# cython:    cdivision      = True
# cython:    boundscheck    = False
# cython:    wraparound     = False

# =============================================================================
# Imports
# =============================================================================

# PLR
from ..bucket._builder cimport set_matrix, set_precedences

# Cython
import cython

# C
from libc.math cimport isnan, fmin, INFINITY

# =============================================================================
# Base builder
# =============================================================================
cdef class TreeBuilder:
    """
        Base class for building a tree.
    """

    def __cinit__(self,
                  Splitter splitter,
                  SIZE_t   max_depth,
                  SIZE_t   min_samples_split,
                  SIZE_t   max_features,
                  object   random_state):
        """
            Constructor.
        """
        # Initialize the hyperparameters of the current object
        self.splitter          = splitter
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.max_features      = max_features
        self.random_state      = random_state

    cpdef Tree build(self,
                     DTYPE_t[:, :]       X,
                     DTYPE_t[:, :]       Y,
                     DTYPE_t[:, :, :]    gbl_precedences,
                     DTYPE_t[:, :, :, :] ind_precedences,
                     DTYPE_t[:, :, :]    ind_pair_order_matrices,
                     SIZE_t[:]           consensus,
                     DTYPE_t[:]          sample_weight,
                     SIZE_t[:, :]        sorted_indexes):
        """
            Initialize the data structures for building the tree.
        """
        # Initialize some values from the input arrays
        cdef SIZE_t n_samples = X.shape[0]

        # Define some values to be employed

        # Attributes
        cdef REMOVED_ATTS_t removed_atts

        # Tree
        cdef Tree tree
        
        # Impurity for the root of the tree
        cdef DTYPE_t impurity

        # Define the indexes
        cdef SIZE_t sample

        # Iterate all the samples to fill the contents of the global precedences matrix and
        # the individual precedences and pair order matrices
        for sample in range(n_samples):
            # Fill the contents of the precedences matrices
            set_precedences(y = Y[sample], weight = sample_weight[sample], precedences = gbl_precedences)
            set_precedences(y = Y[sample], weight = 1.0,                   precedences = ind_precedences[sample])
            # Fill the contents of the pair order matrices
            set_matrix(y = Y[sample], pair_order_matrix = ind_pair_order_matrices[sample])
        
        # Obtain the impurity and bucket order for the root of the tree
        impurity = self.splitter.criterion.node_impurity_consensus(precedences         = gbl_precedences,
                                                                   pair_order_matrices = ind_pair_order_matrices,
                                                                   consensus           = consensus,
                                                                   sample_weight       = sample_weight,
                                                                   sorted_indexes      = sorted_indexes[0],
                                                                   start_index         = 0,
                                                                   end_index           = n_samples)

        # Build the tree recursively
        tree = self._build(X                       = X,
                           removed_atts            = removed_atts,
                           Y                       = Y,
                           gbl_precedences         = gbl_precedences,
                           ind_precedences         = ind_precedences,
                           ind_pair_order_matrices = ind_pair_order_matrices,
                           sample_weight           = sample_weight,
                           sorted_indexes          = sorted_indexes,
                           consensus               = consensus,
                           impurity                = impurity,
                           level                   = 0)

        # Return the obtained tree
        return tree
            
    cdef Tree _build(self,
                     DTYPE_t[:, :]       X,
                     REMOVED_ATTS_t      removed_atts,
                     DTYPE_t[:, :]       Y,
                     DTYPE_t[:, :, :]    gbl_precedences,
                     DTYPE_t[:, :, :, :] ind_precedences,
                     DTYPE_t[:, :, :]    ind_pair_order_matrices,
                     DTYPE_t[:]          sample_weight,
                     SIZE_t[:, :]        sorted_indexes,
                     SIZE_t[:]           consensus,
                     DTYPE_t             impurity,
                     SIZE_t              level):
        """
            Recursively build the tree.
        """
        # Initialize some values from the input arrays
        cdef SIZE_t n_samples = sorted_indexes[0].shape[0]
        cdef SIZE_t n_classes = Y.shape[1]

        # Define some values from the input arrays, that are going to be computed afterwards
        cdef SIZE_t n_features
        cdef SIZE_t all_bucket_orders_equal 

        # Define some values to be employed

        # Tree
        cdef Node root
        cdef list childs
        cdef Tree tree

        # Partition
        cdef USEFUL_ATTS_t useful_atts
        cdef SIZE_t[:]     selected_atts
        cdef Parameters    parameters
        
        # First, remove the useless attributes, that is, remove those whose all values are the same
        self._remove_useless_attributes(X = X, removed_atts = &removed_atts, useful_atts = &useful_atts, sorted_indexes = sorted_indexes)

        # Once checked, obtain the current number of features and
        # whether all the bucket orders are the same
        n_features              = useful_atts.size()
        all_bucket_orders_equal = self._are_all_bucket_orders_equal(Y = Y, sorted_indexes = sorted_indexes[0])

        # Second, check the stopping condition
        # If it fulfils, create a tree with root a leaf node
        if self._check_stopping_condition(n_samples               = n_samples,
                                          n_features              = n_features,
                                          all_bucket_orders_equal = all_bucket_orders_equal,
                                          level                   = level):
            # Solve the OBOP if the criterion does not need to compute it before
            if not self.splitter.criterion.require_obop:
                self.splitter.criterion.node_consensus(precedences = gbl_precedences,
                                                       consensus   = consensus)
                
            # Initialize the root
            root = LeafNode(impurity  = impurity,
                            n_samples = n_samples,
                            y         = consensus)
            
            # Initialize the childs
            childs = list()

        # Otherwise, split the indexes accordingly
        else:
            # Get the selected attributes
            selected_atts = self.random_state.choice(a       = useful_atts,
                                                     size    = int(fmin(n_features, self.max_features)),
                                                     replace = False)

            # Initialize the inner data structures of the splitter
            self.splitter.init(n_classes)

            # Compute the parameters according to the defined algorithms of the subclasses
            self._partition_parameters(X                       = X,
                                       selected_atts           = selected_atts,
                                       Y                       = Y,
                                       gbl_precedences         = gbl_precedences,
                                       ind_precedences         = ind_precedences,
                                       ind_pair_order_matrices = ind_pair_order_matrices,
                                       sample_weight           = sample_weight,
                                       sorted_indexes          = sorted_indexes)

            # Split the indexes
            self.splitter.split_indexes(sorted_indexes)

            # IMPORTANT: TO AVOID THE LOSS OF THE PARAMETERS BETWEEN RECURSIVE CALLS, STORE THEM
            # MOREOVER, INSTEAD OF CREATING THEM IN A CONSTRUCTOR, MANUALLY ASSIGN THE VALUES
            # SINCE IT SEEMS THAT A CYTHON BUG IS RECOGNIZED THE IMPURITIES MEMORY VIEW AS TUPLE
            parameters                = Parameters()
            parameters.n_splits       = self.splitter.n_splits
            parameters.impurities     = self.splitter.impurities
            parameters.precedences    = self.splitter.precedences
            parameters.consensus      = self.splitter.consensus
            parameters.sorted_indexes = self.splitter.sorted_indexes

            # Initialize the root of the tree
            root = (InnerNode(impurity  = impurity,
                              n_samples = n_samples,
                              y         = consensus)
                              .init(att        = self.splitter.att,
                                    thresholds = self.splitter.thresholds_values))

            # Initialize the childs
            childs = [self._build(X                       = X,
                                  removed_atts            = removed_atts,
                                  Y                       = Y,
                                  gbl_precedences         = parameters.precedences[child],
                                  ind_precedences         = ind_precedences,
                                  ind_pair_order_matrices = ind_pair_order_matrices,
                                  sample_weight           = sample_weight,
                                  sorted_indexes          = parameters.sorted_indexes[child],
                                  consensus               = parameters.consensus[child],
                                  impurity                = parameters.impurities[child],
                                  level                   = level + 1)
                      for child in range(parameters.n_splits)]
                    
        # Build the tree
        tree = Tree(root   = root,
                    childs = childs)

        # Return the obtained tree
        return tree

    cdef void _remove_useless_attributes(self,
                                         DTYPE_t[:, :]  X,
                                         REMOVED_ATTS_t *removed_atts,
                                         USEFUL_ATTS_t  *useful_atts,
                                         SIZE_t[:, :]   sorted_indexes) nogil:
        """
            Remove the useless attributes.
        """
        # Initialize some values from the input arrays
        cdef SIZE_t n_samples  = sorted_indexes[0].shape[0]
        cdef SIZE_t n_features = X.shape[1]

        # Define some values to be employed
        cdef BOOL_t useful

        # Define the indexes
        cdef SIZE_t att
        cdef SIZE_t sample    

        # Iterate through all the attributes to check if kept or not
        for att in range(n_features):
            # Continue if the attribute has been already removed
            if removed_atts.find(att) != removed_atts.end():
                continue
            # Reinitialize the useful boolean variable
            useful = False
            # Iterate all the samples and check whether the attribute is useful
            # by looking that there is, at least, two different samples
            for sample in range(1, n_samples):
                # Different value of the attribute, it is useful
                if X[sorted_indexes[att, 0], att] != X[sorted_indexes[att, sample], att]:
                    useful = True
                    break
            # Useful, keep the attribute
            if useful:
                useful_atts.push_back(att)
            # Otherwise, add to the corresponding set
            else:
                removed_atts.insert(att)
    
    cdef BOOL_t _are_all_bucket_orders_equal(self,
                                             DTYPE_t[:, :] Y,
                                             SIZE_t[:]     sorted_indexes) nogil:
        """
            Check if all the bucket orders are the same, taking into account
            that NaN is equal to NaN.
        """
        # Initialize some values from the input arrays
        cdef SIZE_t n_samples = sorted_indexes.shape[0]
        cdef SIZE_t n_classes = Y.shape[1]

        # Define the indexes
        cdef SIZE_t label
        cdef SIZE_t sample

        # Initialize the boolean variable
        cdef BOOL_t all_bucket_orders_equal = True

        # Look for all the labels
        for label in range(n_classes):
            # Look for all the samples
            for sample in range(1, n_samples):
                # Non equal
                if Y[sorted_indexes[0], label] != Y[sorted_indexes[sample], label] or isnan(Y[sorted_indexes[0], label]) and isnan(Y[sorted_indexes[sample], label]):
                    all_bucket_orders_equal = False
                    break
            # Break as soon as there is a different bucket order
            if not all_bucket_orders_equal:
                break

        # Return the obtained boolean variable
        return all_bucket_orders_equal

    cdef BOOL_t _check_stopping_condition(self,
                                          SIZE_t n_samples,
                                          SIZE_t n_features,
                                          BOOL_t all_bucket_orders_equal,
                                          SIZE_t level) nogil:
        """
            Check the stopping condition.
        """
        return (n_features == 0                    or # No useful partition
                all_bucket_orders_equal            or # All the bucket orders are the same
                n_samples < self.min_samples_split or # The number of samples is less than specified
                level == self.max_depth)              # The level of the tree is equal to the maximum one
    
    cdef void _partition_parameters(self,
                                    DTYPE_t[:, :]       X,
                                    SIZE_t[:]           selected_atts,
                                    DTYPE_t[:, :]       Y,
                                    DTYPE_t[:, :, :]    gbl_precedences,
                                    DTYPE_t[:, :, :, :] ind_precedences,
                                    DTYPE_t[:, :, :]    ind_pair_order_matrices,
                                    DTYPE_t[:]          sample_weight,
                                    SIZE_t[:, :]        sorted_indexes) nogil:
        pass

# =============================================================================
# Greedy builder
# =============================================================================
@cython.final
cdef class GreedyBuilder(TreeBuilder):
    """
        Class for building a tree using the state-of-the-art greedy algorithm.
    """
    
    cdef void _partition_parameters(self,
                                    DTYPE_t[:, :]       X,
                                    SIZE_t[:]           selected_atts,
                                    DTYPE_t[:, :]       Y,
                                    DTYPE_t[:, :, :]    gbl_precedences,
                                    DTYPE_t[:, :, :, :] ind_precedences,
                                    DTYPE_t[:, :, :]    ind_pair_order_matrices,
                                    DTYPE_t[:]          sample_weight,
                                    SIZE_t[:, :]        sorted_indexes) nogil:
        """
            Obtain the parameters for the decision tree according to the given set of attributes and bucket orders.
        """
        # Initialize some values from the input arrays
        cdef SIZE_t n_features = selected_atts.shape[0]

        # Define some values to be employed
        cdef SIZE_t selected_att

        # Define the indexes
        cdef SIZE_t att

        # Iterate through the selected attributes to obtain the best parameters
        for att in range(n_features):
            # Obtain the selected attribute
            selected_att = selected_atts[att]
            # Obtain the best parameters that can be obtained from the partition generated by the splitter
            self.splitter.partition_parameters(X                       = X[:, selected_att],
                                               att                     = selected_att,
                                               Y                       = Y,
                                               gbl_precedences         = gbl_precedences,
                                               ind_precedences         = ind_precedences,
                                               ind_pair_order_matrices = ind_pair_order_matrices,
                                               sample_weight           = sample_weight,
                                               sorted_indexes          = sorted_indexes[selected_att])
