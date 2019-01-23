# cython:    language_level = 3
# cython:    cdivision      = True
# cython:    boundscheck    = False
# cython:    wraparound     = False

# =============================================================================
# Imports
# =============================================================================

# PLR
from ._splitter cimport Splitter
from ._tree     cimport Tree, Node, InnerNode, LeafNode
from .._types   cimport BOOL_t, DTYPE_t, SIZE_t
from ._types    cimport REMOVED_ATTS_t, USEFUL_ATTS_t
from ._types    cimport Parameters

# Cython
from cpython.list cimport list

# =============================================================================
# Base builder
# =============================================================================
cdef class TreeBuilder:
    """
        Base class for building a tree.
    """

    # Hyperparameters
    cdef Splitter splitter
    cdef SIZE_t   max_depth
    cdef SIZE_t   min_samples_split
    cdef SIZE_t   max_features
    cdef object   random_state

    # Methods
    cpdef Tree build(self,
                     DTYPE_t[:, :]       X,
                     DTYPE_t[:, :]       Y,
                     DTYPE_t[:, :, :]    gbl_precedences,
                     DTYPE_t[:, :, :, :] ind_precedences,
                     DTYPE_t[:, :, :]    ind_pair_order_matrices,
                     SIZE_t[:]           consensus,
                     DTYPE_t[:]          sample_weight,
                     SIZE_t[:, :]        sorted_indexes)

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
                     SIZE_t              level)

    cdef void _remove_useless_attributes(self,
                                         DTYPE_t[:, :]  X,
                                         REMOVED_ATTS_t *removed_atts,
                                         USEFUL_ATTS_t  *useful_atts,
                                         SIZE_t[:, :]   sorted_indexes) nogil
                                        
    cdef BOOL_t _are_all_bucket_orders_equal(self,
                                             DTYPE_t[:, :] Y,
                                             SIZE_t[:]     sorted_indexes) nogil

    cdef BOOL_t _check_stopping_condition(self,
                                          SIZE_t n_samples,
                                          SIZE_t n_features,
                                          BOOL_t all_bucket_orders_equal,
                                          SIZE_t level) nogil

    cdef void _partition_parameters(self,
                                    DTYPE_t[:, :]       X,
                                    SIZE_t[:]           selected_atts,
                                    DTYPE_t[:, :]       Y,
                                    DTYPE_t[:, :, :]    gbl_precedences,
                                    DTYPE_t[:, :, :, :] ind_precedences,
                                    DTYPE_t[:, :, :]    ind_pair_order_matrices,
                                    DTYPE_t[:]          sample_weight,
                                    SIZE_t[:, :]        sorted_indexes) nogil

# =============================================================================
# Greedy builder
# =============================================================================
cdef class GreedyBuilder(TreeBuilder):
    """
        Class for building a tree using the state-of-the-art greedy algorithm.
    """
    pass
    
