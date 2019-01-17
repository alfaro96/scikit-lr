# cython:    language_level = 3
# cython:    cdivision      = True
# cython:    boundscheck    = False
# cython:    wraparound     = False

# =============================================================================
# Imports
# =============================================================================

# PLR
from ._criterion cimport Criterion
from .._types    cimport BOOL_t, DTYPE_t, SIZE_t, UINT8_t
from ._types     cimport INDEXES_t

# =============================================================================
# Base splitter
# =============================================================================
cdef class Splitter:
    """
        Base class for splitting the node of a tree.
    """

    # Hyperparameters
    cdef Criterion criterion
    cdef SIZE_t    max_splits
    cdef object    random_state

    # Attributes
    cdef SIZE_t              att
    cdef DTYPE_t             impurity
    cdef SIZE_t              n_splits
    cdef DTYPE_t[:]          impurities
    cdef SIZE_t[:]           thresholds
    cdef DTYPE_t[:]          thresholds_values
    cdef DTYPE_t[:, :, :, :] precedences
    cdef DTYPE_t[:, :, :, :] aux_precedences
    cdef SIZE_t[:, :]        consensus
    cdef SIZE_t[:, :]        aux_consensus
    cdef list                sorted_indexes

    # Methods
    cdef void init(self,
                   SIZE_t n_classes)
                   
    cdef void _init(self,
                    SIZE_t n_classes,
                    SIZE_t n_splits)

    cdef void partition_parameters(self,
                                   DTYPE_t[:]          X,
                                   SIZE_t              att,
                                   DTYPE_t[:, :]       Y,
                                   DTYPE_t[:, :, :]    gbl_precedences,
                                   DTYPE_t[:, :, :, :] ind_precedences,
                                   DTYPE_t[:, :, :]    ind_pair_order_matrices,
                                   DTYPE_t[:]          sample_weight,
                                   SIZE_t[:]           sorted_indexes) nogil

    cdef void _subtract_precedences(self,
                                    DTYPE_t[:, :, :] first_operand,
                                    DTYPE_t[:, :, :] second_operand,
                                    DTYPE_t[:, :, :] output_operand,
                                    DTYPE_t          sample_weight) nogil

    cdef void _add_precedences(self,
                               DTYPE_t[:, :, :] first_operand,
                               DTYPE_t[:, :, :] second_operand,
                               DTYPE_t[:, :, :] output_operand,
                               DTYPE_t          sample_weight) nogil

    cdef void split_indexes(self,
                            SIZE_t[:, :] sorted_indexes)

    cdef void _in1d(self,
                    SIZE_t[:]  element,
                    INDEXES_t  *test_elements,
                    UINT8_t[:] matrix) nogil
                    
    cdef void _memory_view_to_set(self,
                                  SIZE_t[:] test_elements,
                                  INDEXES_t *test_elements_set) nogil

# =============================================================================
# Binary splitter
# =============================================================================
cdef class BinarySplitter(Splitter):
    """
        Class for splitting the node of a tree selecting the best threshold for each attribute.
    """
    pass
