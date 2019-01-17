# cython: language_level = 3
# cython: cdivision      = True
# cython: boundscheck    = False
# cython: wraparound     = False

# =============================================================================
# Imports
# =============================================================================

# Numpy
import  numpy as np
cimport numpy as np

# Always include this statement after cimporting numpy, to avoid
# segmentation faults
np.import_array()

# PLR
from ..bucket._builder cimport distance, normalize_matrix, set_matrix

# Cython
import cython

# C
from libc.math cimport log2

# =============================================================================
# Base criterion
# =============================================================================
cdef class Criterion:
    """
        Base class for computing the impurity of a node.
    """
    def __cinit__(self,
                  OBOP_ALGORITHM bucket,
                  DTYPE_t        beta,
                  BOOL_t         require_obop,
                  object         random_state):
        """
            Constructor.
        """
        # Initialize the hyperparameters of the current object
        self.bucket       = bucket
        self.beta         = beta
        self.require_obop = require_obop
        self.random_state = random_state

    cpdef Criterion init(self,
                         SIZE_t n_classes):
        """
            Initialize some data structures to solve the OBOP.
            Since the only thing that is needed at each step is the
            consensus bucket order, that must be stored in a different
            memory location, it is possible to use this self preinitializes
            structures just to avoid multiple copies of this same objects.
        """
        # Initialize the arrays
        self.items             = list(range(n_classes))
        self.pair_order_matrix = np.zeros((n_classes, n_classes), dtype = np.float64)
        self.utopian_matrix    = np.zeros((n_classes, n_classes), dtype = np.float64)
        self.dani_indexes      = np.zeros(n_classes, dtype = np.float64)

        # Initialize the builders
        self.utopian_matrix_builder = UtopianMatrixBuilder()
        self.dani_indexes_builder   = DANIIndexesBuilder()
        self.obop_builder           = OptimalBucketOrderProblemBuilder(algorithm = self.bucket, beta = self.beta, random_state = self.random_state)

        # Initialize an empty pair order matrix for a possible consensus bucket order,
        # that it is not necessary to be returned but computed
        self.consensus_pair_order_matrix = np.zeros((n_classes, n_classes), dtype = np.float64)

        # Return the current object already initialized
        return self

    cdef DTYPE_t node_impurity_consensus(self,
                                         DTYPE_t[:, :, :] precedences,
                                         DTYPE_t[:, :, :] pair_order_matrices,
                                         SIZE_t[:]        consensus,
                                         DTYPE_t[:]       sample_weight,
                                         SIZE_t[:]        sorted_indexes,
                                         SIZE_t           start_index,
                                         SIZE_t           end_index) nogil:
        """
            Compute the impurity and consensus bucket order for the currrent node.
        """
        # Define some values to be employed
        cdef DTYPE_t impurity

        # Obtain the consensus bucket order (if corresponds)
        if self.require_obop:
            self.node_consensus(precedences = precedences,
                                consensus   = consensus)

        # Obtain the impurity from the corresponding subclasses
        impurity = self.node_impurity(precedences         = precedences,
                                      pair_order_matrices = pair_order_matrices,
                                      consensus           = consensus,
                                      sample_weight       = sample_weight,
                                      sorted_indexes      = sorted_indexes,
                                      start_index         = start_index,
                                      end_index           = end_index)

        # Return obtained impurity
        return impurity

    cdef DTYPE_t node_impurity(self,
                               DTYPE_t[:, :, :] precedences,
                               DTYPE_t[:, :, :] pair_order_matrices,
                               SIZE_t[:]        consensus,
                               DTYPE_t[:]       sample_weight,
                               SIZE_t[:]        sorted_indexes,
                               SIZE_t           start_index,
                               SIZE_t           end_index) nogil:
        pass

    cdef void node_consensus(self,
                             DTYPE_t[:, :, :] precedences,
                             SIZE_t[:]        consensus) nogil:
        """
            Compute the consensus bucket order of a node.
        """
        # Initialize some values from the input arrays
        cdef SIZE_t n_classes = precedences.shape[0]

        # IMPORTANT: BEFORE SOLVING THE OBOP, REINITIALIZE ALL THE DATA STRUCTURES
        self.pair_order_matrix[:]           = 0
        self.utopian_matrix[:]              = 0
        self.dani_indexes[:]                = 0
        self.consensus_pair_order_matrix[:] = 0

        # Normalize the pair order matrix from the precedences one
        normalize_matrix(precedences       = precedences,
                         pair_order_matrix = self.pair_order_matrix)

        # Compute the utopian matrix and DANI indexes (if necessary)
        if self.obop_builder.algorithm == OBOP_ALGORITHM.BPA_LIA_SG or \
           self.obop_builder.algorithm == OBOP_ALGORITHM.BPA_LIA_MP or \
           self.obop_builder.algorithm == OBOP_ALGORITHM.BPA_LIA_MP2:
           # Obtain the utopian matrix
           self.utopian_matrix_builder.build(pair_order_matrix = self.pair_order_matrix,
                                             utopian_matrix    = self.utopian_matrix)
           # Obtain the DANI indexes
           self.dani_indexes_builder.build(pair_order_matrix = self.pair_order_matrix,
                                           utopian_matrix    = self.utopian_matrix,
                                           dani_indexes      = self.dani_indexes)

        # Solve the OBOP, calling to the corresponding builder
        self.obop_builder.build(items             = self.items,
                                y                 = consensus,
                                pair_order_matrix = self.pair_order_matrix,
                                utopian_matrix    = self.utopian_matrix,
                                dani_indexes      = self.dani_indexes)

        # Normalize the pair order matrix for the consensus bucket order
        set_matrix(y                 = consensus,
                   pair_order_matrix = self.consensus_pair_order_matrix)

# =============================================================================
# Disagreements criterion
# =============================================================================
@cython.final
cdef class DisagreementsCriterion(Criterion):
    """
        Class for computing the impurity of a node using the disagreements criterion.
    """

    cdef DTYPE_t node_impurity(self,
                               DTYPE_t[:, :, :] precedences,
                               DTYPE_t[:, :, :] pair_order_matrices,
                               SIZE_t[:]        consensus,
                               DTYPE_t[:]       sample_weight,
                               SIZE_t[:]        sorted_indexes,
                               SIZE_t           start_index,
                               SIZE_t           end_index) nogil:
        """
            Compute the impurity of a node with the disagremeents criterion.
        """
        # Initialize some values from the input arrays
        cdef SIZE_t n_classes = precedences.shape[0]

        # Initialize some values to be employed
        cdef DTYPE_t disagreements = 0.0
        cdef DTYPE_t agreements    = 0.0
        cdef DTYPE_t impurity      = 0.0

        # Initialize the indexes
        cdef SIZE_t label
        cdef SIZE_t f_class
        cdef SIZE_t s_class

        # Obtain the disagreements and agreements with the bucket orders of the current partition
        for f_class in range(n_classes):
            for s_class in range(f_class + 1, n_classes):
                if consensus[f_class] < consensus[s_class]:
                    disagreements += precedences[s_class, f_class, 0] + precedences[f_class, s_class, 1]
                    agreements    += precedences[f_class, s_class, 0]
                elif consensus[f_class] == consensus[s_class]:
                    disagreements += precedences[f_class, s_class, 0] + precedences[s_class, f_class, 0]
                    agreements    += precedences[f_class, s_class, 1]
                elif consensus[f_class] > consensus[s_class]:
                    disagreements += precedences[f_class, s_class, 0] + precedences[f_class, s_class, 1]
                    agreements    += precedences[s_class, f_class, 0]
            # Compute the impurity
            impurity = disagreements / (disagreements + agreements)

        # Return the obtained impurity
        return impurity

# =============================================================================
# Distance criterion
# =============================================================================
@cython.final
cdef class DistanceCriterion(Criterion):
    """
        Class for computing the impurity of a node using the distance criterion.
    """

    cdef DTYPE_t node_impurity(self,
                               DTYPE_t[:, :, :] precedences,
                               DTYPE_t[:, :, :] pair_order_matrices,
                               SIZE_t[:]        consensus,
                               DTYPE_t[:]       sample_weight,
                               SIZE_t[:]        sorted_indexes,
                               SIZE_t           start_index,
                               SIZE_t           end_index) nogil:
        """
            Compute the impurity of a node with the distance criterion.
        """
        # Initialize some values from the input arrays
        cdef SIZE_t n_samples = (end_index - start_index)

        # Initialize some values to be employed
        cdef DTYPE_t impurity = 0.0

        # Define the indexes
        cdef SIZE_t sample
        cdef SIZE_t index

        # Iterate all the sample to obtain the distance between the matrices
        for sample in range(n_samples):
            # Obtain the index
            index = sorted_indexes[start_index + sample]
            # Compute the impurity
            impurity += sample_weight[index] * distance(matrix_1 = self.consensus_pair_order_matrix,
                                                        matrix_2 = pair_order_matrices[index])
        impurity /= n_samples

        # Return the obtained impurity
        return impurity

# =============================================================================
# Entropy criterion
# =============================================================================
@cython.final
cdef class EntropyCriterion(Criterion):
    """
        Class for computing the impurity of a node using the entropy criterion.
    """

    cdef DTYPE_t node_impurity(self,
                               DTYPE_t[:, :, :] precedences,
                               DTYPE_t[:, :, :] pair_order_matrices,
                               SIZE_t[:]        consensus,
                               DTYPE_t[:]       sample_weight,
                               SIZE_t[:]        sorted_indexes,
                               SIZE_t           start_index,
                               SIZE_t           end_index) nogil:
        """
            Compute the impurity of a node with the disagrements criterion.
        """
        # Initialize some values from the input arrays
        cdef SIZE_t n_classes = precedences.shape[0]

        # Initialize the indexes
        cdef SIZE_t f_class
        cdef SIZE_t s_class

        # Initialize some values to be employed
        cdef DTYPE_t impurity        = 0.0
        cdef DTYPE_t precedences_sum = 0.0

        # Iterate to obtain the impurity
        for f_class in range(n_classes):
            for s_class in range(f_class + 1, n_classes):
                # Obtain the number of precedences between the current labels
                precedences_sum = precedences[f_class, s_class, 0] + precedences[f_class, s_class, 1] + precedences[s_class, f_class, 0]
                # Avoid errors when there is no bucket order where the label "f_class" precedes "s_class"
                if precedences[f_class, s_class, 0] > 0:
                    impurity -= precedences[f_class, s_class, 0] / precedences_sum * log2(precedences[f_class, s_class, 0] / precedences_sum)
                # Avoid errors when there is no bucket order where the label "f_class" is tied with "s_class"
                if precedences[f_class, s_class, 1] > 0: 
                    impurity -= precedences[f_class, s_class, 1] / precedences_sum * log2(precedences[f_class, s_class, 1] / precedences_sum)
                # Avoid errors when there is no bucket order where the label "s_class" precedes "f_class"
                if precedences[s_class, f_class, 0] > 0:
                    impurity -= precedences[s_class, f_class, 0] / precedences_sum * log2(precedences[s_class, f_class, 0] / precedences_sum)
            impurity /= ((n_classes * (n_classes - 1)) / 2)

        # Return the obtained impurity
        return impurity 
