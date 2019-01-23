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
from ..bucket._builder cimport OBOP_ALGORITHM, PairOrderMatrixBuilder, UtopianMatrixBuilder, DANIIndexesBuilder, OptimalBucketOrderProblemBuilder
from ..bucket._types   cimport BUCKET_t
from .._types          cimport DTYPE_t, SIZE_t

# Cython
import cython

# =============================================================================
# Base builder for the nearest-neighbors paradigm
# =============================================================================
cdef class BaseNeighborsBuilder:
    """
        Base builder for the nearest-neighbors algorithm.
    """

    def __cinit__(self,
                  OBOP_ALGORITHM bucket,
                  DTYPE_t        beta,
                  object         random_state):
        """
            Constructor.
        """
        # Initialize the hyperparameters of the current object
        self.bucket       = bucket
        self.beta         = beta
        self.random_state = random_state

    cpdef BaseNeighborsBuilder init(self,
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
        self.precedences       = np.zeros((n_classes, n_classes, 2), dtype = np.float64)
        self.pair_order_matrix = np.zeros((n_classes, n_classes),    dtype = np.float64)
        self.utopian_matrix    = np.zeros((n_classes, n_classes),    dtype = np.float64)
        self.dani_indexes      = np.zeros(n_classes, dtype = np.float64)

        # Initialize the builders
        self.pair_order_matrix_builder = PairOrderMatrixBuilder()
        self.utopian_matrix_builder    = UtopianMatrixBuilder()
        self.dani_indexes_builder      = DANIIndexesBuilder()
        self.obop_builder              = OptimalBucketOrderProblemBuilder(algorithm = self.bucket, beta = self.beta, random_state = self.random_state)

        # Return the current object already initialized
        return self

# =============================================================================
# Builder for the k-nearest-neighbors paradigm
# =============================================================================
@cython.final
cdef class KNeighborsBuilder(BaseNeighborsBuilder):
    """
        Builder for the k-nearest-neighbors algorithm.
    """
    
    cpdef void predict(self,
                       DTYPE_t[:, :, :] Y,
                       DTYPE_t[:, :]    sample_weight,
                       SIZE_t[:, :]     predictions) nogil:
        """
            Predict according to the given k-nearest buckets and sample weight for each instance,
            that is, solve the OBOP.
        """
        # Initialize some values from the input arrays
        cdef SIZE_t n_samples = Y.shape[0]

        # Define the indexes
        cdef SIZE_t sample

        # Iterate all the samples to solve the OBOP
        # according to the k-nearest neighbors of each instance
        for sample in range(n_samples):
            # IMPORTANT: BEFORE SOLVING THE OBOP, REINITIALIZE ALL THE DATA STRUCTURES
            self.precedences[:]       = 0
            self.pair_order_matrix[:] = 0
            self.utopian_matrix[:]    = 0
            self.dani_indexes[:]      = 0

            # Obtain the pair order matrix for the nearest neighbors of the current instance
            self.pair_order_matrix_builder.build(Y                 = Y[sample],
                                                 sample_weight     = sample_weight[sample],
                                                 precedences       = self.precedences,
                                                 pair_order_matrix = self.pair_order_matrix)

            # Compute the utopian matrix and DANI indexes (if necessary)
            if self.obop_builder.algorithm == OBOP_ALGORITHM.BPA_LIA_SG  or \
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
                                    y                 = predictions[sample],
                                    pair_order_matrix = self.pair_order_matrix,
                                    utopian_matrix    = self.utopian_matrix,
                                    dani_indexes      = self.dani_indexes)
