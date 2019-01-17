# cython: language_level = 3
# cython: cdivision      = True
# cython: boundscheck    = False
# cython: wraparound     = False

# =============================================================================
# Imports
# =============================================================================

# Cython
import cython

# C
from libc.math cimport fabs, isnan, NAN, INFINITY

# =============================================================================
# Pair order matrix builder
# =============================================================================
@cython.final
cdef class PairOrderMatrixBuilder:
    """
        Builder for a pair order matrix.
    """

    cpdef void build(self,
                     DTYPE_t[:, :]    Y,
                     DTYPE_t[:]       sample_weight,
                     DTYPE_t[:, :, :] precedences,
                     DTYPE_t[:, :]    pair_order_matrix) nogil:
        """
            Build a pair order matrix from the dataset Y.
        """
        # Initialize some values from the input arrays
        cdef SIZE_t n_samples = Y.shape[0]
        cdef SIZE_t n_classes = Y.shape[1]

        # Define the indexes
        cdef SIZE_t sample
        cdef SIZE_t f_class
        cdef SIZE_t s_class

        # Normalize the sample weight making them sum the number of samples
        normalize_sample_weight(sample_weight)

        # Iterate all the samples to build the precedences matrix
        for sample in range(n_samples):
            # Set the values into the precedences matrix
            set_precedences(y           = Y[sample],
                            weight      = sample_weight[sample],
                            precedences = precedences)

        # Normalize the precedences matrix to obtain the pair order matrix
        normalize_matrix(precedences       = precedences,
                         pair_order_matrix = pair_order_matrix)

# =============================================================================
# Utopian matrix builder
# =============================================================================
@cython.final
cdef class UtopianMatrixBuilder:
    """
        Builder for an utopian matrix.
    """

    cpdef void build(self,
                     DTYPE_t[:, :] pair_order_matrix,
                     DTYPE_t[:, :] utopian_matrix) nogil:
        """
            Build an utopian matrix from the given pair order matrix.
        """
        # Initialize some values from the input arrays
        cdef SIZE_t n_classes = pair_order_matrix.shape[0]

        # Define the indexes
        cdef SIZE_t f_class
        cdef SIZE_t s_class

        # Iterate all the entries of the pair order matrix to build the utopian matrix
        for f_class in range(n_classes):
            for s_class in range(f_class, n_classes):
                if pair_order_matrix[f_class, s_class] > 0.75:
                    utopian_matrix[f_class, s_class] = 1.0
                    utopian_matrix[s_class, f_class] = 0.0
                elif pair_order_matrix[f_class, s_class] >= 0.25 and pair_order_matrix[f_class, s_class] <= 0.75:
                    utopian_matrix[f_class, s_class] = 0.5
                    utopian_matrix[s_class, f_class] = 0.5
                elif pair_order_matrix[f_class, s_class] < 0.25:
                    utopian_matrix[f_class, s_class] = 0.0
                    utopian_matrix[s_class, f_class] = 1.0

# =============================================================================
# Anti-utopian matrix builder
# =============================================================================
@cython.final
cdef class AntiUtopianMatrixBuilder:
    """
        Builder for an anti-utopian matrix.
    """

    cpdef void build(self,
                     DTYPE_t[:, :] pair_order_matrix,
                     DTYPE_t[:, :] anti_utopian_matrix) nogil:
        """
            Build an anti-utopian matrix from the given pair order matrix.
        """
        # Initialize some values from the input arrays
        cdef SIZE_t n_classes = pair_order_matrix.shape[0]

        # Define the indexes
        cdef SIZE_t f_class
        cdef SIZE_t s_class

        # Iterate all the entries of the pair order matrix to build the anti-utopian matrix
        for f_class in range(n_classes):
            for s_class in range(n_classes):
                if pair_order_matrix[f_class, s_class] > 0.5:
                    anti_utopian_matrix[f_class, s_class] = 0.0
                elif pair_order_matrix[f_class, s_class] <= 0.5:
                    anti_utopian_matrix[f_class, s_class] = 1.0

# =============================================================================
# DANI indexes builder
# =============================================================================
@cython.final
cdef class DANIIndexesBuilder:
    """
        Builder for the DANI indexes.
    """

    # Methods
    cpdef void build(self,
                     DTYPE_t[:, :] pair_order_matrix,
                     DTYPE_t[:, :] utopian_matrix,
                     DTYPE_t[:]    dani_indexes) nogil:
        """
            Build the DANI indexes from the pair order and utopian matrix.
        """
        # Initialize some values from the input arrays
        cdef SIZE_t n_classes = utopian_matrix.shape[0]

        # Define the indexes
        cdef SIZE_t f_class
        cdef SIZE_t s_class

        # Iterate all the entries of the pair order matrix and utopian matrix to build the DANI indexes
        for f_class in range(n_classes):
            for s_class in range(n_classes):
                if f_class != s_class:
                    dani_indexes[f_class] += fabs(pair_order_matrix[f_class, s_class] - utopian_matrix[f_class, s_class])
            dani_indexes[f_class] /= (n_classes - 1)

# =============================================================================
# Optimal Bucket Order Problem builder
# =============================================================================
@cython.final
cdef class OptimalBucketOrderProblemBuilder:
    """
        Builder for the Optimal Bucket Order Problem.
    """

    def __cinit__(self,
                  OBOP_ALGORITHM algorithm,
                  DTYPE_t        beta,
                  object         random_state):
        """
            Constructor.
        """
        # Initialize the hyperparameters of the current object
        self.algorithm    = algorithm
        self.beta         = beta
        self.random_state = random_state

    cpdef void build(self,
                     BUCKET_t      items,
                     SIZE_t[:]     y,
                     DTYPE_t[:, :] pair_order_matrix,
                     DTYPE_t[:, :] utopian_matrix,
                     DTYPE_t[:]    dani_indexes) nogil:
        """
            Build the buckets for the Optimal Bucket Order Problem.
        """        
        # Initialize some values from the input arrays
        cdef SIZE_t n_classes = items.size()

        # Define the indexes
        cdef SIZE_t bucket
        cdef SIZE_t label

        # Define some values to be employed
        cdef BUCKETS_t buckets
        cdef SIZE_t    n_buckets
        
        # Build the buckets recursively
        buckets = self._build(items             = items,
                              pair_order_matrix = pair_order_matrix,
                              utopian_matrix    = utopian_matrix,
                              dani_indexes      = dani_indexes)
        
        # Obtain the number of buckets
        n_buckets = buckets.size()

        # Build the bucket order
        for bucket in range(n_buckets):
            for label in buckets[bucket]:
                y[label] = (bucket + 1)

    cdef BUCKETS_t _build(self,
                          BUCKET_t      items,
                          DTYPE_t[:, :] pair_order_matrix,
                          DTYPE_t[:, :] utopian_matrix,
                          DTYPE_t[:]    dani_indexes) nogil:
        """
            Recursive function to build the buckets for the Optimal Bucket Order Problem.
        """
        # Initialize some values from the input arrays
        cdef SIZE_t n_items = items.size()

        # Initialize the pivot (index)
        cdef SIZE_t pivot       = 0
        cdef SIZE_t pivot_index = 0

        # Define the left, central and right buckets
        cdef BUCKET_t left
        cdef BUCKET_t left_prime
        cdef BUCKET_t central
        cdef BUCKET_t central_prime
        cdef DTYPE_t  central_averaged
        cdef BUCKET_t right
        cdef BUCKET_t right_prime
        cdef BUCKET_t left_right_prime

        # Define the buckets being recursively created
        cdef BUCKETS_t left_buckets
        cdef BUCKETS_t central_buckets
        cdef BUCKETS_t right_buckets
        cdef BUCKETS_t buckets

        # Define the indexes
        cdef SIZE_t item
        cdef SIZE_t label

        # Define and initialize some auxiliar values
        cdef SIZE_t  counter
        cdef DTYPE_t lower_dani = INFINITY

        # Return an empty vector (list) if there are no more items
        if n_items == 0:
            return buckets

        # If the algorithm is the original one, pick the pivot at random
        if (self.algorithm == OBOP_ALGORITHM.BPA_ORIGINAL_SG or
            self.algorithm == OBOP_ALGORITHM.BPA_ORIGINAL_MP or
            self.algorithm == OBOP_ALGORITHM.BPA_ORIGINAL_MP2):
                with gil:
                    pivot = items[self.random_state.randint(n_items)]
        # Otherwise, use the one minimizing the DANI indexes
        else:
            for item in items:
                if dani_indexes[item] < lower_dani:
                    pivot      = item
                    lower_dani = dani_indexes[item]

        # Initialize central with the pivot
        central.push_back(pivot)

        # If the algorithm is with single pivot, use the pivot to decide the precedence relation
        if (self.algorithm == OBOP_ALGORITHM.BPA_ORIGINAL_SG or
            self.algorithm == OBOP_ALGORITHM.BPA_LIA_SG):
            for item in items:
                # Avoid the pivot
                if item == pivot:
                    continue
                # Preference for the current item rather than the pivot, so move it to the left bucket
                elif pair_order_matrix[pivot, item] < 0.5 - self.beta:
                    left.push_back(item)
                # Tie between the current item and the pivot, so move it to the central pivot
                elif 0.5 - self.beta <= pair_order_matrix[pivot, item] and pair_order_matrix[pivot, item] <= 0.5 + self.beta:
                    central.push_back(item)
                # Otherwise, prefer the pivot rather than the current item, so move it to the right bucket
                else:
                    right.push_back(item)
        # Otherwise, it is with multi-pivot
        # Use all the elements in the central list
        else:
            for item in items:
                # Initialize the values for the central bucket and its number of items
                central_averaged = 0.0
                counter          = 0
                # Compute the average of the items contained in the central bucket
                for label in central:
                    central_averaged += pair_order_matrix[label, item]
                    counter          += 1
                central_averaged /= counter
                
                # Avoid the pivot
                if item == pivot:
                    continue
                # Preference for the current item rather than those in the central bucket, so move to the left one
                elif central_averaged < 0.5 - self.beta:
                    left.push_back(item)
                # Tie between the current item and those in the central bucket, so move to the central one
                elif 0.5 - self.beta <= central_averaged and central_averaged <= 0.5 + self.beta:
                    central.push_back(item)
                # Otherwise, prefer the items in the central bucket rather than the current one, so move it to the right bucket
                else:
                    right.push_back(item)

        # Check whether a two-stage is applied
        if (self.algorithm == OBOP_ALGORITHM.BPA_ORIGINAL_MP2 or
            self.algorithm == OBOP_ALGORITHM.BPA_LIA_MP2):
            # Initialize the auxiliar left, central and right buckets,
            # copying the original left, central and right buckets
            # and initializing left and right empty
            left_prime    = left
            central_prime = central
            right_prime   = right
            left.clear()
            right.clear()

            # Concatenate the auxiliar left and right buckets to apply the two-stage procedure
            left_right_prime.insert(left_right_prime.end(), left_prime.begin(), left_prime.end())
            left_right_prime.insert(left_right_prime.end(), right_prime.begin(), right_prime.end())

            # Apply the algorithm standard algorithm with the auxiliar buckets
            for item in left_right_prime:
                # Initialize the values for the central bucket and its number of items
                central_averaged = 0.0
                counter          = 0
                # Compute the average of the items contained in the central bucket
                for label in central:
                    central_averaged += pair_order_matrix[label, item]
                    counter          += 1
                central_averaged /= counter
                
                # Avoid the pivot
                if item == pivot:
                    continue
                # Preference for the current item rather than those in the central bucket, so move to the left one
                elif central_averaged < 0.5 - self.beta:
                    left.push_back(item)
                # Tie between the current item and those in the central bucket, so move to the central one
                elif 0.5 - self.beta <= central_averaged and central_averaged <= 0.5 + self.beta:
                    central.push_back(item)
                # Otherwise, prefer the items in the central bucket rather than the current one, so move it to the right bucket
                else:
                    right.push_back(item)

        # Apply the algorithm recursively

        # Left bucket
        left_buckets = self._build(items             = left,
                                   pair_order_matrix = pair_order_matrix,
                                   utopian_matrix    = utopian_matrix,
                                   dani_indexes      = dani_indexes)

        # Central bucket
        central_buckets.push_back(central)

        # Right bucket
        right_buckets = self._build(items             = right,
                                    pair_order_matrix = pair_order_matrix,
                                    utopian_matrix    = utopian_matrix,
                                    dani_indexes      = dani_indexes)

        # Concatenation of the buckets
        # Append to the end of "buckets" the content (from the beginning to the end)
        # of "left_buckets", "central_buckets" and "right_buckets"
        buckets.insert(buckets.end(), left_buckets.begin(),    left_buckets.end())
        buckets.insert(buckets.end(), central_buckets.begin(), central_buckets.end())
        buckets.insert(buckets.end(), right_buckets.begin(),   right_buckets.end())

        # Return the obtained buckets
        return buckets

# =============================================================================
# Global methods
# =============================================================================

cpdef DTYPE_t distance(DTYPE_t[:, :] matrix_1,
                       DTYPE_t[:, :] matrix_2) nogil:
    """
        Compute the distance between two matrices.
    """
    # Initialize some values from the input arrays
    cdef SIZE_t n_classes = matrix_1.shape[0]

    # Initialize some values to be employed
    cdef DTYPE_t distance = 0.0

    # Define the indexes
    cdef SIZE_t f_class
    cdef SIZE_t s_class

    # Iterate the elements above the main diagonal of both
    # matrices to compute the distance between them, taking
    # into account that the elements below the main diagonal
    # are symettric with respect to both matrices
    for f_class in range(n_classes):
        for s_class in range(n_classes):
            distance += fabs(matrix_1[f_class, s_class] - matrix_2[f_class, s_class])

    # Return the distance between the matrices
    return distance

cpdef void normalize_matrix(DTYPE_t[:, :, :] precedences,
                            DTYPE_t[:, :]    pair_order_matrix) nogil:
    """
        Normalize the precedences matrix to obtain the pair order matrix.
    """
    # Initialize some values from the input arrays
    cdef SIZE_t n_classes = precedences.shape[0]

    # Define the indexes
    cdef SIZE_t f_class
    cdef SIZE_t s_class

    # Normalize
    for f_class in range(n_classes):
        for s_class in range(f_class, n_classes):
            pair_order_matrix[f_class, s_class] = ((1.0 * precedences[f_class, s_class, 0] + 0.5 * precedences[f_class, s_class, 1]) /                                                               (precedences[f_class, s_class, 0] + precedences[f_class, s_class, 1] + precedences[s_class, f_class, 0]))
            pair_order_matrix[s_class, f_class] = 1.0 - pair_order_matrix[f_class, s_class]

cpdef void normalize_sample_weight(DTYPE_t[:] sample_weight) nogil:
    """
        Normalize the weights to make them sum the number of samples.
    """
    # Initialize some values from the input arrays
    cdef SIZE_t n_samples = sample_weight.shape[0]

    # Initialize some values to be employed
    cdef DTYPE_t sample_weight_sum = 0.0

    # Define the indexes
    cdef SIZE_t sample

    # Normalize the weights making them sum the number of samples
    for sample in range(n_samples):
        sample_weight_sum += sample_weight[sample]
    for sample in range(n_samples):
        sample_weight[sample] = (sample_weight[sample] / sample_weight_sum) * n_samples

cpdef void set_matrix(DTYPE_OR_SIZE_t[:] y,
                      DTYPE_t[:, :]      pair_order_matrix) nogil:
    """
        Fill the pair order matrix according to the given bucket order.
    """
    # Initialize some values from the input arrays
    cdef SIZE_t n_classes = y.shape[0]

    # Define the indexes
    cdef SIZE_t f_class
    cdef SIZE_t s_class

    # Build the precedences matrix iterating the classes
    for f_class in range(n_classes):
        for s_class in range(f_class, n_classes):
            # Same label, put 0.5 in main diagonal
            if f_class == s_class:
                pair_order_matrix[f_class, s_class] = 0.5
            # Otherwise, apply the standard procedure 
            else:
                # "f_class" precedes "s_class", put 1.0 in the entry where
                # counting the precedences of "f_class" regarding "s_class"
                # and 0.0 in the entry where counting the precedences of
                # "s_class" regarding "f_class"
                if y[f_class] < y[s_class]:
                    pair_order_matrix[f_class, s_class] = 1.0
                    pair_order_matrix[s_class, f_class] = 0.0
                # "f_class" is tied with "s_class", put 0.5 in the entry where
                # counting the precedences of "f_class" regarding "s_class"
                # and 0.5 in the entry where counting the precedences of
                # "s_class" regarding "f_class"
                elif y[f_class] == y[s_class]:
                    pair_order_matrix[f_class, s_class] = 0.5
                    pair_order_matrix[s_class, f_class] = 0.5
                # "s_class" precedes "f_class", put 0.0 in the entry where
                # counting the precedences of "f_class" regarding "s_class"
                # and 1.0 in the entry where counting the precedences of
                # "s_class" regarding "f_class"
                else:
                    pair_order_matrix[f_class, s_class] = 0.0
                    pair_order_matrix[s_class, f_class] = 1.0

cpdef void set_precedences(DTYPE_OR_SIZE_t[:] y,
                           DTYPE_t            weight,
                           DTYPE_t[:, :, :]   precedences) nogil:
    """
        Set the precedences of the current bucket order into
        the corresponding precedecences matrix according to the given weight.
    """
    # Initialize some values from the input arrays
    cdef SIZE_t n_classes = y.shape[0]

    # Define the indexes
    cdef SIZE_t f_class
    cdef SIZE_t s_class

    # Build the precedences matrix iterating the classes
    for f_class in range(n_classes):
        for s_class in range(f_class, n_classes):
            # Same label, put "weight" in the second entry (tied entry) of "f_class" regarding "s_class"
            if f_class == s_class:
                precedences[f_class, s_class, 1] += weight
            # Otherwise, apply the standard procedure 
            else:
                # "f_class" precedes "s_class", put "weight" in the entry where
                # counting the precedences of "f_class" regarding "s_class"
                if y[f_class] < y[s_class]:
                    precedences[f_class, s_class, 0] += weight
                # "f_class" is tied with "s_class", put "weight" in the entries where
                # counting the ties of "f_class" and "s_class"
                elif y[f_class] == y[s_class]:
                    precedences[f_class, s_class, 1] += weight
                    precedences[s_class, f_class, 1] += weight
                # "s_class" precedes "f_class", put "weight" in the entry
                # where counting the precedences of "s_class" regarding "f_class"
                elif y[f_class] > y[s_class]:
                    precedences[s_class, f_class, 0] += weight
                    
                # Missing label detected
                if isnan(y[f_class]) or isnan(y[s_class]):
                    # Since either "f_class" or "s_class" is missed, it is considered
                    # that either "f_class" could precede "s_class" or viceversa
                    # Moreover, it is also considered that they can be tied
                    precedences[f_class, s_class, 0] += weight
                    precedences[s_class, f_class, 0] += weight
                    precedences[f_class, s_class, 1] += weight
                    precedences[s_class, f_class, 1] += weight
