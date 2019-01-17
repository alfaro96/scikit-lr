# cython: language_level = 3
# cython: cdivision      = True
# cython: boundscheck    = False
# cython: wraparound     = False

# =============================================================================
# Imports
# =============================================================================

# NumPy
import  numpy as np
cimport numpy as np

# Always include this statement after cimporting numpy, to avoid
# segmentation faults
np.import_array()

# Cython
import cython
from   cython.operator cimport dereference as deref

# C
from libc.math cimport fabs, isnan, NAN, INFINITY

# =============================================================================
# ClusterProbability transformer
# =============================================================================
@cython.final
cdef class ClusterProbabilityTransformer:
    """
        Transformer for probability distributions to bucket orders.
    """
    
    def __cinit__(self,
                  DTYPE_t threshold,
                  object  metric):
        """
            Constructor.
        """
        # Initialize the hyperparameters of the current object
        self.threshold = threshold
        self.metric    = metric

    cpdef void transform(self,
                         DTYPE_t[:, :]    Y,
                         DTYPE_t[:, :]    prev_prob_dists,
                         DTYPE_t[:, :]    new_prob_dists,
                         UINT8_t[:, :, :] matrices) nogil:
        """
            Transform the given probability distributions to bucket orders.
        """
        # Initialize some values from the input arrays
        cdef SIZE_t n_samples = Y.shape[0]
        cdef SIZE_t n_classes = Y.shape[1]

        # Define the indexes
        cdef SIZE_t sample
        cdef SIZE_t label

        # Fill the main diagonal of the matrices to "True"
        for sample in range(n_samples):
            for label in range(n_classes):
                matrices[sample, label, label] = True

        # Transform each bucket order
        for sample in range(n_samples):
            self._transform(ranking        = Y[sample],
                            prev_prob_dist = prev_prob_dists[sample],
                            new_prob_dist  = new_prob_dists[sample],
                            matrix         = matrices[sample])

    cdef void _transform(self,
                         DTYPE_t[:]    ranking,
                         DTYPE_t[:]    prev_prob_dist,
                         DTYPE_t[:]    new_prob_dist,
                         UINT8_t[:, :] matrix) nogil:
        """
            Recursively transforms a probability distribution to a bucket order.
        """
        # Initialize some values from the input arrays
        cdef SIZE_t n_classes = prev_prob_dist.shape[0]

        # Define and initialize some values to be employed

        # To compute the closer cluster
        cdef DTYPE_t difference
        cdef DTYPE_t min_difference      = INFINITY
        cdef DTYPE_t mean_value_clusters = 0.0

        # To gather the selected clusters
        cdef SIZE_t         left_closer_cluster  = -1
        cdef SIZE_t         right_closer_cluster = -1
        cdef vector[SIZE_t] left_cluster
        cdef vector[SIZE_t] right_cluster

        # To gather the value obtained from the metric
        cdef DTYPE_t value

        # Define the indexes
        cdef SIZE_t label
        cdef SIZE_t f_class
        cdef SIZE_t s_class

        # Copy the probability distribution
        prev_prob_dist[:] = new_prob_dist

        # Only transform if there are classes to merge
        if not self._all_same_bucket(matrix):
            # Obtain the indexes of the classes whose probabilities are closer
            for f_class in range(n_classes):
                for s_class in range(f_class, n_classes):
                    # Not cluster yet and different label
                    if not matrix[f_class, s_class] and f_class != s_class:
                        difference = fabs(prev_prob_dist[f_class] - prev_prob_dist[s_class])
                        if difference < min_difference:
                            min_difference       = difference
                            left_closer_cluster  = f_class
                            right_closer_cluster = s_class

            # Obtain the left, right cluster and the mean value of the probabilities in them 
            for label in range(n_classes):
                # The class is in the left cluster
                if matrix[left_closer_cluster, label]:
                    mean_value_clusters += prev_prob_dist[label]
                    left_cluster.push_back(label)
                # The class is in the right cluster
                elif matrix[right_closer_cluster, label]:
                    mean_value_clusters += prev_prob_dist[label]
                    right_cluster.push_back(label)
            # Compute the mean value
            mean_value_clusters /= (left_cluster.size() + right_cluster.size())

            # Obtain the new probability distribution like if the clusters are merge
            for label in left_cluster:
                new_prob_dist[label] = mean_value_clusters
            for label in right_cluster:
                new_prob_dist[label] = mean_value_clusters

            # Obtain the value of the metric (with the GIL)
            with gil:
                value = self.metric(probs_true = np.asarray(prev_prob_dist[None, :]),
                                    probs_pred = np.asarray(new_prob_dist[None, :]))

            # If the metric used to obtain the distance between the probability
            # distributions is less or equal than the specified threshold, merge
            if value <= self.threshold:
                # Merge
                for f_class in left_cluster:
                    for s_class in right_cluster: 
                        matrix[f_class, s_class] = True
                        matrix[s_class, f_class] = True
                
                # Apply recursively
                self._transform(ranking        = ranking,
                                prev_prob_dist = prev_prob_dist,
                                new_prob_dist  = new_prob_dist,
                                matrix         = matrix)
            # Otherwise, create the bucket order
            else:
                rank_data(data    = prev_prob_dist,
                          ranking = ranking,
                          inverse = True)

        # Otherwise, create the bucket order
        else:
            rank_data(data    = prev_prob_dist,
                      ranking = ranking,
                      inverse = True)

    cdef BOOL_t _all_same_bucket(self,
                                 UINT8_t[:, :] matrix) nogil:
        """
            Check if all the classes are in the same bucket.
        """
        # Initialize some values from the input arrays
        cdef SIZE_t n_classes = matrix.shape[0]

        # Initialize some values to be employed
        cdef BOOL_t all_same_bucket = True

        # Define the indexes
        cdef SIZE_t f_class
        cdef SIZE_t s_class

        # Iterate all the entries of the matrix to check if all the classes belong
        # to the same bucket
        for f_class in range(n_classes):
            for s_class in range(f_class + 1, n_classes):
                if not matrix[f_class, s_class]:
                    all_same_bucket = False
                    break
            # Break if any pair of classes are in different buckets
            if not all_same_bucket:
                break

        # Return if all belong to the same bucket or not
        return all_same_bucket

# =============================================================================
# Top-K transformer
# =============================================================================
@cython.final
cdef class TopKTransformer:
    """
        Transformer for bucket orders to new ones using top-k process.
    """

    def __cinit__(self,
                  perc):
        """
            Constructor.
        """
        # Initialize the hyperparameters of the current object
        self.perc = perc

    cpdef void transform_from_probs(self,
                                    DTYPE_t[:, :] Y,
                                    DTYPE_t[:, :] prob_dists,
                                    SIZE_t[:, :]  sorted_prob_dists) nogil:
        """
            Transform a set of bucket orders using a top-k process given the probability distribution of each instance.
        """
        # Initialize some values from the input arrays
        cdef SIZE_t n_samples = Y.shape[0]
        cdef SIZE_t n_classes = Y.shape[1]

        # Define some values to be employed
        cdef DTYPE_t missed_prob

        # Define the indexes
        cdef SIZE_t sample
        cdef SIZE_t label

        # Obtain the number of classes to be missed
        cdef SIZE_t n_missed_classes = int(self.perc * n_classes)

        # Only miss if the number of classes to be missed is greater than 0
        if n_missed_classes > 0:
            # Transform the bucket orders
            for sample in range(n_samples):
                # Obtain the top-k probability of the current sample
                missed_prob = prob_dists[sample, sorted_prob_dists[sample, n_missed_classes - 1]]
                # Iterate each class to check if it must be missed
                for label in range(n_classes):
                    # Transform if the probability of the current class
                    # is less or equal than the one in the top-k
                    if prob_dists[sample, label] <= missed_prob:
                        Y[sample, label] = INFINITY

    cpdef void transform_from_bucket_orders(self,
                                            DTYPE_t[:, :] Y,
                                            SIZE_t[:, :]  sorted_Y) nogil:
        """
            Transform a set of bucket orders using a top-k process without the probability distributions.
        """
        # Initialize some values from the input arrays
        cdef SIZE_t n_samples = Y.shape[0]
        cdef SIZE_t n_classes = Y.shape[1]

        # Define some values to be employed
        cdef SIZE_t missed_class

        # Define the indexes
        cdef SIZE_t sample
        cdef SIZE_t label

        # Obtain the number of classes to be missed
        cdef SIZE_t n_missed_classes = int(self.perc * n_classes)

        # Only miss if the number of classes to be missed is greater than 0
        if n_missed_classes > 0:
            # Transform the bucket orders
            for sample in range(n_samples):
                for label in range(n_missed_classes):
                    # Obtain the class to be missed
                    missed_class = sorted_Y[sample, label]
                    # Miss such class
                    Y[sample, missed_class] = INFINITY

# =============================================================================
# MissRandom transformer
# =============================================================================
@cython.final
cdef class MissRandomTransformer:
    """
        Transformer for bucket orders missing labels in a random way.
    """

    def __cinit__(self,
                  DTYPE_t perc,
                  object  random_state):
        """
            Constructor.
        """
        # Initialize the hyperparameters of the current object
        self.perc         = perc
        self.random_state = random_state

    cpdef void transform(self,
                         DTYPE_t[:, :] Y) nogil:
        """
            Transform the given bucket orders to incomplete ones.
        """
        # Initialize the values from the input array
        cdef SIZE_t n_samples = Y.shape[0]
        cdef SIZE_t n_classes = Y.shape[1]

        # Define the indexes
        cdef SIZE_t sample
        cdef SIZE_t label
        cdef SIZE_t counter

        # Transform each ranking
        for sample in range(n_samples):
            counter = 0
            for label in range(n_classes):
                # At least two classes ranked
                if (n_classes - counter) == 2:
                    break
                # Check if the label will be missed
                with gil:
                    if self.random_state.rand() < self.perc:
                        Y[sample, label] = NAN
                        counter         += 1
            # Rank again the data after transforming the ranking
            rank_data(ranking = Y[sample],
                      data    = Y[sample],
                      inverse = False)

# =============================================================================
# Global methods
# =============================================================================

cpdef void rank_data(DTYPE_t[:] ranking,
                     DTYPE_t[:] data,
                     BOOL_t     inverse) nogil:
    """
        Rank the data into ranking, avoiding the NaN values
    """
    # Initialize some values from the input arrays
    cdef SIZE_t n_classes = data.shape[0]

    # Define and initialize some values to be employed
    cdef SIZE_t  rnk
    cdef DTYPE_t value
    cdef (unordered_set[DTYPE_t])* unique_data = new unordered_set[DTYPE_t]()

    # Define the indexes
    cdef SIZE_t  label
    cdef DTYPE_t f_value
    cdef DTYPE_t s_value

    # Obtain the unique data
    for label in range(n_classes):
        # Only insert the complete labels
        if not isnan(data[label]) and data[label] != INFINITY:
            deref(unique_data).insert(data[label])

    # Rank the data
    for label in range(n_classes):
        # Initialize the ranking being assigned to the current label
        rnk = 1
        # Get the current value
        f_value = data[label]
        # Avoid the missing labels
        if isnan(f_value) or f_value == INFINITY:
            continue
        # Otherwise, look the ranking of the current label
        for s_value in deref(unique_data):
            if ((s_value < f_value and not inverse) or (s_value > f_value and inverse)):
                rnk += 1
        # Assign the obtained ranking
        ranking[label] = rnk
