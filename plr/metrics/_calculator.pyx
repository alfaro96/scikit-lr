# cython: language_level = 3
# cython: cdivision      = True
# cython: boundscheck    = False
# cython: wraparound     = False

# =============================================================================
# Imports
# =============================================================================

# C
from libc.math cimport sqrt, log

# =============================================================================
# Global methods
# =============================================================================

cpdef DTYPE_t bhattacharyya_distance_calculator(DTYPE_t[:, :] probs_true,
                                                DTYPE_t[:, :] probs_pred,
                                                DTYPE_t[:]    sample_weight) nogil:
    """
        Calculator for the Bhattacharyya distance.
    """
    # Initialize some values from the input arrays
    cdef SIZE_t n_samples = probs_true.shape[0]

    # Initialize and define some variables to be employed
    cdef DTYPE_t distance       = 0.0
    cdef DTYPE_t local_distance

    # Define the indexes
    cdef SIZE_t sample

    # Normalize the sample weight
    normalize_sample_weight(sample_weight)

    # Iterate through all the sample to obtain the distance one by one
    for sample in range(n_samples):
        # Avoid errors when the sample weight is zero
        if sample_weight[sample] != 0:
            distance += sample_weight[sample] * -log(bhattacharyya_score_calculator(probs_true    = probs_true[sample, None],
                                                                                    probs_pred    = probs_pred[sample, None],
                                                                                    sample_weight = sample_weight[sample, None]))
    
    # Return the obtained distance
    return distance

cpdef DTYPE_t bhattacharyya_score_calculator(DTYPE_t[:, :] probs_true,
                                             DTYPE_t[:, :] probs_pred,
                                             DTYPE_t[:]    sample_weight) nogil:
    """
        Calculator for the Bhattacharyya score.
    """
    # Initialize some values from the input arrays
    cdef SIZE_t n_samples = probs_true.shape[0]
    cdef SIZE_t n_classes = probs_true.shape[1]

    # Define the indexes
    cdef SIZE_t sample
    cdef SIZE_t label

    # Initialize and define some variables to be employed
    cdef DTYPE_t coefficient       = 0.0
    cdef DTYPE_t local_coefficient

    # Normalize the sample weight
    normalize_sample_weight(sample_weight)

    # Iterate all the samples to obtain the coefficients
    for sample in range(n_samples):
        # Reinitialize the local coefficient to zero
        local_coefficient = 0.0
        # Iterate all the classes
        for label in range(n_classes):
            local_coefficient += sqrt(probs_true[sample, label] * probs_pred[sample, label])
        # Obtain the coefficient of the current probability distribution, weighting accordingly
        coefficient += sample_weight[sample] * local_coefficient

    # Return the obtained coefficient
    return coefficient
    
cpdef DTYPE_t kendall_distance_calculator(SIZE_t[:, :] Y_true,
                                          SIZE_t[:, :] Y_pred,
                                          DTYPE_t[:]   sample_weight,
                                          BOOL_t       normalize) nogil:
    """
        Calculator for the Kendall distance.
    """
    # Initialize some values from the input arrays
    cdef SIZE_t n_samples = Y_true.shape[0]
    cdef SIZE_t n_classes = Y_true.shape[1]

    # Initialize and define some variables to be employed
    cdef DTYPE_t penalization   = 0.5 
    cdef DTYPE_t distance       = 0.0
    cdef DTYPE_t local_distance

    # Define the indexes
    cdef SIZE_t sample
    cdef SIZE_t f_class
    cdef SIZE_t s_class

    # Normalize the sample weight
    normalize_sample_weight(sample_weight)

    # Iterate all the samples to obtain the distance
    for sample in range(n_samples):
        # Reinitialize the local distance to zero
        local_distance = 0.0
        # Iterate pair of classes
        for f_class in range(n_classes):
            for s_class in range(f_class + 1, n_classes):
                # Disagreement
                if (Y_true[sample, f_class] < Y_true[sample, s_class] and Y_pred[sample, f_class] > Y_pred[sample, s_class] or # X < Y in one and X > Y in the other
                    Y_true[sample, f_class] > Y_true[sample, s_class] and Y_pred[sample, f_class] < Y_pred[sample, s_class]):  # X > Y in one and X < Y in the other
                    local_distance += 1
                # Tied in one but different bucket in the other
                elif (Y_true[sample, f_class] == Y_true[sample, s_class] and Y_pred[sample, f_class] <  Y_pred[sample, s_class] or # X == Y in one and X < Y  in the other
                      Y_true[sample, f_class] == Y_true[sample, s_class] and Y_pred[sample, f_class] >  Y_pred[sample, s_class] or # X == Y in one and X > Y  in the other
                      Y_true[sample, f_class]  < Y_true[sample, s_class] and Y_pred[sample, f_class] == Y_pred[sample, s_class] or # X < Y  in one and X == Y in the other
                      Y_true[sample, f_class]  > Y_true[sample, s_class] and Y_pred[sample, f_class] == Y_pred[sample, s_class]):  # X > Y  in one and X == Y in the other
                    local_distance += penalization
        # Obtain the global distance weighting the local one accordingly
        if normalize:
            distance += sample_weight[sample] * (local_distance / ((n_classes * (n_classes - 1)) / 2))
        else:
            distance += sample_weight[sample] * local_distance

    # Return the obtained distance
    return distance

cpdef DTYPE_t tau_x_score_calculator(SIZE_t[:, :] Y_true,
                                     SIZE_t[:, :] Y_pred,
                                     DTYPE_t[:]   sample_weight) nogil:
    """
        Calculator for the tau x score.
    """
    # Initialize some values from the input arrays
    cdef SIZE_t n_samples = Y_true.shape[0]
    cdef SIZE_t n_classes = Y_true.shape[1]

    # Initialize and define some variables to be employed
    cdef DTYPE_t coefficient       = 0.0
    cdef DTYPE_t local_coefficient

    # Define the indexes
    cdef SIZE_t sample
    cdef SIZE_t f_class
    cdef SIZE_t s_class

    # Normalize the sample weight
    normalize_sample_weight(sample_weight)

    # Iterate all the samples to obtain the coeffiecient
    for sample in range(n_samples):
        # Reinitialize the local coefficient to zero
        local_coefficient = 0.0
        # Iterate pair of classes
        for f_class in range(n_classes):
            for s_class in range(n_classes):
                # Same label
                if f_class == s_class:
                    pass
                # Agreeement
                elif (Y_true[sample, f_class]  < Y_true[sample, s_class] and Y_pred[sample, f_class]  < Y_pred[sample, s_class] or # X < Y in both rankings
                      Y_true[sample, f_class]  < Y_true[sample, s_class] and Y_pred[sample, f_class] == Y_pred[sample, s_class] or # X < Y in the first and X = Y in the other
                      Y_true[sample, f_class] == Y_true[sample, s_class] and Y_pred[sample, f_class]  < Y_pred[sample, s_class] or # X = Y in the first and X < Y in the other
                      Y_true[sample, f_class] == Y_true[sample, s_class] and Y_pred[sample, f_class] == Y_pred[sample, s_class] or # X = Y in both rankings
                      Y_true[sample, f_class]  > Y_true[sample, s_class] and Y_pred[sample, f_class]  > Y_pred[sample, s_class]):  # X > Y in both rankings
                    local_coefficient += 1
                # Disagreement
                else:
                    local_coefficient -= 1
        # Obtain the global coefficient, weighting the local one accordingly
        coefficient += sample_weight[sample] * (local_coefficient / (n_classes * (n_classes - 1)))

    # Return the obtained coefficient
    return coefficient

cdef void normalize_sample_weight(DTYPE_t[:] sample_weight) nogil:
    """
        Normalize the sample weight to make them sum 1.
    """
    # Initialize some values from the input arrays
    cdef SIZE_t n_samples = sample_weight.shape[0]

    # Initialize some values to be employed
    cdef DTYPE_t sample_weight_sum = 0.0

    # Define the indexes
    cdef SIZE_t sample

    # Normalize the weights making them sum one
    for sample in range(n_samples):
        sample_weight_sum += sample_weight[sample]
    for sample in range(n_samples):
        sample_weight[sample] = (sample_weight[sample] / sample_weight_sum)
