# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False


# =============================================================================
# Imports
# =============================================================================

# Third party
from libc.math cimport exp, fabs, fmin, log
import numpy as np
cimport numpy as np

# Always include this statement after cimporting
# NumPy, to avoid segmentation faults
np.import_array()

# Local application
from ..metrics._label_ranking_fast cimport _kendall_distance_fast


# =============================================================================
# Constants
# =============================================================================

# The number of iterations, the minimum theta, the maximum
# theta and a convergence value for the Mallows criterion
cdef INT64_t N_ITERS = 10
cdef DTYPE_t LOWER_THETA = -20.0
cdef DTYPE_t UPPER_THETA = 10.0
cdef DTYPE_t EPSILON = 1e-5

# Distance measures that can be
# employed for the Mallows criterion
DISTANCES = {
    "kendall": 0
}


# =============================================================================
# Classes
# =============================================================================

# =============================================================================
# Criterion
# =============================================================================
cdef class Criterion:
    """Interface for impurity criteria.

    This object stores methods on how to calculate how good a split is using
    different metrics."""

    def __cinit__(self, RankAggregationAlgorithm rank_algorithm,
                  DISTANCE distance=KENDALL):
        """Constructor."""
        # Initialize the hyperparameters
        self.rank_algorithm = rank_algorithm
        self.distance = distance

        # Initialize the rankings
        # and the sample weights
        self.Y = None
        self.sample_weight = None

        # Initialize the number of classes
        # and the number of children
        self.n_classes = 0
        self.n_children = 0

        # Initialize the sample indexes and the
        # position of the samples in the children
        self.samples = None
        self.pos = None

        # Initialize the number of samples in total and
        # the number of samples in the node and
        # the number of samples in the children
        self.n_samples = 0
        self.n_node_samples = 0
        self.n_children_samples = None

        # Initialize the weighted number of samples in total,
        # the weighted number of samples in the node and
        # the weighted number of samples in the children
        self.weighted_n_samples = 0
        self.weighted_n_node_samples = 0
        self.weighted_n_children_samples = None

        # Initialize the impurity of the node
        # and the impurity in the children
        self.impurity_node = 0.0
        self.impurity_children = None

        # Initialize the consensus ranking in the node
        # the consensus ranking in the children
        self.consensus_node = None
        self.consensus_children = None

        # Initialize the rankings in the node, a copy of the
        # rankings in the node and the sample weights in the node
        self.Y_node = None
        self.Y_node_copy = None
        self.sample_weight_node = None

        # Initialize the count in the node
        # and the count in the children
        self.count_node = None
        self.count_children = None

        # Initialize the precedences matrix in the node
        # and the precedences matrix in the children
        self.precedences_matrix_node = None
        self.precedences_matrix_children = None

    cdef void init_from_rankings(self, INT64_t_2D Y, DTYPE_t_1D sample_weight,
                                 INT64_t max_children):
        """Initialize common attributes for the criteria
        from the rankings and the sample weight."""
        # Initialize some values from the input arrays
        cdef INT64_t n_samples = Y.shape[0]
        cdef INT64_t n_classes = Y.shape[1]

        # Define the indexes
        cdef SIZE_t sample

        # Initialize the Rank Aggregation algorithm. Even if
        # the required parameters will be directly passed to
        # the corresponding methods, for some of them the
        # inner data structures are needed by them to work
        self.rank_algorithm.init(n_classes)

        # Initialize the rankings and the sample weights
        # (not copy, directly the reference)
        self.Y = Y
        self.sample_weight = sample_weight

        # Initialize the number of classes
        # and the number of children
        self.n_classes = n_classes
        self.n_children = 1

        # Initialize the sample indexes and
        # the position of the samples in the child
        self.samples = np.zeros(n_samples, dtype=np.intp)
        self.pos = np.zeros(max_children + 1, dtype=np.intp)

        # Initialize the number of samples in total,
        # the number of samples in the node and
        # the number of samples in the children
        self.n_samples = n_samples
        self.n_node_samples = n_samples
        self.n_children_samples = np.zeros(max_children, dtype=np.int64)

        # Initialize the weighted number of samples in total,
        # the weighted number of samples in the node
        # and the weighted number of samples in the children
        self.weighted_n_samples = 0.0
        self.weighted_n_node_samples = 0.0
        self.weighted_n_children_samples = np.zeros(
            max_children, dtype=np.float64)

        # Initialize the impurity of the node
        # and the impurity in the children
        self.impurity_node = 0.0
        self.impurity_children = np.zeros(max_children, dtype=np.float64)

        # Initialize the consensus ranking in the node
        # and the consensus ranking in the children
        self.consensus_node = np.zeros(
            n_classes, dtype=np.int64)
        self.consensus_children = np.zeros(
            (max_children, n_classes), dtype=np.int64)

        # Compute the sample indexes and the
        # weighted number of samples in total
        for sample in range(n_samples):
            self.samples[sample] = sample
            self.weighted_n_samples += sample_weight[sample]
            self.weighted_n_node_samples += sample_weight[sample]

        # Compute the parameters to obtain the consensus ranking
        # using the underlying RankAggregationAlgorithm object.
        # This computation is automatically handled by the object
        # that will automatically determine the parameters to build
        self.rank_algorithm.build_params(
            Y, sample_weight, self.count_node, self.precedences_matrix_node)

        # Compute the impurity of the node
        # using the corresponding criterion
        self.node_impurity()

    cdef void init_from_record(self, INT64_t n_samples,
                               DTYPE_t weighted_n_samples, DTYPE_t impurity,
                               INT64_t_1D consensus, DTYPE_t_1D count,
                               DTYPE_t_3D precedences_matrix):
        """Initialize common attributes of the
        criteria from the information of a record."""
        # Initialize the number of samples and the
        # weighted number of samples in the node
        self.n_node_samples = n_samples
        self.weighted_n_node_samples = weighted_n_samples

        # Initialize the impurity and the
        # consensus ranking in the node
        self.impurity_node = impurity
        self.consensus_node[:] = consensus

    cdef void reset(self, SIZE_t_1D samples) nogil:
        """Reset the criterion to the initial position."""
        # Reset the number of children to one,
        # since the default behaviour is
        # that the node is not yet split
        self.n_children = 1

        # Reset the sample indexes (copy the reference
        # and not the contents) and the position of
        # the samples in the children (setting the
        # position of the left child to zero and the
        # position of the right child to the number
        # of samples in this node)
        self.samples = samples
        self.pos[:] = 0
        self.pos[1] = self.n_node_samples

        # Reset the number of samples in the children
        # (setting the number of samples in the left child
        # to zero and the number of samples in the right
        # child to the number of samples in this node)
        self.n_children_samples[:] = 0
        self.n_children_samples[0] = self.n_node_samples

        # Reset the weighted number of samples in the children
        # (setting the weighted number of samples in the left
        # child to zero and the weighted number of samples in
        # the right child to the weighted number of samples
        # in this node)
        self.weighted_n_children_samples[:] = 0.0
        self.weighted_n_children_samples[0] = self.weighted_n_node_samples

        # Reset the impurity in the children
        # (setting the impurity in the left
        # child to zero and the impurity in the
        # right child to the impurity in this node)
        self.impurity_children[:] = 0.0
        self.impurity_children[0] = self.impurity_node

        # Reset the consensus ranking in the children
        # (setting the consensus ranking in this node
        # to the all zeros and the consensus ranking
        # in the right child to the consensus ranking
        # in this node)
        self.consensus_children[:] = 0
        self.consensus_children[0, :] = self.consensus_node

    cdef void update(self, SIZE_t new_pos) nogil:
        """Update statistics by moving
        samples[pos:new_pos] to the left child."""
        # Define some values to be employed
        cdef INT64_t_1D y
        cdef DTYPE_t weight
        cdef DTYPE_t_1D count_left
        cdef DTYPE_t_1D count_right
        cdef DTYPE_t_3D precedences_matrix_left
        cdef DTYPE_t_3D precedences_matrix_right

        # Define the indexes
        cdef SIZE_t pos
        cdef SIZE_t left_child
        cdef SIZE_t right_child
        cdef SIZE_t sample
        cdef SIZE_t sample_idx

        # Initialize the index of the left
        # child and the index of the right child
        left_child = self.n_children - 2
        right_child = self.n_children - 1

        # Initialize the position of
        # the last evaluated sample
        pos = self.pos[self.n_children - 1]

        # Initialize the count and the precedences matrix for the left
        # and right child to None and hereafter they will be properly set
        count_left = None
        count_right = None
        precedences_matrix_left = None
        precedences_matrix_right = None

        # Set either the count or the precedences matrix
        # for the left child and the right child by
        # properly managing the None values, that are
        # an indicator of which one is necessary
        if self.count_children is not None:
            count_left = (
                self.count_children[left_child])
            count_right = (
                self.count_children[right_child])
        else:
            precedences_matrix_left = (
                self.precedences_matrix_children[left_child])
            precedences_matrix_right = (
                self.precedences_matrix_children[right_child])

        # Update the statistics for the left child and the right child.
        # In fact, the statistics are directly updated using the
        # RankAggregationalgorithm, which automatically determines
        # the statistics to be updated
        for sample in range(pos, new_pos):
            # Initialize the identifier of the
            # sample, the ranking and its weight
            sample_idx = self.samples[sample]
            y = self.Y[sample_idx]
            weight = self.sample_weight[sample_idx]
            # Update the weighted number of
            # samples in the left and right child
            # (positive weight to the left child
            # and negative weight to the right child)
            self.weighted_n_children_samples[left_child] += weight
            self.weighted_n_children_samples[right_child] -= weight
            # Update the count or the precedences matrix
            # for the left and right child (positively
            # weighting the ranking for the left child
            # and negatively weighting the ranking for
            # the right child)
            self.rank_algorithm.update_params(
                y, weight, count_left, precedences_matrix_left)
            self.rank_algorithm.update_params(
                y, -weight, count_right, precedences_matrix_right)

        # Update the position of
        # the samples in the children
        self.pos[self.n_children - 1] = new_pos

        # Update the number of samples in the children
        # (moving the number of positions from
        # the right child to the left child)
        self.n_children_samples[self.n_children - 2] += (new_pos - pos)
        self.n_children_samples[self.n_children - 1] -= (new_pos - pos)

    cdef void node_children(self) nogil:
        """Add a new children to the node."""
        # Define the indexes
        cdef SIZE_t left_child
        cdef SIZE_t right_child

        # Increase the number of children
        self.n_children += 1

        # Initialize the index of the left child
        # and the index of the right child
        left_child = self.n_children - 2
        right_child = self.n_children - 1

        # Move the position of the samples in the children
        # (setting the position of the samples in the new right child
        # to the position of the samples in the previous right child
        # and the same for the new and previous left children)
        self.pos[right_child + 1] = self.pos[right_child]
        self.pos[right_child] = self.pos[left_child]

        # Move the number of samples in the children
        # (setting the number of the samples in the
        # new right child to the number of samples in
        # the previous left child and the number of
        # samples in the new left child to zero)
        self.n_children_samples[right_child] = (
            self.n_children_samples[left_child])
        self.n_children_samples[left_child] = 0

        # Move the weighted number of samples in the children
        # (setting the weighted number of samples in the
        # new right child to the weighted number of samples in
        # the previous left child and the weighted number of
        # samples in the new left child to zero)
        self.weighted_n_children_samples[right_child] = (
            self.weighted_n_children_samples[left_child])
        self.weighted_n_children_samples[left_child] = 0.0

        # Move the impurity in the children
        # (setting the impurity in the new right
        # child to the impurity of the previous left
        # child and the impurity of the new left
        # child to zero)
        self.impurity_children[right_child] = (
            self.impurity_children[left_child])
        self.impurity_children[left_child] = 0.0

        # Move the consensus in the children
        # (setting the consensus ranking in the
        # new right child to the consensus ranking
        # of the previous left child and the consensus
        # ranking of the new left child to all zero)
        self.consensus_children[right_child, :] = (
            self.consensus_children[left_child])
        self.consensus_children[left_child, :] = 0

    cdef void node_impurity(self) nogil:
        """Placeholder for calculating the impurity of the node.

        Placeholder for a method which will evaluate the impurity of the
        current node, i.e. the impurity of samples. This is the primary
        function of the criterion class."""

    cdef void children_impurity(self) nogil:
        """Placeholder for calculating the impurity of children.

        Placeholder for a method which evaluates the impurity in children
        nodes, i.e. the impurity of samples indexed by the the start
        and end positions of each child."""

    cdef DTYPE_t impurity_improvement(self) nogil:
        """Compute the improvement in impurity.

        This method computes the improvement
        in impurity when a split occurs."""
        # Define some values to be employed
        cdef DTYPE_t impurity_improvement

        # Define the indexes
        cdef SIZE_t child

        # Initialize the improvement in impurity to
        # the maximum value, that is, the impurity
        # before the split (i.e., the impurity in
        # the node)
        impurity_improvement = self.impurity_node

        # Compute the impurity of the children
        self.children_impurity()

        # Compute the improvement in impurity,
        # that is, the impurity after splitting
        # the node in these positions
        for child in range(self.n_children):
            impurity_improvement -= (self.weighted_n_children_samples[child]
                                     / self.weighted_n_node_samples
                                     * self.impurity_children[child])
        impurity_improvement *= (self.weighted_n_node_samples
                                 / self.weighted_n_samples)

        # Return the computed
        # improvement in impurity
        return impurity_improvement


# =============================================================================
# Label Ranking criterion
# =============================================================================
cdef class LabelRankingCriterion(Criterion):
    """Abstract criterion for Label Ranking."""

    cdef void init_from_rankings(self, INT64_t_2D Y, DTYPE_t_1D sample_weight,
                                 INT64_t max_children):
        """Initialize the attributes for this criterion
        from the rankings and the sample weight."""
        # Initialize some values from the input arrays
        cdef INT64_t n_classes = Y.shape[1]

        # For Label Ranking criteria, also initialize the count in
        # the node and the count in the children. By this way, it
        # is avoided that several precedences matrices are also
        # initialized when they are not necessary
        self.count_node = np.zeros(
            n_classes, dtype=np.float64)
        self.count_children = np.zeros(
            (max_children, n_classes), dtype=np.float64)

        # Call to the method of the parent
        # to initialize the rest of attributes
        Criterion.init_from_rankings(self, Y, sample_weight, max_children)

    cdef void init_from_record(self, INT64_t n_samples,
                               DTYPE_t weighted_n_samples, DTYPE_t impurity,
                               INT64_t_1D consensus, DTYPE_t_1D count,
                               DTYPE_t_3D precedences_matrix):
        """Initialize the attributes of the
        criterion from the information of a record."""
        # Call to the method of the parent
        # to initialize the common attributes
        Criterion.init_from_record(
            self, n_samples, weighted_n_samples,
            impurity, consensus,
            count, precedences_matrix)

        # Initialize the count in the node
        self.count_node[:] = count

    cdef void reset(self, SIZE_t_1D samples) nogil:
        """Reset the criterion to the initial position."""
        # Call to the method of the parent
        # to reset common attributes
        Criterion.reset(self, samples)

        # Reset the count in the children (setting the
        # count of the first child to the count in the node)
        self.count_children[:] = 0.0
        self.count_children[0, :] = self.count_node

    cdef void node_children(self) nogil:
        """Add a new children to the node."""
        # Call to the method of the parent
        # to move the common attributes
        Criterion.node_children(self)

        # Define the indexes
        cdef SIZE_t left_child
        cdef SIZE_t right_child

        # Initialize the left child and the right child
        left_child = self.n_children - 2
        right_child = self.n_children - 1

        # Move the count in the children (setting the
        # count in the new right child to the count
        # in the previous left child and the count
        # in the new left child to zero)
        self.count_children[right_child, :] = (
            self.count_children[left_child])
        self.count_children[left_child, :] = 0.0


# =============================================================================
# Mallows criterion
# =============================================================================
cdef class Mallows(LabelRankingCriterion):
    """Mallows impurity criterion."""

    cdef void init_from_rankings(self, INT64_t_2D Y, DTYPE_t_1D sample_weight,
                                 INT64_t max_children):
        """Initialize the attributes for this criterion
        from the rankings and the sample weight."""
        # Initialize some values from the input arrays
        cdef INT64_t n_samples = Y.shape[0]
        cdef INT64_t n_classes = Y.shape[1]

        # Initialize the rankings in the node,
        # a copy of the rankings in the node
        # and the sample weights in the node
        self.Y_node = np.zeros(
            (n_samples, n_classes), dtype=np.int64)
        self.Y_node_copy = np.zeros(
            (n_samples, n_classes), dtype=np.int64)
        self.sample_weight_node = np.zeros(
            n_samples, dtype=np.float64)

        # Copy the rankings and the samples
        # weights to the previous views
        self.Y_node[:] = Y
        self.Y_node_copy[:] = Y
        self.sample_weight_node[:] = sample_weight

        # Call to the method of the parent
        # to initialize the common attributes
        LabelRankingCriterion.init_from_rankings(
            self, Y, sample_weight, max_children)

    cdef void reset(self, SIZE_t_1D samples) nogil:
        """Reset the Mallows criterion."""
        # Call to the method of the parent
        # to initialize the common attributes
        LabelRankingCriterion.reset(self, samples)

        # Define the indexes
        cdef SIZE_t sample
        cdef SIZE_t sample_idx

        # For efficiency purposes, once the node is reset copy,
        # in sorted order, the rankings and the sample weight
        for sample in range(self.n_node_samples):
            # Initialize the identifier of the sample
            sample_idx = self.samples[sample]
            # Copy in sorted order the rankings in the node,
            # the copy of the rankings in the node
            # and the sample weight in the node
            self.Y_node[sample, :] = self.Y[sample_idx]
            self.Y_node_copy[sample, :] = self.Y[sample_idx]
            self.sample_weight_node[sample] = self.sample_weight[sample_idx]

    cdef DTYPE_t function(self, DTYPE_t theta, DTYPE_t distance) nogil:
        """Function for the Netwon-Raphson method."""
        # Define some values to be employed
        cdef DTYPE_t new_theta

        # Define the indexes
        cdef SIZE_t label

        # Initialize the new theta to zero
        # (since it is the default value to compute it)
        new_theta = 0.0

        # Compute the new theta according to
        # the distance function to be employed
        # and taking into account the proper function
        for label in range(self.n_classes, 1, -1):
            # Kendall
            if self.distance == KENDALL:
                new_theta += ((label * exp(-theta*label))
                              / (1 - exp(-theta*label)))

        # Compute the last step of the function where
        # the sign of the new theta value is decided

        # Kendall
        if self.distance == KENDALL:
            new_theta = ((((self.n_classes-1) / (exp(theta)-1))
                         - distance) - new_theta)

        # Return the new theta
        return new_theta

    cdef DTYPE_t derivative_function(self, DTYPE_t theta) nogil:
        """Derivative of the function for the Netwon-Raphson method."""
        # Define some values to be employed
        cdef DTYPE_t new_theta

        # Define the indexes
        cdef SIZE_t label

        # Initialize the new theta to zero
        # (since it is the default value to compute it)
        new_theta = 0.0

        # Compute the new theta according to
        # the distance function to be employed
        # and taking into account the proper
        # derivative of the function
        for label in range(self.n_classes, 1, -1):
            # Kendall
            if self.distance == KENDALL:
                new_theta += ((label**2 * exp(-theta*label))
                              / (1 - exp(-theta*label))**2)

        # Compute the last step of the derivative of the function
        # where the sign of the new theta value is decided

        # Kendall
        if self.distance == KENDALL:
            new_theta = ((((-self.n_classes+1) * exp(theta))
                         / (exp(theta)-1)**2) + new_theta)

        # Return the new theta
        return new_theta

    cdef DTYPE_t newton_raphson(self, DTYPE_t distance) nogil:
        """Execute the Netwon-Raphson method to compute the impurity."""
        # Define some values to be employed
        cdef DTYPE_t x1, x2
        cdef DTYPE_t dx, dxold
        cdef DTYPE_t temp, xl, xh
        cdef DTYPE_t f, df, fl, fh
        cdef BOOL_t out, decreasing
        cdef DTYPE_t theta

        # Define the indexes
        cdef SIZE_t label
        cdef SIZE_t iteration

        # Initialize the lower and upper values
        # for the theta parameter (impurity)
        x1 = LOWER_THETA
        x2 = UPPER_THETA

        # Initialize the lower and upper values
        # for the function and its derivative
        fl = self.function(x1, distance)
        fh = self.derivative_function(x2)

        # Some initial checks, to maintain the theta
        # parameter in the correspoding range
        if fl == 0.0:
            theta = x1
        elif fh == 0.0:
            theta = x2
        else:
            if fl < 0.0:
                xl = x1
                xh = x2
            else:
                xh = x1
                xl = x2
            # Initialize some needed values to
            # apply the Newton-Raphson method
            theta = x1
            dxold = fabs(x2 - x1)
            dx = dxold
            f = self.function(theta, distance)
            df = self.derivative_function(theta)

            # Execute the specified number of iterations for the
            # Newton-Raphson method (although an early break
            # will done the job when convergence)
            for iteration in range(N_ITERS):
                # Check whether theta is out of range
                # and not decreasing fast enough
                out = ((((theta-xh) * df - f)
                       * ((theta-xl) * df - f)) > 0.0)
                decreasing = (fabs(2.0*f) > fabs(dxold*df))
                # If the theta parameter is out of range or theta
                # parameter not decreasing fast enough, bisect
                if out or decreasing:
                    dxold = dx
                    dx = 0.5 * (xh-xl)
                    theta = xl + dx
                    if xl == theta:
                        break
                # Otherwise, update the theta parameter
                # according to the Newton-Raphson method
                else:
                    dxold = dx
                    dx = f / df
                    temp = theta
                    theta -= dx
                    if temp == theta:
                        break
                # Change is negligible (convergence
                # has been found), break
                if fabs(dx) < EPSILON:
                    break
                # Update theta parameters for the function
                # and the derivative of the function
                f = self.function(theta, distance)
                df = self.derivative_function(theta)
                # If the bracket is not in the root,
                # update the lower and upper theta values
                if f < 0.0:
                    xl = theta
                else:
                    xh = theta

        # Return the theta parameter as impurity
        return theta

    cdef DTYPE_t impurity(self, INT64_t_2D Y, DTYPE_t_1D sample_weight,
                          DTYPE_t weighted_n_samples, DTYPE_t_1D count,
                          INT64_t_1D consensus) nogil:
        """Compute the impurity using the Mallows criterion."""
        # Initialize some values from the input arrays
        cdef INT64_t n_samples = Y.shape[0]

        # Define some values to be employed
        cdef DTYPE_t distance
        cdef DTYPE_t impurity

        # Define the indexes
        cdef SIZE_t sample

        # Initialize the distance between the
        # rankings and the consensus ranking to zero
        distance = 0.0

        # Aggregate the rankings using the MLE to obtain the
        # completed rankings and also its corresponding consensus
        self.rank_algorithm.aggregate_mle(
            Y, sample_weight, consensus, count)

        # Compute the distance between the completed
        # rankings and its corresponding consensus
        for sample in range(n_samples):
            # Kendall
            if self.distance == KENDALL:
                distance += ((sample_weight[sample] / weighted_n_samples)
                             * _kendall_distance_fast(
                                 y_true=Y[sample],
                                 y_pred=consensus,
                                 normalize=False))

        # Compute the impurity using the Netwon-Raphson method
        # (inverse of theta, the highest value, the lowest uncertainty)
        impurity = 1 / self.newton_raphson(distance)

        # Return the impurity
        return impurity

    cdef void node_impurity(self) nogil:
        """Compute the impurity of the node with the Mallows criterion."""
        # To compute the impurity in the node, it is
        # necessary to compute the consensus ranking
        self.rank_algorithm.aggregate_params(
            self.consensus_node, self.count_node, self.precedences_matrix_node)

        # Then, compute the impurity in the node
        self.impurity_node = self.impurity(
            self.Y_node, self.sample_weight_node,
            self.weighted_n_node_samples, self.count_node,
            self.consensus_node)

        # Copy the rankings (since, due to the MLE
        # process, they may have been modified)
        self.Y_node[:] = self.Y_node_copy

    cdef void children_impurity(self) nogil:
        """Compute the impurity of the children with the Mallows criterion."""
        # Define the indexes
        cdef SIZE_t start
        cdef SIZE_t end
        cdef SIZE_t child

        # Compute the impurity in the children by
        # computing the theta parameter on each child
        for child in range(self.n_children):
            # Initialize the start and end position
            # of the samples in this child
            start = self.pos[child]
            end = self.pos[child + 1]
            # Compute the consensus ranking in this
            # child before computing the impurity
            self.rank_algorithm.aggregate_params(
                self.consensus_children[child],
                self.count_children[child],
                self.precedences_matrix_children[child])
            # Then, compute the impurity in this child
            self.impurity_children[child] = self.impurity(
                self.Y_node[start:end], self.sample_weight_node[start:end],
                self.weighted_n_children_samples[child],
                self.count_children[child],
                self.consensus_children[child])

        # Copy the rankings (since, due to the MLE
        # process, they may have been modified)
        self.Y_node[:] = self.Y_node_copy


# =============================================================================
# Partial Label Ranking criterion
# =============================================================================
cdef class PartialLabelRankingCriterion(Criterion):
    """Abstract criterion for Partial Label Ranking."""

    cdef void init_from_rankings(self, INT64_t_2D Y, DTYPE_t_1D sample_weight,
                                 INT64_t max_children):
        """Initialize the attributes for this criterion
        from the rankings and the sample weight."""
        # Initialize some values from the input arrays
        cdef INT64_t n_classes = Y.shape[1]

        # Initialize the precedences matrix in the node
        # and the precedences matrix in the children
        self.precedences_matrix_node = np.zeros(
            (n_classes, n_classes, 2),
            dtype=np.float64)
        self.precedences_matrix_children = np.zeros(
            (max_children, n_classes, n_classes, 2),
            dtype=np.float64)

        # Call to the method of the parent
        # to initialize the common attributes
        Criterion.init_from_rankings(self, Y, sample_weight, max_children)

    cdef void init_from_record(self, INT64_t n_samples,
                               DTYPE_t weighted_n_samples,
                               DTYPE_t impurity, INT64_t_1D consensus,
                               DTYPE_t_1D count,
                               DTYPE_t_3D precedences_matrix):
        """Initialize the attributes of the
        criterion from the information of a record."""
        # Call to the method of the parent
        # to initialize the common attributes
        Criterion.init_from_record(
            self, n_samples, weighted_n_samples,
            impurity, consensus,
            count, precedences_matrix)

        # Initialize the precedences matrix in the node
        self.precedences_matrix_node[:] = precedences_matrix

    cdef void reset(self, SIZE_t_1D samples) nogil:
        """Reset the criterion to the initial position."""
        # Call to the method of the parent
        # to reset the common attributes
        Criterion.reset(self, samples)

        # Reset the precedences matrix in the children
        # (setting the precedences matrix of the first
        # child to the precedences matrix in the node)
        self.precedences_matrix_children[:] = 0.0
        self.precedences_matrix_children[0, :] = self.precedences_matrix_node

    cdef void node_children(self) nogil:
        """Add a new children to the node."""
        # Call to the method of the parent
        # to move the common attributes
        Criterion.node_children(self)

        # Define the indexes
        cdef SIZE_t left_child
        cdef SIZE_t right_child

        # Initialize the index of the left child
        # and the index of the right child
        left_child = self.n_children - 2
        right_child = self.n_children - 1

        # Move the precedences matrix in the children
        # (setting the precedences matrix in the new
        # right child to the precedences matrix of the
        # previous left child and the precedences matrix
        # of the new left child to zero)
        self.precedences_matrix_children[right_child, :] = (
            self.precedences_matrix_children[left_child])
        self.precedences_matrix_children[left_child, :] = 0.0

    cdef DTYPE_t impurity(self, DTYPE_t_3D precedences_matrix,
                          INT64_t_1D consensus) nogil:
        """Placeholder for a method that will compute the impurity."""

    cdef void node_impurity(self) nogil:
        """Compute the impurity of the node
        with Partial Label Ranking criteria."""
        # To compute the impurity in the node, it is
        # necessary to compute the consensus ranking
        self.rank_algorithm.aggregate_params(
            self.consensus_node, self.count_node, self.precedences_matrix_node)

        # Then, compute the impurity in the node
        self.impurity_node = self.impurity(
            self.precedences_matrix_node, self.consensus_node)

    cdef void children_impurity(self) nogil:
        """Compute the impurity of the children
        with Partial Label Ranking criteria."""
        # Define the indexes
        cdef SIZE_t start
        cdef SIZE_t end
        cdef SIZE_t child

        # Compute the impurity in the children
        for child in range(self.n_children):
            # Compute the consensus ranking in this
            # child before computing the impurity
            self.rank_algorithm.aggregate_params(
                self.consensus_children[child],
                self.count_children[child],
                self.precedences_matrix_children[child])
            # Then, compute the impurity in this child
            self.impurity_children[child] = self.impurity(
                self.precedences_matrix_children[child],
                self.consensus_children[child])


# =============================================================================
# Disagreements criterion
# =============================================================================
cdef class Disagreements(PartialLabelRankingCriterion):
    """Disagreements impurity criterion."""

    cdef DTYPE_t impurity(self, DTYPE_t_3D precedences_matrix,
                          INT64_t_1D consensus) nogil:
        """Compute the impurity using the disagreements criterion."""
        # Define some values to be employed
        cdef DTYPE_t disagreements
        cdef INT64_t sum_total
        cdef DTYPE_t f_precedences
        cdef DTYPE_t ties
        cdef DTYPE_t s_precedences
        cdef DTYPE_t total_precedences

        # Define the indexes
        cdef SIZE_t f_class
        cdef SIZE_t s_class

        # Initialize the disagreements impurity to zero
        disagreements = 0.0

        # Initialize the total sum of pairs to zero
        sum_total = 0

        # Compute the impurity counting the frequency
        # of disagreements between the rankings
        # in this node and the consensus ranking
        for f_class in range(self.n_classes - 1):
            for s_class in range(f_class + 1, self.n_classes):
                # Extract the number of times that "f_class"
                # precedes "s_class", the number of times that
                # "f_class" is tied with "s_class" and the
                # number of times that "s_class" precedes
                # "f_class". Also, obtain the sum of these values
                f_precedences = precedences_matrix[f_class, s_class, 0]
                ties = precedences_matrix[f_class, s_class, 1]
                s_precedences = precedences_matrix[s_class, f_class, 0]
                total_precedences = f_precedences + ties + s_precedences
                # Compute the weighted disagreements, taking into account
                # that if "f_class" precedes "s_class" in the consensus
                # ranking, then, the number of disagreements is given
                # by the number of times that "f_class" is tied with
                # "s_class" plus the number of times that "s_class"
                # precedes "f_class". The same reasoning applies
                # for the rest of possibilities
                if consensus[f_class] < consensus[s_class]:
                    disagreements += ((ties + s_precedences)
                                      / total_precedences)
                elif consensus[f_class] == consensus[s_class]:
                    disagreements += ((f_precedences+s_precedences)
                                      / total_precedences)
                else:
                    disagreements += ((f_precedences + ties)
                                      / total_precedences)
                # Increase the total sum of pairs to
                # properly reduce the disagreements
                # by the number of possible
                # combinations between the classes
                sum_total += 1
        disagreements /= sum_total

        # Return the disagreements impurity
        return disagreements


# =============================================================================
# Distance criterion
# =============================================================================
cdef class Distance(PartialLabelRankingCriterion):
    """Distance impurity criterion."""

    cdef DTYPE_t impurity(self, DTYPE_t_3D precedences_matrix,
                          INT64_t_1D consensus) nogil:
        """Compute the impurity using the distance criterion."""
        # Define some values to be employed
        cdef DTYPE_t distance
        cdef DTYPE_t f_precedences
        cdef DTYPE_t ties
        cdef DTYPE_t s_precedences
        cdef DTYPE_t total_precedences

        # Define the indexes
        cdef SIZE_t f_class
        cdef SIZE_t s_class

        # Initialize the distance impurity to zero
        distance = 0.0

        # Compute the impurity counting the distance between the
        # pair order matrix associated with the consensus ranking
        # and the pair order matrices associated with the rankings
        # in this node. In fact, it is not necessary to iterate all
        # these rankings, since this information can be extracted
        # from the precedences matrix
        for f_class in range(self.n_classes - 1):
            for s_class in range(f_class + 1, self.n_classes):
                # Extract the number of times that "f_class"
                # precedes "s_class", the number of times that
                # "f_class" is tied with "s_class" and the
                # number of times that "s_class" precedes
                # "f_class". Also, obtain the sum of these values
                f_precedences = precedences_matrix[f_class, s_class, 0]
                ties = precedences_matrix[f_class, s_class, 1]
                s_precedences = precedences_matrix[s_class, f_class, 0]
                total_precedences = f_precedences + ties + s_precedences
                # Compute the weighted distance, taking into account that
                # if "f_class" precedes "s_class", then, the distance
                # between the corresponding pair order matrices is
                # the number of times that "f_class" is tied with
                # "s_class" plus two times the number of times that
                # "s_class" precedes "f_class". The same applies
                # for the rest of possibilities
                if consensus[f_class] < consensus[s_class]:
                    distance += ((ties + 2 * s_precedences)
                                 / total_precedences)
                elif consensus[f_class] == consensus[s_class]:
                    distance += ((f_precedences + s_precedences)
                                 / total_precedences)
                else:
                    distance += ((2 * f_precedences + ties)
                                 / total_precedences)

        # Return the distance impurity
        return distance


# =============================================================================
# Entropy criterion
# =============================================================================
cdef class Entropy(PartialLabelRankingCriterion):
    """Entropy impurity criterion."""

    cdef DTYPE_t impurity(self, DTYPE_t_3D precedences_matrix,
                          INT64_t_1D consensus) nogil:
        """Compute the impurity using the entropy criterion."""
        # Define some values to be employed
        cdef DTYPE_t entropy
        cdef INT64_t sum_total
        cdef DTYPE_t total_precedences
        cdef DTYPE_t prob_f_precedes
        cdef DTYPE_t prob_tied
        cdef DTYPE_t prob_s_precedes

        # Define the indexes
        cdef SIZE_t f_class
        cdef SIZE_t s_class

        # Initialize the entropy impurity to zero
        entropy = 0.0

        # Initialize the total sum of pairs to zero
        sum_total = 0

        # Compute the impurity using the entropy
        # of the probability distribution defined
        # by the probability that a class precedes,
        # is tied and is behind another class.
        # This information is directly obtained
        # from the precedences matrix
        for f_class in range(self.n_classes - 1):
            for s_class in range(f_class + 1, self.n_classes):
                # Compute the probability that the
                # first class precedes, is tied
                # and is behind the second class
                total_precedences = (precedences_matrix[f_class, s_class, 0]
                                     + precedences_matrix[f_class, s_class, 1]
                                     + precedences_matrix[s_class, f_class, 0])
                prob_f_precedes = (precedences_matrix[f_class, s_class, 0]
                                   / total_precedences)
                prob_tied = (precedences_matrix[f_class, s_class, 1]
                             / total_precedences)
                prob_s_precedes = (precedences_matrix[s_class, f_class, 0]
                                   / total_precedences)
                # Compute the weighted entropy (avoiding zero
                # probabilities which causes errors with log)
                if prob_f_precedes >= EPSILON:
                    entropy -= prob_f_precedes * log2(prob_f_precedes)
                if prob_tied >= EPSILON:
                    entropy -= prob_tied * log2(prob_tied)
                if prob_s_precedes >= EPSILON:
                    entropy -= prob_s_precedes * log2(prob_s_precedes)
                # Increase the total sum of pairs to properly
                # reduce the entropy by the number of
                # possible combinations between classes
                sum_total += 1
        entropy /= sum_total

        # Return the entropy impurity
        return entropy


# =============================================================================
# Methods
# =============================================================================

cdef DTYPE_t log2(DTYPE_t x) nogil:
    """Compute the logarithm in base 2."""
    return log(x) / log(2.0)
