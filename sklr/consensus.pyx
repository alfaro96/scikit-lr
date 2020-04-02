# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False


# =============================================================================
# Imports
# =============================================================================

# Standard
from numbers import Integral, Real
from warnings import warn

# Third party
from libc.math cimport fabs
from libc.math cimport INFINITY
from libc.stdlib cimport free, malloc
import numpy as np
cimport numpy as np

# Always include this statement after cimporting NumPy,
# to avoid possible segmentation faults
np.import_array()

# Local application
from .utils._memory cimport copy_pointer_INT64_1D, copy_pointer_INT64_2D
from .utils._ranking cimport rank_data_pointer
from .utils._ranking cimport RANK_METHOD
from .utils.ranking import (
    check_label_ranking_targets, check_partial_label_ranking_targets,
    _transform_rankings)
from .utils.validation import (
    check_array, check_consistent_length, check_sample_weight)


# =============================================================================
# Constants
# =============================================================================

# Mapping to the Rank Aggregation algorithms
ALGORITHM_MAPPING = {
    "borda_count": BordaCountAlgorithm,
    "bpa_lia_mp2": BPALIAMP2Algorithm
}

# List of the available Rank Aggregation algorithms
ALGORITHM_LIST = sorted(list(ALGORITHM_MAPPING.keys()))

# Epsilon value to add a little bit of helpful noise
cdef DTYPE_t EPSILON = np.finfo(np.float64).eps


# =============================================================================
# Classes
# =============================================================================

# =============================================================================
# Rank Aggregation algorithm
# =============================================================================
cdef class RankAggregationAlgorithm:
    """Provides a uniform interface to fast Rank Aggregation algorithms.

    This class provides a uniform interface to fast Rank Aggregation
    algorithms. The various algorithms can be accessed via the
    :meth:`get_algorithm` class method and the algorithm string
    identifier (see below).

    For example, to use the Borda count algorithm:

    >>> import numpy as np
    >>> from sklr.consensus import RankAggregationAlgorithm
    >>> rank_algorithm = RankAggregationAlgorithm.get_algorithm("borda_count")
    >>> Y = np.array([[1, 2, 3], [2, 3, 1], [2, 3, 1]])
    >>> rank_algorithm.aggregate(Y)
    array([1, 3, 2])

    Available algorithms

    The following lists the string algorithm identifiers and the associated
    Rank Aggregation algorithm classes:

    =============  ===================  ====
    identifier     class name           args
    -------------  -------------------  ----
    "borda_count"  BordaCountAlgorithm  -
    "bpa_lia_mp2"  BPALIAMP2Algorithm   beta
    =============  ===================  ====

    References
    ----------
    .. [1] `J. Borda, Memoire sur les elections au scrutin.
            Histoire de l’Academie Royal des Sciences, 1770.`_

    .. [2] `P. Emerson, "The original Borda count and partial voting",
            Social Choice and Welfare, vol. 40, pp. 353-358, 2013.`_

    .. [3] `J. A. Aledo, J. A. Gámez, y D. Molina, "Using extension
            sets to aggregate partial rankings in a flexible setting",
            Applied Mathematics and Computation,
            vol. 290, pp. 208-223, 2016.`_

    .. [4] `A. Gionis, H. Mannila, K. Puolamäki, and A. Ukkonen,
            "Algorithms for discovering bucket orders from data",
            In Proceedings of the 12th ACM SIGKDD international
            conference on Knowledge discovery and data mining,
            2006, pp. 561-566.`_

    .. [5] `J. A. Aledo, J. A. Gámez, y A. Rosete, "Utopia in the
            solution of the Optimal Bucket Order Problem",
            Decision Support Systems, vol. 97, pp. 69-80, 2017.`_
    """

    def __cinit__(self):
        """Constructor."""
        # Initialize the hyperparameters
        self.beta = 0.25

        # Initialize the attributes
        self.count_builder = CountBuilder()
        self.pair_order_matrix_builder = PairOrderMatrixBuilder()
        self.utopian_matrix_builder = UtopianMatrixBuilder()
        self.dani_indexes_builder = DANIIndexesBuilder()

    def __init__(self):
        """Constructor."""
        # Avoid that this abstract class is instantiated
        if self.__class__ is RankAggregationAlgorithm:
            raise NotImplementedError("RankAggregationAlgorithm "
                                      "is an abstract class.")

    @classmethod
    def get_algorithm(cls, algorithm, **kwargs):
        """Get the Rank Aggregation algorithm from the string identifier.

        See the docstring of RankAggregationAlgorithm
        for a list of available algorithms.

        Parameters
        ----------
        algorithm : {"borda_count", "bpa_lia_mp2"}
            The algorithm to use.

        **kwargs : dict
            Additional arguments will be passed to the requested algorithm.

        Returns
        -------
        rank_algorithm : object
            The Rank Aggregation algorithm.

        Raises
        ------
        ValueError
            If the provided algorithm is not available.
        """
        # Map the algorithm string identifier to the Rank Aggregation class
        if (isinstance(algorithm, type) and
                issubclass(algorithm, RankAggregationAlgorithm)):
            pass
        else:
            try:
                rank_algorithm = ALGORITHM_MAPPING[algorithm]
            except KeyError:
                raise ValueError("Unrecognized algorithm: '{}'. "
                                 "Valid algorithms are: {}."
                                 .format(algorithm, ALGORITHM_LIST))

        # For the Bucket Pivot Algorithm with multiple pivots
        # and two-stages, the beta hyperparameter must be set
        # if it is not provided as keyworded argument
        if rank_algorithm is BPALIAMP2Algorithm:
            if "beta" not in kwargs:
                kwargs["beta"] = 0.25

        # Initialize the Rank Aggregation algorithm
        # (mapping the hyperparameters to their values)
        rank_algorithm = rank_algorithm(**kwargs)

        # Return it
        return rank_algorithm

    cdef void init(self, INT64_t n_classes):
        """Placeholder for a method that will
        initialize the Rank Aggregation algorithm."""

    cdef void reset(self, DTYPE_t_1D count=None,
                    DTYPE_t_3D precedences_matrix=None) nogil:
        """Placeholder for a method that will
        reset the Rank Aggregation algorithm."""

    cdef void update_params(self, INT64_t_1D y, DTYPE_t weight,
                            DTYPE_t_1D count=None,
                            DTYPE_t_3D precedences_matrix=None) nogil:
        """Placeholder for a method that will update the
        parameters for the Rank Aggregation algorithm."""

    cdef void build_params(self, INT64_t_2D Y, DTYPE_t_1D sample_weight,
                           DTYPE_t_1D count=None,
                           DTYPE_t_3D precedences_matrix=None) nogil:
        """Placeholder for a method that will build the
        parameters for the Rank Aggregation algorithm."""

    cdef void aggregate_params(self, INT64_t_1D consensus,
                               DTYPE_t_1D count=None,
                               DTYPE_t_3D precedences_matrix=None) nogil:
        """Placeholder for a method that will build the consensus ranking
        from the parameters of the Rank Aggregation algorithm."""

    cdef void aggregate_rankings(self, INT64_t_2D Y, DTYPE_t_1D sample_weight,
                                 INT64_t_1D consensus) nogil:
        """Aggregate the rankings to build the consensus ranking."""
        # First, build the parameters of the Rank Aggregation algorithm
        # (and so reuse some of the already implemented methods)
        self.build_params(Y, sample_weight)

        # Second, aggregate the parameters built on the
        # previous step to obtain the consensus ranking
        self.aggregate_params(consensus)

    cdef void complete_rankings(self, INT64_t_2D Y,
                                INT64_t_1D consensus) nogil:
        """Placeholder for a method that will complete
        the rankings using the Borda count algorithm."""

    cdef void aggregate_mle(self, INT64_t_2D Y, DTYPE_t_1D sample_weight,
                            INT64_t_1D consensus,
                            DTYPE_t_1D count=None,
                            BOOL_t replace=True) nogil:
        """Placeholder for a method that will build the
        consensus ranking with a MLE process using the
        Borda count algorithm. If specified, the rankings
        in Y will be replaced by the completed ones."""

    def check_targets(self, Y):
        """Check whether the rankings in Y can be
        aggregated with the Rank Aggregation algorithm."""

    def aggregate(self, Y, sample_weight=None):
        """Aggregate the rankings in Y according to the sample weights.

        Parameters
        ----------
        Y : np.ndarray of shape (n_samples, n_classes)
            Input rankings.

        sample_weight : np.ndarray of shape (n_samples,), \
                default=None
            Sample weights. If None, then samples are equally weighted.

        Returns
        -------
        consensus : np.ndarray of shape (n_classes,)
            The consensus ranking.
        """
        # Check the format of the input rankings
        # (allowing classes with infinite values since the
        # consensus ranking may be obtained from missed classes)
        Y = check_array(Y, force_all_finite=False)

        # Check whether the rankings can be
        # aggregated with the selected algorithm
        self.check_targets(Y)

        # Initialize some values from the input arrays
        cdef INT64_t n_samples = Y.shape[0]
        cdef INT64_t n_classes = Y.shape[1]

        # Check the sample weights
        sample_weight = check_sample_weight(sample_weight, Y)

        # Transform the rankings for
        # properly handle them in Cython
        Yt = _transform_rankings(Y)

        # Initialize the consensus ranking
        cdef np.ndarray[INT64_t, ndim=1] consensus = np.zeros(
            n_classes, dtype=np.int64)

        # Initialize the Rank Aggregation algorithm, that is,
        # create the data structures to build the parameters
        self.init(n_classes)

        # Aggregate the rankings to build the consensus ranking
        # (properly applying the MLE process when it corresponds)
        if self.__class__ is BordaCountAlgorithm:
            self.aggregate_mle(Yt, sample_weight, consensus)
        else:
            self.aggregate_rankings(Yt, sample_weight, consensus)

        # Return the built consensus ranking (and,
        # if specified, the completed rankings)
        return consensus


# =============================================================================
# Borda count algorithm
# =============================================================================
cdef class BordaCountAlgorithm(RankAggregationAlgorithm):
    """Borda count algorithm."""

    cdef void init(self, INT64_t n_classes):
        """Initialize the Borda count algorithm."""
        self.count_builder.init(n_classes)

    cdef void reset(self, DTYPE_t_1D count=None,
                    DTYPE_t_3D precedences_matrix=None) nogil:
        """Reset the Borda count algorithm."""
        self.count_builder.reset(count)

    cdef void update_params(self, INT64_t_1D y, DTYPE_t weight,
                            DTYPE_t_1D count=None,
                            DTYPE_t_3D precedences_matrix=None) nogil:
        """Update the parameters for the Borda count algorithm."""
        self.count_builder.update(y, weight, count)

    cdef void build_params(self, INT64_t_2D Y, DTYPE_t_1D sample_weight,
                           DTYPE_t_1D count=None,
                           DTYPE_t_3D precedences_matrix=None) nogil:
        """Build the parameters for the Borda count algorithm."""
        self.count_builder.build(Y, sample_weight, count)

    cdef void aggregate_params(self, INT64_t_1D consensus,
                               DTYPE_t_1D count=None,
                               DTYPE_t_3D precedences_matrix=None) nogil:
        """Build the consensus ranking from the
        parameters of the Borda count algorithm."""
        # If the count is not provided, then,
        # the count of the builder must be used
        if count is None:
            count = self.count_builder.count

        # Initialize some values from the input arrays
        cdef INT64_t n_classes = consensus.shape[0]

        # Define some values to be employed
        cdef DTYPE_t *negative_count

        # Define the indexes
        cdef SIZE_t label

        # Allocate memory for the negative count to allow that the
        # count is sorted in decreasing order (since the desired
        # behaviour is that classes with higher count has higher ranking)
        negative_count = <DTYPE_t*> malloc(n_classes * sizeof(DTYPE_t))

        # Change the values of the count to
        # negative for sorting decreasingly
        for label in range(n_classes):
            negative_count[label] = -count[label]

        # Rank the negative count to obtain the consensus ranking
        rank_data_pointer(
            data=negative_count,
            y=&consensus[0],
            method=RANK_METHOD.ORDINAL,
            n_classes=n_classes)

        # The negative count is not needed anymore,
        # so release the allocated memory
        free(negative_count)

    cdef void complete_rankings(self, INT64_t_2D Y,
                                INT64_t_1D consensus) nogil:
        """Complete the rankings using the Borda count algorithm."""
        # Initialize some values from the input arrays
        cdef INT64_t n_samples = Y.shape[0]
        cdef INT64_t n_classes = Y.shape[1]

        # Define some values to be employed
        cdef INT64_t n_ranked
        cdef INT64_t best_position
        cdef DTYPE_t best_distance
        cdef DTYPE_t local_distance
        cdef DTYPE_t *y

        # Define the indexes
        cdef SIZE_t sample
        cdef SIZE_t i
        cdef SIZE_t j
        cdef SIZE_t k

        # Allocate memory for a ranking that will hold the
        # completed ranking for the corresponding iteration
        y = <DTYPE_t*> malloc(n_classes * sizeof(DTYPE_t))

        # Complete each ranking
        for sample in range(n_samples):
            # Reinitialize the number of
            # ranked classes in this ranking
            n_ranked = 0
            # As a preliminary step, compute the number
            # of ranked classes in this ranking
            for i in range(n_classes):
                if (Y[sample, i] != RANK_TYPE.RANDOM and
                        Y[sample, i] != RANK_TYPE.TOP):
                    n_ranked += 1
            # Check the classes that must be completed
            for i in range(n_classes):
                # Randomly missed classes can be at any
                # possible position, so that the one
                # minimizing the distance with respect
                # to the ranked classes is selected
                if Y[sample, i] == RANK_TYPE.RANDOM:
                    # Reinitialize the best position and
                    # the best distance for this class
                    best_position = 0
                    best_distance = INFINITY
                    # Check the optimal position
                    # where the class must be inserted
                    # (j = 0 means before the first and
                    # j = m after the last label)
                    for j in range(n_classes + 1):
                        # Reinitialize the local distance
                        local_distance = 0.0
                        # Compute the distance w.r.t.
                        # the consensus ranking when
                        # this class is inserted between
                        # those on position j and j + 1
                        for k in range(n_classes):
                            # Only computes the distance
                            # for non missed classes
                            if Y[sample, k] != RANK_TYPE.RANDOM:
                                # Disagreement when inserting the class
                                # between those on position j and j + 1,
                                # increase the local distance
                                if (Y[sample, k] <= j and
                                        consensus[k] > consensus[i] or
                                        Y[sample, k] > j and
                                        consensus[k] < consensus[i]):
                                    local_distance += 1.0
                        # If the local distance is strictly less
                        # (because in case of a tie, the position
                        # with the smallest index is chosen)
                        # than the best distance found until now,
                        # change the best position and distance
                        if local_distance < best_distance:
                            best_position = j
                            best_distance = local_distance
                    # Insert the class in the best possible
                    # position according to the computed distance
                    y[i] = best_position
                # Top-k missed classes can only be at the latest positions
                # of the ranking. Therefore, set their positions to the
                # number of ranked items (plus one) and break the ties
                # according to the consensus ranking (since this will
                # minimize the distance with respect to it)
                elif Y[sample, i] == RANK_TYPE.TOP:
                    y[i] = n_ranked + 1
                # For ranked classes, directly copy
                # the position in the ranking
                else:
                    y[i] = Y[sample, i]

            # Add a little bit of noise based on the consensus ranking
            # to achieve that those classes inserted at the same position
            # are put in the same order than in the consensus ranking
            for i in range(n_classes):
                y[i] += EPSILON * consensus[i]

            # Rank again using the order of the consensus ranking to break
            # the ties when two classes inserted at the same position
            rank_data_pointer(
                data=y,
                y=&Y[sample, 0],
                method=RANK_METHOD.ORDINAL,
                n_classes=n_classes)

        # Once all the rankings have been completed,
        # release the allocated memory for the auxiliar
        # ranking since it is not needed anymore
        free(y)

    cdef void aggregate_mle(self, INT64_t_2D Y, DTYPE_t_1D sample_weight,
                            INT64_t_1D consensus,
                            DTYPE_t_1D count=None,
                            BOOL_t replace=True) nogil:
        """Build the consensus ranking with a MLE process
        using the Borda count algorithm. If specified, the
        rankings in Y will be replaced by the completed ones."""
        # Initialize some values from the input arrays
        cdef INT64_t n_samples = Y.shape[0]
        cdef INT64_t n_classes = Y.shape[1]

        # Define some values to be employed
        cdef INT64_t *consensus_copy
        cdef INT64_t *Y_copy

        # Define the indexes
        cdef SIZE_t sample
        cdef SIZE_t label

        # Allocate memory for the copy of the consensus ranking
        # and the copy of the rankings, since they will be necessary
        # to check whether there is a convergence in the consensus rankings
        consensus_copy = <INT64_t*> malloc(n_classes * sizeof(INT64_t))
        Y_copy = <INT64_t*> malloc(n_samples * n_classes * sizeof(INT64_t))

        # Compute an initial consensus ranking using the incomplete
        # rankings since it is necessary to execute the MLE process
        if count is None:
            self.aggregate_rankings(Y, sample_weight, consensus)
        else:
            self.aggregate_params(consensus, count)

        # Copy the rankings from the memory view to the pointer for
        # being able to apply the MLE process with the original rankings
        copy_pointer_INT64_2D(&Y[0, 0], Y_copy, m=n_samples, n=n_classes)

        # Apply the MLE until convergence in the consensus ranking of
        # the previous iteration and the consensus ranking in this one
        while True:
            # Copy the consensus ranking from the memory view to the
            # pointer for checking the convergence between iterations
            copy_pointer_INT64_1D(&consensus[0], consensus_copy, m=n_classes)
            # Complete the rankings using the consensus
            # ranking found in the previous iteration
            self.complete_rankings(Y, consensus)
            # Compute a new consensus ranking
            # with the completed rankings
            self.aggregate_rankings(Y, sample_weight, consensus)
            # Check the convergence with respect to
            # the previous iteration. If so, break
            if are_consensus_equal(&consensus[0], consensus_copy, n_classes):
                break
            # Otherwise, repeat the MLE process
            # with the original rankings
            else:
                # Copy the rankings from the pointer to the memory view
                # to apply the MLE process with the original rankings
                copy_pointer_INT64_2D(
                    Y_copy, &Y[0, 0], m=n_samples, n=n_classes)

        # If the input rankings must not be replaced, copy
        # the rankings from the pointer to the memory view
        if not replace:
            copy_pointer_INT64_2D(
                Y_copy, &Y[0, 0], m=n_samples, n=n_classes)

        # Free the copy of the consensus ranking and the copy
        # of the rankings since they are not needed anymore
        free(consensus_copy)
        free(Y_copy)

    def check_targets(self, Y):
        """Check whether the rankings in Y can be
        aggregated with the Borda count algorithm."""
        # The Borda count algorithm can
        # only aggregate Label Ranking targets
        check_label_ranking_targets(Y)


# =============================================================================
# Bucket Pivot Algorithm with multiple pivots and two-stages
# =============================================================================
cdef class BPALIAMP2Algorithm(RankAggregationAlgorithm):
    """Bucket Pivot Algorithm with multiple pivots and two-stages."""

    def __init__(self, beta):
        """Constructor."""
        # Check that the beta hyperparameter is an
        # integer or floating value greater or equal
        # than zero and less or equal than one
        if (not isinstance(beta, (Real, np.floating)) and
                not isinstance(beta, (Integral, np.integer))):
            raise TypeError("Expected beta to take a float "
                            "or int value. Got {}."
                            .format(type(beta).__name__))
        elif beta < 0 or beta > 1:
            raise ValueError("Expected beta to be greater than zero "
                             "and less than one. Got beta = {}."
                             .format(beta))

        # Initialize the hyperparameters
        self.beta = beta

    cdef void init(self, INT64_t n_classes):
        """Initialize the Bucket Pivot Algorithm
        with multiple pivots and two-stages."""
        self.pair_order_matrix_builder.init(n_classes)
        self.utopian_matrix_builder.init(n_classes)
        self.dani_indexes_builder.init(n_classes)

    cdef void reset(self, DTYPE_t_1D count=None,
                    DTYPE_t_3D precedences_matrix=None) nogil:
        """Reset the Bucket Pivot Algorithm
        with multiple pivots and two-stages."""
        self.pair_order_matrix_builder.reset(precedences_matrix)
        self.utopian_matrix_builder.reset()
        self.dani_indexes_builder.reset()

    cdef void update_params(self, INT64_t_1D y, DTYPE_t weight,
                            DTYPE_t_1D count=None,
                            DTYPE_t_3D precedences_matrix=None) nogil:
        """Update the parameters for the Bucket Pivot
        Algorithm with multiple pivots and two-stages."""
        self.pair_order_matrix_builder.update_precedences_matrix(
            y, weight, precedences_matrix)

    cdef void build_params(self, INT64_t_2D Y, DTYPE_t_1D sample_weight,
                           DTYPE_t_1D count=None,
                           DTYPE_t_3D precedences_matrix=None) nogil:
        """Build the parameters for the Bucket Pivot
        Algorithm with multiple pivots and two-stages."""
        self.pair_order_matrix_builder.build_precedences_matrix(
            Y, sample_weight, precedences_matrix)

    cdef SIZE_t pick_pivot(self, DTYPE_t_1D dani_indexes,
                           BUCKET_t items) nogil:
        """Pick the pivot from the available items
        using the information of the DANI indexes."""
        # Define some values to be employed
        cdef SIZE_t pivot
        cdef DTYPE_t lower_dani

        # Define the indexes
        cdef SIZE_t item

        # Initialize the lower DANI value that
        # has been found to the highest possible one
        lower_dani = INFINITY

        # Initialize the pivot to zero, even if another
        # one is selected since it is ensured that at
        # least one item can be selected as pivot
        pivot = 0

        # Obtain the pivot selecting the item
        # that minimizes the DANI indexes value
        for item in items:
            if dani_indexes[item] < lower_dani:
                pivot = item
                lower_dani = dani_indexes[item]

        # Return the picked pivot
        return pivot

    cdef DTYPE_t mean_bucket(self, DTYPE_t_2D pair_order_matrix,
                             SIZE_t item, BUCKET_t bucket) nogil:
        """Compute the mean of the pair order matrix with
        respect to the item and the items in the bucket."""
        # Initialize some values from the input arrays
        cdef INT64_t n_items = bucket.size()

        # Define some values to be employed
        cdef DTYPE_t mean

        # Define the indexes
        cdef SIZE_t element

        # Initialize to zero the mean of the pair order matrix
        # with respect to the item and the items in the bucket
        mean = 0.0

        # Compute the mean of the pair order matrix with
        # respect to the item and the items in the bucket
        for element in bucket:
            mean += pair_order_matrix[element, item]

        # Divide such mean by the number of items in the bucket
        # (it is sure that, at least, one item is in the bucket
        # so that division by zero errors cannot appear)
        mean /= n_items

        # Return the computed mean
        return mean

    cdef void first_step(self, DTYPE_t_2D pair_order_matrix, SIZE_t pivot,
                         BUCKET_t items, BUCKET_t *left_bucket,
                         BUCKET_t *central_bucket,
                         BUCKET_t *right_bucket) nogil:
        """Apply the first step of the Bucket Pivot Algorithm
        with multiple pivots and two-stages, that is, insert the
        items using the information of the pair order matrix."""
        # Define some values to be employed
        cdef DTYPE_t mean

        # Define the indexes
        cdef SIZE_t item

        # Insert the items in the corresponding bucket
        for item in items:
            # Avoid the pivot, since it is already
            # inserted in the central bucket
            if item == pivot:
                continue
            # Compute the mean of the pair order matrix with respect
            # to the item and the items in the central bucket
            mean = self.mean_bucket(
                pair_order_matrix, item, central_bucket[0])
            # Preference for this item rather than those in
            # the central bucket, so insert it in the left one
            if mean < (0.5-self.beta):
                left_bucket.push_back(item)
            # Tie between this item and those in the central
            # bucket, so insert it in the central one
            elif ((0.5-self.beta) <= mean and
                    mean <= (0.5+self.beta)):
                central_bucket.push_back(item)
            # Otherwise, prefer the items in the central bucket rather
            # than this one, so insert it in the right bucket
            else:
                right_bucket.push_back(item)

    cdef void second_step(self, DTYPE_t_2D pair_order_matrix, SIZE_t pivot,
                          BUCKET_t *left_bucket, BUCKET_t *central_bucket,
                          BUCKET_t *right_bucket) nogil:
        """Apply the second step of the Bucket Pivot Algorithm
        with multiple pivots and two-stages, that is, re-insert
        the items using the pair order matrix."""
        # Define some values to be employed
        cdef BUCKET_t new_left_bucket
        cdef BUCKET_t new_central_bucket
        cdef BUCKET_t new_right_bucket
        cdef BUCKET_t new_left_right_bucket

        # Copy the contents of the left bucket, central bucket
        # and right bucket to auxiliar data structures to avoid
        # that the information of those buckets is modified
        new_left_bucket = left_bucket[0]
        new_central_bucket = central_bucket[0]
        new_right_bucket = right_bucket[0]

        # Clear the left bucket and the right bucket
        # (since they are the ones that are going to be re-inserted)
        left_bucket.clear()
        right_bucket.clear()

        # Concatenate the information of the left bucket
        # with the information of the right bucket
        new_left_right_bucket.insert(
            new_left_right_bucket.end(),
            new_left_bucket.begin(),
            new_left_bucket.end())

        new_left_right_bucket.insert(
            new_left_right_bucket.end(),
            new_right_bucket.begin(),
            new_right_bucket.end())

        # To re-insert the items, just apply the
        # first step but using the proper buckets
        self.first_step(
            pair_order_matrix, pivot,
            items=new_left_right_bucket,
            left_bucket=left_bucket,
            central_bucket=central_bucket,
            right_bucket=right_bucket)

    cdef void insert(self, DTYPE_t_2D pair_order_matrix, SIZE_t pivot,
                     BUCKET_t items, BUCKET_t *left_bucket,
                     BUCKET_t *central_bucket, BUCKET_t *right_bucket) nogil:
        """Insert the items into the corresponding buckets,
        that is, apply the first step and the second step."""
        # First, insert the items using the
        # information of the pair order matrix
        self.first_step(
            pair_order_matrix, pivot, items,
            left_bucket, central_bucket, right_bucket)

        # Second, re-insert the items using the
        # information of the pair order matrix
        self.second_step(
            pair_order_matrix, pivot,
            left_bucket, central_bucket, right_bucket)

    cdef void concatenate(self, BUCKETS_t *left_buckets,
                          BUCKETS_t *central_buckets,
                          BUCKETS_t *right_buckets,
                          BUCKETS_t *buckets) nogil:
        """Concatenate the buckets."""
        # Append to the end of "buckets" the content
        # (from the beginning to the end) of "left_buckets",
        # "central_buckets" and "right_buckets"
        buckets.insert(
            buckets.end(), left_buckets.begin(), left_buckets.end())
        buckets.insert(
            buckets.end(), central_buckets.begin(), central_buckets.end())
        buckets.insert(
            buckets.end(), right_buckets.begin(), right_buckets.end())

    cdef BUCKETS_t build_buckets(self, DTYPE_t_2D pair_order_matrix,
                                 DTYPE_t_1D dani_indexes,
                                 BUCKET_t items) nogil:
        """Recursively build the buckets for the consensus ranking."""
        # Initialize some values from the input arrays
        cdef INT64_t n_items = items.size()

        # Define some values to be employed
        cdef BUCKET_t left_bucket
        cdef BUCKETS_t left_buckets
        cdef BUCKET_t central_bucket
        cdef BUCKETS_t central_buckets
        cdef BUCKET_t right_bucket
        cdef BUCKETS_t right_buckets
        cdef BUCKETS_t buckets
        cdef SIZE_t pivot

        # Return an empty vector if there are no
        # more items that can be inserted in buckets
        if n_items == 0:
            return buckets

        # Pick the pivot using the information
        # provided by the DANI indexes
        pivot = self.pick_pivot(dani_indexes, items)

        # Initialize central bucket with
        # the previously picked pivot
        central_bucket.push_back(pivot)

        # Insert the items
        self.insert(
            pair_order_matrix, pivot, items,
            &left_bucket, &central_bucket, &right_bucket)

        # Apply the algorithm recursively for
        # each one of the computed buckets
        # (except for the central one, that remains equal)

        # Left buckets
        left_buckets = self.build_buckets(
            pair_order_matrix, dani_indexes, left_bucket)

        # Central buckets
        central_buckets.push_back(central_bucket)

        # Right buckets
        right_buckets = self.build_buckets(
            pair_order_matrix, dani_indexes, right_bucket)

        # Concatenate the buckets
        self.concatenate(
            &left_buckets, &central_buckets, &right_buckets, &buckets)

        # Return the built buckets
        return buckets

    cdef void aggregate_params(self, INT64_t_1D consensus,
                               DTYPE_t_1D count=None,
                               DTYPE_t_3D precedences_matrix=None) nogil:
        """Build the consensus ranking from the parameters of the
        Bucket Pivot Algorithm with multiple pivots and two-stages."""
        # Initialize some values from the input arrays
        cdef INT64_t n_classes = consensus.shape[0]

        # Define some values to be employed
        cdef BUCKET_t items
        cdef BUCKET_t bucket
        cdef BUCKETS_t buckets

        # Define the indexes
        cdef SIZE_t item
        cdef SIZE_t pos

        # Initialize the items inserting
        # the indexes of the classes
        for item in range(n_classes):
            items.push_back(item)

        # Build the pair order matrix from the precedences
        # matrix of the pair order matrix builder
        self.pair_order_matrix_builder.build_pair_order_matrix(
            precedences_matrix)

        # Build the utopian matrix from the pair
        # order matrix of the pair order matrix builder
        self.utopian_matrix_builder.build(
            self.pair_order_matrix_builder.pair_order_matrix)

        # Build the DANI indexes from the pair order
        # atrix of the pair order matrix builder and
        # the utopian matrix of the utopian matrix builder
        self.dani_indexes_builder.build(
            self.pair_order_matrix_builder.pair_order_matrix,
            self.utopian_matrix_builder.utopian_matrix)

        # Build the buckets using the information
        # of the pair order matrix and the DANI indexes
        buckets = self.build_buckets(
            self.pair_order_matrix_builder.pair_order_matrix,
            self.dani_indexes_builder.dani_indexes,
            items)

        # Once the buckets have been obtained, it is necessary to change
        # this information to the format managed in the scikit-lr package
        pos = 0

        # Build the consensus ranking
        # using the found buckets
        for bucket in buckets:
            pos += 1
            for item in bucket:
                consensus[item] = pos

    def check_targets(self, Y):
        """Check whether the rankings in Y can be
        aggregated with the Bucket Pivot Algorithm
        with multiple pivots and two-stages."""
        # The Bucket Pivot Algorithm with multiple pivots and
        # two-stages can aggregate Partial Label Ranking targets
        check_partial_label_ranking_targets(Y)


# =============================================================================
# Count builder
# =============================================================================
cdef class CountBuilder:
    """Count builder."""

    def __cinit__(self):
        """Constructor."""
        # Initialize the attributes
        self.count = None

    cdef void init(self, INT64_t n_classes):
        """Initialize the count builder."""
        # Initialize the count of this object
        # (even if it may be externally provided)
        self.count = np.zeros(n_classes, dtype=np.float64)

    cdef void reset(self, DTYPE_t_1D count=None) nogil:
        """Reset the count builder."""
        # If the count is not provided, then, the
        # one stored in this object must be reset
        if count is None:
            count = self.count

        # Reset the count
        count[:] = 0.0

    cdef void update(self, INT64_t_1D y, DTYPE_t weight,
                     DTYPE_t_1D count=None) nogil:
        """Update the count."""
        # If the count is not provided, then, the
        # one stored in this object must be updated
        if count is None:
            count = self.count

        # Initialize some values from the input arrays
        cdef INT64_t n_classes = y.shape[0]

        # Define some values to be employed
        cdef DTYPE_t rest_count
        cdef DTYPE_t curr_count
        cdef INT64_t n_ranked
        cdef INT64_t n_random
        cdef INT64_t n_top

        # Define the indexes
        cdef SIZE_t label

        # Initialize the remaining count being assigned
        # to the classes of this ranking to zero
        rest_count = 0.0

        # Initialize the number of ranked and
        # missed classes (random and top-k) to zero
        n_ranked = n_random = n_top = 0

        # Compute the total count of this ranking
        # and set this value to the remaining count
        for label in range(n_classes):
            rest_count += (label + 1)

        # Compute the number of ranked and
        # missed classes in this ranking
        for label in range(n_classes):
            if y[label] == RANK_TYPE.RANDOM:
                n_random += 1
            elif y[label] == RANK_TYPE.TOP:
                n_top += 1
            else:
                n_ranked += 1

        # Update the count of the ranked and randomly missed classes
        # (the count for top-k classes must be set with the remaining count)
        for label in range(n_classes):
            if y[label] == RANK_TYPE.TOP:
                continue
            elif y[label] == RANK_TYPE.RANDOM:
                curr_count = (n_classes+1.0) / 2.0
            else:
                curr_count = ((((n_classes-n_random) - y[label] + 1.0)
                              * (n_classes + 1.0))
                              / ((n_classes-n_random) + 1.0))
            # Update the remaining count and the count of
            # this class (weighting the sample accordingly)
            rest_count -= curr_count
            count[label] += weight * curr_count

        # Update the count of the top-k missed classes
        # equitably assigning the remaining count
        for label in range(n_classes):
            if y[label] == RANK_TYPE.TOP:
                count[label] += weight * (rest_count / n_top)

    cdef void build(self, INT64_t_2D Y, DTYPE_t_1D sample_weight,
                    DTYPE_t_1D count=None) nogil:
        """Build the count."""
        # Initialize some values from the input arrays
        cdef INT64_t n_samples = Y.shape[0]

        # Define the indexes
        cdef SIZE_t sample

        # For sanity, it is necessary to ensure that
        # the count is reset before building it
        self.reset(count)

        # Compute the count
        # (updating it for each input ranking)
        for sample in range(n_samples):
            self.update(
                y=Y[sample], weight=sample_weight[sample],
                count=count)


# =============================================================================
# Pair order matrix builder
# =============================================================================
cdef class PairOrderMatrixBuilder:
    """Pair order matrix builder."""

    def __cinit__(self):
        """Constructor."""
        # Initialize the attributes
        self.precedences_matrix = None
        self.pair_order_matrix = None

    cdef void init(self, INT64_t n_classes):
        """Initialize the pair order matrix builder."""
        # Initialize the precedences matrix and the pair order matrix
        # (even if they may be externally provided)
        self.precedences_matrix = np.zeros(
            (n_classes, n_classes, 2), dtype=np.float64)
        self.pair_order_matrix = np.zeros(
            (n_classes, n_classes), dtype=np.float64)

    cdef void reset(self, DTYPE_t_3D precedences_matrix=None,
                    DTYPE_t_2D pair_order_matrix=None) nogil:
        """Reset the pair order matrix builder."""
        # If the precedences matrix and the pair order matrix are not
        # provided, then, the ones stored in this object must be reset
        if precedences_matrix is None:
            precedences_matrix = self.precedences_matrix
        if pair_order_matrix is None:
            pair_order_matrix = self.pair_order_matrix

        # Reset the precedences matrix and the pair order matrix
        precedences_matrix[:] = 0.0
        pair_order_matrix[:] = 0.0

    cdef void update_precedences_matrix(
            self, INT64_t_1D y, DTYPE_t weight,
            DTYPE_t_3D precedences_matrix=None) nogil:
        """Update the precedences matrix."""
        # If the precedences matrix is not provided, then,
        # the one stored in this object must be updated
        if precedences_matrix is None:
            precedences_matrix = self.precedences_matrix

        # Initialize some values from the input arrays
        cdef INT64_t n_classes = y.shape[0]

        # Define the indexes
        cdef SIZE_t f_class
        cdef SIZE_t s_class

        # Add the information of this ranking
        # to the precedences matrix
        for f_class in range(n_classes):
            for s_class in range(f_class, n_classes):
                # Same class, put the corresponding weight in the second
                # entry (tied entry) of "f_class" regarding "s_class"
                if f_class == s_class:
                    precedences_matrix[f_class, s_class, 1] += weight
                # Otherwise, apply the standard procedure
                else:
                    # By-pass the information of this precedence relation if
                    # either "f_class" or "s_class" is missed in this ranking
                    if (y[f_class] == RANK_TYPE.RANDOM or
                            y[s_class] == RANK_TYPE.RANDOM):
                        pass
                    # "f_class" precedes "s_class", put the
                    # corresponding weight in the entry
                    # where counting the precedences
                    # of "f_class" regarding "s_class"
                    elif y[f_class] < y[s_class]:
                        precedences_matrix[f_class, s_class, 0] += weight
                    # "f_class" is tied with "s_class", put the
                    # corresponding weight in the entries where
                    # counting the ties of "f_class" and "s_class"
                    elif y[f_class] == y[s_class]:
                        precedences_matrix[f_class, s_class, 1] += weight
                        precedences_matrix[s_class, f_class, 1] += weight
                    # "s_class" precedes "f_class", put the
                    # corresponding weight in the entry
                    # where counting the precedences
                    # of "s_class" regarding "f_class"
                    else:
                        precedences_matrix[s_class, f_class, 0] += weight

    cdef void build_precedences_matrix(
            self, INT64_t_2D Y, DTYPE_t_1D sample_weight,
            DTYPE_t_3D precedences_matrix=None) nogil:
        """Build the precedences matrix."""
        # Initialize some values from the input arrays
        cdef INT64_t n_samples = Y.shape[0]

        # Define the indexes
        cdef SIZE_t sample

        # For sanity, it is necessary to reset the
        # precedences matrix before building it
        self.reset(precedences_matrix)

        # Build the precedences matrix
        # (updating it for each input ranking)
        for sample in range(n_samples):
            self.update_precedences_matrix(
                y=Y[sample], weight=sample_weight[sample],
                precedences_matrix=precedences_matrix)

    cdef void build_pair_order_matrix(
            self, DTYPE_t_3D precedences_matrix=None,
            DTYPE_t_2D pair_order_matrix=None) nogil:
        """Build the pair order matrix from the precedences matrix."""
        # If the precedences matrix and the pair order matrix are not
        # provided, then, the ones stored in this object must be used
        if precedences_matrix is None:
            precedences_matrix = self.precedences_matrix
        if pair_order_matrix is None:
            pair_order_matrix = self.pair_order_matrix

        # Initialize some values from the input arrays
        cdef INT64_t n_classes = precedences_matrix.shape[0]

        # Define some values to be employed
        cdef DTYPE_t f_precedences
        cdef DTYPE_t s_precedences
        cdef DTYPE_t ties
        cdef DTYPE_t total

        # Define the indexes
        cdef SIZE_t f_class
        cdef SIZE_t s_class

        # Normalize the precedences matrix by counting the number
        # of times that each class precedes another and then,
        # dividing by the total number of precedences
        for f_class in range(n_classes):
            for s_class in range(f_class, n_classes):
                # Obtain the precedences of "f_class"
                # regarding "s_class", the ties of
                # "f_class" and "s_class" and the precedences
                # of "s_class" regarding "f_class"
                # (adding a little bit of noise to avoid
                # division by zero errors)
                f_precedences = (precedences_matrix[f_class, s_class, 0]
                                 + EPSILON)
                ties = (precedences_matrix[f_class, s_class, 1]
                        + EPSILON)
                s_precedences = (precedences_matrix[s_class, f_class, 0]
                                 + EPSILON)
                # Obtain the sum of these entries
                total = f_precedences + ties + s_precedences
                # Compute the pair order matrix
                pair_order_matrix[f_class, s_class] = (
                    (f_precedences + 0.5*ties)
                    / total)
                pair_order_matrix[s_class, f_class] = (
                    1.0
                    - pair_order_matrix[f_class, s_class])


# =============================================================================
# Utopian matrix builder
# =============================================================================
cdef class UtopianMatrixBuilder:
    """Utopian matrix builder."""

    def __cinit__(self):
        """Constructor."""
        # Initialize the attributes
        self.utopian_matrix = None

    cdef void init(self, INT64_t n_classes):
        """Initialize the utopian matrix builder."""
        # Initialize the utopian matrix
        # (even if it may be externally provided)
        self.utopian_matrix = np.zeros(
            (n_classes, n_classes), dtype=np.float64)

    cdef void reset(self, DTYPE_t_2D utopian_matrix=None) nogil:
        """Reset the utopian matrix builder."""
        # If the utopian matrix is not provided, then,
        # the one stored in this object must be reset
        if utopian_matrix is None:
            utopian_matrix = self.utopian_matrix

        # Reset the utopian matrix
        utopian_matrix[:] = 0.0

    cdef void build(self, DTYPE_t_2D pair_order_matrix,
                    DTYPE_t_2D utopian_matrix=None) nogil:
        """Build the utopian matrix from the pair order matrix."""
        # If the utopian matrix is not provided, then,
        # the one stored in this object must be built
        if utopian_matrix is None:
            utopian_matrix = self.utopian_matrix

        # Initialize some values from the input arrays
        cdef INT64_t n_classes = pair_order_matrix.shape[0]

        # Define the indexes
        cdef SIZE_t f_class
        cdef SIZE_t s_class

        # Build the utopian matrix associated with the pair order matrix
        for f_class in range(n_classes):
            for s_class in range(f_class, n_classes):
                if pair_order_matrix[f_class, s_class] > 0.75:
                    utopian_matrix[f_class, s_class] = 1.0
                    utopian_matrix[s_class, f_class] = 0.0
                elif (pair_order_matrix[f_class, s_class] >= 0.25 and
                        pair_order_matrix[f_class, s_class] <= 0.75):
                    utopian_matrix[f_class, s_class] = 0.5
                    utopian_matrix[s_class, f_class] = 0.5
                elif pair_order_matrix[f_class, s_class] < 0.25:
                    utopian_matrix[f_class, s_class] = 0.0
                    utopian_matrix[s_class, f_class] = 1.0


# =============================================================================
# DANI indexes builder
# =============================================================================
cdef class DANIIndexesBuilder:
    """DANI indexes builder."""

    def __cinit__(self):
        """Constructor."""
        # Initialize the attributes
        self.dani_indexes = None

    cdef void init(self, INT64_t n_classes):
        """Initialize the DANI indexes builder."""
        # Initialize the DANI indexes
        # (even if it may be externally provided)
        self.dani_indexes = np.zeros(n_classes, dtype=np.float64)

    cdef void reset(self, DTYPE_t_1D dani_indexes=None) nogil:
        """Reset the DANI indexes builder."""
        # If the DANI indexes is not provided, then,
        # the one stored in this object must be reset
        if dani_indexes is None:
            dani_indexes = self.dani_indexes

        # Reset the DANI indexes
        dani_indexes[:] = 0.0

    cdef void build(self, DTYPE_t_2D pair_order_matrix,
                    DTYPE_t_2D utopian_matrix,
                    DTYPE_t_1D dani_indexes=None) nogil:
        """Build the DANI indexes from the pair
        order matrix and the utopian matrix."""
        # If the DANI indexes is not provided, then,
        # the one stored in this object must be built
        if dani_indexes is None:
            dani_indexes = self.dani_indexes

        # Initialize some values from the input arrays
        cdef INT64_t n_classes = pair_order_matrix.shape[0]

        # Define the indexes
        cdef SIZE_t f_class
        cdef SIZE_t s_class

        # For sanity, it is necessary to reset
        # the DANI indexes before building it
        dani_indexes[:] = 0.0

        # Build the DANI indexes
        for f_class in range(n_classes):
            for s_class in range(n_classes):
                if f_class != s_class:
                    dani_indexes[f_class] += fabs(
                        pair_order_matrix[f_class, s_class]
                        - utopian_matrix[f_class, s_class])
            dani_indexes[f_class] /= n_classes - 1


# =============================================================================
# Methods
# =============================================================================

cdef BOOL_t are_consensus_equal(INT64_t *consensus,
                                INT64_t *new_consensus,
                                INT64_t n_classes) nogil:
    """Check whether the consensus rankings are equal."""
    # Define some values to be employed
    cdef BOOL_t equal

    # Define the indexes
    cdef SIZE_t label

    # Initialize the consensus as if they are equal,
    # for being able to apply short-circuit as far
    # as a different ranking position is found
    equal = True

    # Check whether the consensus rankings are equal
    for label in range(n_classes):
        equal = consensus[label] == new_consensus[label]
        if not equal:
            break

    # Return whether the consensus rankings are equal
    return equal
