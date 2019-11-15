# cython language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False


# =============================================================================
# Imports
# =============================================================================

# Standard
from numbers import Integral, Real

# Third party
from libc.math cimport fabs
from libc.math cimport INFINITY
from libc.stdlib cimport free, malloc
import numpy as np
cimport numpy as np

# Always include this statement after cimporting
# NumPy, to avoid segmentation faults
np.import_array()

# Local application
from .utils._memory cimport copy_pointer_INT64_1D, copy_pointer_INT64_2D
from .utils._ranking_fast cimport RANK_METHOD
from .utils._ranking_fast cimport rank_data_pointer
from .utils.ranking import _transform_rankings
from .utils.ranking import (
    check_label_ranking_targets, check_partial_label_ranking_targets)
from .utils.validation import check_array, check_consistent_length


# =============================================================================
# Constants
# =============================================================================

# Available Rank Aggregation algorithms
ALGORITHM_MAPPING = {
    "borda_count": BordaCountAlgorithm,
    "borda": BordaCountAlgorithm,
    "label_ranking": BordaCountAlgorithm,
    "lr": BordaCountAlgorithm,
    "bpa_lia_mp2": BPALIAMP2Algorithm,
    "partial_label_ranking": BPALIAMP2Algorithm,
    "plr": BPALIAMP2Algorithm
}

# Epsilon value employed to add a little bit of noise in the rankings
# to ensure that the classes are ranked according to the consensus ranking
cdef DTYPE_t EPSILON = np.finfo(np.float64).eps


# =============================================================================
# Classes
# =============================================================================

# =============================================================================
# Rank aggregation algorithm
# =============================================================================
cdef class RankAggregationAlgorithm:
    """Provides a uniform interface to fast Rank Aggregation algorithms.

    This class provides a uniform interface to fast Rank Aggregation algorithms
    The various algorithms can be accessed via the :meth:`get_algorithm`
    class method and the algorithm string identifier (see below).
    For example, to use the Borda count algorithm:

    >>> import numpy as np
    >>> from sklr.consensus import RankAggregationAlgorithm
    >>> rank_algorithm = RankAggregationAlgorithm.get_algorithm("borda_count")
    >>> Y = np.array([[1, 2, 3], [2, 3, 1], [2, 3, 1]])
    >>> rank_algorithm.aggregate(Y)
    array([1, 2, 3])

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
        # to some default values
        self.beta = 0.25

        # Initialize the attributes
        self.count_builder = CountBuilder()
        self.pair_order_matrix_builder = PairOrderMatrixBuilder()
        self.utopian_matrix_builder = UtopianMatrixBuilder()
        self.dani_indexes_builder = DANIIndexesBuilder()

    def __init__(self):
        """Constructor."""
        if self.__class__ is RankAggregationAlgorithm:
            raise NotImplementedError("RankAggregationAlgorithm is "
                                      "an abstract class.")

    cdef void init(self, INT64_t n_classes):
        """Placeholder for a method that will
        initialize the Rank Aggregation algorithm."""


    cdef void reset(self, DTYPE_t_1D count=None,
                    DTYPE_t_3D precedences_matrix=None) nogil:
        """Placeholder for a method that will
        reset the Rank Aggregation algorithm."""


    @classmethod
    def get_algorithm(cls, algorithm, **kwargs):
        """Get the Rank Aggregation algorithm from the string identifier.

        See the docstring of RankAggregationAlgorithm
        for a list of available algorithms.

        Parameters
        ----------
        algorithm : str
            The algorithm to use.

        **kwargs : dict
            Additional arguments will be passed to the requested algorithm.

        Returns
        -------
        rank_algorithm : RankAggregationAlgorithm
            The Rank Aggregation algorithm.

        Raises
        ------
        ValueError
            If the provided algorithm is not available.
        """
        # Map the algorithm string ID to the Rank Aggregation class
        # raising the proper error when the algorithm is not available
        if (isinstance(algorithm, type) and
                issubclass(algorithm, RankAggregationAlgorithm)):
            pass
        else:
            try:
                rank_algorithm = ALGORITHM_MAPPING[algorithm]
            except KeyError:
                raise ValueError(
                    "Unrecognized algorithm: '{}'. Valid algorithms are: {}."
                    .format(algorithm, sorted(list(ALGORITHM_MAPPING.keys()))))

        # Also initialize the beta hyperparameter to a default value of 0.25
        # for the Bucket Pivot Algorithm with multiple pivots and two-stages
        if rank_algorithm is BPALIAMP2Algorithm:
            if "beta" not in kwargs:
                kwargs["beta"] = 0.25

        # Initialize the callable Rank Aggregation algorithm
        rank_algorithm = rank_algorithm(**kwargs)

        # Return it
        return rank_algorithm

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
        # First, it is necessary to build the parameters of the
        # Rank Aggregation algorithm to reuse the implemented methods
        self.build_params(Y, sample_weight)

        # Second, aggregate the parameters built the
        # previous step to obtain the consensus ranking
        self.aggregate_params(consensus)

    cdef void complete_rankings(self, INT64_t_2D Y,
                                INT64_t_1D consensus) nogil:
        """Placeholder for a method that will complete
        the rankings using the Rank Aggregation algorithm."""


    cdef void aggregate_mle(self, INT64_t_2D Y, DTYPE_t_1D sample_weight,
                            INT64_t_1D consensus,
                            DTYPE_t_1D count=None,
                            DTYPE_t_3D precedences_matrix=None,
                            BOOL_t replace=True) nogil:
        """Build the consensus ranking with a MLE process
        using the Rank Aggregation algorithm. If specified,
        the rankings in Y will be replaced by the completed ones."""
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
        # to check the convergence between iterations
        consensus_copy = <INT64_t*> malloc(n_classes * sizeof(INT64_t))
        Y_copy = <INT64_t*> malloc(n_samples * n_classes * sizeof(INT64_t))

        # Compute an initial consensus ranking from the possibly incomplete
        # rankings since it is necessary to execute the MLE process
        if count is None and precedences_matrix is None:
            self.aggregate_rankings(Y, sample_weight, consensus)
        else:
            self.aggregate_params(consensus, count, precedences_matrix)

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

    def aggregate(self, Y, sample_weight=None,
                  apply_mle=False, return_Yt=False):
        """Aggregate the rankings in Y according to the sample weights.

        Parameters
        ----------
        Y : np.ndarray of shape (n_samples, n_classes)
            Input rankings.

        sample_weight : {None, np.ndarray} of shape (n_samples,),
                optional (default=None)
            Sample weights.

        apply_mle : bool, optional (default=False)
            If True, apply a MLE process to complete the
            rankings before obtaining the consensus ranking,
            else aggregate the rankings without completing.

        return_Yt : bool, optional (default=False)
            If ``apply_mle`` and True, return the completed rankings
            according to the MLE process carry out to obtain the
            consensus ranking.

        Returns
        -------
        consensus : np.ndarray of shape (n_classes,)
            The consensus ranking of the ones in Y.

        Yt : np.ndarray of shape (n_samples, n_classes)
            If ``apply_mle and return_Y``, return the
            completed rankings according to the MLE process.

        Raises
        ------
        NotImplementedError
            If the rankings are for the Partial Label Ranking
            problem and ``apply_mle``.
        """
        # Check the format of the input rankings
        # (allowing missing classes since a consensus
        # ranking can also be obtained from them)
        Y = check_array(Y, force_all_finite=False)

        # Check whether the rankings can be
        # aggregated with the selected algorithm
        if self.__class__ is BordaCountAlgorithm:
            check_label_ranking_targets(Y)
        else:
            check_partial_label_ranking_targets(Y)

        # TODO: For the Bucket Pivot Algorithm with multiple pivots and
        # two-stages, the MLE process cannot be used to aggregate the rankings
        if apply_mle and self.__class__ is BPALIAMP2Algorithm:
            raise NotImplementedError("The MLE process cannot be carried out "
                                      "with Bucket Pivot Algorithm with "
                                      "multiple pivots and two-stages.")

        # Initialize some values from the input arrays
        cdef INT64_t n_samples = Y.shape[0]
        cdef INT64_t n_classes = Y.shape[1]

        # If the sample weights are not provided, output an array of ones
        if sample_weight is None:
            sample_weight = np.ones(n_samples, dtype=np.float64)
        # Otherwise, check their format, ensuring that they are
        # a 1-D array with the same number of samples than the rankings
        else:
            # Check the format of the sample weights
            sample_weight = check_array(
                sample_weight, ensure_2d=False, dtype=np.float64)
            # Ensure that they are a 1-D array
            if sample_weight.ndim != 1:
                raise ValueError("Sample weights must be 1-D array. "
                                 "Got {}-D array."
                                 .format(sample_weight.ndim))
            # Ensure that the rankings and the sample
            # weights have the number of samples
            check_consistent_length(Y, sample_weight)

        # Transform the rankings for properly handle them in Cython
        Yt = _transform_rankings(Y)

        # Initialize the consensus ranking
        cdef np.ndarray[INT64_t, ndim=1] consensus = np.zeros(
            n_classes, dtype=np.int64)

        # Initialize the Rank Aggregation algorithm, i.e., create
        # the data structures of the parameters needed by it
        self.init(n_classes)

        # Aggregate the rankings to build the consensus one
        # (either applying the MLE process or the direct method)
        if apply_mle:
            self.aggregate_mle(Yt, sample_weight, consensus)
        else:
            self.aggregate_rankings(Yt, sample_weight, consensus)

        # Return the built consensus ranking and, if
        # specified, also return the completed rankings
        if apply_mle and return_Yt:
            return (consensus, Yt)
        else:
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
        # If the count is not provided, then, the count of the builder must
        # be used. This will allow that other classes provide their own count
        if count is None:
            count = self.count_builder.count

        # Initialize some values from the input arrays
        cdef INT64_t n_classes = consensus.shape[0]

        # Define some values to be employed
        cdef DTYPE_t *negative_count

        # Define the indexes
        cdef SIZE_t label

        # Allocate memory for the negative count to allow
        # that the count is sorted in decreasing order
        # (since the desired behaviour is that classes
        # with higher count has higher ranking)
        negative_count = <DTYPE_t*> malloc(n_classes * sizeof(DTYPE_t))

        # Change the values of the count to a negative
        # one for sorting in decreasing order
        for label in range(n_classes):
            negative_count[label] = -count[label]

        # Rank the negative count to
        # obtain the consensus ranking
        rank_data_pointer(
            data=negative_count, y=consensus, method=RANK_METHOD.ORDINAL)

        # The negative count is not needed
        # anymore, so release the allocated memory
        free(negative_count)

    cdef void complete_rankings(self, INT64_t_2D Y,
                                INT64_t_1D consensus) nogil:
        """Complete the rankings using the Borda count algorithm."""
        # Initialize some values from the input arrays
        cdef INT64_t n_samples = Y.shape[0]
        cdef INT64_t n_classes = Y.shape[1]

        # Define some values to be employed
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
            # Check the classes that must be completed
            for i in range(n_classes):
                # Only complete missing labels
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
                # Directly copy the position in the ranking for the
                # classes that have not been randomly missed classes
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
                data=y, y=Y[sample],
                method=RANK_METHOD.ORDINAL)

        # Once all the rankings have been completed,
        # release the allocated memory for the auxiliar
        # ranking since it is not needed anymore
        free(y)


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
            raise TypeError("Expected beta to take a float or "
                            "int value. Got {}."
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

        # Insert the items in the corresponding bucket,
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
        # First step
        self.first_step(
            pair_order_matrix, pivot, items,
            left_bucket, central_bucket, right_bucket)

        # Second step
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
        # Initialize the count, just in case it is needed since
        # it may be provided in the corresponding method. Thus,
        # the extra memory overhead should not be an issue
        self.count = np.zeros(n_classes, dtype=np.float64)

    cdef void reset(self, DTYPE_t_1D count=None) nogil:
        """Reset the count builder."""
        # Reset the provided count or
        # the count stored in this object
        if count is None:
            self.count[:] = 0.0
        else:
            count[:] = 0.0

    cdef void update(self, INT64_t_1D y, DTYPE_t weight,
                     DTYPE_t_1D count=None) nogil:
        """Update the count."""
        # Determine whether to use the count stored
        # in this object or the input count
        if count is None:
            count = self.count

        # Initialize some values from the input arrays
        cdef INT64_t n_classes = y.shape[0]

        # Define some values to be employed
        cdef INT64_t n_random
        cdef DTYPE_t curr_count

        # Define the indexes
        cdef SIZE_t label

        # Initialize the number of randomly missed classes to zero,
        # (by default, the ranking does not contain missed classes)
        n_random = 0

        # Obtain the number of randomly missed classes to
        # properly compute the count assigned to each class
        for label in range(n_classes):
            if y[label] == RANK_TYPE.RANDOM:
                n_random += 1

        # Update the count
        for label in range(n_classes):
            # If the class is randomly missed, consider
            # that it can be at any possible position
            if y[label] == RANK_TYPE.RANDOM:
                curr_count = (n_classes+1.0) / 2.0
            # If the class is ranked, assign
            # the count as it corresponds
            else:
                curr_count = ((((n_classes-n_random) - y[label] + 1.0)
                              * (n_classes + 1.0))
                              / ((n_classes-n_random) + 1.0))
            # Update the count weighting
            # the sample accordingly
            count[label] += weight * curr_count

    cdef void build(self, INT64_t_2D Y, DTYPE_t_1D sample_weight,
                    DTYPE_t_1D count=None) nogil:
        """Build the count."""
        # Initialize some values from the input arrays
        cdef INT64_t n_samples = Y.shape[0]

        # Define the indexes
        cdef SIZE_t sample

        # IMPORTANT: FOR SANITY, IT IS NECESSARY TO
        # RESET THE COUNT BEFORE BUILDING IT
        self.reset(count)

        # Compute the count by updating
        # it for each input ranking
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
        # even if they can be provided in the corresponding methods.
        # Thus, the extra memory overhead should not be an issue
        self.precedences_matrix = np.zeros(
            (n_classes, n_classes, 2), dtype=np.float64)

        self.pair_order_matrix = np.zeros(
            (n_classes, n_classes), dtype=np.float64)

    cdef void reset(self, DTYPE_t_3D precedences_matrix=None) nogil:
        """Reset the pair order matrix builder."""
        # Reset the precedences matrix stored
        # of this object or the provided one
        if precedences_matrix is None:
            self.precedences_matrix[:] = 0.0
        else:
            precedences_matrix[:] = 0.0

        # Reset the pair order matrix of this object
        self.pair_order_matrix[:] = 0.0

    cdef void update_precedences_matrix(
            self, INT64_t_1D y, DTYPE_t weight,
            DTYPE_t_3D precedences_matrix=None) nogil:
        """Update the precedences matrix."""
        # Determine whether to use the precedences matrix
        # of this object or the input precedences matrix
        if precedences_matrix is None:
            precedences_matrix = self.precedences_matrix

        # Initialize some values from the input arrays
        cdef INT64_t n_classes = y.shape[0]

        # Define the indexes
        cdef SIZE_t f_class
        cdef SIZE_t s_class

        # Add the information of the ranking
        # to the precedences matrix
        for f_class in range(n_classes):
            for s_class in range(f_class, n_classes):
                # Same class, put the corresponding weight in the second
                # entry (tied entry) of "f_class" regarding "s_class"
                if f_class == s_class:
                    precedences_matrix[f_class, s_class, 1] += weight
                # Otherwise, apply the standard procedure
                else:
                    # Randomly missed class detected.
                    # Since either "f_class" or "s_class" is missed,
                    # that either "f_class" could precede "s_class"
                    # or viceversa. Moreover, it is also considered
                    # that they can be tied
                    if (y[f_class] == RANK_TYPE.RANDOM or
                            y[s_class] == RANK_TYPE.RANDOM):
                        precedences_matrix[f_class, s_class, 0] += weight
                        precedences_matrix[s_class, f_class, 0] += weight
                        precedences_matrix[f_class, s_class, 1] += weight
                        precedences_matrix[s_class, f_class, 1] += weight
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
            self, INT64_t_2D Y,
            DTYPE_t_1D sample_weight,
            DTYPE_t_3D precedences_matrix=None) nogil:
        """Build the precedences matrix."""
        # Initialize some values from the input arrays
        cdef INT64_t n_samples = Y.shape[0]

        # Define the indexes
        cdef SIZE_t sample

        # IMPORTANT: FOR SANITY, IT IS NECESSARY TO
        # RESET PRECEDENCES MATRIX BEFORE BUILDING IT
        self.reset(precedences_matrix)

        # Build the precedences matrix by
        # updating it using each ranking
        for sample in range(n_samples):
            self.update_precedences_matrix(
                y=Y[sample], weight=sample_weight[sample],
                precedences_matrix=precedences_matrix)

    cdef void build_pair_order_matrix(
            self,
            DTYPE_t_3D precedences_matrix=None) nogil:
        """Build the pair order matrix from the precedences matrix."""
        # Determine whether to use the precedences matrix
        # of this object or the input precedences matrix
        if precedences_matrix is None:
            precedences_matrix = self.precedences_matrix

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
                self.pair_order_matrix[f_class, s_class] = (
                    (f_precedences + 0.5*ties)
                    / total)
                self.pair_order_matrix[s_class, f_class] = (
                    1.0
                    - self.pair_order_matrix[f_class, s_class])


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
        self.utopian_matrix = np.zeros(
            (n_classes, n_classes), dtype=np.float64)

    cdef void reset(self) nogil:
        """Reset the utopian matrix builder."""
        # Reset the utopian matrix
        self.utopian_matrix[:] = 0.0

    cdef void build(self, DTYPE_t_2D pair_order_matrix) nogil:
        """Build the utopian matrix from the pair order matrix."""
        # Initialize some values from the input arrays
        cdef INT64_t n_classes = pair_order_matrix.shape[0]

        # Define the indexes
        cdef SIZE_t f_class
        cdef SIZE_t s_class

        # Build the utopian matrix associated
        # with the pair order matrix
        for f_class in range(n_classes):
            for s_class in range(f_class, n_classes):
                if pair_order_matrix[f_class, s_class] > 0.75:
                    self.utopian_matrix[f_class, s_class] = 1.0
                    self.utopian_matrix[s_class, f_class] = 0.0
                elif (pair_order_matrix[f_class, s_class] >= 0.25 and
                        pair_order_matrix[f_class, s_class] <= 0.75):
                    self.utopian_matrix[f_class, s_class] = 0.5
                    self.utopian_matrix[s_class, f_class] = 0.5
                elif pair_order_matrix[f_class, s_class] < 0.25:
                    self.utopian_matrix[f_class, s_class] = 0.0
                    self.utopian_matrix[s_class, f_class] = 1.0


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
        self.dani_indexes = np.zeros(n_classes, dtype=np.float64)

    cdef void reset(self) nogil:
        """Reset the DANI indexes builder."""
        # Reset the DANI indexes
        self.dani_indexes[:] = 0.0

    cdef void build(self, DTYPE_t_2D pair_order_matrix,
                    DTYPE_t_2D utopian_matrix) nogil:
        """Build the DANI indexes from the pair
        order matrix and the utopian matrix."""
        # Initialize some values from the input arrays
        cdef INT64_t n_classes = pair_order_matrix.shape[0]

        # Define the indexes
        cdef SIZE_t f_class
        cdef SIZE_t s_class

        # IMPORTANT: FOR SANITY, RESET THE
        # UTOPIAN MATRIX BEFORE BUILDING IT
        self.dani_indexes[:] = 0.0

        # Build the DANI indexes
        for f_class in range(n_classes):
            for s_class in range(n_classes):
                if f_class != s_class:
                    self.dani_indexes[f_class] += fabs(
                        pair_order_matrix[f_class, s_class]
                        - utopian_matrix[f_class, s_class])
            self.dani_indexes[f_class] /= n_classes - 1


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
