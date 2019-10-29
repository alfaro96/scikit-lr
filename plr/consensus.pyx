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

# Always include this statement after cimporting NumPy, to avoid faults
np.import_array()

# Local application
from cython.operator cimport dereference as deref
from .utils._memory cimport (
    view_to_pointer_INT64_1D, view_to_pointer_INT64_2D,
    pointer_to_view_INT64_2D)
from .utils._ranking_fast cimport RANK_METHOD
from .utils._ranking_fast cimport rank_pointer_fast
from .utils.ranking import _transform_rankings
from .utils.ranking import (
    check_label_ranking_targets, check_partial_label_ranking_targets)
from .utils.validation import check_array, check_consistent_length


# =============================================================================
# Constants
# =============================================================================

# Available algorithms
ALGORITHM_MAPPING = {
    "borda_count": BordaCountAlgorithm,
    "borda": BordaCountAlgorithm,
    "label_ranking": BordaCountAlgorithm,
    "lr": BordaCountAlgorithm,
    "bpa_lia_mp2": BPALIAMP2Algorithm,
    "partial_label_ranking": BPALIAMP2Algorithm,
    "plr": BPALIAMP2Algorithm
}

# Epsilon
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
    >>> from plr.consensus import RankAggregationAlgorithm
    >>> alg = RankAggregationAlgorithm.get_algorithm("borda_count")
    >>> Y = np.array([[1, 2, 3], [2, 3, 1], [2, 3, 1]])
    >>> alg.aggregate(Y)
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
        # Initialize the hyperparameters to some default values
        self.beta = 0.25

        # Initialize the attributes
        self.count_builder = CountBuilder()
        self.pair_order_matrix_builder = PairOrderMatrixBuilder()
        self.utopian_matrix_builder = UtopianMatrixBuilder()
        self.dani_indexes_builder = DANIIndexesBuilder()

    def __init__(self):
        """Constructor."""
        if self.__class__ is RankAggregationAlgorithm:
            raise NotImplementedError(
                "RankAggregationAlgorithm is an abstract class.")

    cdef void init(self, INT64_t n_classes):
        """Placeholder for a method that will
        initialize the Rank Aggregation algorithm."""
        pass

    cdef void reset(self, DTYPE_t_1D count=None,
                    DTYPE_t_3D precedences_matrix=None) nogil:
        """Placeholder for a method that will
        reset the Rank Aggregation algorithm."""
        pass

    @classmethod
    def get_algorithm(cls, algorithm, **kwargs):
        """Get the Rank Aggregation algorithm from the string identifier.

        See the docstring of RankAggregationAlgorithm
        for a list of available algorithms.

        Parameters
        ----------
        algorithm: string
            The algorithm to use.

        **kwargs:
            Additional arguments will be passed to the requested algorithm.

        Returns
        -------
        rank_algorithm: RankAggregationAlgorithm
            The Rank Aggregation algorithm.

        Raises
        ------
        ValueError
            If the provided algorithm is not available.
        """
        # Map the algorithm string ID to the Rank Aggregation class
        if (isinstance(algorithm, type) and
                issubclass(algorithm, RankAggregationAlgorithm)):
            pass
        else:
            try:
                rank_algorithm = ALGORITHM_MAPPING[algorithm]
            except KeyError:
                raise ValueError(
                    "Unrecognized algorithm: \"{}\". Valid algorithms are: {}."
                    .format(algorithm, sorted(list(ALGORITHM_MAPPING.keys()))))

        # In Bucket Pivot Algorithm with multiple pivots
        # and two-stages, initialize the beta hyperparameter
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
        pass

    cdef void build_params(self, INT64_t_2D Y, DTYPE_t_1D sample_weight,
                           DTYPE_t_1D count=None,
                           DTYPE_t_3D precedences_matrix=None) nogil:
        """Placeholder for a method that will build the
        parameters for the Rank Aggregation algorithm."""
        pass

    cdef void aggregate_params(self, INT64_t_1D consensus,
                               DTYPE_t_1D count=None,
                               DTYPE_t_3D precedences_matrix=None) nogil:
        """Placeholder for a method that will build the consensus ranking
        from the parameters of the Rank Aggregation algorithm."""
        pass

    cdef void aggregate_rankings(self, INT64_t_2D Y, DTYPE_t_1D sample_weight,
                                 INT64_t_1D consensus) nogil:
        """Aggregate the rankings to build the consensus ranking."""
        # First, build the parameters of the Rank Aggregation algorithm
        self.build_params(Y, sample_weight)

        # Second, aggregate the built parameters
        # to obtain the consensus ranking
        self.aggregate_params(consensus)

    cdef void complete_rankings(self, INT64_t_2D Y,
                                INT64_t_1D consensus) nogil:
        """Placeholder for a method that will complete
        the rankings using the Rank Aggregation algorithm."""
        pass

    cdef void aggregate_mle(self, INT64_t_2D Y, DTYPE_t_1D sample_weight,
                            INT64_t_1D consensus,
                            DTYPE_t_1D count=None,
                            DTYPE_t_3D precedences_matrix=None) nogil:
        """Placeholder for a method that will build the consensus
        ranking with a MLE process using the Rank Aggregation algorithm."""
        pass

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
        Y = check_array(Y, force_all_finite=False)

        # Check whether the rankings can be
        # aggregated with the selected algorithm
        if self.__class__ is BordaCountAlgorithm:
            check_label_ranking_targets(Y)
        else:
            check_partial_label_ranking_targets(Y)

        # Check whether the MLE process must be applied
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

        # Initialize the Rank Aggregation algorithm
        self.init(n_classes)

        # Aggregate the rankings to build the consensus one
        if apply_mle:
            self.aggregate_mle(Yt, sample_weight, consensus)
        else:
            self.aggregate_rankings(Yt, sample_weight, consensus)

        # Revert the top-k missed classes to their original value,
        # since they must be fixed to be managed by Python methods
        if apply_mle and return_Yt:
            Yt = Yt.astype(np.float64)
            Yt[np.isinf(Y)] = np.inf

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
        # Initialize the count builder
        self.count_builder.init(n_classes)

    cdef void reset(self, DTYPE_t_1D count=None,
                    DTYPE_t_3D precedences_matrix=None) nogil:
        """Reset the Borda count algorithm."""
        # Reset the count builder
        self.count_builder.reset(count)

    cdef void update_params(self, INT64_t_1D y, DTYPE_t weight,
                            DTYPE_t_1D count=None,
                            DTYPE_t_3D precedences_matrix=None) nogil:
        """Update the parameters for the Borda count algorithm."""
        # Update the count
        self.count_builder.update(y, weight, count)

    cdef void build_params(self, INT64_t_2D Y, DTYPE_t_1D sample_weight,
                           DTYPE_t_1D count=None,
                           DTYPE_t_3D precedences_matrix=None) nogil:
        """Build the parameters for the Borda count algorithm."""
        # Build the count
        self.count_builder.build(Y, sample_weight, count)

    cdef void aggregate_params(self, INT64_t_1D consensus,
                               DTYPE_t_1D count=None,
                               DTYPE_t_3D precedences_matrix=None) nogil:
        """Build the consensus ranking from the
        parameters of the Borda count algorithm."""
        # Determine whether the count of the builder must be used
        if count is None:
            count = self.count_builder.count

        # Initialize some values from the input arrays
        cdef INT64_t n_classes = consensus.shape[0]

        # Define some values to be employed
        cdef DTYPE_t *negative_count

        # Define the indexes
        cdef SIZE_t label

        # Allocate memory for the negative count
        negative_count = <DTYPE_t*> malloc(n_classes * sizeof(DTYPE_t))

        # Change the count to sort in decreasing order
        for label in range(n_classes):
            negative_count[label] = -count[label]

        # Rank the count to obtain the consensus ranking
        rank_pointer_fast(
            data=negative_count, y=consensus, method=RANK_METHOD.ORDINAL)

        # Free the negative count
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

        # Allocate memory for the ranking
        y = <DTYPE_t*> malloc(n_classes * sizeof(DTYPE_t))

        # Complete each ranking
        for sample in range(n_samples):
            # Check those that must be completed
            for i in range(n_classes):
                # Only complete missing labels
                if Y[sample, i] == RANK_TYPE.RANDOM:
                    # Reinitialize the best position and the best distance
                    best_position = 0
                    best_distance = INFINITY
                    # Check the optimal position
                    # where the label must be inserted
                    # (j = 0 means before the first and
                    # j = m after the last label)
                    for j in range(n_classes + 1):
                        # Reinitialize the local distance
                        local_distance = 0.0
                        # Compute the distance w.r.t.
                        # the consensus ranking when the
                        # label is inserted between those
                        # on position j and j + 1
                        for k in range(n_classes):
                            # Only computes the distance
                            # for non missing labels
                            if Y[sample, k] != RANK_TYPE.RANDOM:
                                # Disagreement when inserting the label
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
                    # Insert the label into the optimal position
                    y[i] = best_position
                # For the non randomly missed classes,
                # directly copy the ranking
                else:
                    y[i] = Y[sample, i]

            # Add a little bit of noise based on the consensus ranking
            # to achieve that those labels inserted at the same position
            # are put in the same order as in the consensus ranking
            for i in range(n_classes):
                y[i] += EPSILON * consensus[i]

            # Rank again using the order of the consensus ranking to break
            # the ties when two labels inserted at the same position
            rank_pointer_fast(
                data=y, y=Y[sample],
                method=RANK_METHOD.ORDINAL)

        # Free the ranking
        free(y)

    cdef void aggregate_mle(self, INT64_t_2D Y, DTYPE_t_1D sample_weight,
                            INT64_t_1D consensus,
                            DTYPE_t_1D count=None,
                            DTYPE_t_3D precedences_matrix=None) nogil:
        """Build the consensus ranking with a MLE
        process using the Borda count algorithm."""
        # Initialize some values from the input arrays
        cdef INT64_t n_samples = Y.shape[0]
        cdef INT64_t n_classes = Y.shape[1]

        # Define some values to be employed
        cdef INT64_t *consensus_copy
        cdef INT64_t *Y_copy

        # Define the indexes
        cdef SIZE_t sample
        cdef SIZE_t label

        # Allocate memory for the copy of the consensus
        # ranking and the copy of the rankings
        consensus_copy = <INT64_t*> malloc(n_classes * sizeof(INT64_t))
        Y_copy = <INT64_t*> malloc(n_samples * n_classes * sizeof(INT64_t))

        # Compute an initial consensus ranking
        if count is None:
            self.aggregate_rankings(Y, sample_weight, consensus)
        else:
            self.aggregate_params(consensus, count, precedences_matrix)

        # Copy the rankings (from the memory view to the pointer)
        view_to_pointer_INT64_2D(Y, &Y_copy)

        # Apply the MLE until convergence in the consensus rankings
        while True:
            # Copy the consensus ranking (from the memory view to the pointer)
            view_to_pointer_INT64_1D(consensus, &consensus_copy)
            # Complete the rankings
            self.complete_rankings(Y, consensus)
            # Compute the consensus ranking with the completed rankings
            self.aggregate_rankings(Y, sample_weight, consensus)
            # Convergence found, break
            if are_consensus_equal(consensus, consensus_copy):
                break
            # Otherwise, repeat the MLE process
            else:
                # Copy the rankings (from the pointer to the memory view)
                pointer_to_view_INT64_2D(Y_copy, Y)

        # Free the copy of the consensus ranking and the copy of the rankings
        free(consensus_copy)
        free(Y_copy)


# =============================================================================
# Bucket Pivot Algorithm with multiple pivots and two-stages
# =============================================================================
cdef class BPALIAMP2Algorithm(RankAggregationAlgorithm):
    """Bucket Pivot Algorithm with multiple pivots and two-stages."""

    def __init__(self, beta):
        """Constructor."""
        # Check the beta hyperparameter
        if (not isinstance(beta, (Real, np.floating)) and
                not isinstance(beta, (Integral, np.integer))):
            raise TypeError(
                "Expected beta to take a float or int value. Got {}."
                .format(type(beta).__name__))
        elif beta < 0 or beta > 1:
            raise ValueError(
                "Expected beta to be greater than zero and less than one. "
                "Got beta = {}."
                .format(beta))

        # Initialize the hyperparameters
        self.beta = beta

    cdef void init(self, INT64_t n_classes):
        """Initialize the Bucket Pivot Algorithm
        with multiple pivots and two-stages."""
        # Initialize the pair order matrix builder, the utopian
        # matrix builder and the DANI indexes builder
        self.pair_order_matrix_builder.init(n_classes)
        self.utopian_matrix_builder.init(n_classes)
        self.dani_indexes_builder.init(n_classes)

    cdef void reset(self, DTYPE_t_1D count=None,
                    DTYPE_t_3D precedences_matrix=None) nogil:
        """Reset the Bucket Pivot Algorithm
        with multiple pivots and two-stages."""
        # Reset the pair order matrix builder, the utopian
        # matrix builder and the DANI indexes builder
        self.pair_order_matrix_builder.reset(precedences_matrix)
        self.utopian_matrix_builder.reset()
        self.dani_indexes_builder.reset()

    cdef void update_params(self, INT64_t_1D y, DTYPE_t weight,
                            DTYPE_t_1D count=None,
                            DTYPE_t_3D precedences_matrix=None) nogil:
        """Update the parameters for the Bucket Pivot
        Algorithm with multiple pivots and two-stages."""
        # Update the precedences matrix
        self.pair_order_matrix_builder.update_precedences_matrix(
            y, weight, precedences_matrix)

    cdef void build_params(self, INT64_t_2D Y, DTYPE_t_1D sample_weight,
                           DTYPE_t_1D count=None,
                           DTYPE_t_3D precedences_matrix=None) nogil:
        """Build the parameters for the Bucket Pivot
        Algorithm with multiple pivots and two-stages."""
        # Build the precedences matrix
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

        # Initialize the lower DANI to the highest possible value
        lower_dani = INFINITY

        # Initialize the pivot to zero, even if another one is selected
        pivot = 0

        # Obtain the pivot
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
        # Define some values to be employed
        cdef DTYPE_t mean
        cdef INT64_t counter

        # Define the indexes
        cdef SIZE_t element

        # Initialize the mean to zero
        mean = 0.0

        # Initialize the counter to zero
        counter = 0

        # Compute the mean of the pair order matrix with
        # respect to the item and the items in the bucket
        for element in bucket:
            mean += pair_order_matrix[element, item]
            counter += 1
        if counter > 0:
            mean /= counter

        # Return the computed mean
        return mean

    cdef void first_step(self, DTYPE_t_2D pair_order_matrix, SIZE_t pivot,
                         BUCKET_t items,
                         BUCKET_t *left_bucket,
                         BUCKET_t *central_bucket,
                         BUCKET_t *right_bucket) nogil:
        """Insert the items using the pair order matrix."""
        # Define some values to be employed
        cdef DTYPE_t mean

        # Define the indexes
        cdef SIZE_t item

        # Insert the items
        for item in items:
            # Avoid the pivot
            if item == pivot:
                continue
            # Compute the mean of the pair order matrix with respect
            # to the item and the items in the central bucket
            mean = self.mean_bucket(
                pair_order_matrix, item, deref(central_bucket))
            # Preference for the current item rather than those
            # in the central bucket, so move to the left one
            if mean < (0.5-self.beta):
                deref(left_bucket).push_back(item)
            # Tie between the current item and those in the
            # central bucket, so move to the central one
            elif ((0.5-self.beta) <= mean and
                    mean <= (0.5+self.beta)):
                deref(central_bucket).push_back(item)
            # Otherwise, prefer the items in the central bucket rather
            # than the current one, so move it to the right bucket
            else:
                deref(right_bucket).push_back(item)

    cdef void second_step(self, DTYPE_t_2D pair_order_matrix, SIZE_t pivot,
                          BUCKET_t *left_bucket,
                          BUCKET_t *central_bucket,
                          BUCKET_t *right_bucket) nogil:
        """Re-insert the items using the pair order matrix."""
        # Define some values to be employed
        cdef BUCKET_t new_left_bucket
        cdef BUCKET_t new_central_bucket
        cdef BUCKET_t new_right_bucket
        cdef BUCKET_t new_left_right_bucket

        # Initialize the new left bucket, new central bucket and new
        # right bucket, by copying the contents from the original ones
        new_left_bucket = deref(left_bucket)
        new_central_bucket = deref(central_bucket)
        new_right_bucket = deref(right_bucket)

        # Clear the left bucket and the right bucket
        left_bucket.clear()
        right_bucket.clear()

        # Concatenate the new left bucket and the new right bucket
        new_left_right_bucket.insert(
            new_left_right_bucket.end(),
            new_left_bucket.begin(),
            new_left_bucket.end())

        new_left_right_bucket.insert(
            new_left_right_bucket.end(),
            new_right_bucket.begin(),
            new_right_bucket.end())

        # Apply the first step with the new items
        self.first_step(
            pair_order_matrix, pivot,
            items=new_left_right_bucket,
            left_bucket=left_bucket,
            central_bucket=central_bucket,
            right_bucket=right_bucket)

    cdef void insert(self, DTYPE_t_2D pair_order_matrix, SIZE_t pivot,
                     BUCKET_t items,
                     BUCKET_t *left_bucket,
                     BUCKET_t *central_bucket,
                     BUCKET_t *right_bucket) nogil:
        """Insert the items into the corresponding buckets."""
        # Apply the first step
        self.first_step(
            pair_order_matrix, pivot, items,
            left_bucket, central_bucket, right_bucket)

        # Apply the second step
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

        # Return an empty vector if there are no more items
        if n_items == 0:
            return buckets

        # Pick the pivot
        pivot = self.pick_pivot(dani_indexes, items)

        # Initialize central bucket with the pivot
        central_bucket.push_back(pivot)

        # Insert the items
        self.insert(
            pair_order_matrix, pivot, items,
            &left_bucket, &central_bucket, &right_bucket)

        # Apply the algorithm recursively

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

        # Initialize the items
        for item in range(n_classes):
            items.push_back(item)

        # Build the pair order matrix
        self.pair_order_matrix_builder.build_pair_order_matrix(
            precedences_matrix)

        # Build the utopian matrix
        self.utopian_matrix_builder.build(
            self.pair_order_matrix_builder.pair_order_matrix)

        # Build the DANI indexes
        self.dani_indexes_builder.build(
            self.pair_order_matrix_builder.pair_order_matrix,
            self.utopian_matrix_builder.utopian_matrix)

        # Build the buckets using the information
        # of the pair order matrix and the DANI indexes
        buckets = self.build_buckets(
            self.pair_order_matrix_builder.pair_order_matrix,
            self.dani_indexes_builder.dani_indexes,
            items)

        # Initialize the position to zero
        pos = 0

        # Build the consensus ranking from the buckets
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
        # Initialize the count
        self.count = np.zeros(n_classes, dtype=np.float64)

    cdef void reset(self, DTYPE_t_1D count=None) nogil:
        """Reset the count builder."""
        # Reset the count
        if count is None:
            self.count[:] = 0.0
        else:
            count[:] = 0.0

    cdef void update(self, INT64_t_1D y, DTYPE_t weight,
                     DTYPE_t_1D count=None) nogil:
        """Update the count."""
        # Determine whether to use the count of the object
        if count is None:
            count = self.count

        # Initialize some values from the input arrays
        cdef INT64_t n_classes = y.shape[0]

        # Define some values to be employed
        cdef INT64_t n_random
        cdef INT64_t n_top

        cdef DTYPE_t curr_count
        cdef DTYPE_t total_count

        # Define the indexes
        cdef SIZE_t label

        # Initialize the number of randomly missed classes
        # and the number of top-k classes to zero
        n_random = 0
        n_top = 0

        # Initialize th total count to zero
        total_count = 0

        # Obtain the number of randomly missed classes
        # and the number of classes in the top-k.
        # Moreover, obtain the total count
        for label in range(n_classes):
            if y[label] == RANK_TYPE.RANDOM:
                n_random += 1
            elif y[label] == RANK_TYPE.TOP:
                n_top += 1
            total_count += label + 1

        # Update the count for the ranked classes
        # and the randomly missed classes
        for label in range(n_classes):
            # Does not update the top-k classes
            if y[label] == RANK_TYPE.TOP:
                continue
            # If the class is randomly missed, consider
            # that it can be at any possible position
            elif y[label] == RANK_TYPE.RANDOM:
                curr_count = (n_classes+1.0) / 2.0
            # If the class is ranked, assign the count as corresponding
            else:
                curr_count = ((((n_classes-n_random) - y[label] + 1.0)
                              * (n_classes + 1.0))
                              / ((n_classes-n_random) + 1.0))
            # Update the count
            count[label] += weight * curr_count
            total_count -= curr_count

        # Assign the remaining count to the top-k classes
        for label in range(n_classes):
            if y[label] == RANK_TYPE.TOP:
                count[label] += weight * (total_count/n_top)

    cdef void build(self, INT64_t_2D Y, DTYPE_t_1D sample_weight,
                    DTYPE_t_1D count=None) nogil:
        """Build the count."""
        # Initialize some values from the input arrays
        cdef INT64_t n_samples = Y.shape[0]

        # Define the indexes
        cdef SIZE_t sample

        # IMPORTANT: FOR SANITY, RESET THE COUNT BEFORE BUILDING IT
        self.reset(count)

        # Compute the count
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
        # Initialize the precedences matrix
        self.precedences_matrix = np.zeros(
            (n_classes, n_classes, 2), dtype=np.float64)

        # Initialize the pair order matrix
        self.pair_order_matrix = np.zeros(
            (n_classes, n_classes), dtype=np.float64)

    cdef void reset(self, DTYPE_t_3D precedences_matrix=None) nogil:
        """Reset the pair order matrix builder."""
        # Reset the precedences matrix
        if precedences_matrix is None:
            self.precedences_matrix[:] = 0.0
        else:
            precedences_matrix[:] = 0.0

        # Reset the pair order matrix
        self.pair_order_matrix[:] = 0.0

    cdef void update_precedences_matrix(
            self, INT64_t_1D y, DTYPE_t weight,
            DTYPE_t_3D precedences_matrix=None) nogil:
        """Update the precedences matrix."""
        # Determine whether to use the precedences matrix of the object
        if precedences_matrix is None:
            precedences_matrix = self.precedences_matrix

        # Initialize some values from the input arrays
        cdef INT64_t n_classes = y.shape[0]

        # Define the indexes
        cdef SIZE_t f_class
        cdef SIZE_t s_class

        # Add the information of the ranking to the precedences matrix
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

        # IMPORTANT: FOR SANITY, RESET THE
        # PRECEDENCES MATRIX BEFORE BUILDING IT
        self.reset(precedences_matrix)

        # Build the precedences matrix
        for sample in range(n_samples):
            self.update_precedences_matrix(
                y=Y[sample], weight=sample_weight[sample],
                precedences_matrix=precedences_matrix)

    cdef void build_pair_order_matrix(
            self,
            DTYPE_t_3D precedences_matrix=None) nogil:
        """Build the pair order matrix from the precedences matrix."""
        # Determine whether to use the precedences matrix of the object
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

        # Normalize the precedences matrix
        for f_class in range(n_classes):
            for s_class in range(f_class, n_classes):
                # Obtain the precedences of "f_class"
                # regarding "s_class", the ties of
                # "f_class" and "s_class" and the precedences
                # of "s_class" regarding "f_class"
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

        # Build the utopian matrix associated with the pair order matrix
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

        # IMPORTANT: FOR SANITY, RESET THE UTOPIAN MATRIX BEFORE BUILDING IT
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

cdef BOOL_t are_consensus_equal(INT64_t_1D consensus_view,
                                INT64_t *consensus_pointer) nogil:
    """Check whether the consensus rankings are equal."""
    # Initialize some values from the input arrays
    cdef INT64_t n_classes = consensus_view.shape[0]

    # Define some values to be employed
    cdef BOOL_t equal

    # Define the indexes
    cdef SIZE_t label

    # Initialize the consensus as if they are equal
    equal = True

    # Check whether the consensus rankings are equal
    for label in range(n_classes):
        equal = consensus_view[label] == consensus_pointer[label]
        if not equal:
            break

    # Return whether the consensus rankings are equal
    return equal
