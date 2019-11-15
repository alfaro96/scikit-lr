# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False


# =============================================================================
# Imports
# =============================================================================

# Third party
from libc.math cimport INFINITY
from libc.stdlib cimport free, malloc
import numpy as np
cimport numpy as np

# Always include this statement after cimporting
# NumPy, to avoid segmentation faults
np.import_array()


# =============================================================================
# Classes
# =============================================================================

# =============================================================================
# Splitter
# =============================================================================
cdef class Splitter:
    """Abstract splitter class."""

    def __cinit__(self, Criterion criterion, INT64_t max_features,
                  INT64_t max_splits, object random_state):
        """Constructor."""
        # Initialize the hyperparameters
        self.criterion = criterion
        self.max_features = max_features
        self.max_splits = max_splits
        self.random_state = random_state

        # Initialize the data, the sample weights
        # and the sorted indexes for the data
        self.X = None
        self.sample_weight = None
        self.X_idx_sorted = None

        # Initialize the number of features
        # and the features indexes (that will
        # hold the features to be considered
        # on each node)
        self.n_features = 0
        self.features = None

        # Force the number of splits to
        # two for the binary splitter
        # (to save memory)
        if isinstance(self, BinarySplitter):
            self.max_splits = 2

    cdef void init(self, DTYPE_t_2D X, INT64_t_2D Y,
                   DTYPE_t_1D sample_weight, SIZE_t_2D X_idx_sorted):
        """Initialize the splitter."""
        # Initialize some values from the input arrays
        cdef INT64_t n_features = X.shape[1]

        # Define the indexes
        cdef SIZE_t feature

        # Initialize the data, the sample weights and
        # the sorted indexes for the data (setting the
        # reference, without copying the contents)
        self.X = X
        self.sample_weight = sample_weight
        self.X_idx_sorted = X_idx_sorted

        # Initialize the number of features
        # and the features indexes
        self.n_features = n_features
        self.features = np.zeros(self.n_features, dtype=np.intp)

        # Fill the features indexes
        # and the useful features
        for feature in range(n_features):
            self.features[feature] = feature
            self.useful_features.push_back(feature)

        # Initialize the criterion from
        # the rankings and the sample weight
        self.criterion.init_from_rankings(Y, sample_weight, self.max_splits)

    cdef void reset(self, SIZE_t_2D X_idx_sorted,
                    SplitRecord record, INT64_t child):
        """Reset the splitter."""
        # Define some values to be employed
        cdef INT64_t n_samples
        cdef DTYPE_t weighted_n_samples
        cdef DTYPE_t impurity
        cdef INT64_t_1D consensus
        cdef DTYPE_t_1D count
        cdef DTYPE_t_3D precedences_matrix

        # Reset the sorted indexes for the data (directy
        # setting the reference without copying the contents)
        self.X_idx_sorted = X_idx_sorted

        # Initialize the attributes that will be
        # passed from the record to the criterion
        n_samples = record.samples[child]
        weighted_n_samples = record.weights[child]
        impurity = record.impurities[child]
        consensus = record.consensus[child]
        count = record.counts[child]
        precedences_matrix = record.precedences_matrices[child]

        # Reset the criterion from the information
        # provided by the record of a child
        self.criterion.init_from_record(
            n_samples, weighted_n_samples,
            impurity, consensus,
            count, precedences_matrix)

    cdef void drawn(self):
        """Take random features with replacement
        needed by the node_split method."""
        # Define some values to be employed
        cdef INT64_t n_useful_features

        # Extract the number of useful features
        # from the set with the useful features
        n_useful_features = self.useful_features.size()

        # Obtain the random features (choice as most as
        # possible features, taking into account that it
        # is not possible to take more features than the
        # useful ones and, also, it is not possible to take
        # more features than the maximum number of features)
        self.features = self.random_state.choice(
            a=self.useful_features,
            size=min(n_useful_features, self.max_features),
            replace=False)

    cdef void init_params(self, SIZE_t feature) nogil:
        """Placeholder for a method that will
        initialize the parameters for the splitter."""

    cdef void update_params(self, SIZE_t feature, SIZE_t sample,
                            BOOL_t add_split,
                            BOOL_t update_criterion,
                            BOOL_t check_improvement) nogil:
        """Placeholder for a method that will
        update the parameters for the splitter."""

    cdef BOOL_t add_split(self, SIZE_t feature, SIZE_t sample) nogil:
        """Placeholder for a method that will
        determine whether to add an split to the node."""

    cdef BOOL_t update_criterion(self, SIZE_t feature, SIZE_t sample,
                                 BOOL_t add_split) nogil:
        """Placeholder for a method that will
        determine whether to update the criterion."""

    cdef BOOL_t check_improvement(self, SIZE_t feature, SIZE_t sample,
                                  BOOL_t add_split,
                                  BOOL_t update_criterion) nogil:
        """Placeholder for a method that will determine
        whether to check the improvement in impurity."""

    cdef void node_split(self, SplitRecord split) nogil:
        """Find the best split on node."""
        # Initialize the number of samples from the criterion
        # and the number of features to be tested
        # from the stored value in the feature indexes
        cdef INT64_t n_samples = self.criterion.n_node_samples
        cdef INT64_t n_features = self.features.shape[0]

        # Define some values to be employed
        cdef DTYPE_t_2D X
        cdef DTYPE_t_1D sample_weight
        cdef SIZE_t_2D X_idx_sorted
        cdef SIZE_t_1D samples
        cdef DTYPE_t current_improvement
        cdef DTYPE_t best_improvement
        cdef BOOL_t add_split
        cdef BOOL_t update_criterion
        cdef BOOL_t check_improvement

        # Define the indexes
        cdef SIZE_t feature
        cdef SIZE_t feature_idx
        cdef SIZE_t sample

        # Initialize the current improvement in impurity and
        # the best improvement in impurity to the lowest value
        # (for properly managing the improvement in impurity)
        current_improvement = -INFINITY
        best_improvement = -INFINITY

        # Initialize the memory view for the data, the sample
        # weights and the sorted indexes for the data
        # (just to save lines of code)
        X = self.X
        sample_weight = self.sample_weight
        X_idx_sorted = self.X_idx_sorted

        # Obtain the feature that maximizes
        # the improvement in impurity
        for feature in range(n_features):
            # Initialize the identifier of this feature
            # (since they have been randomly choice)
            feature_idx = self.features[feature]
            # Initialize the samples to consider
            # for this feature (already sorted
            # from the lowest to the highest value)
            samples = self.X_idx_sorted[feature_idx]
            # Reset the criterion using
            # the samples for this feature
            self.criterion.reset(samples)
            # Initialize the parameters
            # needed to evaluate the improvement
            # in impurity for this feature
            self.init_params(feature_idx)
            # Obtain the thresholds that maximizes the
            # improvement in impurity for this feature
            for sample in range(1, n_samples):
                # Check whether to add an split, update the criterion
                # and compute the improvement in impurity
                add_split = self.add_split(
                    feature_idx, sample)
                update_criterion = self.update_criterion(
                    feature_idx, sample, add_split)
                check_improvement = self.check_improvement(
                    feature_idx, sample, add_split, update_criterion)
                # Add an split point to the
                # node (if corresponds)
                if add_split:
                    self.criterion.node_children()
                # Update the criterion by moving the
                # position of the samples in the node
                # to the actual sample (if corresponds)
                if update_criterion:
                    self.criterion.update(sample)
                # Compute the improvement in impurity and checks
                # whether a better split point has been found
                if check_improvement:
                    current_improvement = self.criterion.impurity_improvement()
                    # The current improvement in impurity is best than
                    # the improvement in impurity found so far. So,
                    # change the best improvement in impurity and, also,
                    # store the corresponding parameters in the split record
                    if current_improvement > best_improvement:
                        best_improvement = current_improvement
                        split.record(
                            self.criterion,
                            feature_idx, X, X_idx_sorted)
                # Update the parameters needed by
                # this feature to properly work
                self.update_params(
                    feature_idx, sample, add_split,
                    update_criterion, check_improvement)

    cdef list node_indexes(self, SplitRecord record):
        """Obtain the new indexes according to the split record."""
        # Initialize the number of samples from the criterion,
        # the number of features from the stored value and
        # the number of splits from the record
        cdef INT64_t n_samples = self.criterion.n_node_samples
        cdef INT64_t n_features = self.n_features
        cdef INT64_t n_splits = record.n_splits

        # Define some values to be employed
        cdef SIZE_t_2D X_idx_sorted
        cdef SIZE_t best_feature
        cdef SIZE_t_1D pos
        cdef INT64_t_1D samples
        cdef INDEXES_t indexes
        cdef UINT8_t *matrix
        cdef list new_X_idx_sorted

        # Define the indexes
        cdef SIZE_t split
        cdef SIZE_t feature
        cdef SIZE_t sample
        cdef SIZE_t index

        # Initialize the view with the sorted indexes
        # for the data and the feature used for splitting
        X_idx_sorted = self.X_idx_sorted
        best_feature = record.feature

        # Initialize the view with the sample positions
        # and the number of samples on each split
        pos = record.pos
        samples = record.samples

        # Initialize the matrix where the indexes
        # to keep for each split will be stored
        matrix = <UINT8_t*> malloc(n_samples * sizeof(UINT8_t))

        # Create an empty list where the sorted
        # indexes for each child will hold
        new_X_idx_sorted = list()

        # Initialize empty arrays with
        # the sorted indexes for each split
        for split in range(n_splits):
            new_X_idx_sorted.append(
                np.zeros((n_features, samples[split]), dtype=np.intp))

        # Obtain the sorted indexes
        # for each possible split
        for split in range(n_splits):
            # For sanity, clear the contents of
            # the previously stored indexes to
            # properly compute them for this split
            indexes.clear()
            # Copy the contents from the view to the
            # set of indexes to find, in an efficient
            # way, the indexes that belongs to this split
            view_to_set_pointer_SIZE_1D(
                view=X_idx_sorted[best_feature, pos[split]:pos[split + 1]],
                pointer=&indexes)
            # Obtain the sorted indexes of
            # this split for each feature
            for feature in range(n_features):
                # Compute the boolean matrix containing
                # the sorted indexes for this feature
                in1d(X_idx_sorted[feature], indexes, &matrix)
                # Initialize a counter to know
                # the position of the index
                index = 0
                # Insert the elements into the memory view, checking
                # the contents against the computed boolean matrix
                for sample in range(n_samples):
                    if matrix[sample]:
                        new_X_idx_sorted[split][feature, index] = (
                            X_idx_sorted[feature, sample])
                        index += 1

        # The matrix is not needed anymore,
        # so release the allocated memory
        free(matrix)

        # Return the new sorted indexes
        return new_X_idx_sorted


# =============================================================================
# Binary splitter
# =============================================================================
cdef class BinarySplitter(Splitter):
    """Splitter for finding the best binary split."""

    cdef BOOL_t add_split(self, SIZE_t feature, SIZE_t sample) nogil:
        """Determine whether to add an split on the binary splitter."""
        # Add a split at the very
        # beginning of the process
        return sample == 1

    cdef BOOL_t update_criterion(self, SIZE_t feature, SIZE_t sample,
                                 BOOL_t add_split) nogil:
        """Determine whether to update the criterion on the binary splitter."""
        # Update the criterion when there is a change on the feature value
        return (self.X[self.criterion.samples[sample - 1], feature] !=
                self.X[self.criterion.samples[sample], feature])

    cdef BOOL_t check_improvement(self, SIZE_t feature, SIZE_t sample,
                                  BOOL_t add_split,
                                  BOOL_t update_criterion) nogil:
        """Determine whether to check the improvement
        in impurity on the binary splitter."""
        # Whenever the criterion is updated,
        # check the improvement in impurity
        return update_criterion


# =============================================================================
# Frequency splitter
# =============================================================================
cdef class FrequencySplitter(Splitter):
    """Splitter for finding the best equal-frequency split."""

    cdef void init_params(self, SIZE_t feature) nogil:
        """Initialize the parameters for the equal-frequency splitter."""
        # Initialize the frequency (dividing the weighted number of
        # samples in this node by the maximum number of possible splits).
        # In fact, it is necessary to substract the weight of
        # the first sample since the process starts at one
        self.frequency = ((self.criterion.weighted_n_node_samples
                           / self.max_splits)
                          - self.criterion.sample_weight[
                            self.criterion.samples[0]])

    cdef void update_params(self, SIZE_t feature, SIZE_t sample,
                            BOOL_t add_split,
                            BOOL_t update_criterion,
                            BOOL_t check_improvement) nogil:
        """Update the parameters for the equal-frequency splitter."""
        # Define some values to be employed
        cdef DTYPE_t weight
        cdef DTYPE_t weighted_n_node_samples
        cdef DTYPE_t weighted_n_child_samples

        # Define the indexes
        cdef SIZE_t child

        # Initialize the index of this child
        child = self.criterion.n_children - 1

        # Initialize the weight of the current sample, the
        # weighted number of samples in the node and the
        # weighted number of samples in the current child
        weight = (
            self.sample_weight[self.criterion.samples[sample]])
        weighted_n_node_samples = (
            self.criterion.weighted_n_node_samples)
        weighted_n_child_samples = (
            self.criterion.weighted_n_children_samples[child])

        # Initialize the new frequency if a new split
        # is added or only update in any other case
        if add_split:
            self.frequency = ((weighted_n_node_samples -
                               - weighted_n_child_samples)
                              / (self.max_splits - child))
        else:
            self.frequency -= weight

    cdef BOOL_t add_split(self, SIZE_t feature, SIZE_t sample) nogil:
        """Determine whether to add an split
        point on the equal-frequency splitter."""
        # Define some values to be employed
        cdef DTYPE_t prev_sample_value
        cdef DTYPE_t curr_sample_value
        cdef DTYPE_t last_sample_value

        # Initialize the feature value of the previous sample,
        # of the current sample and the final sample
        prev_sample_value = self.X[
            self.X_idx_sorted[feature, sample - 1],
            feature]
        curr_sample_value = self.X[
            self.X_idx_sorted[feature, sample],
            feature]
        last_sample_value = self.X[
            self.X_idx_sorted[feature, self.criterion.n_node_samples - 1],
            feature]

        # Add an split point when:
        #   - The feature value of the previous and the current sample changes.
        #   - The frequency is less than zero.
        # or:
        #   - The feature value of the previous and the current sample changes.
        #   - The feature value of the current sample is equal to the last one.
        #   - The number of splits is less than the maximum number of splits.
        return (((prev_sample_value != curr_sample_value and
                 self.frequency <= 0) or
                 (prev_sample_value != curr_sample_value and
                  curr_sample_value == last_sample_value)) and
                self.criterion.n_children < self.max_splits)

    cdef BOOL_t update_criterion(self, SIZE_t feature, SIZE_t sample,
                                 BOOL_t add_split) nogil:
        """Determine whether to update the
        criterion on the equal-frequency splitter."""
        # Update the criterion when
        # a new split is added
        return add_split

    cdef BOOL_t check_improvement(self, SIZE_t feature, SIZE_t sample,
                                  BOOL_t add_split,
                                  BOOL_t update_criterion) nogil:
        """Determine whether to check the improvement
        in impurity on the Frequency splitter.
        """
        # Check the improvement in
        # impurity on the last sample
        return (sample + 1) == self.criterion.n_node_samples


# =============================================================================
# Width splitter
# =============================================================================
cdef class WidthSplitter(Splitter):
    """Splitter for finding the best equal-width split."""

    cdef void init_params(self, SIZE_t feature) nogil:
        """Initialize the parameters for the equal-width splitter."""
        # Define some values to be employed
        cdef DTYPE_t first_sample_value
        cdef DTYPE_t last_sample_value

        # Initialize the feature value of the first sample
        # and and the feature value of the final sample
        first_sample_value = self.X[
            self.X_idx_sorted[feature, 0],
            feature]
        last_sample_value = self.X[
            self.X_idx_sorted[feature, self.criterion.n_node_samples - 1],
            feature]

        # Initialize the width (dividing the value
        # of the last sample minus the value of the
        # first sample by the maximum number of splits)
        self.width = ((last_sample_value - first_sample_value)
                      / self.max_splits)

    cdef BOOL_t add_split(self, SIZE_t feature, SIZE_t sample) nogil:
        """Determine whether to add an split point on the width spliter."""
        # Define some values to be employed
        cdef DTYPE_t first_sample_value
        cdef DTYPE_t prev_sample_value
        cdef DTYPE_t curr_sample_value
        cdef DTYPE_t last_sample_value

        # Initialize the feature value of the first sample, of the
        # current sample, of the next sample and the final sample
        first_sample_value = self.X[
            self.X_idx_sorted[feature, 0],
            feature]
        prev_sample_value = self.X[
            self.X_idx_sorted[feature, sample - 1],
            feature]
        curr_sample_value = self.X[
            self.X_idx_sorted[feature, sample],
            feature]
        last_sample_value = self.X[
            self.X_idx_sorted[feature, self.criterion.n_node_samples - 1],
            feature]

        # Add an split point when:
        #   - The feature value of the current and the next sample changes.
        #   - The width of the interval has been passed out.
        #   - The number of splits is less than the maximum number of splits.
        return (prev_sample_value != curr_sample_value and
                (curr_sample_value > (first_sample_value
                                      + self.width
                                      * self.criterion.n_children)) and
                self.criterion.n_children < self.max_splits)

    cdef BOOL_t update_criterion(self, SIZE_t feature, SIZE_t sample,
                                 BOOL_t add_split) nogil:
        """Determine whether to update the criterion on the qidth splitter."""
        # Update the criterion
        # when a new split is added
        return add_split

    cdef BOOL_t check_improvement(self, SIZE_t feature, SIZE_t sample,
                                  BOOL_t add_split,
                                  BOOL_t update_criterion) nogil:
        """Determine whether to check the improvement
        in impurity on the equal-width splitter."""
        # Check the improvement in
        # impurity on the last sample
        return (sample + 1) == self.criterion.n_node_samples


# =============================================================================
# Split record
# =============================================================================
cdef class SplitRecord:
    """Split record."""

    def __cinit__(self, Criterion criterion,
                  INT64_t max_splits, INT64_t n_classes):
        """Initialize the split record."""
        # Initialize the number of splits
        self.n_splits = 1

        # Initialize the feature used for splitting
        # and the threshold value on each split
        self.feature = 0
        self.thresholds = np.zeros(max_splits + 1, dtype=np.float64)

        # Initialize the sample positions on each split
        self.pos = np.zeros(max_splits + 1, dtype=np.intp)

        # Initialize the number of samples on each split
        self.samples = np.zeros(max_splits, dtype=np.int64)

        # Initialize the weight on each split
        self.weights = np.zeros(max_splits, dtype=np.float64)

        # Initialize the impurity on each split
        self.impurities = np.zeros(max_splits, dtype=np.float64)

        # Initialize the consensus ranking on each split
        self.consensus = np.zeros(
            (max_splits, n_classes), dtype=np.int64)

        # Initialize the counts and the precedences matrices
        # to None and hereafter they will be properly set
        self.counts = None
        self.precedences_matrices = None

        # Initialize the count or the precedences matrix on
        # each split (depending on whether the count must be
        # used instead of the precedences matrix or viceversa)
        if criterion.count_children is not None:
            self.counts = np.zeros(
                (max_splits, n_classes), dtype=np.float64)
        else:
            self.precedences_matrices = np.zeros(
                (max_splits, n_classes, n_classes, 2), dtype=np.float64)

    cdef void record(self, Criterion criterion, SIZE_t feature,
                     DTYPE_t_2D X, SIZE_t_2D X_idx_sorted) nogil:
        """Record the split."""
        # Define the indexes
        cdef SIZE_t split

        # Initialize the number of splits
        self.n_splits = criterion.n_children

        # Initialize the feature used for splitting
        self.feature = feature

        # Initialize the sample positions on each split
        self.pos[:] = criterion.pos

        # Initialize the number of samples on each split and
        # the weighted number of samples on each split
        self.samples[:] = criterion.n_children_samples
        self.weights[:] = criterion.weighted_n_children_samples

        # Initialize the impurity on each split and
        # the consensus ranking on each split
        self.impurities[:] = criterion.impurity_children
        self.consensus[:] = criterion.consensus_children

        # Initialize the count or the precedences matrix
        # (depending on whether the count must be used
        # instead of the precedences matrix or viceversa)
        if criterion.count_children is not None:
            self.counts[:] = (
                criterion.count_children)
        else:
            self.precedences_matrices[:] = (
                criterion.precedences_matrix_children)

        # Initialize the threshold value on each split (to
        # reduce the variance, the middle value between
        # the sample used to split and the next one is used)
        for split in range(self.n_splits - 1):
            self.thresholds[split + 1] = ((X[X_idx_sorted[
                                             feature, self.pos[split + 1] - 1],
                                             feature]
                                           + X[X_idx_sorted[
                                             feature, self.pos[split + 1]],
                                             feature])
                                          / 2)

        # Initialize the thresholds on the extremes
        # (the left extreme to the lowest possible value
        # and the right extreme to the highest possible one)
        self.thresholds[0] = -INFINITY
        self.thresholds[self.n_splits] = INFINITY


# =============================================================================
# Methods
# =============================================================================

cdef void view_to_set_pointer_SIZE_1D(SIZE_t_1D view,
                                      INDEXES_t *pointer) nogil:
    """Copy the contents of the 1-D integer view to the set pointer."""
    # Initialize some values from the input arrays
    cdef SIZE_t n_samples = view.shape[0]

    # Define the indexes
    cdef SIZE_t sample

    # Copy the contents (from the
    # view to the pointer set)
    for sample in range(n_samples):
        pointer[0].insert(view[sample])


cdef void in1d(SIZE_t_1D element, INDEXES_t test_elements,
               UINT8_t **matrix) nogil:
    """Fast version of the in1d method."""
    # Initialize some values from the input arrays
    cdef INT64_t n_samples = element.shape[0]

    # Define the indexes
    cdef SIZE_t sample
    cdef SIZE_t index

    # Obtain the elements to
    # keep from test_elements
    for sample in range(n_samples):
        if test_elements.find(element[sample]) != test_elements.end():
            matrix[0][sample] = True
        else:
            matrix[0][sample] = False
