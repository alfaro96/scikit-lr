"""Bagging meta-estimator."""


# =============================================================================
# Imports
# =============================================================================

# Standard
from abc import ABC, abstractmethod
from numbers import Real, Integral

# Third party
import numpy as np

# Local application
from ._base import BaseEnsemble
from ._base import _indexes_to_mask
from ._base import MAX_RAND_SEED
from ..base import LabelRankerMixin, PartialLabelRankerMixin
from ..tree import DecisionTreeLabelRanker, DecisionTreePartialLabelRanker
from ..utils.validation import (
    check_is_fitted, check_random_state, has_fit_parameter)


# =============================================================================
# Methods
# =============================================================================

def _generate_indexes(random_state, bootstrap, n_population, n_samples):
    """Draw randomly sampled indexes."""
    return random_state.choice(n_population, n_samples, bootstrap)


def _generate_bagging_indexes(random_state, bootstrap_features,
                              bootstrap_samples, n_features, n_samples,
                              max_features, max_samples):
    """Randomly draw feature and sample indexes."""
    # Obtain the random state
    random_state = check_random_state(random_state)

    # Draw random features
    feature_indexes = _generate_indexes(
        random_state, bootstrap_features,
        n_features, max_features)

    # Draw random samples
    sample_indexes = _generate_indexes(
        random_state, bootstrap_samples,
        n_samples, max_samples)

    # Return the randomly
    # drawn features and samples
    return (feature_indexes, sample_indexes)


def _build_estimators(n_estimators, ensemble, X, Y, sample_weight, seeds):
    """Private function used to build estimators."""
    # Initialize some values from the input arrays
    (n_samples, n_features) = (
        X.shape)
    (max_samples, max_features) = (
        ensemble._max_samples, ensemble._max_features)
    (bootstrap, bootstrap_features) = (
        ensemble.bootstrap, ensemble.bootstrap_features)

    # Check whether the base estimator of
    # the ensemble support sample weighting,
    # raising the proper error when they
    # are provided and the base estimator
    # does not supports it
    support_sample_weight = has_fit_parameter(
        ensemble.base_estimator_, "sample_weight")
    if not support_sample_weight and sample_weight is not None:
        raise ValueError("The base estimator does not support sample weight.")

    # Initialize the estimators and the
    # features used for building the estimators
    estimators = []
    estimators_features = []

    # Build the list of
    # estimators for the ensemble
    for i in range(n_estimators):
        # Make an estimator, properly
        # seeding for reproducible experiments
        estimator = ensemble._make_estimator(
            append=False, random_state=seeds[i])
        # Draw random features and samples
        (features, indexes) = _generate_bagging_indexes(
            seeds[i],
            bootstrap_features, bootstrap,
            n_features, n_samples,
            max_features, max_samples)
        # If the base estimator supports sample weights,
        # then, obtain the sample weights using the
        # bootstrap indexes and then fit
        if support_sample_weight:
            # Check the sample weights, initializing them to an
            # uniform distribution if they are not provided and,
            # if provided, copying them to properly weight the
            # samples according to the bootstrap indexes
            if sample_weight is None:
                curr_sample_weight = np.ones(n_samples, dtype=np.float64)
            else:
                curr_sample_weight = np.array(sample_weight, dtype=np.float64)
            # Obtain the sample weights according to the bootstrap
            # indexes. If the samples are obtained with replacement,
            # then, weight the sample weights by the number of times
            # that each sample appears on the indexes. Otherwise,
            # mantain the sample weight but setting to zero the weight
            # of the samples that have not been obtained in the indexes
            if bootstrap:
                sample_counts = np.bincount(indexes, minlength=n_samples)
                curr_sample_weight *= sample_counts
            else:
                not_indexes_mask = ~_indexes_to_mask(indexes, n_samples)
                curr_sample_weight[not_indexes_mask] = 0
            # Fit the estimator using the sample weight
            estimator.fit(X[:, features], Y, curr_sample_weight)
        # Otherwise, directly fit the
        # estimator using the indexes
        else:
            estimator.fit(X[indexes][:, features], Y[indexes])
        # Append the built estimator to the list
        # of estimators and the features employed
        # to build the estimator to the list of features
        estimators.append(estimator)
        estimators_features.append(features)

    # Return the built estimators
    # and estimators features
    return (estimators, estimators_features)


def _predict_estimators(ensemble, X):
    """Private function used to compute the predictions."""
    # Obtain the prediction of each estimator over all the samples
    predictions = np.array([
        estimator.predict(X[:, features]) for (estimator, features)
        in zip(ensemble.estimators_, ensemble.estimators_features_)])

    # Aggregate the predictions of the estimators on each sample
    predictions = np.array([
        ensemble._rank_algorithm.aggregate(predictions[:, sample])
        for sample in range(X.shape[0])])

    # Return the predictions
    return predictions


# =============================================================================
# Classes
# =============================================================================

# =============================================================================
# Base Bagging
# =============================================================================
class BaseBagging(BaseEnsemble, ABC):
    """Base class for Bagging meta-estimator.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    @abstractmethod
    def __init__(self,
                 base_estimator=None,
                 n_estimators=10,
                 max_samples=1.0,
                 max_features=1.0,
                 bootstrap=True,
                 bootstrap_features=False,
                 random_state=None):
        """Constructor."""
        # Call to the constructor of the parent
        super().__init__(base_estimator=base_estimator,
                         n_estimators=n_estimators)

        # Initialize the hyperparameters
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.random_state = random_state

    def _fit(self, X, Y, max_samples=None, sample_weight=None):
        """Build a Bagging ensemble of estimators from the training set."""
        # Check the training data, the training rankings
        # and, if provided, the sample weights. In fact,
        # they are not stored (only checked) since they
        # will be modified when fitting the estimators
        (X, Y, _) = self._validate_training_data(
            X, Y, sample_weight)

        # Obtain the random state
        random_state = check_random_state(self.random_state)

        # Check the estimator
        self._validate_estimator()

        # Check that the maximum number of samples
        # is an integer type or floating type
        # greater than zero and less than or equal
        # the number of samples on the training dataset
        if isinstance(max_samples, (Integral, np.integer)):
            self._max_samples = max_samples
        elif isinstance(max_samples, (Real, np.floating)):
            self._max_samples = int(max_samples * self.n_samples_)
        else:
            raise TypeError("max_samples must be int or float.")
        if self._max_samples <= 0 or self._max_samples > self.n_samples_:
            raise ValueError("max_samples must be in (0, n_samples].")
        self._max_samples = max(1, int(self._max_samples))

        # Check that the maximum number of features
        # is an integer type or floating type
        # greater than zero and less than or equal
        # the number of features on the training dataset
        if isinstance(self.max_features, (Integral, np.integer)):
            self._max_features = self.max_features
        elif isinstance(self.max_features, (Real, np.floating)):
            self._max_features = self.max_features * self.n_features_
        else:
            raise TypeError("max_features must be int or float.")
        if self._max_features <= 0 or self._max_features > self.n_features_:
            raise ValueError("max_features must be in (0, n_features].")
        self._max_features = max(1, int(self._max_features))

        # Obtain the seeds to ensure that
        # the experiments are reproducible
        self._seeds = random_state.randint(
            MAX_RAND_SEED, size=self.n_estimators)

        # Build the estimators, also obtaining the features used to fit
        # the estimators (since they are needed in the predict method)
        (self.estimators_, self.estimators_features_) = _build_estimators(
            self.n_estimators, self, X, Y, sample_weight, self._seeds)

        # Return the built Bagging ensemble
        return self

    def fit(self, X, Y, sample_weight=None):
        """Build a Bagging ensemble of estimators from the training
        set (X, Y).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The training input samples.

        Y : np.ndarray of shape (n_samples, n_classes)
            The target rankings.

        sample_weight : {None, np.ndarray} of shape (n_samples,),
                optional (default=None)
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if the base estimator supports
            sample weighting.

        Returns
        -------
        self : object
        """
        return self._fit(X, Y, self.max_samples, sample_weight=sample_weight)

    def predict(self, X):
        """Predict rankings for X.

        The predicted ranking of an input sample is the aggregation
        of the predicted rankings for each estimator of the ensemble.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        Y : np.ndarray of shape (n_samples, n_classes)
            The predicted rankings.
        """
        # Check if the model is fitted
        check_is_fitted(self)

        # Check the test data
        X = self._validate_test_data(X)

        # Obtain the predictions (using the private
        # method for future parallelism purposes)
        predictions = _predict_estimators(self, X)

        # Return the predictions
        return predictions


# =============================================================================
# Bagging Label Ranker
# =============================================================================
class BaggingLabelRanker(LabelRankerMixin, BaseBagging):
    """A Bagging Label Ranker.

    A Bagging Label Ranker is an ensemble meta-estimator that fits base
    Label Rankers each on random subsets of the original dataset and then
    aggregate their individual predictions to form a final prediction.
    Such a meta-estimator can typically be used as a way to reduce the
    variance of a black-box estimator (e.g., a decision tree), by introducing
    randomization into its construction procedure and then making an
    ensemble out of it.

    This algorithm encompasses several works from the literature. When random
    subsets of the dataset are drawn as random subsets of the samples, then
    this algorithm is known as Pasting [1]_. If samples are drawn with
    replacement, then the method is known as Bagging [2]_. When random subsets
    of the dataset are drawn as random subsets of the features, then the method
    is known as Random Subspaces [3]_. Finally, when base estimators are built
    on subsets of both samples and features, then the method is known as
    Random Patches [4]_.

    Hyperparamters
    --------------
    base_estimator : {object, None}, optional (default=None)
        The base estimator to fit on random subsets of the dataset.
        If None, then the base estimator is a decision tree.

    n_estimators : int, optional (default=10)
        The number of base estimators in the ensemble.

    max_samples : {int, float}, optional (default=1.0)
        The number of samples to draw from X to train each base estimator.

            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples * X.shape[0]` samples.

    max_features : {int, float}, optional (default=1.0)
        The number of features to draw from X to train each base estimator.

        - If int, then draw `max_features` features.
        - If float, then draw `max_features * X.shape[1]` features.

    bootstrap : bool, optional (default=True)
        Whether samples are drawn with replacement. If False, sampling
        without replacement is performed.

    bootstrap_features : bool, optional (default=False)
        Whether features are drawn with replacement.

    random_state : {int, RandomState instance, None}, optional (default=None)
        - If int, random_state is the seed used by the random number generator.
        - If RandomState instance, random_state is the random number generator.
        - If None, the random number generator is the RandomState instance used
          by `np.random`.

    Attributes
    ----------
    base_estimator_ : estimator
        The base estimator from which the ensemble is grown.

    n_samples_ : int
        The number of samples when ``fit`` is performed.

    n_features_ : int
        The number of features when ``fit`` is performed.

    n_classes_ : int
        The number of classes when ``fit`` is performed.

    estimators_ : list of estimators
        The collection of fitted base estimators.

    estimators_features_ : list of np.ndarrays
        The subset of drawn features for each base estimator.

    See also
    --------
    BaggingPartialLabelRanker

    References
    ----------
    .. [1] `L. Breiman, "Pasting small votes for classification in large
            databases and on-line", Machine Learning, vol. 36, pp. 85-103,
            1999.`_

    .. [2] `L. Breiman, "Bagging predictors", Machine Learning, vol. 24,
            pp. 123-140, 1996.`_

    .. [3] `T. Ho, "The random subspace method for constructing decision
            forests", Pattern Analysis and Machine Intelligence, vol. 20,
            pp. 832-844, 1998.`_

    .. [4] `G. Louppe and P. Geurts, "Ensembles on Random Patches", In
            Proceedings of teh Joint European Conference on Machine Learning
            and Knowledge Discovery in Databases, 2012, pp. 346-361.`_

    .. [5] `Juan A. Aledo, José A. Gámez and D. Molina, "Tackling the
            supervised label ranking problem by bagging weak learners",
            Information Fusion, vol. 35, pp. 38-50, 2017.`_

    Examples
    --------
    >>> import numpy as np
    >>> from sklr.ensemble import BaggingLabelRanker
    >>> X = np.array([[0, 1, 0], [1, 0, 1], [1, 1, 0], [0, 1, 0]])
    >>> Y = np.array([[1, 2, 3], [2, 1, 3], [1, 2, 3], [3, 1, 2]])
    >>> model = BaggingLabelRanker(random_state=0)
    >>> clf = model.fit(X, Y)
    >>> clf.predict(np.array([[0, 1, 0]]))
    np.array([[2, 1, 3]])
    """

    def __init__(self,
                 base_estimator=None,
                 n_estimators=10,
                 max_samples=1.0,
                 max_features=1.0,
                 bootstrap=True,
                 bootstrap_features=False,
                 random_state=None):
        """Constructor."""
        # Call to the constructor of the parent
        super().__init__(base_estimator,
                         n_estimators=n_estimators,
                         max_samples=max_samples,
                         max_features=max_features,
                         bootstrap=bootstrap,
                         bootstrap_features=bootstrap_features,
                         random_state=random_state)

    def _validate_estimator(self):
        """Check the estimator and set the base_estimator_ attribute."""
        # Call to the method of the parent with a Decision
        # Tree Label Ranking as default estimator
        super()._validate_estimator(default=DecisionTreeLabelRanker())


# =============================================================================
# Bagging Partial Label Ranker
# =============================================================================
class BaggingPartialLabelRanker(PartialLabelRankerMixin, BaseBagging):
    """A Bagging Partial Label Ranker.

    A Bagging Partial Label Ranker is an ensemble meta-estimator that fits base
    Partial Label Rankers each on random subsets of the original dataset and
    then aggregate their individual predictions to form a final prediction.
    Such a meta-estimator can typically be used as a way to reduce the variance
    of a black-box estimator (e.g., a decision tree), by introducing
    randomization into its construction procedure and then making an ensemble
    out of it.

    This algorithm encompasses several works from the literature. When random
    subsets of the dataset are drawn as random subsets of the samples, then
    this algorithm is known as Pasting [1]_. If samples are drawn with
    replacement, then the method is known as Bagging [2]_. When random subsets
    of the dataset are drawn as random subsets of the features, then the method
    is known as Random Subspaces [3]_. Finally, when base estimators are built
    on subsets of both samples and features, then the method is known as
    Random Patches [4]_.

    Hyperparamters
    --------------
    base_estimator : {object, None}, optional (default=None)
        The base estimator to fit on random subsets of the dataset.
        If None, then the base estimator is a decision tree.

    n_estimators : int, optional (default=10)
        The number of base estimators in the ensemble.

    max_samples : {int, float}, optional (default=1.0)
        The number of samples to draw from X to train each base estimator.

            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples * X.shape[0]` samples.

    max_features : {int, float}, optional (default=1.0)
        The number of features to draw from X to train each base estimator.

        - If int, then draw `max_features` features.
        - If float, then draw `max_features * X.shape[1]` features.

    bootstrap : bool, optional (default=True)
        Whether samples are drawn with replacement. If False, sampling
        without replacement is performed.

    bootstrap_features : bool, optional (default=False)
        Whether features are drawn with replacement.

    random_state : {int, RandomState instance, None}, optional (default=None)
        - If int, random_state is the seed used by the random number generator.
        - If RandomState instance, random_state is the random number generator.
        - If None, the random number generator is the RandomState instance used
          by `np.random`.

    Attributes
    ----------
    base_estimator_ : estimator
        The base estimator from which the ensemble is grown.

    n_samples_ : int
        The number of samples when ``fit`` is performed.

    n_features_ : int
        The number of features when ``fit`` is performed.

    n_classes_ : int
        The number of classes when ``fit`` is performed.

    estimators_ : list of estimators
        The collection of fitted base estimators.

    estimators_features_ : list of np.ndarrays
        The subset of drawn features for each base estimator.

    See also
    --------
    BaggingLabelRanker

    References
    ----------
    .. [1] `L. Breiman, "Pasting small votes for classification in large
            databases and on-line", Machine Learning, vol. 36, pp. 85-103,
            1999.`_

    .. [2] `L. Breiman, "Bagging predictors", Machine Learning, vol. 24,
            pp. 123-140, 1996.`_

    .. [3] `T. Ho, "The random subspace method for constructing decision
            forests", Pattern Analysis and Machine Intelligence, vol. 20,
            pp. 832-844, 1998.`_

    .. [4] `G. Louppe and P. Geurts, "Ensembles on Random Patches", In
            Proceedings of teh Joint European Conference on Machine Learning
            and Knowledge Discovery in Databases, 2012, pp. 346-361.`_

    Examples
    --------
    >>> import numpy as np
    >>> from sklr.ensemble import BaggingPartialLabelRanker
    >>> X = np.array([[0, 1, 0], [1, 0, 1], [1, 1, 0], [0, 1, 0]])
    >>> Y = np.array([[1, 2, 3], [2, 1, 3], [1, 2, 3], [3, 1, 2]])
    >>> model = BaggingPartialLabelRanker(random_state=0)
    >>> clf = model.fit(X, Y)
    >>> clf.predict(np.array([[0, 1, 0]]))
    np.array([[1, 1, 2]])
    """

    def __init__(self,
                 base_estimator=None,
                 n_estimators=10,
                 max_samples=1.0,
                 max_features=1.0,
                 bootstrap=True,
                 bootstrap_features=False,
                 random_state=None):
        """Constructor."""
        # Call to the constructor of the parent
        super().__init__(base_estimator,
                         n_estimators=n_estimators,
                         max_samples=max_samples,
                         max_features=max_features,
                         bootstrap=bootstrap,
                         bootstrap_features=bootstrap_features,
                         random_state=random_state)

    def _validate_estimator(self):
        """Check the estimator and set the base_estimator_ attribute."""
        # Call to the method of the parent with a Decision
        # Tree Partial Label Ranking as base estimator
        super()._validate_estimator(default=DecisionTreePartialLabelRanker())
