"""Weight Boosting.

This module contains weight boosting estimators for both Label Ranking.

The module structure is the following:

- The ``BaseWeightBoosting`` base class implements a common ``fit`` method
  for all the estimators in the module. Label Ranking and Partial Label
  Ranking only differ from each other in the loss function that is optimized.

- ``AdaBoostLabelRanker`` implements adaptive boosting for
  Label Ranking problems.
"""


# =============================================================================
# Imports
# =============================================================================

# Standard
from abc import ABC, abstractmethod
from numbers import Integral, Real

# Third party
import numpy as np

# Local application
from ._base import BaseEnsemble
from ._base import MAX_RAND_SEED
from ..base import LabelRankerMixin, PartialLabelRankerMixin
from ..metrics import kendall_distance as error_lr
from ..tree import DecisionTreeLabelRanker, DecisionTreePartialLabelRanker
from ..utils.validation import (
    check_is_fitted, check_random_state, has_fit_parameter)


# =============================================================================
# Classes
# =============================================================================

# =============================================================================
# Base Weight Boosting
# =============================================================================
class BaseWeightBoosting(BaseEnsemble, ABC):
    """Base class for AdaBoost estimators.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    @abstractmethod
    def __init__(self,
                 base_estimator=None,
                 n_estimators=50,
                 estimator_hyperparams=tuple(),
                 learning_rate=1.0,
                 random_state=None):
        """Constructor."""
        # Call to the constructor of the parent
        super().__init__(base_estimator=base_estimator,
                         n_estimators=n_estimators,
                         estimator_hyperparams=estimator_hyperparams)

        # Initialize the hyperparameters
        self.learning_rate = learning_rate
        self.random_state = random_state

    def fit(self, X, Y, sample_weight=None):
        """Build a weighted boosting from the training set (X, Y).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The training input samples.

        Y : np.ndarray of shape (n_samples, n_classes)
            The target rankings.

        sample_weight : np.ndarray of shape (n_samples,),
                optional (default=None)
            Sample weights. If None, the sample weights are initialized to
            1 / n_samples.

        Returns
        -------
        self : object
        """
        # Validate the training data and rankings
        (X, Y, sample_weight) = self._validate_training_data(
            X, Y, sample_weight)

        # Normalize the provided sample
        # weights to make them sum one
        sample_weight /= np.sum(sample_weight)

        # Obtain the random state
        random_state = check_random_state(self.random_state)

        # Check the estimator
        self._validate_estimator()

        # Check that the learning rate is an
        # integer or floating type greater than zero
        if (not isinstance(self.learning_rate, (Integral, np.integer)) and
                not isinstance(self.learning_rate, (Real, np.floating))):
            raise TypeError("learning_rate must be int or float.")
        elif self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero.")

        # Initialize the estimators, the weight of the
        # estimators an the errors of the estimators, that
        # will be used when obtaining the predictions
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64)

        # Obtain the seeds to ensure that
        # the experiments are reproducible
        self._seeds = random_state.randint(
            MAX_RAND_SEED, size=self.n_estimators)

        # Iteratively build the estimators
        for iboost in range(self.n_estimators):
            # Execute one boosting step, obtaining the sample
            # weight to use in the next iteration and also
            # the weight and error of this estimator
            (sample_weight, estimator_weight, estimator_error) = self._boost(
                iboost, X, Y, sample_weight, random_state)
            # Set the weight of the built estimator and its error
            self.estimator_weights_[iboost] = estimator_weight
            self.estimator_errors_[iboost] = estimator_error
            # Early termination if the estimator has achieved a
            # perfect fit or the base estimator is not good enough
            if estimator_error == 0 or sample_weight is None:
                break
            # Normalize the sample weight to make them sum one
            if iboost < self.n_estimators - 1:
                sample_weight /= np.sum(sample_weight)

        # Return the built AdaBoost ensemble
        return self

    @abstractmethod
    def _boost(self, iboost, X, Y, sample_weight, random_state):
        """Implement a single boost.

        Warning: This method needs to be overridden by subclasses.
        """

    def predict(self, X):
        """Predict rankings for X.

        The predicted ranking of an input sample is the aggregation
        of the predicted rankings for each estimator of the ensemble
        weighting according to the importance of the estimator.

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

        # Obtain the prediction of each estimator over all the samples
        predictions = np.array([
            estimator.predict(X) for estimator in self.estimators_])

        # Aggregate the predictions of the estimators on each sample
        # weighting according to the importance of the estimator
        predictions = np.array([
            self._rank_algorithm.aggregate(
                predictions[:, sample],
                self.estimator_weights_[:len(self.estimators_)])
            for sample in range(X.shape[0])])

        # Return the predictions
        return predictions


# =============================================================================
# AdaBoost Label Ranker
# =============================================================================
class AdaBoostLabelRanker(LabelRankerMixin, BaseWeightBoosting):
    """An AdaBoost Label Ranker.

    AdaBoost [1] Label Ranker is a meta-estimator that begins by fitting
    a Label Ranker on the original dataset and then fits additional copies
    of the Label Ranker on the same dataset but where the weights of
    incorrectly classified instances are adjusted such that subsequent
    Label Rankers focus more on difficult cases.

    Hyperparameters
    ---------------
    base_estimator : object, optional (default=None)
        The base estimator from which the boosted ensemble is built.
        Support for sample weighting is required. If ``None``, then
        the base estimator is ``DecisionTreeLabelRanker(max_depth=3)``.

    n_estimators : int, optional (default=50)
        The maximum number of estimators at which boosting is terminated.
        In case of perfect fit, the learning procedure is stopped early.

    learning_rate : float, optional (default=1.0)
        Learning rate shrinks the contribution of each estimator by
        ``learning_rate``. There is a trade-off between ``learning_rate`` and
        ``n_estimators``.

    random_state : {int, RandomState instance, None}, optional (default=None)
        - If int, random_state is the seed used by the random number generator.
        - If RandomState instance, random_state is the random number generator.
        - If None, the random number generator is the RandomState instance used
          by `np.random`.

    Attributes
    ----------
    base_estimator_ : estimator
        The base estimator from which the ensemble is grown.

    estimators_ : list of estimators
        The collection of fitted sub-estimators.

    n_samples_ : int
        The number of samples when ``fit`` is performed.

    n_features_ : int
        The number of features when ``fit`` is performed.

    n_classes_ : int
        The number of classes when ``fit`` is performed.

    estimator_weights_ : np.ndarray of shape (n_estimators)
        Weights for each estimator in the boosted ensemble.

    estimator_errors_ : np.ndarray of shape (n_estimators)
        Error for each estimator in the boosted ensemble.

    See also
    --------
    DecisionTreeLabelRanker, AdaBoostPartialLabelRanker

    References
    ----------
    .. [1] `Y. Freund and R. Schapire, "A Decision-Theoretic Generalization of
            On-Line Learning and an Application to Boosting", Journal of
            Computer and System Sciences, vol. 55, pp.119-139, 1997.`_

    Examples
    --------
    >>> import numpy as np
    >>> from sklr.ensemble import AdaBoostLabelRanker
    >>> X = np.array([[0, 1, 0], [1, 0, 1], [1, 1, 0], [0, 1, 0]])
    >>> Y = np.array([[1, 2, 3], [2, 1, 3], [1, 2, 3], [3, 1, 2]])
    >>> model = AdaBoostLabelRanker(random_state=0)
    >>> clf = model.fit(X, Y)
    >>> clf.predict(np.array([[0, 1, 0]]))
    np.array([[2, 1, 3]])
    """

    def __init__(self,
                 base_estimator=None,
                 n_estimators=50,
                 learning_rate=1.0,
                 random_state=None):
        """Constructor."""
        # Call to the constructor of the parent
        super().__init__(base_estimator=base_estimator,
                         n_estimators=n_estimators,
                         learning_rate=learning_rate,
                         random_state=random_state)

    def _validate_estimator(self):
        """Check the estimator and set the base_estimator_ attribute."""
        # Call to the method of the parent using
        # as default estimator a Decision Tree
        # Label Ranker with a maximum depth of one
        super()._validate_estimator(
            default=DecisionTreeLabelRanker(max_depth=1))

        # Check that the estimator support sample
        # weighting, raising the corresponding
        # exception when it is not supported
        if not has_fit_parameter(self.base_estimator_, "sample_weight"):
            raise ValueError("{} does not support sample_weight."
                             .format(self.base_estimator_.__class__.__name__))

    def _boost(self, iboost, X, Y, sample_weight, random_state):
        """Implement a single boost for Label Ranking."""
        # Make and fit the estimator
        # using the provided sample weights
        estimator = self._make_estimator(random_state=random_state)
        estimator.fit(X, Y, sample_weight)

        # Obtain the predictions of the
        # estimator over the training dataset
        Y_predict = estimator.predict(X)

        # Complete the rankings to properly evaluate the
        # error of the estimator over the training dataset
        (_, Y_completed) = self._rank_algorithm.aggregate(
            Y, sample_weight, apply_mle=True, return_Yt=True)

        # Obtain the error of the estimator and the
        # error of the estimator each sample of the
        # training dataset to properly update the weights
        (estimator_error, instances_error) = error_lr(
            Y_completed, Y_predict,
            sample_weight=sample_weight, return_dists=True)

        # Early termination if the estimator has achieved a
        # perfect fit or the estimator is not good enough
        if estimator_error <= 0.0:
            estimator_weight = 1.0
            estimator_error = 0.0
        elif estimator_error >= 0.5:
            sample_weight = None
            estimator_weight = 0.0
        # Otherwise, boost the sample weights,
        # that is, update them according to the
        # error of the estimator over all the samples
        else:
            # Obtain the beta parameter and,
            # using it, the weight of the estimator
            beta = estimator_error / (1/estimator_error)
            estimator_weight = self.learning_rate * np.log(1/beta)
            # Boost (update) the weights
            if iboost != self.n_estimators - 1:
                sample_weight *= np.power(
                    beta,
                    (1-instances_error) * self.learning_rate)

        # Return the boost sample weights, the weight of
        # the estimator and the error of the estimator
        return (sample_weight, estimator_weight, estimator_error)
