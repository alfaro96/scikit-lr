"""Forest of trees-based ensemble methods.

Those methods include random forests.

The module structure is the following:

- The ``BaseForest`` base class implements a common ``fit`` method for all
  the estimators in the module. The ``fit`` method of the base ``Forest``
  class calls the ``fit`` method of each sub-estimator on random samples
  (with replacement, a.k.a. bootstrap) of the training set.

  The init of the sub-estimator is further delegated to the
  ``BaseEnsemble`` constructor.

- The ``ForestLabelRanker`` and ``ForestPartialLabelRanker`` base classes
  further implement the prediction logic by aggregating the predicted
  outcomes of the sub-estimators.

- The ``RandomForestLabelRanker`` and ``RandomForestPartialLabelRanker``
  derived classes provide the user with concrete implementations of
  the forest ensemble method using classical, deterministic
  ``DecisionTreeLabelRanker`` and ``DecisionTreePartialLabelRanker`` as
  sub-estimator implementations.

Label Ranking and Partial Label Ranking problems are both handled.
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
from ..tree import DecisionTreeLabelRanker, DecisionTreePartialLabelRanker
from ..utils.validation import check_random_state
from ..utils.validation import check_is_fitted, check_random_state


# =============================================================================
# Methods
# =============================================================================

def _get_n_samples_bootstrap(n_samples, max_samples):
    """Get the number of samples in a bootstrap sample."""
    # Check that the number of samples is an integer type
    # greater than or equal one and less than or equal
    # the number of samples. Also, it is possible to
    # provide a floating value greater than zero and
    # less than one. If None, then, use the number
    # of samples of the training dataset
    if max_samples is None:
        return n_samples
    elif isinstance(max_samples, (Integral, np.integer)):
        if max_samples < 1 or max_samples > n_samples:
            raise ValueError("`max_samples` must be in range "
                             "[1, {}]. Got {}."
                             .format(n_samples, max_samples))
        return max_samples
    elif isinstance(max_samples, (Real, np.floating)):
        if max_samples <= 0 or max_samples >= 1:
            raise ValueError("`max_samples` must be in range "
                             "(0, 1). Got {}."
                             .format(max_samples))
        return int(round(n_samples * max_samples))
    else:
        raise TypeError("`max_samples` should be int or float. "
                        "Got '{}'."
                        .format(type(max_samples).__name__))


def _generate_sample_indexes(random_state, n_samples, n_samples_bootstrap):
    """Private function used to _build_trees function."""
    # Obtain the random state
    random_state = check_random_state(random_state)

    # Obtain the indexes for the samples taking
    # into account the total number of samples
    # and the number of samples to be taken
    sample_indexes = random_state.randint(0, n_samples, n_samples_bootstrap)

    # Return them
    return sample_indexes


def _build_trees(tree, forest, X, Y, sample_weight, tree_idx, n_trees,
                 n_samples_bootstrap=None):
    """Private function used to fit a single tree."""
    # Initialize the number of samples input data
    n_samples = X.shape[0]

    # If the samples are drawn with replacement, then,
    # weight the sample weights by the number of times
    # that each sample appears on the indexes
    if forest.bootstrap:
        # Check the sample weights, initializing them to an
        # uniform distribution if they are not provided and,
        # if provided, copying them to properly weight the
        # samples according to the bootstrap indexes
        if sample_weight is None:
            curr_sample_weight = np.ones(n_samples, dtype=np.float64)
        else:
            curr_sample_weight = np.array(sample_weight, dtype=np.float64)
        # Obtain the sample weights
        # from to the bootstrap indexes
        indexes = _generate_sample_indexes(tree.random_state, n_samples,
                                           n_samples_bootstrap)
        sample_counts = np.bincount(indexes, minlength=n_samples)
        curr_sample_weight *= sample_counts
        # Fit the estimator using the sample weight
        # obtained from the bootstrap indexes
        tree.fit(X, Y, curr_sample_weight)
    # Otherwise, directly use the sample
    # weight provided in the fit method
    else:
        tree.fit(X, Y, sample_weight)

    # Return the built tree
    return tree


# =============================================================================
# Classes
# =============================================================================

# =============================================================================
# Base Forest
# =============================================================================
class BaseForest(BaseEnsemble, ABC):
    """Base class for forests of trees.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    @abstractmethod
    def __init__(self,
                 base_estimator,
                 n_estimators=100,
                 estimator_hyperparams=tuple(),
                 bootstrap=False,
                 random_state=None,
                 max_samples=None):
        """Constructor."""
        # Call to the constructor of the parent
        super().__init__(base_estimator=base_estimator,
                         n_estimators=n_estimators,
                         estimator_hyperparams=estimator_hyperparams)

        # Initialize the hyperparameters
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.max_samples = max_samples

    def fit(self, X, Y, sample_weight=None):
        """Build a forest of trees from the training set (X, Y).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The training input samples.

        Y : np.ndarray of shape (n_samples, n_classes)
            The target rankings.

        sample_weight : np.ndarray of shape (n_samples,),
                optional, (default=None)
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node.

        Returns
        -------
        self : object
        """
        # Check the training data and rankings
        (X, Y, _) = self._validate_training_data(
            X, Y, sample_weight)

        # Obtain the random state
        random_state = check_random_state(self.random_state)

        # Check the estimator
        self._validate_estimator()

        # Get the sample size
        # for the bootstrap indexes
        n_samples_bootstrap = _get_n_samples_bootstrap(
            n_samples=X.shape[0],
            max_samples=self.max_samples)

        # Generate the seeds to ensure that
        # the experiments are reproducible
        self._seeds = random_state.randint(
            MAX_RAND_SEED, size=self.n_estimators)

        # Configure the trees to the hyperparameters
        self.estimators_ = [
            self._make_estimator(append=False, random_state=self._seeds[i])
            for i in range(self.n_estimators)
        ]

        # Build them (using the private method
        # for future parallelism purposes)
        self.estimators_ = [
            _build_trees(
                t, self, X, Y, sample_weight, i, len(self.estimators_),
                n_samples_bootstrap)
            for (i, t) in enumerate(self.estimators_)
        ]

        # Return the built forest ensemble
        return self

    def predict(self, X):
        """Predict ranking for X.

        The predicted ranking of an input sample is an aggregation
        by the trees in the forest. That is, the predicted ranking
        is the one obtained by aggregating the estimate across the trees.

        Parameters
        ----------
        X: np.ndarray of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        Y: np.ndarray of shape (n_samples, n_classes)
            The predicted rankings.
        """
        # Check if the model is fitted
        check_is_fitted(self)

        # Check the test data
        X = self._validate_test_data(X)

        # Obtain the prediction of each tree over all the samples
        predictions = np.array([
            tree.predict(X) for tree in self.estimators_])

        # Aggregate the predictions of the trees on each sample
        predictions = np.array([
            self._rank_algorithm.aggregate(predictions[:, sample])
            for sample in range(X.shape[0])])

        # Return the predictions
        return predictions


# =============================================================================
# Forest Label Ranker
# =============================================================================
class ForestLabelRanker(LabelRankerMixin, BaseForest, ABC):
    """Base class for forest of trees-based Label Rankers.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    @abstractmethod
    def __init__(self,
                 base_estimator,
                 n_estimators=100,
                 estimator_hyperparams=tuple(),
                 bootstrap=False,
                 random_state=None,
                 max_samples=None):
        """Constructor."""
        # Call to the constructor of the parent
        super().__init__(base_estimator,
                         n_estimators=n_estimators,
                         estimator_hyperparams=estimator_hyperparams,
                         bootstrap=bootstrap,
                         random_state=random_state,
                         max_samples=max_samples)


# =============================================================================
# Forest Partial Label Ranker
# =============================================================================
class ForestPartialLabelRanker(PartialLabelRankerMixin, BaseForest, ABC):
    """Base class for forest of trees-based Partial Label Rankers.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    @abstractmethod
    def __init__(self,
                 base_estimator,
                 n_estimators=100,
                 estimator_hyperparams=tuple(),
                 bootstrap=False,
                 random_state=None,
                 max_samples=None):
        """Constructor."""
        # Call to the constructor of the parent
        super().__init__(base_estimator,
                         n_estimators=n_estimators,
                         estimator_hyperparams=estimator_hyperparams,
                         bootstrap=bootstrap,
                         random_state=random_state,
                         max_samples=max_samples)


# =============================================================================
# Random Forest Label Ranker
# =============================================================================
class RandomForestLabelRanker(ForestLabelRanker):
    """A random forest Label Ranker.

    A random forest is a meta estimator that fits a number of decision tree
    Label Rankers on various sub-samples of the dataset and uses aggregating to
    improve the predictive power and control over-fitting.
    The sub-sample size is always the same as the original
    input sample size but the samples are drawn with replacement if
    `bootstrap=True` (default).

    Hyperparameters
    ---------------
    n_estimators : int, optional (default=100)
        The number of trees in the forest.

    criterion : str, optional (default="mallows")
        The function to measure the quality of a split. Supported criteria are
        "mallows" for the Mallows impurity.

    distance : str, optional (default="kendall")
        The distance function to measure the proximity between rankings.
        Supported distances are "kendall" for the Kendall distance.
        This is only employed if ``criterion="mallows"``.

    splitter : str, optional (default="binary")
        The strategy used to choose the split at each node. Supported
        strategies are "binary" to choose the best binary split, "width"
        to choose the best equal-width split and "frequency" to choose
        the best equal-frequency split.

    max_depth : {int, None}, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : {int, float}, optional (default=2)
        The minimum number of samples required to split an internal node:

            - If int, then consider `min_samples_split` as the minimum number.
            - If float, then `min_samples_split` is a fraction and
              `ceil(min_samples_split * n_samples)` are the minimum
              number of samples for each split.

    max_features : {int, float, str, None}, optional (default=None)
        The number of features to consider when looking for the best split:

            - If int, then consider `max_features` features at each split.
            - If float, then `max_features` is a fraction and
              `int(max_features * n_features)` features are considered at each
              split.
            - If "auto", then `max_features=sqrt(n_features)`.
            - If "sqrt", then `max_features=sqrt(n_features)`.
            - If "log2", then `max_features=log2(n_features)`.
            - If `None`, then `max_features=n_features`.

        Note: The search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_splits : int, optional (default=2)
        The maximum number of splits.

    random_state : {int, RandomState instance, None}, optional (default=None)
        - If int, random_state is the seed used by the random number generator.
        - If RandomState instance, random_state is the random number generator.
        - If None, the random number generator is the RandomState instance used
          by `np.random`.

     max_samples: {int, float}, optional (default=None)
        If bootstrap is True, the number of samples to draw from X
        to train each base estimator.

        - If None (default), then draw `X.shape[0]` samples.
        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` samples. Thus,
          `max_samples` should be in the interval `(0, 1)`.

    Attributes
    ----------
    base_estimator_ : DecisionTreeLabelRanker
        The child estimator template used to create the collection of fitted
        sub-estimators.

    estimators_ : list of DecisionTreeLabelRanker
        The collection of fitted sub-estimators.

    n_samples_ : int
        The number of samples when ``fit`` is performed.

    n_features_ : int
        The number of features when ``fit`` is performed.

    n_classes_ : int
        The number of classes when ``fit`` is performed.

    Notes
    -----
    The default values for the hyperparameters controlling the size of the
    trees (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown
    and unpruned trees which can potentially be very large on some data sets.
    To reduce memory consumption, the complexity and size of the trees should
    be controlled by setting those hyperparameter values.

    The features are always randomly permuted at each split. Therefore,
    the best found split may vary, even with the same training data and
    ``max_features=n_features``, if the improvement of the criterion is
    identical for several splits enumerated during the search of the best
    split. To obtain a deterministic behaviour during fitting,
    ``random_state`` has to be fixed.

    See also
    --------
    DecisionTreeLabelRanker, RandomForestPartialLabelRanker

    References
    ----------
    .. [1] `L. Breiman, "Random Forests", Machine Learning, vol. 45, pp. 5-32,
            2001.`_

    Examples
    --------
    >>> import numpy as np
    >>> from sklr.ensemble import RandomForestLabelRanker
    >>> X = np.array([[0, 1, 0], [1, 0, 1], [1, 1, 0], [0, 1, 0]])
    >>> Y = np.array([[1, 2, 3], [2, 1, 3], [1, 2, 3], [3, 1, 2]])
    >>> model = RandomForestLabelRanker(random_state=0)
    >>> clf = model.fit(X, Y)
    >>> clf.predict(np.array([[0, 1, 0]]))
    np.array([[2, 1, 3]])
    """

    def __init__(self,
                 n_estimators=100,
                 criterion="mallows",
                 distance="kendall",
                 splitter="binary",
                 max_depth=None,
                 min_samples_split=2,
                 max_features="auto",
                 bootstrap=True,
                 random_state=None,
                 max_samples=None):
        """Constructor."""
        # Call to the constructor of the parent
        super().__init__(base_estimator=DecisionTreeLabelRanker(),
                         n_estimators=n_estimators,
                         estimator_hyperparams=(
                             "criterion", "distance", "splitter",
                             "max_depth", "min_samples_split",
                             "max_features", "random_state"
                        ),
                         bootstrap=bootstrap,
                         random_state=random_state,
                         max_samples=max_samples)

        # Initialize the hyperparameters of the tree
        self.criterion = criterion
        self.distance = distance
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features


# =============================================================================
# Random Forest Partial Label Ranker
# =============================================================================
class RandomForestPartialLabelRanker(ForestPartialLabelRanker):
    """A random forest Partial Label Ranker.

    A random forest is a meta estimator that fits a number of decision tree
    Partial Label Rankers on various sub-samples of the dataset and uses
    aggregating to improve the predictive power and control over-fitting.
    The sub-sample size is always the same as the original
    input sample size but the samples are drawn with replacement if
    `bootstrap=True` (default).

    Hyperparameters
    ---------------
    n_estimators : int, optional (default=100)
        The number of trees in the forest.

    criterion : str, optional (default="entropy")
        The function to measure the quality of a split. Supported criteria are
        disagreements for disagreements impurity, "distance" for distance
        impurity and "entropy" for the entropy impurity.

    splitter : str, optional (default="binary")
        The strategy used to choose the split at each node. Supported
        strategies are "binary" to choose the best binary split, "width"
        to choose the best equal-width split and "frequency" to choose
        the best equal-frequency split.

    max_depth : {int, None}, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : {int, float}, optional (default=2)
        The minimum number of samples required to split an internal node:

            - If int, then consider `min_samples_split` as the minimum number.
            - If float, then `min_samples_split` is a fraction and
              `ceil(min_samples_split * n_samples)` are the minimum
              number of samples for each split.

    max_features : {int, float, string, None}, optional (default=None)
        The number of features to consider when looking for the best split:

            - If int, then consider `max_features` features at each split.
            - If float, then `max_features` is a fraction and
              `int(max_features * n_features)` features are considered at each
              split.
            - If "auto", then `max_features=sqrt(n_features)`.
            - If "sqrt", then `max_features=sqrt(n_features)`.
            - If "log2", then `max_features=log2(n_features)`.
            - If `None`, then `max_features=n_features`.

        Note: The search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_splits : int, optional (default=2)
        The maximum number of splits.

    random_state : {int, RandomState instance, None}, optional (default=None)
        - If int, random_state is the seed used by the random number generator.
        - If RandomState instance, random_state is the random number generator.
        - If None, the random number generator is the RandomState instance used
          by `np.random`.

     max_samples : {int, float}, optional (default=None)
        If bootstrap is True, the number of samples to draw from X
        to train each base estimator.

        - If None (default), then draw `X.shape[0]` samples.
        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` samples. Thus,
          `max_samples` should be in the interval `(0, 1)`.

    Attributes
    ----------
    base_estimator_ : DecisionTreePartialLabelRanker
        The child estimator template used to create the collection of fitted
        sub-estimators.

    estimators_ : list of DecisionTreePartialLabelRanker
        The collection of fitted sub-estimators.

    n_samples_ : int
        The number of samples when ``fit`` is performed.

    n_features_ : int
        The number of features when ``fit`` is performed.

    n_classes_ : int
        The number of classes when ``fit`` is performed.

    Notes
    -----
    The default values for the hyperparameters controlling the size of the
    trees (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown
    and unpruned trees which can potentially be very large on some data sets.
    To reduce memory consumption, the complexity and size of the trees should
    be controlled by setting those hyperparameter values.

    The features are always randomly permuted at each split. Therefore,
    the best found split may vary, even with the same training data and
    ``max_features=n_features``, if the improvement of the criterion is
    identical for several splits enumerated during the search of the best
    split. To obtain a deterministic behaviour during fitting,
    ``random_state`` has to be fixed.

    See also
    --------
    DecisionTreePartialLabelRanker, RandomForestLabelRanker

    References
    ----------
    .. [1] `L. Breiman, "Random Forests", Machine Learning, vol. 45, pp. 5-32,
            2001.`_

    Examples
    --------
    >>> import numpy as np
    >>> from sklr.ensemble import RandomForestPartialLabelRanker
    >>> X = np.array([[0, 1, 0], [1, 0, 1], [1, 1, 0], [0, 1, 0]])
    >>> Y = np.array([[1, 2, 3], [2, 1, 3], [1, 2, 3], [3, 1, 2]])
    >>> model = RandomForestPartialLabelRanker(random_state=0)
    >>> clf = model.fit(X, Y)
    >>> clf.predict(np.array([[0, 1, 0]]))
    np.array([[1, 1, 1]])
    """

    def __init__(self,
                 n_estimators=100,
                 criterion="entropy",
                 splitter="binary",
                 max_depth=None,
                 min_samples_split=2,
                 max_features="auto",
                 bootstrap=True,
                 random_state=None,
                 max_samples=None):
        """Constructor."""
        # Call to the constructor of the parent
        super().__init__(base_estimator=DecisionTreePartialLabelRanker(),
                         n_estimators=n_estimators,
                         estimator_hyperparams=(
                             "criterion", "splitter",
                             "max_depth", "min_samples_split",
                             "max_features", "random_state"
                         ),
                         bootstrap=bootstrap,
                         random_state=random_state,
                         max_samples=max_samples)

        # Initialize the hyperparameters of the tree
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
