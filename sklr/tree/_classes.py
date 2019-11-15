"""
This module gathers tree-based methods, including Label Ranking and
Partial Label Ranking trees. Label Ranking and Partial Label Ranking
problems are both handled.
"""


# =============================================================================
# Imports
# =============================================================================

# Standard
from abc import ABC, abstractmethod
from math import ceil
from numbers import Integral

# Third party
import numpy as np

# Local application
from . import _criterion, _splitter, _tree
from ._criterion import Criterion
from ._criterion import DISTANCES
from ._splitter import Splitter
from ._tree import Tree, TreeBuilder
from ..base import (
    BaseEstimator, LabelRankerMixin, PartialLabelRankerMixin)
from ..base import is_label_ranker
from ..utils.ranking import _transform_rankings
from ..utils.validation import check_is_fitted, check_random_state


# =============================================================================
# Constants
# =============================================================================

# The criteria that can be used to compute the impurity
# in a node of a decision tree for Label Ranking and
# Partial Label Ranking targets (respectively)
CRITERIA_LR = {
    "mallows": _criterion.Mallows
}

CRITERIA_PLR = {
    "disagreements": _criterion.Disagreements,
    "distance": _criterion.Distance,
    "entropy": _criterion.Entropy
}

# The splitters that can be used to split
# an internal node in the decision trees
SPLITTERS = {
    "binary": _splitter.BinarySplitter,
    "frequency": _splitter.FrequencySplitter,
    "width": _splitter.WidthSplitter
}


# =============================================================================
# Classes
# =============================================================================

# =============================================================================
# Base decision tree
# =============================================================================
class BaseDecisionTree(BaseEstimator, ABC):
    """Base class for decision trees.

    Warning: This class should not be used directly.
    Use derived classes instead."""

    @abstractmethod
    def __init__(self,
                 criterion,
                 distance,
                 splitter,
                 max_depth,
                 min_samples_split,
                 max_features,
                 max_splits,
                 random_state):
        """Constructor."""
        # Initialize the hyperparameters
        self.criterion = criterion
        self.distance = distance
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.max_splits = max_splits
        self.random_state = random_state

    def get_depth(self):
        """Returns the depth of the decision tree.

        The depth of a tree is the maximum
        distance between the root and any leaf.
        """
        # Check if the tree is fitted
        check_is_fitted(self)

        # Return the depth of the tree (stored in the
        # max_depth attribute of the underlying tree)
        return self.tree_.max_depth

    def get_n_internal(self):
        """Returns the number of internal nodes of the decision tree."""
        # Check if the tree is fitted
        check_is_fitted(self)

        # Return the number of internal nodes of the tree (stored
        # in the internal_count attribute of the underlying tree)
        return self.tree_.internal_count

    def get_n_leaves(self):
        """Returns the number of leaves of the decision tree."""
        # Check if the tree is fitted
        check_is_fitted(self)

        # Return the number of leaves of the tree (stored in
        # the leaf_count attribute of the underlying tree)
        return self.tree_.leaf_count

    def get_n_nodes(self):
        """Returns the number of nodes of the decision tree."""
        # Check if the tree is fitted
        check_is_fitted(self)

        # Return the number of nodes of the tree (just the
        # sum of the number of internal and leaf nodes)
        return self.tree_.internal_count + self.tree_.leaf_count

    def fit(self, X, Y, sample_weight=None):
        """Fit the decision tree on the training data and rankings."""
        # Validate the training data, the training
        # rankings and also the sample weights
        (X, Y, sample_weight) = self._validate_training_data(
            X, Y, sample_weight)

        # Only considers the samples positively weighted. Also,
        # transform the rankings for properly handle them in Cython
        pos_samples_idx = sample_weight > 0
        X = X[pos_samples_idx]
        Y = _transform_rankings(Y[pos_samples_idx])
        sample_weight = sample_weight[pos_samples_idx]

        # Check the random state
        random_state = check_random_state(self.random_state)

        # Check the hyperparameters

        # Ensure that the maximum depth of the tree is greater than zero
        if self.max_depth is None:
            max_depth = np.iinfo(np.int32).max - 1
        else:
            if self.max_depth < 0:
                raise ValueError("max_depth must be greater than zero.")
            max_depth = self.max_depth

        # Ensure that the minimum number of samples to split an internal
        # node is a floating value greater than zero or less than or
        # equal one or an integer value greater than two
        if isinstance(self.min_samples_split, (Integral, np.integer)):
            if self.min_samples_split < 2:
                raise ValueError("min_samples_split must be an integer "
                                 "greater than 1 or a float in (0.0, 1.0]. "
                                 "Got the integer {}."
                                 .format(self.min_samples_split))
            min_samples_split = self.min_samples_split
        else:
            if self.min_samples_split <= 0 or self.min_samples_split > 1:
                raise ValueError("min_samples_split must be an integer "
                                 "greater than 1 or a float in (0.0, 1.0]. "
                                 "Got the float {}."
                                 .format(self.min_samples_split))
            min_samples_split = max(
                2,
                int(ceil(self.min_samples_split * self.n_samples_)))

        # Ensure that the maximum number of features is one of the
        # available possible strings (forcing al least to be one).
        # Also, it is possible to provide a floating value greater
        # than zero or less than or equal one or an integer value
        # greater than zero and less than or equal the number of features
        if isinstance(self.max_features, str):
            if self.max_features == "auto":
                self.max_features_ = max(1, int(np.sqrt(self.n_features_)))
            elif self.max_features == "sqrt":
                self.max_features_ = max(1, int(np.sqrt(self.n_features_)))
            elif self.max_features == "log2":
                self.max_features_ = max(1, int(np.log2(self.n_features_)))
            else:
                raise ValueError("Invalid value for max_features. Allowed "
                                 "string values are 'auto', 'sqrt' or "
                                 "'log2'.")
        elif self.max_features is None:
            self.max_features_ = self.n_features_
        elif isinstance(self.max_features, (Integral, np.integer)):
            if self.max_features <= 0 or self.max_features > self.n_features_:
                raise ValueError("max_features must be in (0, n_features].")
            else:
                self.max_features_ = self.max_features
        else:
            if self.max_features <= 0.0 or self.max_features > 1.0:
                raise ValueError("max_features must be in (0.0, 1.0].")
            else:
                self.max_features_ = max(
                    1, int(self.max_features * self.n_features_))

        # Ensure that the maximum number of splits
        # considered at each internal node it at least two
        if self.max_splits < 2:
            raise ValueError("max_splits must be an integer "
                             "greater or equal than 2.")
        max_splits = self.max_splits

        # Initialize the criterion (taking into account whether this
        # estimator is a Label Ranker or a Partial Label Ranker)
        if is_label_ranker(self):
            criterion = CRITERIA_LR[self.criterion](self._rank_algorithm,
                                                    DISTANCES[self.distance])
        else:
            criterion = CRITERIA_PLR[self.criterion](self._rank_algorithm)

        # Initialize the splitter
        splitter = SPLITTERS[self.splitter](criterion,
                                            self.max_features_,
                                            max_splits,
                                            random_state)

        # Initialize the tree
        self.tree_ = Tree(self.n_features_, self.n_classes_)

        # Initialize the builder
        builder = TreeBuilder(splitter, min_samples_split, max_depth)

        # Initialize the sorted indexes. In fact "np.argsort"
        # is used instead of C++ since it is more efficient
        X_idx_sorted = np.argsort(X, axis=0).T

        # Build the tree
        builder.build(self.tree_, X, Y, sample_weight, X_idx_sorted)

        # Return it
        return self

    def predict(self, X):
        """Predict rankings for X.

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

        # Obtain the predictions using
        # the underlying tree structure
        predictions = self.tree_.predict(X)

        # Return them
        return predictions


# =============================================================================
# Decision tree Label Ranker
# =============================================================================
class DecisionTreeLabelRanker(LabelRankerMixin, BaseDecisionTree):
    """A decision tree Label Ranker.

    Hyperparameters
    ---------------
    criterion : str, optional (default="mallows")
        The function to measure the quality of a split. Supported criterion is
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

    Attributes
    ----------
    max_features_ : int
        The inferred value of max_features.

    n_samples_ : int
        The number of samples when ``fit`` is performed.

    n_features_ : int
        The number of features when ``fit`` is performed.

    n_classes_ : int
        The number of classes when ``fit`` is performed.

    tree_ : Tree object
        The underlying Tree object. Please refer to
        ``help(sklr.tree._tree.Tree)`` for attributes of Tree object.

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
    DecisionTreePartialLabelRanker

    References
    ----------
    .. [1] `L. Breiman, J. Friedman, R. Olshen, and C. Stone, "Classification
            and Regression Trees", Chapman and Hall, 1984.`_

    .. [2] `W. Cheng and J. Hühn and E. Hüllermeier, "Decision tree and
            instance-based learning for label ranking", In Proceedings
            of the 26th International Conference on Machine Learning,
            2009, pp. 161-168.`_

    Examples
    --------
    >>> import numpy as np
    >>> from sklr.tree import DecisionTreeLabelRanker
    >>> X = np.array([[0, 1, 0], [1, 0, 1], [1, 1, 0], [0, 1, 0]])
    >>> Y = np.array([[1, 2, 3], [2, 1, 3], [1, 2, 3], [3, 1, 2]])
    >>> model = DecisionTreeLabelRanker(random_state=0)
    >>> clf = model.fit(X, Y)
    >>> clf.predict(np.array([[0, 1, 0]]))
    array([[1, 2, 3]])
    """

    def __init__(self,
                 criterion="mallows",
                 distance="kendall",
                 splitter="binary",
                 max_depth=None,
                 min_samples_split=2,
                 max_features=None,
                 max_splits=2,
                 random_state=None):
        """Constructor."""
        # Call to the constructor of the parent
        super().__init__(criterion=criterion,
                         distance=distance,
                         splitter=splitter,
                         max_depth=max_depth,
                         min_samples_split=min_samples_split,
                         max_features=max_features,
                         max_splits=max_splits,
                         random_state=random_state)

    def fit(self, X, Y, sample_weight=None):
        """Build a decision tree Label Ranker from the training set (X, Y).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The training input samples.

        Y : np.ndarray of shape or (n_samples, n_classes)
            The target rankings.

        sample_weight : {None, np.ndarray} of shape (n_samples,),
                optional (default=None)
            Sample weights. If None, then samples are equally weighted.
            Splits that would create child nodes with net zero or negative
            weight are ignored while searching for a split in each node.
            Splits are also ignored if they would result in any single
            class carrying a negative weight in either child node.

        Returns
        -------
        self : object
        """
        # Call to the method of the parent
        super().fit(X, Y, sample_weight)

        # Return the built Label Ranking tree
        return self


# =============================================================================
# Decision tree Partial Label Ranker
# =============================================================================
class DecisionTreePartialLabelRanker(PartialLabelRankerMixin,
                                     BaseDecisionTree):
    """A decision tree Partial Label Ranker.

    Hyperparameters
    ---------------
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

    Attributes
    ----------
    max_features_ : int
        The inferred value of max_features.

    n_samples_ : int
        The number of samples when ``fit`` is performed.

    n_features_ : int
        The number of features when ``fit`` is performed.

    n_classes_ : int
        The number of classes when ``fit`` is performed.

    tree_ : Tree object
        The underlying Tree object. Please refer to
        ``help(sklr.tree._tree.Tree)`` for attributes of Tree object.

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
    DecisionTreeLabelRanker

    References
    ----------
    .. [1] `L. Breiman, J. Friedman, R. Olshen, and C. Stone, "Classification
            and Regression Trees", Chapman and Hall, 1984.`_

    .. [2] `J. C. Alfaro, J. A. Aledo, and J. A. Gámez, "Algoritmos basados en
            árboles de decisión para partial label ranking", In Actas de la
            XVIII Conferencia de la Asociación Española para la Inteligencia
            Artificial, 2018, pp. 15-20.`_

    Examples
    --------
    >>> import numpy as np
    >>> from sklr.tree import DecisionTreePartialLabelRanker
    >>> X = np.array([[0, 1, 0], [1, 0, 1], [1, 1, 0], [0, 1, 0]])
    >>> Y = np.array([[1, 2, 3], [2, 1, 3], [1, 2, 3], [3, 1, 2]])
    >>> model = DecisionTreePartialLabelRanker(random_state=0)
    >>> clf = model.fit(X, Y)
    >>> clf.predict(np.array([[0, 1, 0]]))
    array([[1, 1, 2]])
    """

    def __init__(self,
                 criterion="entropy",
                 splitter="binary",
                 max_depth=None,
                 min_samples_split=2,
                 max_features=None,
                 max_splits=2,
                 random_state=None):
        """Constructor."""
        # Call to the constructor of the parent
        super().__init__(criterion=criterion,
                         distance=None,
                         splitter=splitter,
                         max_depth=max_depth,
                         min_samples_split=min_samples_split,
                         max_features=max_features,
                         max_splits=max_splits,
                         random_state=random_state)

    def fit(self, X, Y, sample_weight=None):
        """Build a decision tree Partial Label Ranker
        from the training set (X, Y).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The training input samples.

        Y : np.ndarray of shape or (n_samples, n_classes)
            The target rankings.

        sample_weight : {None, np.ndarray} of shape (n_samples,),
                optional (default=None)
            Sample weights. If None, then samples are equally weighted.
            Splits that would create child nodes with net zero or negative
            weight are ignored while searching for a split in each node.
            Splits are also ignored if they would result in any single
            class carrying a negative weight in either child node.

        Returns
        -------
        self : object
        """
        # Call to the method of the parent
        super().fit(X, Y, sample_weight)

        # Return the built Partial Label Ranking tree
        return self
