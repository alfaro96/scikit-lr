"""
The :mod:`sklr.dummy` module includes simple baseline estimators to compare
with other real estimators.
"""


# =============================================================================
# Imports
# =============================================================================

# Standard
from abc import ABC
from abc import abstractmethod

# Third party
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted, _check_sample_weight

# Local application
from .utils import check_array
from .base import LabelRankerMixin, PartialLabelRankerMixin


# =============================================================================
# Constants
# =============================================================================

# Define the set of valid strategy types
VALID_STRATEGIES = {"most_frequent", "constant"}


# =============================================================================
# Classes
# =============================================================================

class BaseDummy(BaseEstimator, ABC):
    """Base class for dummy estimators."""

    @abstractmethod
    def __init__(self, strategy, constant):
        """Constructor."""
        self.strategy = strategy
        self.constant = constant

    def fit(self, X, Y, sample_weight):
        """Fit the dummy estimator on the training data and rankings."""
        (X, Y) = self._validate_data(X, Y, multi_output=True)
        sample_weight = _check_sample_weight(sample_weight, X)

        (_, n_classes) = Y.shape

        if self.strategy not in VALID_STRATEGIES:
            raise ValueError("Unknown strategy type: {0}. Expected one of {1}."
                             .format(self.strategy, list(VALID_STRATEGIES)))

        if self.strategy == "constant":
            if self.constant is None:
                raise ValueError("The constant target ranking has to be "
                                 "specified for the constant strategy.")
            elif self.constant.shape[0] != n_classes:
                raise ValueError("The constant target ranking should have "
                                 "shape {0}.".format(n_classes))
            else:
                self.constant = check_array(
                    self.constant, dtype=np.int64, ensure_2d=False)
                # Re-raise a more informative message when the constant
                # target ranking cannot be managed by the estimator
                try:
                    self._rank_algorithm.check_targets(self.constant[None, :])
                except ValueError:
                    raise ValueError("The constant target ranking is not the "
                                     "target type managed by the estimator.")

        self.ranking_ = self._rank_algorithm.aggregate(Y, sample_weight)

        return self

    def predict(self, X):
        """Predict rankings for the provided data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features), dtype=np.float64
            The input samples.

        Returns
        -------
        Y: ndarray of shape (n_samples, n_classes), dtype=np.int64
            The predicted rankings.
        """
        check_is_fitted(self)

        X = self._validate_data(X, reset=False)

        # Repeat the constant target value for all the input samples
        return np.tile((self.constant if self.strategy == "constant"
                        else self.ranking_), reps=(X.shape[0], 1))


class DummyLabelRanker(LabelRankerMixin, BaseDummy):
    """A Dummy Label Ranker that make predictions using simple rules.

    This Label Ranker is useful as a simple baseline to compare with other
    (real) Label Rankers. Do not use it for real problems.

    Hyperparameters
    ---------------
    strategy : {"most_frequent", "constant"}, default="most_frequent"
        The strategy used to generate predictions. The allowed strategies are
        "most_frequent", to always predict the most frequent ranking in the
        training dataset and "constant", to always predict a constant ranking
        that is provided by the user.

    constant : ndarray of shape (n_classes,), dtype=np.int64
        The explicit constant as predicted by the ``"constant"`` strategy.
        This hyperparameter is useful only for the ``"constant"`` strategy.

    Attributes
    ----------
    ranking_ : ndarray of shape (n_classes,), dtype=np.int64
        The most frequent ranking in the training dataset.

    Examples
    --------
    >>> import numpy as np
    >>> from sklr.dummy import DummyLabelRanker
    >>> X = np.array([[0], [1], [2], [3]])
    >>> Y = np.array([[1, 2, 3], [2, 1, 3], [1, 2, 3], [3, 1, 2]])
    >>> dummy_model = DummyLabelRanker(strategy="most_frequent")
    >>> dummy_lr = dummy_model.fit(X, Y)
    >>> dummy_lr.predict(np.array([[1.1], [2.2]]))
    array([[2, 1, 3],
           [2, 1, 3]])
    """

    def __init__(self, strategy="most_frequent", constant=None):
        """Constructor."""
        super(DummyLabelRanker, self).__init__(strategy, constant)

    def fit(self, X, Y, sample_weight=None):
        """Fit the Dummy Label Ranker on the training data and rankings.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features), dtype=np.float64
            The training samples.

        Y : ndarray of shape or (n_samples, n_classes), dtype=np.int64 \
                or dtype=np.float64
            The target rankings.

        sample_weight : ndarray of shape (n_samples,), dtype=np.float64, \
                default=None
            The sample weights. If ``None``, then samples are equally weighted.

        Returns
        -------
        self : DummyLabelRanker
            The fitted Dummy Label Ranker.
        """
        return super(DummyLabelRanker, self).fit(X, Y, sample_weight)


class DummyPartialLabelRanker(PartialLabelRankerMixin, BaseDummy):
    """A Dummy Partial Label Ranker that make predictions using simple rules.

    This Partial Label Ranker is useful as a simple baseline to compare with
    other (real) Partial Label Rankers. Do not use it for real problems.

    Hyperparameters
    ---------------
    strategy : {"most_frequent", "constant"}, default="most_frequent"
        The strategy used to generate predictions. The allowed strategies are
        "most_frequent", to always predict the most frequent ranking in the
        training dataset and "constant", to always predict a constant ranking
        that is provided by the user.

    constant : ndarray of shape (n_classes,), dtype=np.int64
        The explicit constant as predicted by the ``"constant"`` strategy.
        This hyperparameter is useful only for the ``"constant"`` strategy.

    Attributes
    ----------
    ranking_ : ndarray of shape (n_classes,), dtype=np.int64
        The most frequent ranking in the training dataset.

    Examples
    --------
    >>> import numpy as np
    >>> from sklr.dummy import DummyPartialLabelRanker
    >>> X = np.array([[0], [1], [2], [3]])
    >>> Y = np.array([[1, 2, 2], [2, 1, 3], [1, 1, 2], [3, 1, 2]])
    >>> dummy_model = DummyPartialLabelRanker(strategy="most_frequent")
    >>> dummy_plr = dummy_model.fit(X, Y)
    >>> dummy_plr.predict(np.array([[1.1], [2.2]]))
    array([[1, 1, 2],
           [1, 1, 2]])
    """

    def __init__(self, strategy="most_frequent", constant=None):
        """Constructor."""
        super(DummyPartialLabelRanker, self).__init__(strategy, constant)

    def fit(self, X, Y, sample_weight=None):
        """Fit the Dummy Partial Label Ranker on the training data and rankings.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features), dtype=np.float64
            The training samples.

        Y : ndarray of shape or (n_samples, n_classes), dtype=np.int64 \
                or dtype=np.float64
            The target rankings.

        sample_weight : ndarray of shape (n_samples,), dtype=np.float64, \
                default=None
            The sample weights. If ``None``, then samples are equally weighted.

        Returns
        -------
        self : DummyPartialLabelRanker
            The fitted Dummy Partial Label Ranker.
        """
        return super(DummyPartialLabelRanker, self).fit(X, Y, sample_weight)
