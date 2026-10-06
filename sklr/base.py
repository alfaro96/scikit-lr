"""Base classes for all estimators and various utility functions."""

from numpy.typing import ArrayLike
from sklearn.utils import get_tags

from sklr.metrics import kendall_tau_score, kendall_tau_x_score


class LabelRankerMixin:
    """Mixin class for all label rankers in scikit-lr.

    A label ranker predicts a ranking of the labels for each sample, without ties,
    and learns from (possibly incomplete) rankings without ties. This mixin sets the
    ``estimator_type`` tag to ``"label_ranker"`` and states through the target tags
    that ``fit`` requires a target ``y`` with several outputs, one per label, as
    described in :ref:`ranking_representation`. Its :meth:`score` method computes
    Kendall's :math:`\\tau` between the predicted and the true rankings.

    See Also
    --------
    PartialLabelRankerMixin : Mixin class for all partial label rankers.
    is_label_ranker : Return whether an estimator is a label ranker.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.base import BaseEstimator
    >>> from sklr.base import LabelRankerMixin, is_label_ranker
    >>> # Mixin classes should always be on the left-hand side for a correct MRO
    >>> class MyEstimator(LabelRankerMixin, BaseEstimator):
    ...     def fit(self, X, y):
    ...         self.ranking_ = np.asarray(y)[0]
    ...         return self
    ...     def predict(self, X):
    ...         return np.tile(self.ranking_, (len(X), 1))
    >>> estimator = MyEstimator()
    >>> X = np.array([[1, 2], [2, 3], [3, 4]])
    >>> y = np.array([[1, 2, 3], [2, 1, 3], [1, 3, 2]])
    >>> estimator.fit(X, y).predict(X)
    array([[1, 2, 3],
           [1, 2, 3],
           [1, 2, 3]])
    >>> is_label_ranker(estimator)
    True
    """

    def __sklearn_tags__(self):
        # The mixin comes before an estimator class in the bases of the estimator,
        # which defines the method, but pyrefly only sees the bases of the mixin
        tags = super().__sklearn_tags__()  # pyrefly: ignore[missing-attribute]
        tags.estimator_type = "label_ranker"
        tags.target_tags.required = True
        tags.target_tags.multi_output = True
        tags.target_tags.single_output = False
        return tags

    def score(
        self, X: ArrayLike, y: ArrayLike, sample_weight: ArrayLike | None = None
    ) -> float:
        """Return the mean Kendall's :math:`\\tau` on the given data.

        It is computed with :func:`~sklr.metrics.kendall_tau_score`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples, n_labels)
            True rankings for `X`, complete and without ties.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            Mean Kendall's :math:`\\tau` of ``self.predict(X)`` with respect to `y`.
        """
        y_pred = self.predict(X)  # pyrefly: ignore[missing-attribute]
        return kendall_tau_score(y, y_pred, sample_weight=sample_weight)


class PartialLabelRankerMixin:
    """Mixin class for all partial label rankers in scikit-lr.

    A partial label ranker predicts a ranking of the labels for each sample, where
    some labels may be tied, and learns from (possibly incomplete) rankings that may
    have ties. This mixin sets the ``estimator_type`` tag to ``"partial_label_ranker"``
    and states through the target tags that ``fit`` requires a target ``y`` with
    several outputs, one per label, as described in :ref:`ranking_representation`.
    Its :meth:`score` method computes the :math:`\\tau_x` coefficient between the
    predicted and the true rankings.

    See Also
    --------
    LabelRankerMixin : Mixin class for all label rankers.
    is_partial_label_ranker : Return whether an estimator is a partial label ranker.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.base import BaseEstimator
    >>> from sklr.base import PartialLabelRankerMixin, is_partial_label_ranker
    >>> # Mixin classes should always be on the left-hand side for a correct MRO
    >>> class MyEstimator(PartialLabelRankerMixin, BaseEstimator):
    ...     def fit(self, X, y):
    ...         self.ranking_ = np.asarray(y)[0]
    ...         return self
    ...     def predict(self, X):
    ...         return np.tile(self.ranking_, (len(X), 1))
    >>> estimator = MyEstimator()
    >>> X = np.array([[1, 2], [2, 3], [3, 4]])
    >>> y = np.array([[1, 1, 2], [2, 1, 3], [1, 2, 2]])
    >>> estimator.fit(X, y).predict(X)
    array([[1, 1, 2],
           [1, 1, 2],
           [1, 1, 2]])
    >>> is_partial_label_ranker(estimator)
    True
    """

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()  # pyrefly: ignore[missing-attribute]
        tags.estimator_type = "partial_label_ranker"
        tags.target_tags.required = True
        tags.target_tags.multi_output = True
        tags.target_tags.single_output = False
        return tags

    def score(
        self, X: ArrayLike, y: ArrayLike, sample_weight: ArrayLike | None = None
    ) -> float:
        """Return the mean :math:`\\tau_x` coefficient on the given data.

        It is computed with :func:`~sklr.metrics.kendall_tau_x_score`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples, n_labels)
            True rankings for `X`, complete and possibly with ties.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            Mean :math:`\\tau_x` of ``self.predict(X)`` with respect to `y`.
        """
        y_pred = self.predict(X)  # pyrefly: ignore[missing-attribute]
        return kendall_tau_x_score(y, y_pred, sample_weight=sample_weight)


def is_label_ranker(estimator: object) -> bool:
    """Return ``True`` if the given estimator is (probably) a label ranker.

    The estimator is taken for a label ranker if its ``estimator_type`` tag is
    ``"label_ranker"``, which :class:`LabelRankerMixin` sets.

    Parameters
    ----------
    estimator : estimator instance
        Estimator object to test.

    Returns
    -------
    out : bool
        ``True`` if `estimator` is a label ranker and ``False`` otherwise.

    See Also
    --------
    is_partial_label_ranker : Return whether an estimator is a partial label ranker.

    Examples
    --------
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> from sklr.base import is_label_ranker
    >>> is_label_ranker(DecisionTreeClassifier())
    False
    """
    return get_tags(estimator).estimator_type == "label_ranker"


def is_partial_label_ranker(estimator: object) -> bool:
    """Return ``True`` if the given estimator is (probably) a partial label ranker.

    The estimator is taken for a partial label ranker if its ``estimator_type`` tag is
    ``"partial_label_ranker"``, which :class:`PartialLabelRankerMixin` sets.

    Parameters
    ----------
    estimator : estimator instance
        Estimator object to test.

    Returns
    -------
    out : bool
        ``True`` if `estimator` is a partial label ranker and ``False`` otherwise.

    See Also
    --------
    is_label_ranker : Return whether an estimator is a label ranker.

    Examples
    --------
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> from sklr.base import is_partial_label_ranker
    >>> is_partial_label_ranker(DecisionTreeClassifier())
    False
    """
    return get_tags(estimator).estimator_type == "partial_label_ranker"
