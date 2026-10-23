"""Pairwise decomposition label ranker."""

import math
import warnings
from numbers import Integral
from typing import Self

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import check_is_fitted, has_fit_parameter, validate_data

from sklr.base import LabelRankerMixin
from sklr.utils._ranking import _validate_data
from sklr.utils._sklearn_compat import HasMethods, _check_sample_weight, _fit_context


def _fit_pair(estimator, X, first, second, sample_weight):
    """Fit the estimator of a pair of labels to whether the first one is ahead.

    It returns ``None`` if no sample of positive weight ranks both labels, and a
    :class:`~sklearn.dummy.DummyClassifier` if all of them order the labels in the
    same way, because many classifiers cannot be fitted to a single class.
    """
    is_known = ~np.isnan(first) & ~np.isnan(second)
    if sample_weight is not None:
        is_known &= sample_weight > 0
    if not np.any(is_known):
        return None
    is_ahead = (first[is_known] < second[is_known]).astype(np.intp)
    if np.all(is_ahead == is_ahead[0]):
        estimator = DummyClassifier(strategy="prior")
    kwargs = {} if sample_weight is None else {"sample_weight": sample_weight[is_known]}
    return estimator.fit(X[is_known], is_ahead, **kwargs)


def _votes(proba, n_labels):
    """Sum, for each label, the probabilities that it is ahead of the other labels.

    `proba` holds the probability that the first label of each pair is ahead of the
    second one, with the pairs ordered as :func:`numpy.triu_indices` gives them.
    """
    first, second = np.triu_indices(n_labels, k=1)
    identity = np.eye(n_labels)
    return proba @ identity[first] + (1 - proba) @ identity[second]


class PairwiseLabelRanker(MetaEstimatorMixin, LabelRankerMixin, BaseEstimator):
    """Label ranker that learns the preference between each pair of labels.

    The ranking by pairwise comparison of [hullermeier_label_2008]_ fits, for each
    pair of labels, a probabilistic classifier that predicts the probability that
    the first label is ahead of the second one, and ranks the labels by the sum of
    the probabilities that each one is ahead of the others (weighted voting). The
    training rankings may be incomplete, and each one is only used for the pairs
    of labels that it ranks.

    Read more in the :ref:`User Guide <pairwise_comparison>`.

    Parameters
    ----------
    estimator : estimator object, default=None
        Probabilistic binary classifier fitted to each pair of labels, which
        implements :term:`fit` and :term:`predict_proba`. Its class ``1`` means that
        the first label of the pair is ahead of the second one, and its class ``0``
        that it is behind. ``None`` means a
        :class:`~sklearn.linear_model.LogisticRegression` with its default
        parameters. If `sample_weight` is given, its ``fit`` has to support it.

    n_jobs : int, default=None
        Number of jobs to run in parallel to fit the estimators of the pairs of
        labels. ``None`` means 1 unless in a ``joblib.parallel_backend`` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>` for more
        details.

    Attributes
    ----------
    estimators_ : list of estimator objects or None
        Estimator fitted to each pair of labels :math:`y_i` and :math:`y_j`, with
        :math:`i < j`, in the order of :func:`numpy.triu_indices`, or ``None`` for a
        pair that no training ranking of positive weight ranks, whose labels are
        predicted ahead of each other with probability ``0.5``. A pair that all
        those rankings order in the same way has a
        :class:`~sklearn.dummy.DummyClassifier` that predicts that order.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    See Also
    --------
    LabelwiseLabelRanker : Label ranker that learns the position of each label
        separately.
    sklr.neighbors.KNeighborsLabelRanker : Label ranker based on the k-nearest
        neighbors.
    sklr.tree.DecisionTreeLabelRanker : A decision tree label ranker.
    sklearn.multiclass.OneVsOneClassifier : One-vs-one multiclass strategy.

    Notes
    -----
    The estimator of each pair of labels :math:`y_i` and :math:`y_j`, with
    :math:`i < j`, is fitted to the training rankings that rank both labels, and
    predicts the probability :math:`R_{ij}(\\boldsymbol{x})` that :math:`y_i` is
    ahead of :math:`y_j` given an instance :math:`\\boldsymbol{x}`, with
    :math:`R_{ji}(\\boldsymbol{x}) = 1 - R_{ij}(\\boldsymbol{x})`. The votes of
    each label are

    .. math::

        S_i(\\boldsymbol{x}) = \\sum_{j \\neq i} R_{ij}(\\boldsymbol{x}),

    and the predicted ranking sorts the labels by decreasing votes, equations (3.2)
    to (3.4) of [hullermeier_label_2008]_, with the label of smallest index first on
    ties. If each :math:`R_{ij}(\\boldsymbol{x})` is the true probability that
    :math:`y_i` is ahead of :math:`y_j`, the expected position of :math:`y_i` is
    :math:`n - S_i(\\boldsymbol{x})`, where :math:`n` is the number of labels, and
    the predicted ranking minimizes the expected sum of the squared differences
    between its positions and the true ones, an affine transformation of
    Spearman's :math:`\\rho` (Theorem 3 of [hullermeier_label_2008]_).

    A training ranking that does not rank one of the labels of a pair says nothing
    about their preference, so it is left out of the training data of the pair, as
    are the rankings of zero weight. If no ranking is left for a pair, :meth:`fit`
    warns with a :class:`UserWarning`, and its labels are predicted ahead of each
    other with probability ``0.5``. If all of them order the pair in the same way,
    that order is predicted with probability ``1``.

    References
    ----------
    .. [hullermeier_label_2008] E. Hüllermeier, J. Fürnkranz, W. Cheng and K.
       Brinker, "Label ranking by learning pairwise preferences", Artificial
       Intelligence, vol. 172, no. 16-17, pp. 1897-1916, 2008.

    Examples
    --------
    >>> from sklr.reduction import PairwiseLabelRanker
    >>> X = [[0], [1], [2], [3]]
    >>> y = [[1, 2, 3], [2, 1, 3], [1, 2, 3], [3, 1, 2]]
    >>> ranker = PairwiseLabelRanker().fit(X, y)
    >>> ranker.predict([[0.1], [2.9]])
    array([[1, 2, 3],
           [2, 1, 3]])

    The training rankings may be incomplete, with ``np.nan`` for the unranked
    labels:

    >>> import numpy as np
    >>> y = [[1, 2, np.nan], [2, 1, 3], [1, np.nan, 2], [3, 1, 2]]
    >>> ranker = PairwiseLabelRanker().fit(X, y)
    >>> ranker.predict([[0.1], [2.9]])
    array([[2, 1, 3],
           [2, 1, 3]])
    """

    _parameter_constraints: dict = {
        "estimator": [HasMethods(["fit", "predict_proba"]), None],
        "n_jobs": [Integral, None],
    }

    def __init__(
        self, estimator: object | None = None, *, n_jobs: int | None = None
    ) -> None:
        self.estimator = estimator
        self.n_jobs = n_jobs

    @_fit_context(
        # PairwiseLabelRanker.estimator is not validated yet
        prefer_skip_nested_validation=False
    )
    def fit(
        self, X: ArrayLike, y: ArrayLike, sample_weight: ArrayLike | None = None
    ) -> Self:
        """Fit an estimator to the preference between each pair of labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples, n_labels)
            Training rankings, as described in :ref:`ranking_representation`,
            without ties and possibly incomplete.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights, non-negative. ``None`` means that all the samples weigh
            the same.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X, y = _validate_data(self, X, y)
        estimator = self.estimator
        if estimator is None:
            estimator = LogisticRegression()
        if sample_weight is not None:
            if not has_fit_parameter(estimator, "sample_weight"):
                raise ValueError(
                    f"The estimator {estimator!r} does not support sample weights."
                )
            sample_weight = _check_sample_weight(
                sample_weight, X, dtype=np.float64, ensure_non_negative=True
            )
        pairs = list(zip(*np.triu_indices(y.shape[1], k=1)))
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_pair)(clone(estimator), X, y[:, i], y[:, j], sample_weight)
            for i, j in pairs
        )
        unknown = [
            (int(i), int(j))
            for (i, j), est in zip(pairs, self.estimators_)
            if est is None
        ]
        if unknown:
            warnings.warn(
                "No training ranking of positive weight ranks the pairs of labels "
                f"{unknown}, so the labels of each of them are predicted ahead of "
                "each other with probability 0.5.",
                UserWarning,
            )
        return self

    def _predict_pairwise_proba(self, X):
        """Predict the probability that the first label of each pair is ahead."""
        proba = np.full((X.shape[0], len(self.estimators_)), 0.5)
        for pair, estimator in enumerate(self.estimators_):
            if estimator is not None:
                # A pair fitted to a single class has no column of the other one
                is_ahead = estimator.classes_ == 1
                proba[:, pair] = estimator.predict_proba(X)[:, is_ahead].sum(axis=1)
        return proba

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict the rankings of the samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        y : ndarray of shape (n_samples, n_labels)
            Predicted rankings, complete and without ties, as the positions of the
            labels.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        # The number of pairs is n_labels * (n_labels - 1) / 2
        n_labels = (1 + math.isqrt(1 + 8 * len(self.estimators_))) // 2
        votes = _votes(self._predict_pairwise_proba(X), n_labels)
        return rankdata(-votes, method="ordinal", axis=1).astype(np.intp)
