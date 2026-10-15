"""Dummy rankers that predict the same ranking for every sample, as baselines."""

from numbers import Integral, Real
from typing import Literal, Self

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import rankdata
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

# The modules of the consensus are imported instead of their functions, which
# would otherwise look like public names of this public module
from sklr._aggregation import bucket_pivot
from sklr._distributions import mallows, plackett_luce
from sklr.base import LabelRankerMixin, PartialLabelRankerMixin
from sklr.utils._ranking import _validate_data, check_ranking
from sklr.utils._sklearn_compat import (
    Interval,
    StrOptions,
    _check_sample_weight,
    _fit_context,
    _num_samples,
)


class _BaseDummyRanker(BaseEstimator):
    """Common code of the dummy rankers.

    The subclasses set `_allow_ties` and define `_consensus`, which finds the
    consensus ranking of the validated rankings and sample weights.
    """

    _parameter_constraints: dict = {
        "strategy": [StrOptions({"consensus", "constant"})],
        "constant": ["array-like", None],
    }

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        # As the dummy estimators of scikit-learn, the samples are not validated,
        # so any input is accepted, sparse or not
        tags.input_tags.sparse = True
        tags.no_validation = True
        return tags

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(
        self, X: ArrayLike, y: ArrayLike, sample_weight: ArrayLike | None = None
    ) -> Self:
        """Fit the dummy ranker.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. Only its number of samples, its number of features and
            the names of its features are used.

        y : array-like of shape (n_samples, n_labels)
            Training rankings, as described in :ref:`ranking_representation`,
            possibly incomplete, and with ties only for a partial label ranker.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights, non-negative and not all zero. ``None`` means that all
            the samples weigh the same.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X, y = _validate_data(
            self, X, y, allow_ties=self._allow_ties, skip_check_array=True
        )
        sample_weight = _check_sample_weight(
            sample_weight, X, dtype=np.float64, ensure_non_negative=True
        )
        n_labels = y.shape[1]
        if self.strategy == "constant":
            if self.constant is None:
                raise ValueError(
                    "The constant ranking has to be given when strategy='constant', "
                    "got constant=None."
                )
            if np.ndim(self.constant) != 1 or len(self.constant) != n_labels:
                raise ValueError(
                    f"Expected a constant ranking of {n_labels} labels, as y, got "
                    f"constant={self.constant!r}."
                )
            try:
                constant = check_ranking(
                    np.reshape(self.constant, (1, -1)), allow_ties=self._allow_ties
                )
            except ValueError as error:
                # The message of check_ranking speaks of the first row of a target
                ties = "" if self._allow_ties else " without ties"
                raise ValueError(
                    f"Expected a complete ranking{ties} as constant, with the "
                    f"positions of the labels dense from 1, got "
                    f"constant={self.constant!r}."
                ) from error
            self.ranking_ = constant[0].astype(np.intp)
        else:
            self.ranking_ = self._consensus(y, sample_weight)
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict the same ranking for every sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data. It is only used to count its samples.

        Returns
        -------
        y : ndarray of shape (n_samples, n_labels)
            The ranking in `ranking_`, repeated for every sample.
        """
        check_is_fitted(self)
        return np.tile(self.ranking_, (_num_samples(X), 1))


class DummyLabelRanker(LabelRankerMixin, _BaseDummyRanker):
    """Label ranker that predicts the same ranking for every sample.

    It ignores the input features and predicts the consensus of the training
    rankings or a ranking given by the user. It is useful as a simple baseline to
    compare with other label rankers, and should not be used for real problems.

    Read more in the :ref:`User Guide <dummy_estimators>`.

    Parameters
    ----------
    strategy : {"consensus", "constant"}, default="consensus"
        Strategy to choose the predicted ranking. With ``"consensus"``, it is the
        consensus of the training rankings according to `model`, and with
        ``"constant"``, the ranking given by `constant`.

    constant : array-like of shape (n_labels,), default=None
        Ranking predicted with ``strategy="constant"``, complete and without ties.
        Ignored with ``strategy="consensus"``.

    model : {"mallows", "plackett_luce"}, default="mallows"
        Probability model of the training rankings, used with
        ``strategy="consensus"``. With ``"mallows"``, the consensus is the center
        ranking of the Mallows model, found by the weighted Borda count, and by
        Algorithm 1 of [cheng_decision_2009]_ when some rankings are incomplete.
        With ``"plackett_luce"``, the consensus sorts the labels by decreasing
        parameter of the Plackett-Luce model, whose maximum likelihood estimate is
        found by the MM algorithm of [hunter_mm_2004]_. In both cases, ties between
        labels go to the one with the smaller index.

    max_iter : int, default=None
        Maximum number of iterations of the estimation of the model. ``None`` means
        ``100`` with ``model="mallows"`` and ``1000`` with
        ``model="plackett_luce"``. With ``model="mallows"``, the estimation only
        iterates when some ranking is incomplete. A
        :class:`~sklearn.exceptions.ConvergenceWarning` is issued if the estimation
        does not converge, which is common with ``model="plackett_luce"`` when the
        training rankings agree, because the maximum likelihood estimate does not
        exist.

    tol : float, default=1e-6
        Tolerance of the estimation of the Plackett-Luce model, which stops when no
        parameter changes by more than `tol`. Ignored with ``model="mallows"``.

    Attributes
    ----------
    ranking_ : ndarray of shape (n_labels,)
        Ranking predicted for every sample, as the positions of the labels.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    See Also
    --------
    DummyPartialLabelRanker : Partial label ranker that predicts the same bucket
        order for every sample.
    sklearn.dummy.DummyClassifier : Classifier that makes predictions using simple
        rules.

    References
    ----------
    .. [cheng_decision_2009] W. Cheng, J. Hühn and E. Hüllermeier, "Decision tree
       and instance-based learning for label ranking", in Proceedings of the 26th
       International Conference on Machine Learning, Montreal, Canada,
       pp. 161-168, 2009.

    .. [hunter_mm_2004] D. R. Hunter, "MM algorithms for generalized Bradley-Terry
       models", The Annals of Statistics, vol. 32, no. 1, pp. 384-406, 2004.

    Examples
    --------
    >>> from sklr.dummy import DummyLabelRanker
    >>> X = [[0], [1], [2], [3]]
    >>> y = [[1, 2, 3], [2, 1, 3], [1, 2, 3], [3, 1, 2]]
    >>> ranker = DummyLabelRanker()
    >>> ranker.fit(X, y)
    DummyLabelRanker()
    >>> ranker.predict([[1.1], [2.2]])
    array([[2, 1, 3],
           [2, 1, 3]])
    >>> round(ranker.score(X, y), 2)
    0.5
    """

    _parameter_constraints: dict = {
        **_BaseDummyRanker._parameter_constraints,
        "model": [StrOptions({"mallows", "plackett_luce"})],
        "max_iter": [Interval(Integral, 1, None, closed="left"), None],
        "tol": [Interval(Real, 0, None, closed="neither")],
    }

    _allow_ties = False

    def __init__(
        self,
        *,
        strategy: Literal["consensus", "constant"] = "consensus",
        constant: ArrayLike | None = None,
        model: Literal["mallows", "plackett_luce"] = "mallows",
        max_iter: int | None = None,
        tol: float = 1e-6,
    ) -> None:
        self.strategy = strategy
        self.constant = constant
        self.model = model
        self.max_iter = max_iter
        self.tol = tol

    def _consensus(self, y, sample_weight):
        # Without max_iter, each estimation uses its own default
        max_iter = {} if self.max_iter is None else {"max_iter": self.max_iter}
        if self.model == "mallows":
            return mallows.estimate_center(y, sample_weight, **max_iter)
        parameters = plackett_luce.estimate_parameters(
            y, sample_weight, tol=self.tol, **max_iter
        )
        return rankdata(-parameters, method="ordinal").astype(np.intp)


class DummyPartialLabelRanker(PartialLabelRankerMixin, _BaseDummyRanker):
    """Partial label ranker that predicts the same bucket order for every sample.

    It ignores the input features and predicts the consensus of the training
    rankings or a ranking given by the user. With ``strategy="consensus"``, it is
    the Zero-R baseline of [alfaro_learning_2021]_. It is useful as a simple
    baseline to compare with other partial label rankers, and should not be used
    for real problems.

    Read more in the :ref:`User Guide <dummy_estimators>`.

    Parameters
    ----------
    strategy : {"consensus", "constant"}, default="consensus"
        Strategy to choose the predicted ranking. With ``"consensus"``, it is the
        consensus bucket order of the training rankings, found by the Bucket Pivot
        Algorithm of [aledo_utopia_2017]_ as in [alfaro_learning_2021]_, with the
        labels that no ranking of positive weight ranks together with another
        label tied in a last bucket. With ``"constant"``, it is the ranking given by
        `constant`.

    constant : array-like of shape (n_labels,), default=None
        Ranking predicted with ``strategy="constant"``, complete and possibly with
        ties. Ignored with ``strategy="consensus"``.

    Attributes
    ----------
    ranking_ : ndarray of shape (n_labels,)
        Ranking predicted for every sample, as the positions of the labels.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    See Also
    --------
    DummyLabelRanker : Label ranker that predicts the same ranking for every
        sample.
    sklearn.dummy.DummyClassifier : Classifier that makes predictions using simple
        rules.

    References
    ----------
    .. [aledo_utopia_2017] J. A. Aledo, J. A. Gámez and A. Rosete, "Utopia in the
       solution of the Bucket Order Problem", Decision Support Systems, vol. 97,
       pp. 69-80, 2017.

    .. [alfaro_learning_2021] J. C. Alfaro, J. A. Aledo and J. A. Gámez, "Learning
       decision trees for the partial label ranking problem", International
       Journal of Intelligent Systems, vol. 36, no. 2, pp. 890-918, 2021.

    Examples
    --------
    >>> from sklr.dummy import DummyPartialLabelRanker
    >>> X = [[0], [1], [2], [3]]
    >>> y = [[1, 2, 2], [2, 1, 3], [1, 1, 2], [3, 1, 2]]
    >>> ranker = DummyPartialLabelRanker()
    >>> ranker.fit(X, y)
    DummyPartialLabelRanker()
    >>> ranker.predict([[1.1], [2.2]])
    array([[1, 1, 2],
           [1, 1, 2]])
    """

    _allow_ties = True

    def __init__(
        self,
        *,
        strategy: Literal["consensus", "constant"] = "consensus",
        constant: ArrayLike | None = None,
    ) -> None:
        self.strategy = strategy
        self.constant = constant

    def _consensus(self, y, sample_weight):
        return bucket_pivot.aggregate(y, sample_weight)
