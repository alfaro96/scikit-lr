"""Labelwise decomposition label ranker."""

import warnings
from numbers import Integral
from typing import Literal, Self

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import linear_sum_assignment
from scipy.special import comb
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import check_is_fitted, has_fit_parameter, validate_data

from sklr.base import LabelRankerMixin
from sklr.reduction._ordinal import OrdinalLogisticRegression
from sklr.utils._ranking import _validate_data
from sklr.utils._sklearn_compat import (
    HasMethods,
    StrOptions,
    _check_sample_weight,
    _fit_context,
)


def _observation_likelihood(positions, n_ranked, n_labels):
    """Compute the probability of each observed position given each true position.

    It is equation (16) of [cheng_labelwise_2013] without the sum, for a label
    observed at `positions` in rankings of `n_ranked` labels whose missing labels
    are removed at random.
    """
    true_positions = np.arange(1, n_labels + 1)
    positions = positions[:, np.newaxis]
    n_ranked = n_ranked[:, np.newaxis]
    return (
        comb(true_positions - 1, true_positions - positions)
        * comb(
            n_labels - true_positions, n_labels - n_ranked - true_positions + positions
        )
        / comb(n_labels, n_ranked)
    )


def _fit_label(estimator, X, positions, n_ranked, n_labels, sample_weight):
    """Fit the estimator of a label, or return ``None`` if no sample is informative."""
    if not hasattr(estimator, "fit_likelihood"):
        kwargs = {} if sample_weight is None else {"sample_weight": sample_weight}
        return estimator.fit(X, positions.astype(np.intp), **kwargs)
    # The samples that do not rank the label, or rank it alone, observe it with
    # the same probability whatever its position, so they do not change the
    # likelihood
    is_informative = ~np.isnan(positions) & (n_ranked > 1) & (sample_weight > 0)
    if not np.any(is_informative):
        return None
    return estimator.fit_likelihood(
        X[is_informative],
        _observation_likelihood(
            positions[is_informative], n_ranked[is_informative], n_labels
        ),
        np.arange(1, n_labels + 1),
        sample_weight[is_informative],
    )


def _assign(costs):
    """Find the assignment of labels to positions of minimum total cost.

    Among the assignments of minimum cost, it returns the one that gives the
    smallest position to the first label, then to the second one, and so on. Each
    label in turn takes the smallest free position that still allows an assignment
    of minimum cost, which is checked by solving the assignment of the others.
    """
    n_labels = len(costs)
    labels, ranking = linear_sum_assignment(costs)
    best = costs[labels, ranking].sum()
    # The same assignment may sum its costs in another order, so the totals are
    # compared up to the rounding errors of the sums
    tol = 1e-10 * max(1.0, abs(best))
    free = list(range(n_labels))
    fixed_cost = 0.0
    for label in range(n_labels):
        for position in free:
            if position >= ranking[label]:
                break
            others = [other for other in free if other != position]
            others_costs = costs[label + 1 :][:, others]
            rows, cols = linear_sum_assignment(others_costs)
            total = fixed_cost + costs[label, position] + others_costs[rows, cols].sum()
            if total <= best + tol:
                ranking[label] = position
                ranking[label + 1 :] = np.asarray(others)[cols]
                break
        fixed_cost += costs[label, ranking[label]]
        free.remove(ranking[label])
    return ranking + 1


class LabelwiseLabelRanker(MetaEstimatorMixin, LabelRankerMixin, BaseEstimator):
    """Label ranker that learns the position of each label separately.

    The labelwise decomposition of [cheng_labelwise_2013]_ fits, for each label, a
    probabilistic classifier that predicts the probability of each of its positions,
    and predicts the ranking that minimizes the expected `loss` under those
    probabilities, as if the positions of the labels were independent. The training
    rankings may be incomplete, as if their unranked labels had been removed at
    random from complete rankings, with the default `estimator` or any other that
    implements ``fit_likelihood``.

    Read more in the :ref:`User Guide <labelwise_decomposition>`.

    Parameters
    ----------
    estimator : estimator object, default=None
        Probabilistic classifier fitted to the positions of each label, from ``1``
        to the number of labels, which implements :term:`fit` and
        :term:`predict_proba`. The positions are ordered, so an ordinal classifier
        suits them best. ``None`` means the ordinal logistic regression of
        [cheng_labelwise_2013]_.

        To learn from incomplete rankings, the estimator also has to implement
        ``fit_likelihood(X, likelihood, classes, sample_weight=None)``, as the
        default one does, which is then used instead of ``fit`` for all the
        training rankings. ``likelihood[i, k]`` is the probability of what is
        observed of sample ``i`` if its class is ``classes[k]``, and the method
        maximizes the sum over the samples of the logarithm of
        ``likelihood[i] @ proba[i]``, weighted by ``sample_weight``, where
        ``proba[i]`` holds the probabilities of the classes of sample ``i`` that
        :term:`predict_proba` predicts afterwards. It returns the fitted estimator,
        with ``classes_`` set to ``classes``, which are the positions of the label.
        An estimator without ``fit_likelihood`` only accepts complete rankings, and
        predicts a zero probability for the positions that a label never takes in
        them. If `sample_weight` is given, the method that fits the estimator has
        to support it.

    loss : {"absolute_error", "squared_error"}, default="absolute_error"
        Loss between the true and the predicted position of each label, summed
        over the labels, whose expected value the predicted ranking minimizes.
        ``"absolute_error"`` is the absolute difference, whose sum is an affine
        transformation of Spearman's footrule, and the ranking solves an
        assignment problem. ``"squared_error"`` is the squared difference, whose
        sum is an affine transformation of Spearman's :math:`\\rho`, and the
        ranking sorts the labels by increasing expected position. In both cases,
        among the rankings of minimum expected loss, the predicted one gives the
        smallest position to the first label, then to the second one, and so on.

    n_jobs : int, default=None
        Number of jobs to run in parallel to fit the estimators of the labels.
        ``None`` means 1 unless in a ``joblib.parallel_backend`` context. ``-1``
        means using all processors. See :term:`Glossary <n_jobs>` for more
        details.

    Attributes
    ----------
    estimators_ : list of estimator objects or None
        Estimator fitted to the positions of each label, or ``None`` for a label
        that no training ranking of positive weight ranks together with another
        label, whose positions are predicted as equally likely.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    See Also
    --------
    sklr.neighbors.KNeighborsLabelRanker : Label ranker based on the k-nearest
        neighbors.
    sklr.tree.DecisionTreeLabelRanker : A decision tree label ranker.
    sklearn.multioutput.MultiOutputClassifier : Multi target classification by
        fitting one classifier per target.

    Notes
    -----
    The estimator of each label :math:`y_j` predicts the probability
    :math:`P_j(r \\mid \\boldsymbol{x})` that its position is :math:`r`, for
    :math:`r = 1, \\dots, n`, given an instance :math:`\\boldsymbol{x}`. The
    expected loss of putting :math:`y_j` at position :math:`r` is

    .. math::

        L_j(r) = \\sum_{r'=1}^n \\ell(r', r) P_j(r' \\mid \\boldsymbol{x}),

    with :math:`\\ell(r', r) = |r' - r|` for ``loss="absolute_error"`` and
    :math:`\\ell(r', r) = (r' - r)^2` for ``loss="squared_error"``, and the
    predicted ranking is

    .. math::

        \\hat\\pi = \\arg\\min_{\\pi \\in \\Omega} \\sum_{j=1}^n L_j(\\pi(j)),

    where :math:`\\Omega` is the set of the rankings of the :math:`n` labels. It is
    an assignment of the labels to the positions [cheng_labelwise_2013]_. With the
    absolute difference, it is solved by
    :func:`scipy.optimize.linear_sum_assignment`. With the squared difference,
    :math:`\\sum_{j=1}^n \\pi(j)^2` is the same for all the permutations, so
    :math:`\\hat\\pi` sorts the labels by increasing expected position
    :math:`\\sum_{r=1}^n r P_j(r \\mid \\boldsymbol{x})`.

    The default estimator is the cumulative logit model of equation (20) of
    [cheng_labelwise_2013]_, whose coefficients have the :math:`L_2` penalty of
    :class:`~sklearn.linear_model.LogisticRegression` with its default strength,
    and whose thresholds are not penalized. Its likelihood is maximized by L-BFGS-B, and
    :meth:`fit` issues a :class:`~sklearn.exceptions.ConvergenceWarning` if it does
    not converge within ``100`` iterations.

    An incomplete ranking :math:`\\sigma` ranks the :math:`m` labels of
    :math:`\\mathcal{Y}(\\sigma)`, and if the other :math:`n - m` labels were
    removed at random from a complete ranking, the probability of observing a
    ranked label :math:`y_j` at position :math:`\\sigma(j)` is

    .. math::

        \\sum_{r=\\sigma(j)}^{\\sigma(j) + n - m} P_j(r \\mid \\boldsymbol{x})
        \\frac{\\binom{r - 1}{r - \\sigma(j)}
        \\binom{n - r}{n - m - r + \\sigma(j)}}{\\binom{n}{m}},

    equation (16) of [cheng_labelwise_2013]_. The fractions of binomial coefficients
    are the ``likelihood`` that ``fit_likelihood`` receives, so that the model of each
    label maximizes the weighted log-likelihood of the rankings that rank that
    label, equation (17) of [cheng_labelwise_2013]_, which is the usual one for
    complete rankings. The rankings that do not rank the label, or rank it alone,
    observe it with the same probability for any model, so they are left out, as
    are those of zero weight. If no ranking is left for a label, :meth:`fit` warns
    with a :class:`UserWarning`, and the positions of the label are predicted as
    equally likely.

    References
    ----------
    .. [cheng_labelwise_2013] W. Cheng, S. Henzgen and E. Hüllermeier, "Labelwise
       versus pairwise decomposition in label ranking", in Proceedings of the
       Workshop on Lernen, Wissen & Adaptivität, pp. 129-136, 2013.

    Examples
    --------
    >>> from sklr.reduction import LabelwiseLabelRanker
    >>> X = [[0], [1], [2], [3]]
    >>> y = [[1, 2, 3], [2, 1, 3], [1, 2, 3], [3, 1, 2]]
    >>> ranker = LabelwiseLabelRanker().fit(X, y)
    >>> ranker.predict([[1.1], [2.2]])
    array([[1, 2, 3],
           [2, 1, 3]])

    The training rankings may be incomplete, with ``np.nan`` for the unranked
    labels:

    >>> import numpy as np
    >>> y = [[1, 2, np.nan], [2, 1, 3], [1, np.nan, 2], [3, 1, 2]]
    >>> ranker = LabelwiseLabelRanker().fit(X, y)
    >>> ranker.predict([[1.1], [2.2]])
    array([[2, 1, 3],
           [2, 1, 3]])
    """

    _parameter_constraints: dict = {
        "estimator": [HasMethods(["fit", "predict_proba"]), None],
        "loss": [StrOptions({"absolute_error", "squared_error"})],
        "n_jobs": [Integral, None],
    }

    def __init__(
        self,
        estimator: object | None = None,
        *,
        loss: Literal["absolute_error", "squared_error"] = "absolute_error",
        n_jobs: int | None = None,
    ) -> None:
        self.estimator = estimator
        self.loss = loss
        self.n_jobs = n_jobs

    @_fit_context(
        # LabelwiseLabelRanker.estimator is not validated yet
        prefer_skip_nested_validation=False
    )
    def fit(
        self, X: ArrayLike, y: ArrayLike, sample_weight: ArrayLike | None = None
    ) -> Self:
        """Fit an estimator to the positions of each label.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples, n_labels)
            Training rankings, as described in :ref:`ranking_representation`,
            without ties, and incomplete only if `estimator` implements
            ``fit_likelihood``, as the default one does.

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
            estimator = OrdinalLogisticRegression()
        is_ranked = ~np.isnan(y)
        learns_likelihood = hasattr(estimator, "fit_likelihood")
        if not learns_likelihood:
            if not np.all(is_ranked):
                sample = np.flatnonzero(~np.all(is_ranked, axis=1))[0]
                raise ValueError(
                    "Expected complete rankings in y, because the estimator "
                    f"{estimator!r} has no fit_likelihood method to learn from "
                    f"incomplete rankings, got y[{sample}] = {y[sample]}."
                )
            if sample_weight is not None and not has_fit_parameter(
                estimator, "sample_weight"
            ):
                raise ValueError(
                    f"The estimator {estimator!r} does not support sample weights."
                )
        if sample_weight is not None or learns_likelihood:
            sample_weight = _check_sample_weight(
                sample_weight, X, dtype=np.float64, ensure_non_negative=True
            )
        n_ranked = np.sum(is_ranked, axis=1)
        n_labels = y.shape[1]
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_label)(
                clone(estimator), X, y[:, label], n_ranked, n_labels, sample_weight
            )
            for label in range(n_labels)
        )
        uniform = [label for label, est in enumerate(self.estimators_) if est is None]
        if uniform:
            warnings.warn(
                f"No training ranking of positive weight ranks the labels {uniform} "
                "together with another label, so their positions are predicted as "
                "equally likely.",
                UserWarning,
            )
        return self

    def _predict_position_proba(self, X):
        """Predict the probability of each position of each label of each sample."""
        n_labels = len(self.estimators_)
        proba = np.full((X.shape[0], n_labels, n_labels), 1.0 / n_labels)
        for label, estimator in enumerate(self.estimators_):
            if estimator is not None:
                proba[:, label] = 0.0
                positions = np.asarray(estimator.classes_, dtype=np.intp) - 1
                proba[:, label, positions] = estimator.predict_proba(X)
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
        proba = self._predict_position_proba(X)
        positions = np.arange(1, proba.shape[1] + 1)
        if self.loss == "squared_error":
            expected = proba @ positions
            return rankdata(expected, method="ordinal", axis=1).astype(np.intp)
        costs = proba @ np.abs(positions[:, np.newaxis] - positions)
        return np.array([_assign(cost) for cost in costs], dtype=np.intp)
