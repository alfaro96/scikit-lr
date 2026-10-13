"""Nearest neighbors label ranker."""

from collections.abc import Callable
from numbers import Integral, Real
from typing import Literal, Self

import numpy as np
from joblib import effective_n_jobs
from numpy.typing import ArrayLike
from scipy.stats import rankdata
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import get_tags

from sklr._distributions.mallows import estimate_center_batch
from sklr._distributions.plackett_luce import estimate_parameters_batch
from sklr.base import LabelRankerMixin
from sklr.utils._ranking import _validate_data
from sklr.utils._sklearn_compat import (
    Interval,
    KNeighborsMixin,
    NeighborsBase,
    StrOptions,
    _fit_context,
    _get_weights,
)

# The default number of iterations of the estimation of each model
_DEFAULT_MAX_ITER = {"mallows": 100, "plackett_luce": 1000}


def _dudani_weights(dist):
    """Weigh the neighbors of each query by the rule of [dudani_distance_1976]."""
    nearest = dist[:, :1]
    farthest = dist[:, -1:]
    span = farthest - nearest
    return np.divide(farthest - dist, span, out=np.ones_like(dist), where=span > 0)


class KNeighborsLabelRanker(LabelRankerMixin, KNeighborsMixin, NeighborsBase):
    """Label ranker based on the k-nearest neighbors.

    The ranking predicted for a sample is the mode of a probability model of the
    rankings, the Mallows or the Plackett-Luce model, estimated from the rankings of
    its `n_neighbors` nearest neighbors in the training set, as in the
    instance-based label rankers of [cheng_decision_2009]_ and [cheng_label_2010]_.
    The training rankings may be incomplete, and the neighbors may be weighted by
    their distance to the sample.

    Read more in the :ref:`User Guide <neighbors>`.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors of each sample, used by :meth:`predict` and by default
        by :meth:`kneighbors`.

    model : {"mallows", "plackett_luce"}, default="mallows"
        Probability model of the rankings of the neighbors. With ``"mallows"``, the
        predicted ranking is the center ranking of the Mallows model, found by the
        weighted Borda count, and by Algorithm 1 of [cheng_decision_2009]_ when some
        neighbors have incomplete rankings. With ``"plackett_luce"``, the predicted
        ranking sorts the labels by decreasing parameter of the Plackett-Luce model,
        whose maximum likelihood estimate is found by the MM algorithm of
        [hunter_mm_2004]_. In both cases, ties between labels go to the one with the
        smaller index.

    weights : {"uniform", "distance", "dudani"}, callable or None, \
            default="uniform"
        Weight function of the neighbors. With ``"uniform"``, all the neighbors of a
        sample weigh the same. With ``"distance"``, each neighbor weighs the inverse
        of its distance to the sample, and if some neighbors are at zero distance,
        only they weigh. With ``"dudani"``, the weights follow the rule of
        [dudani_distance_1976]_ used by [cheng_decision_2009]_, which decreases
        linearly from one for the nearest neighbor to zero for the farthest one, and
        gives one to all the neighbors if they are at the same distance. A callable
        receives an array of distances and returns an array of the same shape with
        non-negative weights. ``None`` means ``"uniform"``.

        With ``model="mallows"``, the weight of each neighbor is also multiplied by
        the fraction of the labels that its ranking ranks. If all the neighbors of a
        sample weigh zero, they are weighted by that fraction alone with
        ``model="mallows"``, and equally with ``model="plackett_luce"`` or if that
        fraction is also zero.

    algorithm : {"auto", "ball_tree", "kd_tree", "brute"}, default="auto"
        Algorithm used to compute the nearest neighbors, which are found with a
        :class:`~sklearn.neighbors.BallTree` with ``"ball_tree"``, with a
        :class:`~sklearn.neighbors.KDTree` with ``"kd_tree"`` and by brute force
        with ``"brute"``. ``"auto"`` chooses the most appropriate algorithm from the
        data passed to :meth:`fit`. Fitting on sparse input always uses brute force,
        and warns if another algorithm than ``"auto"`` or ``"brute"`` was chosen.

    leaf_size : int, default=30
        Leaf size passed to :class:`~sklearn.neighbors.BallTree` or
        :class:`~sklearn.neighbors.KDTree`. This can affect the speed of the
        construction and query, as well as the memory required to store the tree.
        The optimal value depends on the nature of the problem.

    p : float, default=2
        Power parameter for the Minkowski metric. ``p=1`` gives the Manhattan
        distance and ``p=2`` the Euclidean distance. It must be positive.

    metric : str, DistanceMetric object or callable, default="minkowski"
        Metric to use for distance computation. The default, ``"minkowski"``, gives
        the Euclidean distance with ``p=2``. See the documentation of
        :mod:`scipy.spatial.distance` and the metrics listed in
        :func:`sklearn.metrics.pairwise.distance_metrics` for valid metric values.

        If `metric` is ``"precomputed"``, `X` is assumed to be a distance matrix and
        must be square in :meth:`fit`. `X` may be a sparse graph, in which case
        only its nonzero elements may be considered neighbors.

        If `metric` is a callable, it takes two one-dimensional arrays that represent
        two samples and returns their distance.

    metric_params : dict, default=None
        Additional keyword arguments for the metric function. ``None`` means no
        additional arguments.

    max_iter : int, default=None
        Maximum number of iterations of the estimation of the model of each sample.
        ``None`` means ``100`` with ``model="mallows"`` and ``1000`` with
        ``model="plackett_luce"``. With ``model="mallows"``, the estimation only
        iterates when some neighbor has an incomplete ranking. A
        :class:`~sklearn.exceptions.ConvergenceWarning` is issued if some estimation
        does not converge, which is common with ``model="plackett_luce"`` when the
        rankings of the neighbors agree, because the maximum likelihood estimate
        does not exist.

    tol : float, default=1e-6
        Tolerance of the estimation of the Plackett-Luce model, which stops when no
        parameter changes by more than `tol`. Ignored with ``model="mallows"``.

    n_jobs : int, default=None
        Number of parallel jobs to run for the neighbors search and the estimation
        of the models. ``None`` means 1 unless in a ``joblib.parallel_backend``
        context. ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Attributes
    ----------
    effective_metric_ : str or callable
        Distance metric used. It is the same as the `metric` parameter or a synonym
        of it, such as ``"euclidean"`` if `metric` is ``"minkowski"`` and `p` is
        ``2``.

    effective_metric_params_ : dict
        Additional keyword arguments for the metric function. For most metrics, it
        is the same as the `metric_params` parameter, but it may also contain the
        value of `p` if `effective_metric_` is ``"minkowski"``.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    n_samples_fit_ : int
        Number of samples in the fitted data.

    See Also
    --------
    sklearn.neighbors.KNeighborsClassifier : Classifier implementing the k-nearest
        neighbors vote.
    sklearn.neighbors.KNeighborsRegressor : Regression based on k-nearest
        neighbors.
    sklearn.neighbors.NearestNeighbors : Unsupervised learner for implementing
        neighbor searches.

    Notes
    -----
    Let :math:`d_{(1)} \\leq \\dots \\leq d_{(k)}` be the sorted distances
    :math:`d(\\boldsymbol{x}, \\boldsymbol{x}_i)` of the :math:`k` nearest neighbors
    :math:`\\boldsymbol{x}_1, \\dots, \\boldsymbol{x}_k` of an instance
    :math:`\\boldsymbol{x}`, numbered from the nearest one, whose rankings are
    :math:`\\sigma_1, \\dots, \\sigma_k`. With ``weights="distance"``, the weight of the
    :math:`i`-th nearest neighbor is :math:`w_i = 1 / d_{(i)}`, and with
    ``weights="dudani"`` it is

    .. math::

        w_i = \\frac{d_{(k)} - d_{(i)}}{d_{(k)} - d_{(1)}},

    or ``1`` if :math:`d_{(k)} = d_{(1)}`, as in equation (1) of
    [dudani_distance_1976]_. With ``model="mallows"``, the weight is multiplied by the
    fraction :math:`|\\mathcal{Y}(\\sigma_i)| / n` of the :math:`n` labels ranked by
    :math:`\\sigma_i`, where :math:`\\mathcal{Y}(\\sigma_i)` is the set of labels that
    it ranks, because Algorithm 1 of [cheng_decision_2009]_ completes the incomplete
    rankings, and a completed ranking is less reliable than an observed one. The
    Plackett-Luce model does not complete the rankings, since the probability of an
    incomplete ranking has a closed form [cheng_label_2010]_, so its weights are not
    multiplied by that fraction.

    See :ref:`Nearest Neighbors <sklearn:neighbors>` in the user guide of
    scikit-learn for a discussion of the choice of `algorithm` and `leaf_size`. If
    the :math:`k`-th and the :math:`(k + 1)`-th nearest neighbors of a sample are
    at the same distance but have different rankings, the prediction depends on the
    order of the training data.

    References
    ----------
    .. [cheng_decision_2009] W. Cheng, J. Hühn and E. Hüllermeier, "Decision tree
       and instance-based learning for label ranking", in Proceedings of the 26th
       International Conference on Machine Learning, Montreal, Canada,
       pp. 161-168, 2009.

    .. [cheng_label_2010] W. Cheng, K. Dembczyński and E. Hüllermeier, "Label
       ranking methods based on the Plackett-Luce model", in Proceedings of the
       27th International Conference on Machine Learning, Haifa, Israel,
       pp. 215-222, 2010.

    .. [dudani_distance_1976] S. A. Dudani, "The distance-weighted
       k-nearest-neighbor rule", IEEE Transactions on Systems, Man, and
       Cybernetics, vol. SMC-6, no. 4, pp. 325-327, 1976.

    .. [hunter_mm_2004] D. R. Hunter, "MM algorithms for generalized Bradley-Terry
       models", The Annals of Statistics, vol. 32, no. 1, pp. 384-406, 2004.

    Examples
    --------
    >>> from sklr.neighbors import KNeighborsLabelRanker
    >>> X = [[0], [1], [2], [3]]
    >>> y = [[1, 2, 3], [2, 1, 3], [1, 2, 3], [3, 1, 2]]
    >>> ranker = KNeighborsLabelRanker(n_neighbors=3)
    >>> ranker.fit(X, y)
    KNeighborsLabelRanker(n_neighbors=3)
    >>> ranker.predict([[1.1], [2.2]])
    array([[1, 2, 3],
           [2, 1, 3]])

    With the weights of Dudani, the nearest neighbor of each sample weighs more:

    >>> ranker = KNeighborsLabelRanker(n_neighbors=3, weights="dudani").fit(X, y)
    >>> ranker.predict([[1.1], [2.2]])
    array([[2, 1, 3],
           [1, 2, 3]])
    """

    _parameter_constraints: dict = {
        **NeighborsBase._parameter_constraints,
        "model": [StrOptions({"mallows", "plackett_luce"})],
        "weights": [StrOptions({"uniform", "distance", "dudani"}), callable, None],
        "max_iter": [Interval(Integral, 1, None, closed="left"), None],
        "tol": [Interval(Real, 0, None, closed="neither")],
    }
    _parameter_constraints.pop("radius")

    def __init__(
        self,
        n_neighbors: int = 5,
        *,
        model: Literal["mallows", "plackett_luce"] = "mallows",
        weights: Literal["uniform", "distance", "dudani"]
        | Callable[[np.ndarray], np.ndarray]
        | None = "uniform",
        algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "auto",
        leaf_size: int = 30,
        p: float = 2,
        metric: str | Callable = "minkowski",
        metric_params: dict | None = None,
        max_iter: int | None = None,
        tol: float = 1e-6,
        n_jobs: int | None = None,
    ) -> None:
        super().__init__(
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric=metric,
            p=p,
            metric_params=metric_params,
            n_jobs=n_jobs,
        )
        self.model = model
        self.weights = weights
        self.max_iter = max_iter
        self.tol = tol

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        # For cross-validation routines to split data correctly
        tags.input_tags.pairwise = self.metric == "precomputed"
        return tags

    @_fit_context(
        # KNeighborsLabelRanker.metric is not validated yet
        prefer_skip_nested_validation=False
    )
    def fit(self, X: ArrayLike, y: ArrayLike) -> Self:
        """Fit the k-nearest neighbors label ranker from the training dataset.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features) or \
                (n_samples, n_samples) if metric="precomputed"
            Training data.

        y : array-like of shape (n_samples, n_labels)
            Training rankings, as described in :ref:`ranking_representation`,
            possibly incomplete but without ties.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        ensure_all_finite = "allow-nan" if get_tags(self).input_tags.allow_nan else True
        X, y = _validate_data(
            self,
            X,
            y,
            accept_sparse="csr",
            order="C",
            ensure_all_finite=ensure_all_finite,
        )
        # NeighborsBase._fit validates y unless it is given a fitted estimator,
        # and it would reject the unranked labels
        neighbors = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            metric=self.metric,
            p=self.p,
            metric_params=self.metric_params,
            n_jobs=self.n_jobs,
        ).fit(X)
        return self._fit(neighbors, y)

    def predict(self, X: ArrayLike | None) -> np.ndarray:
        """Predict the rankings for the provided data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_queries, n_features), \
                or (n_queries, n_indexed) if metric="precomputed", or None
            Test samples. If ``None``, the rankings of the training samples are
            predicted, and no training sample is its own neighbor.

        Returns
        -------
        y : ndarray of shape (n_queries, n_labels)
            Predicted rankings, complete and without ties, as the positions of the
            labels.
        """
        if self.weights in {"uniform", None}:
            # The distances are only needed to weigh the neighbors
            neigh_ind = self.kneighbors(X, return_distance=False)
            neigh_dist = None
        else:
            neigh_dist, neigh_ind = self.kneighbors(X)

        y = self._y[neigh_ind]
        weights = self._get_neighbor_weights(neigh_dist, y)
        max_iter = self.max_iter
        if max_iter is None:
            max_iter = _DEFAULT_MAX_ITER[self.model]
        n_threads = effective_n_jobs(self.n_jobs)
        if self.model == "mallows":
            return estimate_center_batch(
                y, weights, max_iter=max_iter, n_threads=n_threads
            )
        parameters = estimate_parameters_batch(
            y, weights, tol=self.tol, max_iter=max_iter, n_threads=n_threads
        )
        return rankdata(-parameters, method="ordinal", axis=1).astype(np.intp)

    def _get_neighbor_weights(self, neigh_dist, y):
        """Weigh the neighbors of each query point, whose rankings are `y`."""
        n_queries, n_neighbors, _ = y.shape
        if self.weights == "dudani":
            weights = _dudani_weights(neigh_dist)
        else:
            weights = _get_weights(neigh_dist, self.weights)
        if weights is None:
            weights = np.ones((n_queries, n_neighbors))
        weights = np.asarray(weights, dtype=np.float64)

        fallbacks = [np.ones((n_queries, n_neighbors))]
        if self.model == "mallows":
            completeness = np.mean(~np.isnan(y), axis=2)
            weights = weights * completeness
            fallbacks.insert(0, completeness)
        for fallback in fallbacks:
            is_zero = ~np.any(weights > 0, axis=1)
            weights[is_zero] = fallback[is_zero]
        return weights
