"""Nearest neighbors partial label ranker."""

from collections.abc import Callable
from typing import Literal, Self

import numpy as np
from joblib import effective_n_jobs
from numpy.typing import ArrayLike
from sklearn.utils import Tags

from sklr._aggregation.bucket_pivot import aggregate_batch
from sklr.base import PartialLabelRankerMixin
from sklr.neighbors._base import _fit_rankings, _kneighbors_weights
from sklr.utils._sklearn_compat import (
    KNeighborsMixin,
    NeighborsBase,
    StrOptions,
    _fit_context,
)


class KNeighborsPartialLabelRanker(
    PartialLabelRankerMixin, KNeighborsMixin, NeighborsBase
):
    """Partial label ranker based on the k-nearest neighbors.

    The bucket order predicted for a sample is the consensus of the partial rankings
    of its `n_neighbors` nearest neighbors in the training set, found by the Bucket
    Pivot Algorithm of [aledo_utopia_2017]_, as in the instance-based partial label
    ranker of [alfaro_learning_2021]_. The training rankings may have ties and be
    incomplete, and the neighbors may be weighted by their distance to the sample.

    Read more in the :ref:`User Guide <partial_label_ranking_neighbors>`.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors of each sample, used by :meth:`predict` and by default
        by :meth:`kneighbors`.

    weights : {"uniform", "distance", "dudani"}, callable or None, \
            default="uniform"
        Weight function of the neighbors. With ``"uniform"``, all the neighbors of a
        sample weigh the same. With ``"distance"``, each neighbor weighs the inverse
        of its distance to the sample, and if some neighbors are at zero distance,
        only they weigh. With ``"dudani"``, the weights follow the rule of
        [dudani_distance_1976]_, which decreases linearly from one for the nearest
        neighbor to zero for the farthest one, and gives one to all the neighbors if
        they are at the same distance. A callable receives an array of distances and
        returns an array of the same shape with non-negative weights. ``None``
        means ``"uniform"``. If all the neighbors of a sample weigh zero, they are
        weighted equally.

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

    n_jobs : int, default=None
        Number of parallel jobs to run for the neighbors search and the consensus
        of the neighbors. ``None`` means 1 unless in a ``joblib.parallel_backend``
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
    KNeighborsLabelRanker : Label ranker based on the k-nearest neighbors.
    sklearn.neighbors.KNeighborsClassifier : Classifier implementing the k-nearest
        neighbors vote.
    sklearn.neighbors.NearestNeighbors : Unsupervised learner for implementing
        neighbor searches.

    Notes
    -----
    Let :math:`\\sigma_1, \\dots, \\sigma_k` be the partial rankings of the
    :math:`k` nearest neighbors of an instance :math:`\\boldsymbol{x}`, with weights
    :math:`w_1, \\dots, w_k` computed from their distances to
    :math:`\\boldsymbol{x}` as described in `weights`. Unlike in
    :class:`KNeighborsLabelRanker` with ``model="mallows"``, the weights are not
    multiplied by the fraction of the labels that each ranking ranks, since the
    consensus does not complete the rankings. For each pair of labels :math:`y_i`
    and :math:`y_j`, the entry :math:`C(y_i, y_j)` of the pair order matrix is the
    weighted fraction of the rankings ranking both labels that put :math:`y_i`
    ahead of :math:`y_j`, with a ranking that ties them counting one half, as in
    equation (3) of [alfaro_learning_2021]_. The predicted bucket order
    :math:`\\hat\\pi` is the solution that the Bucket Pivot Algorithm with the least
    indecision assumption, global DANI indices and two-stage multiple pivots,
    :math:`\\mathrm{LIA}_G^{\\mathrm{MP2}}` in [aledo_utopia_2017]_, gives for this
    matrix, with the threshold :math:`\\beta = 0.25`. When several labels could be
    the pivot, the one with the smaller index is chosen, so the prediction is
    deterministic.

    Only the labels that some neighbor of positive weight ranks together with
    another label are ordered, and :math:`C(y_i, y_j)` is one half for two of them
    that no such neighbor ranks together. The other labels are tied in a last
    bucket, since nothing is known about how they compare with the rest, and if no
    such neighbor ranks two labels, all the labels are tied.

    See :ref:`Nearest Neighbors <sklearn:neighbors>` in the user guide of
    scikit-learn for a discussion of the choice of `algorithm` and `leaf_size`. If
    the :math:`k`-th and the :math:`(k + 1)`-th nearest neighbors of a sample are
    at the same distance but have different rankings, the prediction depends on the
    order of the training data.

    References
    ----------
    .. [aledo_utopia_2017] J. A. Aledo, J. A. Gámez and A. Rosete, "Utopia in the
       solution of the bucket order problem", Decision Support Systems, vol. 97,
       pp. 69-80, 2017.

    .. [alfaro_learning_2021] J. C. Alfaro, J. A. Aledo and J. A. Gámez, "Learning
       decision trees for the partial label ranking problem", International Journal
       of Intelligent Systems, vol. 36, no. 2, pp. 890-918, 2021.

    .. [dudani_distance_1976] S. A. Dudani, "The distance-weighted
       k-nearest-neighbor rule", IEEE Transactions on Systems, Man, and
       Cybernetics, vol. SMC-6, no. 4, pp. 325-327, 1976.

    Examples
    --------
    >>> from sklr.neighbors import KNeighborsPartialLabelRanker
    >>> X = [[0], [1], [2], [3]]
    >>> y = [[1, 2, 2], [1, 1, 2], [1, 2, 3], [2, 1, 3]]
    >>> ranker = KNeighborsPartialLabelRanker(n_neighbors=3)
    >>> ranker.fit(X, y)
    KNeighborsPartialLabelRanker(n_neighbors=3)
    >>> ranker.predict([[0.9], [2.2]])
    array([[1, 2, 3],
           [1, 1, 2]])

    With the weights of Dudani, the nearest neighbor of each sample weighs more:

    >>> ranker = KNeighborsPartialLabelRanker(n_neighbors=3, weights="dudani")
    >>> ranker.fit(X, y).predict([[0.9], [2.2]])
    array([[1, 1, 2],
           [1, 1, 2]])
    """

    _parameter_constraints: dict = {
        **NeighborsBase._parameter_constraints,
        "weights": [StrOptions({"uniform", "distance", "dudani"}), callable, None],
    }
    _parameter_constraints.pop("radius")

    def __init__(
        self,
        n_neighbors: int = 5,
        *,
        weights: Literal["uniform", "distance", "dudani"]
        | Callable[[np.ndarray], np.ndarray]
        | None = "uniform",
        algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "auto",
        leaf_size: int = 30,
        p: float = 2,
        metric: str | Callable = "minkowski",
        metric_params: dict | None = None,
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
        self.weights = weights

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        # For cross-validation routines to split data correctly
        tags.input_tags.pairwise = self.metric == "precomputed"
        return tags

    @_fit_context(
        # KNeighborsPartialLabelRanker.metric is not validated yet
        prefer_skip_nested_validation=False
    )
    def fit(self, X: ArrayLike, y: ArrayLike) -> Self:
        """Fit the k-nearest neighbors partial label ranker from the training dataset.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features) or \
                (n_samples, n_samples) if metric="precomputed"
            Training data.

        y : array-like of shape (n_samples, n_labels)
            Training rankings, as described in :ref:`ranking_representation`,
            possibly with ties and incomplete.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        return _fit_rankings(self, X, y, allow_ties=True)

    def predict(self, X: ArrayLike | None) -> np.ndarray:
        """Predict the bucket orders for the provided data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_queries, n_features), \
                or (n_queries, n_indexed) if metric="precomputed", or None
            Test samples. If ``None``, the bucket orders of the training samples are
            predicted, and no training sample is its own neighbor.

        Returns
        -------
        y : ndarray of shape (n_queries, n_labels)
            Predicted bucket orders, complete and possibly with ties, as the
            positions of the labels.
        """
        neigh_ind, weights = _kneighbors_weights(self, X)
        weights[~np.any(weights > 0, axis=1)] = 1
        return aggregate_batch(
            self._y[neigh_ind], weights, n_threads=effective_n_jobs(self.n_jobs)
        )
