"""Nearest neighbors partial label ranking."""


# =============================================================================
# Imports
# =============================================================================

# Third party
from sklearn.neighbors._base import KNeighborsMixin
from sklearn.neighbors._base import NeighborsBase as BaseNeighbors

# Local application
from ..base import PartialLabelRankerMixin
from ._base import _predict_k_neighbors


# =============================================================================
# Classes
# =============================================================================

class KNeighborsPartialLabelRanker(KNeighborsMixin,
                                   PartialLabelRankerMixin,
                                   BaseNeighbors):
    """:term:`Partial label ranker` implementing the k-nearest neighbors vote.

    Read more in the :ref:`User Guide <k_neighbors>`.

    Parameters
    ----------
    n_neighbors : int, default=5
        The number of neighbors to use by default for :meth:`kneighbors`
        queries.

    weights : {"uniform", "distance"} or callable, default="uniform"
        The weight function used in prediction. The possible values are:

        - "uniform": Uniform weights. All points in each neighborhood are
          weighted equally.

        - "distance": Weigh points by the inverse of their distance.
          In this case, closer neighbors of a query point will have a
          greater influence than neighbors which are further away.

        - callable: A user-defined function which accepts an array of
          distances, and returns an array of the same shape containing
          the weights.

    algorithm : {"auto", "ball_tree", "kd_tree", "brute"}, default="auto"
        The algorithm used to compute the nearest neighbors:

        - "ball_tree" will use :class:`sklearn.neighbors.BallTree`.
        - "kd_tree" will use :class:`sklearn.neighbors.KDTree`.
        - "brute" will use a brute-force search.
        - "auto" will attempt to decide the most appropriate algorithm
          based on the values passed to :meth:`fit` method.

    leaf_size : int, default=30
        The leaf size passed to :class:`sklearn.neighbors.BallTree` or
        :class:`sklearn.neighbors.KDTree`. This can affect the speed of
        the construction and query, as well as the memory required to
        store the tree. The optimal value depends on the nature of the
        problem.

    p : int, default=2
        The power parameter for the Minkowski metric. When ``p=1``, this is
        equivalent to using the Manhattan distace, and Euclidean distance
        for ``p=2``. For arbitrary ``p``, Minkowski distance is used.

    metric : str or callable, default="minkowski"
        The distance metric to use for the tree. The default metric is
        Minkowski, and with ``p=2`` is equivalent to the Euclidean metric.
        See the documentation of :class:`sklearn.neighbors.DistanceMetric`
        for a list of available metrics.

        If ``metric="precomputed"``, ``X`` is assumed to be a distance
        matrix and must be square during :meth:`fit`.

    metric_params : dict, default=None
        The additional keyword arguments for the metric function.

    n_jobs : int, default=None
        The number of parallel jobs to run for neighbors search. ``None``
        means 1 unless in a :obj:`joblib.parallel_backend` context. ``-1``
        means using all processors. See :term:`Glossary <n_jobs>` for more
        details.

        Does not affect :meth:`fit` method.

    Attributes
    ----------
    effective_metric_ : str or callable
        The distance metric used. It will be same as the `metric` parameter
        or a synonym of it, e.g. "euclidean" if ``metric="minkowski"` and
        ``p=2``.

    effective_metric_params_ : dict
        The additional keyword arguments for the metric function. For most
        metrics will be same with `metric_params` parameter, but may also
        contain the `p` parameter value if the `effective_metric_` attribute
        is set to "minkowski".

    n_samples_fit_ : int
        The number of samples in the fitted data.

    See Also
    --------
    KNeighborsLabelRanker : A k-nearest neighbors label ranker.

    Notes
    -----
    Regarding the nearest neighbors algorithms, if it is found that two
    neighbors, neighbor ``k + 1`` and ``k``, have identical distances but
    different :term:`rankings`, the results will depend on the ordering of
    the training data.

    References
    ----------
    .. [1] X. Wu and V. Kumar, "The Top Ten Algorithms in Data Mining",
           Chapman and Hall, 2009.

    .. [2] J. C. Alfaro, J. A. Aledo and J. A. Gámez, "Learning decision tres
           for the partial label ranking problem", International Journal of
           Intelligent Systems, vol. 36, pp. 890-918, 2021.

    Examples
    --------
    >>> from sklr.neighbors import KNeighborsPartialLabelRanker
    >>> X = [[0], [1], [2], [3]]
    >>> Y = [[1, 2, 2], [1, 2, 1], [2, 2, 1], [2, 2, 1]]
    >>> neigh = KNeighborsPartialLabelRanker(n_neighbors=3)
    >>> neigh.fit(X, Y)
    KNeighborsPartialLabelRanker(n_neighbors=3)
    >>> neigh.predict([[1.1]])
    array([[1, 2, 1]])
    """

    def __init__(self,
                 n_neighbors=5,
                 *,
                 weights="uniform",
                 algorithm="auto",
                 leaf_size=30,
                 p=2,
                 metric="minkowski",
                 metric_params=None,
                 n_jobs=None,
                 **kwargs):
        """Constructor."""
        super(KNeighborsPartialLabelRanker, self).__init__(n_neighbors,
                                                           algorithm=algorithm,
                                                           leaf_size=leaf_size,
                                                           p=p,
                                                           metric=metric,
                                                           metric_params=metric_params,  # noqa 
                                                           n_jobs=n_jobs,
                                                           **kwargs)

        self.weights = weights

    def fit(self, X, Y):
        """Fit the k-nearest neighbors :term:`partial label ranker`
        from the training dataset.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or \
                (n_samples, n_samples) if metric="precomputed"
            The training data.

        Y : array-like of shape (n_samples, n_classes)
            The target :term:`partial rankings`.

        Returns
        -------
        self : KNeighborsPartialLabelRanker
            The fitted k-nearest neighbors partial label ranker.
        """
        return super(KNeighborsPartialLabelRanker, self)._fit(X, Y)

    def predict(self, X):
        """Predict the target :term:`partial rankings` for the
        provided data.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features) or \
                (n_queries, n_indexed) if metric="precomputed"
            The test samples.

        Returns
        -------
        Y : ndarray of shape (n_queries, n_classes)
            The target partial rankings for each data sample.
        """
        return _predict_k_neighbors(self, X)
