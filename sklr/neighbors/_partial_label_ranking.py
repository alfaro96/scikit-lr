"""Classes for Nearest Neighbors Partial Label Rankers."""


# =============================================================================
# Imports
# =============================================================================

# Local application
from ..base import PartialLabelRankerMixin
from ._base import BaseNeighbors, KNeighborsMixin


# =============================================================================
# Classes
# =============================================================================

class KNeighborsPartialLabelRanker(BaseNeighbors, KNeighborsMixin,
                                   PartialLabelRankerMixin):
    """Partial Label Ranker implementing the k-nearest neighbors vote.

    Hyperparameters
    ---------------
    n_neighbors : int, default=5
        The number of neighbors to use by default for :meth:`k_neighbors`
        queries.

    weights : {"uniform", "distance"}, default="uniform"
        The weight function used in prediction. The possible values are
        "uniform", to weight all points in each neighborhood equally and
        "distance", to weight points by the inverse of their distance, that
        is, closer neighbors of a query point will have a greater influence
        than neighbors which are further away.

    p : int, default=2
        The power parameter for the Minkowski metric. When ``p=1``, this
        is equivalent to using Manhattan distance, and Euclidean distance
        for ``p=2``. For arbitrary ``p``, Minkowski distance is used.

    metric : {"manhattan", "euclidean", "minkowski", "chebyshev"}, \
            default="minkowski"
        The distance metric to use. The default metric is ``"minkowski"``,
        and with ``p=2`` is equivalent to the standard Euclidean metric.

    Notes
    -----
    Regarding the nearest neighbors algorithms, if it is found that two
    neighbors, neighbor `k+1` and `k`, have identical distances but
    different labels, the results will depend on the ordering of the
    training data.

    See also
    --------
    KNeighborsLabelRanker : A K-Nearest Neighbors Label Ranker.

    References
    ----------
    .. [1] `T. Cover and P. Hart, "Nearest neighbor pattern classification",
            IEEE Transactions on Information Theory, vol. 13, pp. 21-27,
            1967.`_

    Examples
    --------
    >>> import numpy as np
    >>> from sklr.neighbors import KNeighborsPartialLabelRanker
    >>> X = np.array([[0], [1], [2], [3]])
    >>> Y = np.array([[1, 2, 2], [2, 1, 3], [1, 1, 2], [3, 1, 2]])
    >>> knn_model = KNeighborsPartialLabelRanker(n_neighbors=3)
    >>> knn_plr = knn_model.fit(X, Y)
    >>> knn_plr.predict(np.array([[1.1], [2.2]]))
    array([[1, 1, 2],
           [2, 1, 2]])
    """

    def __init__(self, n_neighbors=5, weights="uniform",
                 metric="minkowski", p=2):
        """Constructor."""
        super(KNeighborsPartialLabelRanker, self).__init__(
            n_neighbors, weights, metric, p)

    def fit(self, X, Y):
        """Fit the K-Nearest Neighbors Partial Label Ranker on the training
        data and rankings.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features), dtype=np.float64
            The training samples.

        Y : ndarray of shape or (n_samples, n_classes), dtype=np.int64 \
                or dtype=np.float64
            The target rankings.

        Returns
        -------
        self : KNeighborsPartialLabelRanker
            The fitted K-Nearest Neighbors Partial Label Ranker.
        """
        return super(KNeighborsPartialLabelRanker, self).fit(X, Y)
