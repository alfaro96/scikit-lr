"""Nearest Neighbor Label Ranking."""


# =============================================================================
# Imports
# =============================================================================

# Third party
import numpy as np

# Local application
from ._base import BaseNeighbors, KNeighborsMixin
from ._base import _get_weights
from ..base import LabelRankerMixin


# =============================================================================
# Classes
# =============================================================================

# =============================================================================
# K-Nearest Neighbors
# =============================================================================
class KNeighborsLabelRanker(BaseNeighbors, KNeighborsMixin, LabelRankerMixin):
    """Label Ranker implementing the k-nearest neighbors vote.

    Hyperparameters
    ---------------
    n_neighbors : int, optional (default=5)
        The number of neighbors to use by
        default for :meth:`kneighbors` queries.

    weights : str, optional (defauult="uniform")
        Weight function used in prediction. Possible values are:

        - "uniform": Uniform weights. All points in each neighborhood
          are weighted equally.
        - "distance": Weight points by the inverse of their distance.
          In this case, closer neighbors of a query point will have a
          greater influence than neighbors which are further away.

    p : int, optional (default=2)
        The power parameter for the Minkowski metric.
        When p=1, this is equivalent to using Manhattan
        distance (l1-norm), and Euclidean distance (l2-norm)
        for p=2. For arbitrary p, Minkowski distance (l_p-norm) is used.

    metric : str, optional (default="minkowski")
        The distance metric to use. The default metric is
        minkowski, and with p=2 is equivalent to the standard Euclidean
        metric. See the documentation of the DistanceMetric class for a
        list of available metrics.
        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square during fit.

    References
    ----------
    .. [1] `T. Cover and P. Hart, "Nearest neighbor pattern classification",
            IEEE Transactions on Information Theory, vol. 13, pp. 21-27,
            1967.`_

    .. [2] `W. Cheng and J. Hühn and E. Hüllermeier, "Decision tree and
            instance-based learning for label ranking", In Proceedings
            of the 26th International Conference on Machine Learning,
            2009, pp. 161-168.`_

    Attributes
    ----------
    n_samples_ : int
        The number of samples when ``fit`` is performed.

    n_features_ : int
        The number of features when ``fit`` is performed.

    n_classes_ : int
        The number of classes when ``fit`` is performed.

    Examples
    --------
    >>> import numpy as np
    >>> from sklr.neighbors import KNeighborsLabelRanker
    >>> X = np.array([[0], [1], [2], [3]])
    >>> Y = np.array([[1, 2, 3], [2, 1, 3], [1, 2, 3], [3, 1, 2]])
    >>> model = KNeighborsLabelRanker(n_neighbors=3)
    >>> clf = model.fit(X, Y)
    >>> clf.predict(np.array([[1.1]]))
    array([[1, 2, 3]])

    See also
    --------
    KNeighborsPartialLabelRanker

    Notes
    -----

    .. warning::

       Regarding the Nearest Neighbors algorithms, if it is found that two
       neighbors, neighbor `k+1` and `k`, have identical distances
       but different labels, the results will depend on the ordering of the
       training data.
    """

    def __init__(self, n_neighbors=5, weights="uniform",
                 p=2, metric="minkowski"):
        """Constructor."""
        # Call to the constructor of the parent
        super().__init__(n_neighbors, weights, p, metric)

    def predict(self, X):
        """Predict the rankings for the provided data.

        Parameters
        ----------
        X : np.ndarray of shape (n_queries, n_features) or
                (n_queries, n_samples) if metric == "precomputed".
            Test samples.

        Returns
        -------
        Y : np.ndarray of shape (n_queries, n_classes)
            Rankings for each data sample.
        """
        # Obtain the nearest neighbors
        # (both, the indexes and the distances)
        # and the rankings for the test samples
        (neigh_ind, neigh_dist) = self.kneighbors(X)
        neigh_Y = self._Y[neigh_ind]

        # Obtain the weight of each nearest neighbor
        # for all the instances in the test samples
        neigh_sample_weight = _get_weights(neigh_Y, neigh_dist, self.weights)

        # Obtain the predictions by aggregating the rankings of the
        # nearest neighbors and taking into account the sample weights
        predictions = np.array([
            self._rank_algorithm.aggregate(Y, sample_weight, apply_mle=True)
            for (Y, sample_weight) in zip(neigh_Y, neigh_sample_weight)])

        # Return them
        return predictions
