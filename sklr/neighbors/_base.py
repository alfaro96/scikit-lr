"""Common code of the nearest neighbors rankers."""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import get_tags

from sklr.utils._ranking import _validate_data
from sklr.utils._sklearn_compat import _get_weights


def _dudani_weights(dist):
    """Weigh the neighbors of each query by the rule of [dudani_distance_1976]."""
    nearest = dist[:, :1]
    farthest = dist[:, -1:]
    span = farthest - nearest
    return np.divide(farthest - dist, span, out=np.ones_like(dist), where=span > 0)


def _fit_rankings(estimator, X, y, *, allow_ties):
    """Validate the training data of a ranker and fit its neighbor search."""
    ensure_all_finite = (
        "allow-nan" if get_tags(estimator).input_tags.allow_nan else True
    )
    X, y = _validate_data(
        estimator,
        X,
        y,
        allow_ties=allow_ties,
        accept_sparse="csr",
        order="C",
        ensure_all_finite=ensure_all_finite,
    )
    # NeighborsBase._fit validates y unless it is given a fitted estimator,
    # and it would reject the unranked labels
    neighbors = NearestNeighbors(
        n_neighbors=estimator.n_neighbors,
        algorithm=estimator.algorithm,
        leaf_size=estimator.leaf_size,
        metric=estimator.metric,
        p=estimator.p,
        metric_params=estimator.metric_params,
        n_jobs=estimator.n_jobs,
    ).fit(X)
    return estimator._fit(neighbors, y)


def _kneighbors_weights(estimator, X):
    """Find the neighbors of each query and weigh them by `estimator.weights`.

    Returns the indices of the neighbors and their weights, both of shape
    ``(n_queries, n_neighbors)``. The weights may all be zero for some queries.
    """
    weights = estimator.weights
    if weights is None or weights == "uniform":
        # The distances are only needed to weigh the neighbors
        neigh_ind = estimator.kneighbors(X, return_distance=False)
        return neigh_ind, np.ones(neigh_ind.shape)
    neigh_dist, neigh_ind = estimator.kneighbors(X)
    if weights == "dudani":
        neigh_weights = _dudani_weights(neigh_dist)
    else:
        neigh_weights = _get_weights(neigh_dist, weights)
    return neigh_ind, np.array(neigh_weights, dtype=np.float64)
