import dataclasses

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklr._aggregation.tests import _oracles
from sklr._aggregation.tests._oracles import random_partial_rankings
from sklr.metrics import get_scorer
from sklr.neighbors import KNeighborsPartialLabelRanker
from sklr.neighbors.tests.test_label_ranking import (
    _expected_failed_checks,
    _naive_neighbors,
)
from sklr.utils._estimator_checks import parametrize_with_checks

# The example of the docstring, with the partial rankings of four samples of one
# feature
X_EXAMPLE = [[0], [1], [2], [3]]
Y_EXAMPLE = [[1, 2, 2], [1, 1, 2], [1, 2, 3], [2, 1, 3]]


@parametrize_with_checks(
    [
        KNeighborsPartialLabelRanker(n_neighbors=2),
        KNeighborsPartialLabelRanker(n_neighbors=2, weights="dudani"),
        KNeighborsPartialLabelRanker(metric="precomputed"),
    ],
    expected_failed_checks=_expected_failed_checks,
)
def test_estimators(estimator, check):
    check(estimator)


def _naive_weights(dist, weights):
    """Weigh the neighbors of a query one by one, as the docstring describes."""
    if weights == "uniform":
        weights = [1.0] * len(dist)
    elif weights == "distance":
        weights = [1 / d for d in dist]
    elif dist[-1] == dist[0]:
        weights = [1.0] * len(dist)
    else:
        weights = [(dist[-1] - d) / (dist[-1] - dist[0]) for d in dist]
    return weights


def _check_predict_oracle(X_train, X_test, y, n_neighbors, weights):
    """Check the predictions of a ranker against the naive implementations."""
    ranker = KNeighborsPartialLabelRanker(n_neighbors, weights=weights)
    y_pred = ranker.fit(X_train, y).predict(X_test)

    for ranking, (ind, dist) in zip(
        y_pred, _naive_neighbors(X_train, X_test, n_neighbors), strict=True
    ):
        expected = _oracles.aggregate(y[ind], _naive_weights(dist, weights))
        assert_array_equal(ranking, expected)


@pytest.mark.parametrize("weights", ["uniform", "distance", "dudani"])
@pytest.mark.parametrize("missing", [0, 0.3])
def test_predict_oracle(weights, missing):
    # The neighbors are found by brute force, and the consensus of each query is
    # found by the implementation of the Bucket Pivot Algorithm in exact arithmetic
    rng = np.random.RandomState(0)
    n_neighbors, n_labels = 7, 5
    X_train = rng.rand(40, 3)
    X_test = rng.rand(15, 3)
    y = random_partial_rankings(
        len(X_train), n_labels, missing=missing, random_state=rng
    )

    _check_predict_oracle(X_train, X_test, y, n_neighbors, weights)


@pytest.mark.parametrize("weights", ["uniform", "distance", "dudani"])
def test_predict_oracle_tied_distances(weights):
    # All the neighbors of the query are at the same distance
    X_train = [[1, 0], [-1, 0], [0, 1], [0, -1], [5, 5]]
    y = random_partial_rankings(len(X_train), 4, missing=0, random_state=0)
    _check_predict_oracle(np.array(X_train), [[0, 0]], y, 4, weights)


@pytest.mark.parametrize(
    "weights, expected",
    [
        # The neighbors of the first query rank the first label ahead of the second
        # one in two rankings and tie them in the other, so the entry of the pair
        # order matrix is 5/6, above 3/4. Those of the second query rank them in
        # both orders and tie them once, an entry of 1/2
        ("uniform", [[1, 2, 3], [1, 1, 2]]),
        # The Dudani weights are 1, 0.2 and 0 for the first query, which make the
        # entry 0.7/1.2, below 3/4
        ("dudani", [[1, 1, 2], [1, 1, 2]]),
    ],
)
def test_predict_examples(weights, expected):
    ranker = KNeighborsPartialLabelRanker(n_neighbors=3, weights=weights)
    y_pred = ranker.fit(X_EXAMPLE, Y_EXAMPLE).predict([[0.9], [2.2]])
    assert_array_equal(y_pred, expected)
    assert y_pred.dtype == np.intp


@pytest.mark.parametrize(
    "y, expected",
    [
        # A complete ranking, with or without ties, is predicted as it is
        ([2, 1, 2], [2, 1, 2]),
        ([3, 1, 2], [3, 1, 2]),
        # The labels that the ranking does not rank are tied in a last bucket
        ([1, np.nan, 2], [1, 3, 2]),
        ([np.nan, 1, np.nan], [1, 1, 1]),
        ([np.nan, np.nan, np.nan], [1, 1, 1]),
    ],
)
def test_one_neighbor(y, expected):
    # A single training sample is the neighbor of every query
    ranker = KNeighborsPartialLabelRanker(1).fit([[0]], [y])
    assert_array_equal(ranker.predict([[0], [5]]), [expected] * 2)


@pytest.mark.parametrize("weights", ["uniform", "distance", "dudani"])
# A ranking without ties is a partial ranking too
@pytest.mark.parametrize("ranking", [[2, 1, 3, 2, 1], [3, 1, 2, 5, 4]])
def test_predict_same_bucket_order(weights, ranking):
    # If all the neighbors agree, their bucket order is predicted
    rng = np.random.RandomState(0)
    X = rng.rand(10, 2)
    y = np.tile(ranking, (len(X), 1))
    ranker = KNeighborsPartialLabelRanker(4, weights=weights).fit(X, y)
    assert_array_equal(ranker.predict(rng.rand(5, 2)), y[:5])


def test_predict_zero_weights():
    # With the Dudani weights, the farthest neighbor gets a zero weight, so if the
    # nearest one ranks no label, all the labels tie. A callable that gives zero
    # weights to all the neighbors falls back on equal weights
    X = [[0], [1]]
    y = [[np.nan, np.nan, np.nan], [3, 1, 2]]
    ranker = KNeighborsPartialLabelRanker(2, weights="dudani").fit(X, y)
    assert_array_equal(ranker.predict([[0]]), [[1, 1, 1]])
    ranker.set_params(weights=np.zeros_like)
    assert_array_equal(ranker.predict([[0]]), [[3, 1, 2]])


def test_predict_unhashable_callable():
    # A dataclass that compares by value cannot be hashed
    @dataclasses.dataclass
    class InverseDistance:
        def __call__(self, dist):
            return 1 / dist

    rng = np.random.RandomState(0)
    X = rng.rand(20, 2)
    y = random_partial_rankings(len(X), 4, missing=0.2, random_state=rng)
    X_test = rng.rand(5, 2)
    ranker = KNeighborsPartialLabelRanker(5, weights="distance").fit(X, y)
    expected = ranker.predict(X_test)
    ranker.set_params(weights=InverseDistance())
    assert_array_equal(ranker.predict(X_test), expected)


def test_predict_no_ranked_pairs():
    # Without any pair of labels ranked in the neighborhood, all the labels tie
    X = [[0], [1], [5]]
    y = [[1, np.nan, np.nan], [np.nan, np.nan, 1], [3, 2, 1]]
    ranker = KNeighborsPartialLabelRanker(2).fit(X, y)
    assert_array_equal(ranker.predict([[0]]), [[1, 1, 1]])


@pytest.mark.parametrize("weights", ["uniform", "dudani"])
def test_predict_n_jobs(weights):
    rng = np.random.RandomState(0)
    X = rng.rand(60, 4)
    y = random_partial_rankings(len(X), 5, missing=0.3, random_state=rng)
    y_preds = [
        KNeighborsPartialLabelRanker(10, weights=weights, n_jobs=n_jobs)
        .fit(X, y)
        .predict(X)
        for n_jobs in [None, 2, -1]
    ]
    assert_array_equal(y_preds[0], y_preds[1])
    assert_array_equal(y_preds[0], y_preds[2])


def test_predict_invariances():
    # The prediction does not depend on the order of the training samples, which
    # are at different distances from each query, nor on the scale of the weights
    rng = np.random.RandomState(0)
    X = rng.rand(30, 2)
    y = random_partial_rankings(len(X), 4, missing=0.2, random_state=rng)
    X_test = rng.rand(10, 2)
    ranker = KNeighborsPartialLabelRanker(5, weights="dudani")
    y_pred = ranker.fit(X, y).predict(X_test)

    permutation = rng.permutation(len(X))
    assert_array_equal(
        ranker.fit(X[permutation], y[permutation]).predict(X_test), y_pred
    )
    ranker.set_params(weights=lambda dist: 1 / dist)
    expected = ranker.predict(X_test)
    ranker.set_params(weights=lambda dist: 1000 / dist)
    assert_array_equal(ranker.predict(X_test), expected)


def test_predict_none():
    # Without queries, each training sample is predicted from its neighbors
    # but itself
    rng = np.random.RandomState(0)
    X = rng.rand(20, 2)
    y = random_partial_rankings(len(X), 3, missing=0.2, random_state=rng)
    y_pred = KNeighborsPartialLabelRanker(4).fit(X, y).predict(None)
    for i in range(len(X)):
        others = np.delete(np.arange(len(X)), i)
        ranker = KNeighborsPartialLabelRanker(4).fit(X[others], y[others])
        assert_array_equal(y_pred[i], ranker.predict(X[[i]])[0])


def test_precomputed():
    rng = np.random.RandomState(0)
    X = rng.rand(20, 3)
    X_test = rng.rand(5, 3)
    y = random_partial_rankings(len(X), 4, missing=0.2, random_state=rng)
    ranker = KNeighborsPartialLabelRanker(5, weights="dudani")
    expected = ranker.fit(X, y).predict(X_test)

    dist = np.sqrt(((X[:, np.newaxis] - X) ** 2).sum(axis=2))
    dist_test = np.sqrt(((X_test[:, np.newaxis] - X) ** 2).sum(axis=2))
    ranker.set_params(metric="precomputed")
    assert_array_equal(ranker.fit(dist, y).predict(dist_test), expected)


def test_fit_target_dtypes():
    y_int = np.array(Y_EXAMPLE)
    y_float = y_int.astype(np.float64)
    ranker = KNeighborsPartialLabelRanker(3)
    assert_array_equal(
        ranker.fit(X_EXAMPLE, y_int).predict(X_EXAMPLE),
        ranker.fit(X_EXAMPLE, y_float).predict(X_EXAMPLE),
    )


def test_predict_too_many_neighbors():
    ranker = KNeighborsPartialLabelRanker(5).fit(X_EXAMPLE, Y_EXAMPLE)
    with pytest.raises(ValueError, match="Expected n_neighbors <= n_samples_fit"):
        ranker.predict(X_EXAMPLE)


def test_model_selection():
    # The ranker works in a pipeline whose number of neighbors is tuned by
    # cross-validation with a scorer of scikit-lr
    rng = np.random.RandomState(0)
    X = rng.rand(40, 3)
    y = random_partial_rankings(len(X), 4, missing=0, random_state=rng)
    pipeline = make_pipeline(StandardScaler(), KNeighborsPartialLabelRanker())
    search = GridSearchCV(
        pipeline,
        {"kneighborspartiallabelranker__n_neighbors": [3, 5]},
        scoring=get_scorer("tau_x"),
        cv=3,
    )
    search.fit(X, y)
    assert search.best_params_["kneighborspartiallabelranker__n_neighbors"] in {3, 5}
    assert -1 <= search.score(X, y) <= 1
