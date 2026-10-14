import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import rankdata
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklr._distributions.tests import _oracles
from sklr._distributions.tests._oracles import random_rankings
from sklr.metrics import get_scorer
from sklr.neighbors import KNeighborsLabelRanker
from sklr.neighbors._base import _dudani_weights
from sklr.utils._estimator_checks import parametrize_with_checks

MODELS = ["mallows", "plackett_luce"]

# The example of the docstring, with the rankings of four samples of one feature
X_EXAMPLE = [[0], [1], [2], [3]]
Y_EXAMPLE = [[1, 2, 3], [2, 1, 3], [1, 2, 3], [3, 1, 2]]


def _expected_failed_checks(estimator):
    # The checks of sparse data expect the prediction of an estimator with several
    # outputs to have a single column, while a ranking has one per label
    failed_checks = {
        "check_estimator_sparse_array": "Expects predictions of shape (n, 1)",
        "check_estimator_sparse_matrix": "Expects predictions of shape (n, 1)",
    }
    if estimator.metric == "precomputed":
        # scikit-learn expects the first failure from its own neighbors estimators
        # with precomputed distances. With sparse distances, scikit-learn also warns
        # that those of the checks are not sorted by rows, which the tests turn into
        # an error before the shape of the predictions is checked
        unsorted = "EfficiencyWarning on unsorted sparse distances"
        failed_checks |= {
            "check_n_features_in_after_fitting": "Same as scikit-learn (FIXME)",
            "check_estimator_sparse_tag": unsorted,
            "check_estimator_sparse_array": unsorted,
            "check_estimator_sparse_matrix": unsorted,
        }
    return failed_checks


@parametrize_with_checks(
    [
        KNeighborsLabelRanker(n_neighbors=2),
        KNeighborsLabelRanker(n_neighbors=2, model="plackett_luce", weights="dudani"),
        KNeighborsLabelRanker(metric="precomputed"),
    ],
    expected_failed_checks=_expected_failed_checks,
)
def test_estimators(estimator, check):
    check(estimator)


def _naive_neighbors(X_train, X_test, n_neighbors):
    """Find the nearest neighbors of each query and their Euclidean distances."""
    neighbors = []
    for x in X_test:
        dist = [np.sqrt(np.sum((x - x_train) ** 2)) for x_train in X_train]
        order = sorted(range(len(X_train)), key=dist.__getitem__)[:n_neighbors]
        neighbors.append((order, [dist[i] for i in order]))
    return neighbors


def _naive_weights(dist, y, *, weights, model):
    """Weigh the neighbors of a query one by one, as the docstring describes."""
    if weights == "uniform":
        weights = [1.0] * len(dist)
    elif weights == "distance":
        weights = [1 / d for d in dist]
    elif dist[-1] == dist[0]:
        weights = [1.0] * len(dist)
    else:
        weights = [(dist[-1] - d) / (dist[-1] - dist[0]) for d in dist]
    if model == "mallows":
        completeness = [np.sum(~np.isnan(row)) / len(row) for row in y]
        weights = [w * c for w, c in zip(weights, completeness, strict=True)]
        if not any(w > 0 for w in weights):
            weights = completeness
    if not any(w > 0 for w in weights):
        weights = [1.0] * len(dist)
    return np.array(weights)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("weights", ["uniform", "distance", "dudani"])
@pytest.mark.parametrize("missing", [0, 0.4])
# The estimation of the Plackett-Luce model may not converge when the rankings of
# the neighbors agree, but the oracle runs the same iterations
@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_predict_oracle(model, weights, missing):
    # The neighbors are found by brute force, and the model of each query is
    # estimated by the naive implementations of the models
    rng = np.random.RandomState(0)
    n_neighbors, n_labels = 7, 4
    X_train = rng.rand(40, 3)
    X_test = rng.rand(15, 3)
    y = random_rankings(len(X_train), n_labels, missing=missing, random_state=rng)

    ranker = KNeighborsLabelRanker(n_neighbors, model=model, weights=weights)
    y_pred = ranker.fit(X_train, y).predict(X_test)

    for ranking, (ind, dist) in zip(
        y_pred, _naive_neighbors(X_train, X_test, n_neighbors), strict=True
    ):
        y_neigh = y[ind]
        w = _naive_weights(dist, y_neigh, weights=weights, model=model)
        if model == "mallows":
            expected, _ = _oracles.estimate_center(y_neigh, w)
        else:
            parameters, _, _ = _oracles.estimate_plackett_luce(
                y_neigh, w, tol=1e-6, max_iter=1000
            )
            expected = rankdata(-parameters, method="ordinal")
        assert_array_equal(ranking, expected)


@pytest.mark.parametrize(
    "model, weights, expected",
    [
        # The Borda totals are 8, 7 and 3 for the first query and 6, 8 and 4 for
        # the second one
        ("mallows", "uniform", [[1, 2, 3], [2, 1, 3]]),
        # The Dudani weights are 1, 0.2 and 0 for the first query, giving the votes
        # 2.6, 3.4 and 1.2, and 1, 0.4 and 0 for the second one, giving 3.4, 3.2
        # and 1.8
        ("mallows", "dudani", [[2, 1, 3], [1, 2, 3]]),
        # The third label is never ranked above another one by the neighbors of
        # the first query, so its parameter is zero
        ("plackett_luce", "uniform", [[1, 2, 3], [2, 1, 3]]),
    ],
)
@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_predict_examples(model, weights, expected):
    ranker = KNeighborsLabelRanker(n_neighbors=3, model=model, weights=weights)
    y_pred = ranker.fit(X_EXAMPLE, Y_EXAMPLE).predict([[1.1], [2.2]])
    assert_array_equal(y_pred, expected)
    assert y_pred.dtype == np.intp


@pytest.mark.parametrize(
    "dist, expected",
    [
        ([[0.1, 0.9, 1.1]], [[1, 0.2, 0]]),
        # Neighbors at the same distance, or a single one, all get one
        ([[0.5, 0.5, 0.5]], [[1, 1, 1]]),
        ([[0.3]], [[1]]),
    ],
)
def test_dudani_weights(dist, expected):
    assert_allclose(_dudani_weights(np.array(dist)), expected)


@pytest.mark.parametrize("model", MODELS)
def test_predict_zero_weights(model):
    # With the Dudani weights, the farthest neighbor gets a zero weight and the
    # nearest one ranks no label, so the Mallows model falls back on the fraction
    # of ranked labels and predicts the ranking of the farthest neighbor. The
    # Plackett-Luce model does not use that fraction, so the nearest neighbor
    # gets a positive weight and the uninformative ranking predicts the identity
    X = [[0], [1]]
    y = [[np.nan, np.nan, np.nan], [3, 2, 1]]
    ranker = KNeighborsLabelRanker(2, model=model, weights="dudani").fit(X, y)
    expected = [3, 2, 1] if model == "mallows" else [1, 2, 3]
    assert_array_equal(ranker.predict([[0]]), [expected])


@pytest.mark.parametrize("model", MODELS)
def test_predict_zero_weights_callable(model):
    # A callable that gives zero weights falls back on equal weights, or on the
    # fraction of ranked labels for the Mallows model
    X = [[0], [1], [2]]
    y = [[1, 2, 3], [3, 2, 1], [np.nan, 1, 2]]
    expected = KNeighborsLabelRanker(3, model=model).fit(X, y).predict([[0]])
    ranker = KNeighborsLabelRanker(3, model=model, weights=np.zeros_like)
    assert_array_equal(ranker.fit(X, y).predict([[0]]), expected)


@pytest.mark.parametrize("model", MODELS)
def test_predict_no_ranked_labels(model):
    # Without any ranked label in the neighborhood, all the labels tie and the
    # identity is predicted
    X = [[0], [1], [5]]
    y = [[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [3, 2, 1]]
    ranker = KNeighborsLabelRanker(2, model=model, weights="dudani").fit(X, y)
    assert_array_equal(ranker.predict([[0]]), [[1, 2, 3]])


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("weights", ["uniform", "dudani"])
def test_predict_n_jobs(model, weights):
    rng = np.random.RandomState(0)
    X = rng.rand(60, 4)
    y = random_rankings(len(X), 5, missing=0.3, random_state=rng)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        y_preds = [
            KNeighborsLabelRanker(10, model=model, weights=weights, n_jobs=n_jobs)
            .fit(X, y)
            .predict(X)
            for n_jobs in [None, 2, -1]
        ]
    assert_array_equal(y_preds[0], y_preds[1])
    assert_array_equal(y_preds[0], y_preds[2])


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_predict_invariances(model):
    # The prediction does not depend on the order of the training samples, which
    # are at different distances from each query, nor on the order of the labels
    rng = np.random.RandomState(0)
    X = rng.rand(30, 2)
    y = random_rankings(len(X), 4, missing=0.2, random_state=rng)
    X_test = rng.rand(10, 2)
    ranker = KNeighborsLabelRanker(5, model=model, weights="dudani")
    y_pred = ranker.fit(X, y).predict(X_test)

    permutation = rng.permutation(len(X))
    assert_array_equal(
        ranker.fit(X[permutation], y[permutation]).predict(X_test), y_pred
    )
    # Reversing the labels would change how exact ties are broken, but the
    # Dudani weights of these random samples leave none
    labels = np.arange(y.shape[1])[::-1]
    y_pred_reversed = ranker.fit(X, y[:, labels]).predict(X_test)
    assert_array_equal(y_pred_reversed, y_pred[:, labels])


@pytest.mark.parametrize("model", MODELS)
def test_predict_none(model):
    # Without queries, each training sample is predicted from its neighbors
    # but itself
    rng = np.random.RandomState(0)
    X = rng.rand(20, 2)
    y = random_rankings(len(X), 3, missing=0, random_state=rng)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        y_pred = KNeighborsLabelRanker(4, model=model).fit(X, y).predict(None)
        for i in range(len(X)):
            others = np.delete(np.arange(len(X)), i)
            ranker = KNeighborsLabelRanker(4, model=model).fit(X[others], y[others])
            assert_array_equal(y_pred[i], ranker.predict(X[[i]])[0])


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_precomputed(model):
    rng = np.random.RandomState(0)
    X = rng.rand(20, 3)
    X_test = rng.rand(5, 3)
    y = random_rankings(len(X), 4, missing=0.2, random_state=rng)
    ranker = KNeighborsLabelRanker(5, model=model, weights="dudani")
    expected = ranker.fit(X, y).predict(X_test)

    dist = np.sqrt(((X[:, np.newaxis] - X) ** 2).sum(axis=2))
    dist_test = np.sqrt(((X_test[:, np.newaxis] - X) ** 2).sum(axis=2))
    ranker.set_params(metric="precomputed")
    assert_array_equal(ranker.fit(dist, y).predict(dist_test), expected)


@pytest.mark.parametrize("model", MODELS)
def test_fit_target_dtypes(model):
    y_int = np.array(Y_EXAMPLE)
    y_float = y_int.astype(np.float64)
    ranker = KNeighborsLabelRanker(3, model=model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        assert_array_equal(
            ranker.fit(X_EXAMPLE, y_int).predict(X_EXAMPLE),
            ranker.fit(X_EXAMPLE, y_float).predict(X_EXAMPLE),
        )


@pytest.mark.parametrize("model", MODELS)
def test_one_neighbor(model):
    # A single neighbor predicts its own ranking, and a single training sample is
    # the neighbor of every query, also with two labels
    ranker = KNeighborsLabelRanker(1, model=model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        assert_array_equal(
            ranker.fit(X_EXAMPLE, Y_EXAMPLE).predict(X_EXAMPLE), Y_EXAMPLE
        )
        assert_array_equal(
            ranker.fit([[0]], [[2, 1]]).predict([[0], [5]]), [[2, 1]] * 2
        )


def test_fit_rejects_ties():
    with pytest.raises(ValueError, match="Expected rankings without ties in y"):
        KNeighborsLabelRanker(2).fit(X_EXAMPLE, [[1, 1, 2]] * 4)


def test_predict_too_many_neighbors():
    ranker = KNeighborsLabelRanker(5).fit(X_EXAMPLE, Y_EXAMPLE)
    with pytest.raises(ValueError, match="Expected n_neighbors <= n_samples_fit"):
        ranker.predict(X_EXAMPLE)


def test_plackett_luce_convergence_warning():
    # When the rankings of the neighbors agree, the maximum likelihood estimate of
    # the Plackett-Luce model does not exist, but the prediction is their ranking
    X = np.zeros((10, 1))
    y = np.tile(np.arange(1, 11), (10, 1))
    ranker = KNeighborsLabelRanker(10, model="plackett_luce").fit(X, y)
    with pytest.warns(ConvergenceWarning, match="Plackett-Luce model did not converge"):
        y_pred = ranker.predict([[0]])
    assert_array_equal(y_pred, y[:1])
    # More iterations let the estimation converge
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        assert_array_equal(ranker.set_params(max_iter=5000).predict([[0]]), y[:1])


def test_mallows_max_iter():
    # The incomplete rankings make the center change between iterations, and one
    # iteration is not enough to see that it no longer changes
    rng = np.random.RandomState(0)
    X = rng.rand(30, 2)
    y = random_rankings(len(X), 6, missing=0.5, random_state=rng)
    ranker = KNeighborsLabelRanker(10, max_iter=1).fit(X, y)
    with pytest.warns(ConvergenceWarning, match="center ranking did not converge"):
        ranker.predict(X)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_model_selection(model):
    # The ranker works in a pipeline whose number of neighbors is tuned by
    # cross-validation with a scorer of scikit-lr
    rng = np.random.RandomState(0)
    X = rng.rand(40, 3)
    y = random_rankings(len(X), 4, missing=0, random_state=rng)
    pipeline = make_pipeline(StandardScaler(), KNeighborsLabelRanker(model=model))
    search = GridSearchCV(
        pipeline,
        {"kneighborslabelranker__n_neighbors": [3, 5]},
        scoring=get_scorer("kendall_tau"),
        cv=3,
    )
    search.fit(X, y)
    assert search.best_params_["kneighborslabelranker__n_neighbors"] in {3, 5}
    assert -1 <= search.score(X, y) <= 1
