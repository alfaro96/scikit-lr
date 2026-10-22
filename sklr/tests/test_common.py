"""Tests of the rankers with the tools of scikit-learn that take an estimator."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn import config_context
from sklearn.base import clone
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    cross_validate,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import (
    check_dataframe_column_names_consistency,
    set_random_state,
)
from sklearn.utils.validation import has_fit_parameter

from sklr._aggregation.tests._oracles import random_partial_rankings
from sklr._distributions.tests._oracles import random_rankings
from sklr.base import is_partial_label_ranker
from sklr.utils._estimator_checks import _RankingCheck
from sklr.utils._testing import all_rankers

RANKERS = [Ranker() for Ranker in all_rankers()]

# A grid of a parameter of each ranker with more than one value, where it has one
PARAM_GRIDS = {
    "DecisionTreeLabelRanker": {"max_depth": [1, 3]},
    "DecisionTreePartialLabelRanker": {"max_depth": [1, 3]},
    "DummyLabelRanker": {"model": ["mallows", "plackett_luce"]},
    "DummyPartialLabelRanker": {"strategy": ["consensus"]},
    "KNeighborsLabelRanker": {"n_neighbors": [3, 7]},
    "KNeighborsPartialLabelRanker": {"n_neighbors": [3, 7]},
    "LabelwiseLabelRanker": {"loss": ["absolute_error", "squared_error"]},
}

# The rankers that use the Plackett-Luce model may not converge on random rankings
pytestmark = pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")


def _ids(ranker):
    return type(ranker).__name__


def _make_data(ranker, n_samples=60, n_features=3, n_labels=4, random_state=0):
    """Draw features and rankings that `ranker` accepts as training data."""
    rng = np.random.RandomState(random_state)
    X = rng.normal(loc=5, scale=3, size=(n_samples, n_features))
    if is_partial_label_ranker(ranker):
        y = random_partial_rankings(n_samples, n_labels, missing=0, random_state=rng)
    else:
        y = random_rankings(n_samples, n_labels, missing=0, random_state=rng)
    return X, y.astype(np.intp)


def test_all_rankers():
    assert sorted(PARAM_GRIDS) == [_ids(ranker) for ranker in RANKERS]


@pytest.mark.parametrize("ranker", RANKERS, ids=_ids)
def test_pipeline(ranker):
    X, y = _make_data(ranker)
    X_train, X_test, y_train, y_test = X[:40], X[40:], y[:40], y[40:]
    set_random_state(ranker)
    pipeline = make_pipeline(StandardScaler(), clone(ranker)).fit(X_train, y_train)

    scaler = StandardScaler().fit(X_train)
    ranker = clone(ranker).fit(scaler.transform(X_train), y_train)
    X_test_scaled = scaler.transform(X_test)
    assert_array_equal(pipeline.predict(X_test), ranker.predict(X_test_scaled))
    assert pipeline.score(X_test, y_test) == ranker.score(X_test_scaled, y_test)


@pytest.mark.parametrize("ranker", RANKERS, ids=_ids)
@pytest.mark.parametrize("Search", [GridSearchCV, RandomizedSearchCV])
def test_search_n_jobs(ranker, Search):
    # The candidates are fitted in other processes with several jobs, so the
    # results only match if each fit only depends on its random_state
    X, y = _make_data(ranker)
    set_random_state(ranker)
    param_grid = PARAM_GRIDS[_ids(ranker)]
    cv = KFold(n_splits=3, shuffle=True, random_state=0)
    kwargs = {"n_iter": 1, "random_state": 0} if Search is RandomizedSearchCV else {}

    results = [
        Search(ranker, param_grid, cv=cv, n_jobs=n_jobs, **kwargs).fit(X, y)
        for n_jobs in [1, 2]
    ]
    assert_array_equal(
        results[0].cv_results_["mean_test_score"],
        results[1].cv_results_["mean_test_score"],
    )
    assert results[0].best_params_ == results[1].best_params_
    assert_array_equal(results[0].predict(X), results[1].predict(X))


@pytest.mark.parametrize("ranker", RANKERS, ids=_ids)
def test_cross_validate_n_jobs(ranker):
    X, y = _make_data(ranker)
    set_random_state(ranker)
    cv = KFold(n_splits=3, shuffle=True, random_state=0)

    results = [
        cross_validate(ranker, X, y, cv=cv, n_jobs=n_jobs, return_estimator=True)
        for n_jobs in [1, 2]
    ]
    assert_array_equal(results[0]["test_score"], results[1]["test_score"])
    for estimator, other in zip(results[0]["estimator"], results[1]["estimator"]):
        assert_array_equal(estimator.predict(X), other.predict(X))


@pytest.mark.parametrize("ranker", RANKERS, ids=_ids)
def test_metadata_routing_sample_weight(ranker):
    # The sample weights are routed to fit only if the ranker accepts them, and to
    # score always, so the scores are those of a loop over the folds by hand
    X, y = _make_data(ranker)
    sample_weight = np.random.RandomState(0).rand(len(X))
    set_random_state(ranker)
    has_sample_weight = has_fit_parameter(ranker, "sample_weight")
    cv = KFold(n_splits=3, shuffle=True, random_state=0)

    expected = []
    for train, test in cv.split(X):
        fit_params = (
            {"sample_weight": sample_weight[train]} if has_sample_weight else {}
        )
        fitted = clone(ranker).fit(X[train], y[train], **fit_params)
        expected.append(
            fitted.score(X[test], y[test], sample_weight=sample_weight[test])
        )

    with config_context(enable_metadata_routing=True):
        ranker = clone(ranker).set_score_request(sample_weight=True)
        if has_sample_weight:
            ranker = ranker.set_fit_request(sample_weight=True)
        params = {"sample_weight": sample_weight}
        results = cross_validate(ranker, X, y, cv=cv, params=params)
        search = GridSearchCV(ranker, {}, cv=cv).fit(X, y, **params)

    # The weighted mean of floats in another order may differ in the last bits
    assert_allclose(results["test_score"], expected, rtol=1e-12)
    assert_allclose(search.cv_results_["mean_test_score"], np.mean(expected))


@pytest.mark.parametrize(
    "ranker",
    [ranker for ranker in RANKERS if has_fit_parameter(ranker, "sample_weight")],
    ids=_ids,
)
def test_metadata_routing_pipeline(ranker):
    X, y = _make_data(ranker)
    sample_weight = np.random.RandomState(0).rand(len(X))
    set_random_state(ranker)
    X_scaled = StandardScaler().fit_transform(X)
    expected = clone(ranker).fit(X_scaled, y, sample_weight=sample_weight)

    with config_context(enable_metadata_routing=True):
        scaler = StandardScaler().set_fit_request(sample_weight=False)
        ranker = clone(ranker).set_fit_request(sample_weight=True)
        pipeline = make_pipeline(scaler, ranker).fit(X, y, sample_weight=sample_weight)

    assert_array_equal(pipeline.predict(X), expected.predict(X_scaled))


@pytest.mark.parametrize("ranker", RANKERS, ids=_ids)
def test_pandas_column_name_consistency(ranker):
    pytest.importorskip("pandas")
    _RankingCheck(check_dataframe_column_names_consistency, _ids(ranker))(ranker)


@pytest.mark.parametrize("ranker", RANKERS, ids=_ids)
def test_pandas_input(ranker):
    # Data frames of the features and the rankings, and a series of the weights,
    # give the same ranker as their arrays
    pd = pytest.importorskip("pandas")
    X, y = _make_data(ranker)
    sample_weight = np.random.RandomState(0).rand(len(X))
    set_random_state(ranker)
    fit_params = {}
    pandas_fit_params = {}
    if has_fit_parameter(ranker, "sample_weight"):
        fit_params["sample_weight"] = sample_weight
        pandas_fit_params["sample_weight"] = pd.Series(sample_weight)

    X_frame = pd.DataFrame(X, columns=["a", "b", "c"])
    y_frame = pd.DataFrame(y, columns=["w", "x", "y", "z"])
    expected = clone(ranker).fit(X, y, **fit_params).predict(X)
    result = clone(ranker).fit(X_frame, y_frame, **pandas_fit_params).predict(X_frame)
    assert_array_equal(result, expected)
