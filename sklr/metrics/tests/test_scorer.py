import pickle

import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn import config_context
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, cross_validate

from sklr.metrics import (
    get_scorer,
    get_scorer_names,
    kendall_distance,
    kendall_tau_score,
    tau_x_score,
)
from sklr.utils.tests.test_estimator_checks import (
    ConsensusLabelRanker,
    ConsensusPartialLabelRanker,
)

# The metric of each scorer and the sign that the scorer gives to it
SCORERS = {
    "kendall_tau": (kendall_tau_score, 1),
    "tau_x": (tau_x_score, 1),
    "neg_kendall_distance": (kendall_distance, -1),
}


def _data(random_state=0):
    """Draw random samples and complete rankings without ties."""
    rng = np.random.RandomState(random_state)
    X = rng.rand(30, 2)
    y = np.argsort(rng.rand(30, 4), axis=1).argsort(axis=1) + 1
    return X, y


def test_get_scorer_names():
    assert get_scorer_names() == sorted(SCORERS)


@pytest.mark.parametrize("name", SCORERS)
def test_scorer(name):
    metric, sign = SCORERS[name]
    X, y = _data()
    ranker = ConsensusLabelRanker().fit(X, y)
    scorer = get_scorer(name)
    assert scorer(ranker, X, y) == pytest.approx(sign * metric(y, ranker.predict(X)))


@pytest.mark.parametrize("name", SCORERS)
def test_get_scorer_returns_copy(name):
    assert get_scorer(name) is not get_scorer(name)


@pytest.mark.parametrize("name", SCORERS)
def test_scorer_pickle(name):
    X, y = _data()
    ranker = ConsensusLabelRanker().fit(X, y)
    scorer = get_scorer(name)
    unpickled = pickle.loads(pickle.dumps(scorer))
    assert unpickled(ranker, X, y) == scorer(ranker, X, y)


@pytest.mark.parametrize("scoring", [None, kendall_tau_score])
def test_get_scorer_not_string(scoring):
    assert get_scorer(scoring) is scoring


def test_get_scorer_invalid_type():
    with pytest.raises(ValueError, match="The 'scoring' parameter of get_scorer"):
        get_scorer(5)


def test_get_scorer_invalid_name():
    msg = "'accuracy' is not a valid scoring value. Use sklr.metrics.get_scorer_names"
    with pytest.raises(ValueError, match=msg):
        get_scorer("accuracy")


# The metrics of label ranking reject the predictions of a partial label ranker,
# which may have ties
@pytest.mark.parametrize(
    "ranker, name",
    [(ConsensusLabelRanker(), name) for name in SCORERS]
    + [(ConsensusPartialLabelRanker(), "tau_x")],
)
def test_cross_val_score(ranker, name):
    metric, sign = SCORERS[name]
    X, y = _data()
    cv = KFold(n_splits=3)
    scores = cross_val_score(ranker, X, y, scoring=get_scorer(name), cv=cv)
    expected = [
        sign * metric(y[test], ranker.fit(X[train], y[train]).predict(X[test]))
        for train, test in cv.split(X)
    ]
    assert_allclose(scores, expected)


def test_cross_validate_multiple_scorers():
    X, y = _data()
    scoring = {name: get_scorer(name) for name in SCORERS}
    results = cross_validate(ConsensusLabelRanker(), X, y, scoring=scoring, cv=3)
    for name in SCORERS:
        assert results[f"test_{name}"].shape == (3,)
    # The normalized Kendall distance is (1 - tau) / 2
    assert_allclose(
        results["test_neg_kendall_distance"], (results["test_kendall_tau"] - 1) / 2
    )


def test_grid_search():
    X, y = _data()
    search = GridSearchCV(
        ConsensusLabelRanker(), {}, scoring=get_scorer("kendall_tau"), cv=3
    ).fit(X, y)
    expected = cross_val_score(
        ConsensusLabelRanker(), X, y, scoring=get_scorer("kendall_tau"), cv=3
    )
    assert search.best_score_ == pytest.approx(expected.mean())


@pytest.mark.parametrize(
    "ranker, name",
    [
        (ConsensusLabelRanker(), "kendall_tau"),
        (ConsensusPartialLabelRanker(), "tau_x"),
    ],
)
def test_cross_val_score_default_scorer(ranker, name):
    # Without a scorer, the score method of the ranker is used
    X, y = _data()
    assert_allclose(
        cross_val_score(ranker, X, y, cv=3),
        cross_val_score(ranker, X, y, scoring=get_scorer(name), cv=3),
    )


@pytest.mark.parametrize("name", SCORERS)
def test_sample_weight_routing(name):
    metric, sign = SCORERS[name]
    X, y = _data()
    sample_weight = np.random.RandomState(1).randint(1, 5, size=len(y))
    cv = KFold(n_splits=3)
    with config_context(enable_metadata_routing=True):
        scorer = get_scorer(name).set_score_request(sample_weight=True)
        scores = cross_val_score(
            ConsensusLabelRanker(),
            X,
            y,
            scoring=scorer,
            cv=cv,
            params={"sample_weight": sample_weight},
        )
    expected = [
        sign
        * metric(
            y[test],
            ConsensusLabelRanker().fit(X[train], y[train]).predict(X[test]),
            sample_weight=sample_weight[test],
        )
        for train, test in cv.split(X)
    ]
    assert_allclose(scores, expected)
