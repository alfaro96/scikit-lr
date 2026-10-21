import itertools
import math
import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklr._distributions.tests._oracles import random_rankings
from sklr.reduction import LabelwiseLabelRanker
from sklr.reduction._labelwise import _assign, _observation_likelihood
from sklr.reduction._ordinal import OrdinalLogisticRegression
from sklr.utils._estimator_checks import parametrize_with_checks

LOSSES = ["absolute_error", "squared_error"]


def _expected_failed_checks(estimator):
    # The checks of sample weights build a target of classes 1 and 2 by hand,
    # without adapting it to the tags of the estimator, so it is not a valid
    # ranking
    return {
        "check_sample_weights_pandas_series": "Builds a target that is not a ranking",
        "check_sample_weights_not_an_array": "Builds a target that is not a ranking",
    }


@parametrize_with_checks(
    [
        LabelwiseLabelRanker(),
        LabelwiseLabelRanker(loss="squared_error"),
        LabelwiseLabelRanker(DecisionTreeClassifier(random_state=0)),
    ],
    expected_failed_checks=_expected_failed_checks,
)
def test_estimators(estimator, check):
    check(estimator)


def _random_problem(n_samples, n_labels, *, missing, random_state):
    """Draw rankings that depend on two features, possibly incomplete."""
    rng = np.random.RandomState(random_state)
    X = rng.normal(size=(n_samples, 2))
    scores = X @ rng.normal(size=(2, n_labels)) + rng.normal(size=(n_samples, n_labels))
    y = np.argsort(np.argsort(-scores, axis=1), axis=1) + 1.0
    y[rng.rand(n_samples, n_labels) < missing] = np.nan
    for row in y:
        is_ranked = ~np.isnan(row)
        row[is_ranked] = np.argsort(np.argsort(row[is_ranked])) + 1
    return X, y


@pytest.mark.parametrize("n_labels", [2, 3, 4, 5])
def test_observation_likelihood(n_labels):
    """Equation (16) matches the removal of every subset of labels."""
    for n_ranked in range(1, n_labels + 1):
        counts = np.zeros((n_labels, n_labels))
        n_subsets = 0
        # The kept labels are identified by their true positions
        for kept in itertools.combinations(range(1, n_labels + 1), n_ranked):
            n_subsets += 1
            for observed, true in enumerate(kept, start=1):
                counts[observed - 1, true - 1] += 1
        expected = counts / n_subsets
        positions = np.arange(1, n_labels + 1, dtype=np.float64)
        likelihood = np.asarray(
            _observation_likelihood(positions, np.full(n_labels, n_ranked), n_labels)
        )
        assert_allclose(likelihood, expected, atol=1e-12)
        # A label is kept with probability n_ranked / n_labels
        assert_allclose(likelihood.sum(axis=0), n_ranked / n_labels)


def _brute_force_assignment(costs):
    """Find the smallest positions, in order of label, of minimum total cost."""
    n_labels = len(costs)
    best_total, best = np.inf, tuple(range(n_labels))
    # The permutations come in lexicographic order, so a later one only replaces
    # the best one if its total is smaller beyond the rounding errors
    for permutation in itertools.permutations(range(n_labels)):
        total = sum(costs[label, permutation[label]] for label in range(n_labels))
        if total < best_total - 1e-9:
            best_total, best = total, permutation
    return np.array(best) + 1


@pytest.mark.parametrize("n_labels", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("seed", range(20))
def test_assign(n_labels, seed):
    """The assignment matches the brute force, including the ties."""
    rng = np.random.RandomState(seed)
    # Few distinct costs, so that there are many assignments of minimum cost
    costs = rng.randint(3, size=(n_labels, n_labels)) / 3
    assert_array_equal(_assign(costs), _brute_force_assignment(costs))


def test_assign_ties():
    """Labels with the same costs are ranked by their index."""
    assert_array_equal(_assign(np.zeros((4, 4))), [1, 2, 3, 4])
    costs = np.tile([0.4, 0.1, 0.4, 0.9], (4, 1)).T
    assert_array_equal(_assign(costs), [1, 2, 3, 4])


@pytest.mark.parametrize("loss", LOSSES)
def test_predict_minimizes_expected_loss(loss):
    """The predicted ranking minimizes the expected loss over all the rankings."""
    X, y = _random_problem(30, 4, missing=0.0, random_state=0)
    ranker = LabelwiseLabelRanker(loss=loss).fit(X, y)
    proba = ranker._predict_position_proba(X)
    positions = np.arange(1, 5)
    power = 1 if loss == "absolute_error" else 2
    loss_matrix = np.abs(positions[:, np.newaxis] - positions) ** power
    for sample_proba, ranking in zip(proba, ranker.predict(X)):
        costs = sample_proba @ loss_matrix
        assert_array_equal(ranking, _brute_force_assignment(costs))


def test_predict_squared_error_ties():
    """With the squared error, labels of the same expected position keep their order."""
    X = np.random.RandomState(0).normal(size=(30, 2))
    y = random_rankings(30, 3, missing=0.0, random_state=0)
    # Every label is seen at every position, and the estimator gives the same
    # probability to each one
    ranker = LabelwiseLabelRanker(
        DummyClassifier(strategy="uniform"), loss="squared_error"
    ).fit(X, y)
    assert_array_equal(ranker.predict(X), np.tile([1, 2, 3], (30, 1)))


def test_complete_rankings_fit_classes():
    """With complete rankings, each estimator is fitted to the positions of a label."""
    X, y = _random_problem(40, 4, missing=0.0, random_state=0)
    sample_weight = np.random.RandomState(1).uniform(0.5, 2.0, size=40)
    ranker = LabelwiseLabelRanker().fit(X, y, sample_weight)
    for label, estimator in enumerate(ranker.estimators_):
        expected = OrdinalLogisticRegression().fit(X, y[:, label], sample_weight)
        assert_array_equal(estimator.classes_, np.arange(1, 5))
        assert_allclose(estimator.coef_, expected.coef_, rtol=1e-6)
        assert_allclose(estimator.thresholds_, expected.thresholds_, rtol=1e-6)


def _naive_log_likelihood(X, positions, n_ranked, n_labels, coef, thresholds):
    """Compute equation (17) sample by sample, with the probabilities of the model."""
    total = 0.0
    for x, observed, m in zip(X, positions, n_ranked):
        if np.isnan(observed) or m < 2:
            continue
        observed = int(observed)
        cumulative = [0.0, *expit(thresholds + x @ coef), 1.0]
        total += np.log(
            sum(
                (cumulative[r] - cumulative[r - 1])
                * math.comb(r - 1, r - observed)
                * math.comb(n_labels - r, n_labels - m - r + observed)
                / math.comb(n_labels, m)
                for r in range(observed, observed + n_labels - m + 1)
            )
        )
    return total


def test_incomplete_rankings_maximize_likelihood():
    """Each model reaches the optimum of equation (17) found by a naive optimizer."""
    n_labels = 4
    X, y = _random_problem(60, n_labels, missing=0.4, random_state=0)
    n_ranked = np.sum(~np.isnan(y), axis=1)
    ranker = LabelwiseLabelRanker(OrdinalLogisticRegression(tol=1e-10)).fit(X, y)
    sum_weight = np.sum(~np.isnan(y) & (n_ranked[:, np.newaxis] > 1), axis=0)
    for label, estimator in enumerate(ranker.estimators_):

        def objective(params, label=label):
            coef = params[:2]
            thresholds = params[2] + np.concatenate(
                ([0], np.cumsum(np.exp(params[3:])))
            )
            log_likelihood = _naive_log_likelihood(
                X, y[:, label], n_ranked, n_labels, coef, thresholds
            )
            return (-log_likelihood + 0.5 * coef @ coef) / sum_weight[label]

        naive = minimize(
            objective,
            np.zeros(n_labels + 1),
            method="Nelder-Mead",
            options={"xatol": 1e-10, "fatol": 1e-14, "maxiter": 100000},
        )
        params = np.concatenate(
            (
                estimator.coef_,
                estimator.thresholds_[:1],
                np.log(np.diff(estimator.thresholds_)),
            )
        )
        assert objective(params) <= naive.fun + 1e-9


def test_uninformative_rankings_are_ignored():
    """Rankings that leave a label unranked, or rank it alone, do not change it."""
    X, y = _random_problem(40, 4, missing=0.3, random_state=0)
    ranker = LabelwiseLabelRanker().fit(X, y)
    rng = np.random.RandomState(1)
    X_extra = rng.normal(size=(6, 2))
    y_extra = np.full((6, 4), np.nan)
    y_extra[:3, 1] = 1.0
    extended = LabelwiseLabelRanker().fit(
        np.vstack((X, X_extra)), np.vstack((y, y_extra))
    )
    for estimator, expected in zip(extended.estimators_, ranker.estimators_):
        assert_allclose(estimator.coef_, expected.coef_)
        assert_allclose(estimator.thresholds_, expected.thresholds_)


def test_label_without_information():
    """A label never ranked with another one gets equally likely positions."""
    X, y = _random_problem(30, 3, missing=0.0, random_state=0)
    y = np.column_stack((y, np.full(30, np.nan)))
    with pytest.warns(UserWarning, match=r"the labels \[3\]"):
        ranker = LabelwiseLabelRanker().fit(X, y)
    assert ranker.estimators_[3] is None
    assert_allclose(ranker._predict_position_proba(X)[:, 3], 0.25)
    rankings = ranker.predict(X)
    assert_array_equal(np.sort(rankings, axis=1), np.tile(np.arange(1, 5), (30, 1)))


def test_estimator_with_missing_positions():
    """The probabilities of another estimator are put at the positions it has seen."""
    X, y = _random_problem(30, 3, missing=0.0, random_state=0)
    # The first label is never last
    y[y[:, 0] == 3] = [2, 1, 3]
    ranker = LabelwiseLabelRanker(LogisticRegression()).fit(X, y)
    proba = ranker._predict_position_proba(X)
    assert_allclose(proba[:, 0, 2], 0.0)
    assert_allclose(proba.sum(axis=2), 1.0)


class _LikelihoodSpy(ClassifierMixin, BaseEstimator):
    """Classifier that keeps the arguments of fit_likelihood and predicts uniformly."""

    def fit(self, X, y):
        raise AssertionError("fit_likelihood should be used instead")

    def fit_likelihood(self, X, likelihood, classes, sample_weight=None):
        self.X_, self.likelihood_, self.sample_weight_ = X, likelihood, sample_weight
        self.classes_ = classes
        return self

    def predict_proba(self, X):
        return np.full((len(X), len(self.classes_)), 1 / len(self.classes_))


def test_estimator_with_fit_likelihood():
    """An estimator with fit_likelihood gets the likelihood of equation (16)."""
    n_labels = 4
    X, y = _random_problem(30, n_labels, missing=0.4, random_state=0)
    sample_weight = np.random.RandomState(1).uniform(0.0, 2.0, size=30)
    sample_weight[:3] = 0.0
    ranker = LabelwiseLabelRanker(_LikelihoodSpy()).fit(X, y, sample_weight)
    n_ranked = np.sum(~np.isnan(y), axis=1)
    for label, estimator in enumerate(ranker.estimators_):
        rows = [
            i
            for i in range(30)
            if not np.isnan(y[i, label]) and n_ranked[i] > 1 and sample_weight[i] > 0
        ]
        expected = [
            [
                math.comb(r - 1, r - int(y[i, label]))
                * math.comb(n_labels - r, n_labels - n_ranked[i] - r + int(y[i, label]))
                / math.comb(n_labels, n_ranked[i])
                if int(y[i, label]) <= r <= int(y[i, label]) + n_labels - n_ranked[i]
                else 0.0
                for r in range(1, n_labels + 1)
            ]
            for i in rows
        ]
        assert_array_equal(estimator.classes_, np.arange(1, n_labels + 1))
        assert_allclose(estimator.X_, X[rows])
        assert_allclose(estimator.likelihood_, expected)
        assert_allclose(estimator.sample_weight_, sample_weight[rows])


def test_estimator_without_incomplete_rankings():
    X, y = _random_problem(20, 3, missing=0.3, random_state=0)
    with pytest.raises(ValueError, match="has no fit_likelihood method"):
        LabelwiseLabelRanker(LogisticRegression()).fit(X, y)


def test_rankings_with_ties():
    X, y = _random_problem(20, 3, missing=0.0, random_state=0)
    y[0] = [1, 1, 2]
    with pytest.raises(ValueError, match="Expected rankings without ties in y"):
        LabelwiseLabelRanker().fit(X, y)


def test_estimator_without_sample_weight():
    X, y = _random_problem(20, 3, missing=0.0, random_state=0)
    with pytest.raises(ValueError, match="does not support sample weights"):
        LabelwiseLabelRanker(KNeighborsClassifier()).fit(X, y, np.ones(20))
    # Without sample weights, it is fitted
    LabelwiseLabelRanker(KNeighborsClassifier()).fit(X, y)


@pytest.mark.parametrize("missing", [0.0, 0.3])
def test_integer_sample_weight_repeats_samples(missing):
    X, y = _random_problem(30, 4, missing=missing, random_state=0)
    sample_weight = np.random.RandomState(1).randint(0, 3, size=30)
    estimator = OrdinalLogisticRegression(tol=1e-10)
    weighted = LabelwiseLabelRanker(estimator).fit(X, y, sample_weight)
    repeated = LabelwiseLabelRanker(estimator).fit(
        np.repeat(X, sample_weight, axis=0), np.repeat(y, sample_weight, axis=0)
    )
    # The solver stops at a tolerance on the gradient, so the models agree up to
    # the precision that it allows. Their probabilities are compared instead of
    # their parameters, because the threshold of a position that a label never
    # takes keeps decreasing until the solver stops
    assert_allclose(
        weighted._predict_position_proba(X),
        repeated._predict_position_proba(X),
        atol=1e-6,
    )


@pytest.mark.parametrize("loss", LOSSES)
def test_n_jobs_and_dtype(loss):
    """The predictions are the same with several jobs and with float32 data."""
    X, y = _random_problem(40, 4, missing=0.3, random_state=0)
    expected = LabelwiseLabelRanker(loss=loss).fit(X, y).predict(X)
    parallel = LabelwiseLabelRanker(loss=loss, n_jobs=2).fit(X, y)
    assert_array_equal(parallel.predict(X), expected)
    X32 = X.astype(np.float32)
    assert_array_equal(
        LabelwiseLabelRanker(loss=loss).fit(X32, y).predict(X32), expected
    )


@pytest.mark.parametrize("n_labels", [2, 5])
def test_learns_rankings(n_labels):
    """The rankings of a linear model of the features are learned well."""
    X, y = _random_problem(300, n_labels, missing=0.0, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ranker = LabelwiseLabelRanker().fit(X[:200], y[:200])
    baseline = LabelwiseLabelRanker(DummyClassifier()).fit(X[:200], y[:200])
    assert ranker.score(X[200:], y[200:]) > baseline.score(X[200:], y[200:]) + 0.2
