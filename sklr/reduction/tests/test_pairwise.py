import itertools
import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from sklr._distributions.tests._oracles import random_rankings
from sklr.reduction import PairwiseLabelRanker
from sklr.reduction._pairwise import _votes
from sklr.utils._estimator_checks import parametrize_with_checks


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
        PairwiseLabelRanker(),
        PairwiseLabelRanker(DecisionTreeClassifier(random_state=0)),
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


def _naive_votes(proba, n_labels):
    """Sum, for each label, the probabilities that it is ahead of each other label.

    Equations (3.2) and (3.3) of [hullermeier_label_2008], with the models of the
    pairs i < j in the order of numpy.triu_indices.
    """
    pairs = list(itertools.combinations(range(n_labels), 2))
    votes = np.zeros((len(proba), n_labels))
    for sample, sample_proba in enumerate(proba):
        relation = np.zeros((n_labels, n_labels))
        for (i, j), p in zip(pairs, sample_proba):
            relation[i, j], relation[j, i] = p, 1 - p
        votes[sample] = relation.sum(axis=1)
    return votes


def _naive_ranking(votes):
    """Sort the labels by decreasing votes, the smallest index first on ties."""
    order = sorted(range(len(votes)), key=lambda label: (-votes[label], label))
    ranking = np.empty(len(votes), dtype=np.intp)
    ranking[order] = np.arange(1, len(votes) + 1)
    return ranking


class _PairSpy(ClassifierMixin, BaseEstimator):
    """Classifier that keeps its training data and predicts the prior of class 1."""

    def fit(self, X, y, sample_weight=None):
        self.X_, self.y_, self.sample_weight_ = X, y, sample_weight
        self.classes_ = np.unique(y)
        self.prior_ = np.mean(y == 1)
        return self

    def predict_proba(self, X):
        proba = np.tile([1 - self.prior_, self.prior_], (len(X), 1))
        return proba[:, np.isin([0, 1], self.classes_)]


def test_pairwise_training_data():
    """Each pair is fitted to the samples of positive weight that rank both labels."""
    n_labels = 4
    X, y = _random_problem(40, n_labels, missing=0.4, random_state=0)
    sample_weight = np.random.RandomState(1).uniform(0.0, 2.0, size=40)
    sample_weight[:4] = 0.0
    ranker = PairwiseLabelRanker(_PairSpy()).fit(X, y, sample_weight)
    pairs = list(itertools.combinations(range(n_labels), 2))
    assert len(ranker.estimators_) == len(pairs)
    for (i, j), estimator in zip(pairs, ranker.estimators_):
        rows = [
            s
            for s in range(40)
            if not np.isnan(y[s, i]) and not np.isnan(y[s, j]) and sample_weight[s] > 0
        ]
        targets = [int(y[s, i] < y[s, j]) for s in rows]
        assert_allclose(estimator.X_, X[rows])
        assert_array_equal(estimator.y_, targets)
        assert_allclose(estimator.sample_weight_, sample_weight[rows])


def test_samples_without_sample_weight():
    """Without sample weights, the estimators are fitted without them."""
    X, y = _random_problem(30, 3, missing=0.0, random_state=0)
    ranker = PairwiseLabelRanker(_PairSpy()).fit(X, y)
    for estimator in ranker.estimators_:
        assert estimator.sample_weight_ is None
        assert len(estimator.X_) == 30


@pytest.mark.parametrize("n_labels", [2, 3, 5])
def test_votes(n_labels):
    """The votes sum the valued preference relation, and add up to the pairs."""
    n_pairs = n_labels * (n_labels - 1) // 2
    proba = np.random.RandomState(0).rand(20, n_pairs)
    votes = _votes(proba, n_labels)
    assert_allclose(votes, _naive_votes(proba, n_labels))
    assert_allclose(votes.sum(axis=1), n_pairs)


@pytest.mark.parametrize("missing", [0.0, 0.3])
def test_predict_sorts_votes(missing):
    """The prediction sorts the labels by the votes of the fitted estimators."""
    n_labels = 5
    X, y = _random_problem(60, n_labels, missing=missing, random_state=0)
    ranker = PairwiseLabelRanker().fit(X, y)
    proba = np.column_stack(
        [estimator.predict_proba(X)[:, 1] for estimator in ranker.estimators_]
    )
    expected = [_naive_ranking(votes) for votes in _naive_votes(proba, n_labels)]
    assert_array_equal(ranker.predict(X), expected)


def _random_distribution(n_labels, rng):
    """Draw a probability for each ranking of the labels, with many of them zero."""
    rankings = np.array(list(itertools.permutations(range(1, n_labels + 1))))
    proba = rng.rand(len(rankings)) * (rng.rand(len(rankings)) < 0.5)
    proba[rng.randint(len(rankings))] += 0.1
    return rankings, proba / proba.sum()


def _pairwise_marginals(rankings, proba):
    """Probability that the first label of each pair i < j is ahead of the second."""
    i, j = np.triu_indices(rankings.shape[1], k=1)
    return proba @ (rankings[:, i] < rankings[:, j])


@pytest.mark.parametrize("n_labels", [2, 3, 4, 5])
@pytest.mark.parametrize("seed", range(10))
def test_voting_minimizes_expected_spearman_distance(n_labels, seed):
    """Theorem 3 of [hullermeier_label_2008], by brute force over all the rankings.

    With the true pairwise probabilities, the ranking of the votes minimizes the
    expected sum of squared differences of the positions, and the expected
    position of each label is n minus its votes (Lemma 3).
    """
    rankings, proba = _random_distribution(n_labels, np.random.RandomState(seed))
    votes = _votes(_pairwise_marginals(rankings, proba)[np.newaxis], n_labels)[0]
    assert_allclose(n_labels - votes, proba @ rankings)
    expected_loss = [
        proba @ np.sum((rankings - candidate) ** 2, axis=1) for candidate in rankings
    ]
    ranking = rankdata(-votes, method="ordinal")
    loss = proba @ np.sum((rankings - ranking) ** 2, axis=1)
    assert loss <= min(expected_loss) + 1e-12


@pytest.mark.parametrize("n_labels", [3, 4, 6])
def test_voting_on_frequencies_is_borda(n_labels):
    """Section 5.4 of [hullermeier_label_2008]: voting on frequencies is Borda."""
    # The frequencies are multiples of 1 / 16, so the votes are exact and the
    # ties of the Borda count are ties of the votes
    rankings = random_rankings(16, n_labels, missing=0.0, random_state=0)
    frequencies = _pairwise_marginals(rankings, np.full(16, 1 / 16))
    votes = _votes(frequencies[np.newaxis], n_labels)[0]
    borda = np.sum(n_labels - rankings, axis=0)
    assert_array_equal(votes * 16, borda)
    ranking = rankdata(-votes, method="ordinal")
    assert_array_equal(ranking, _naive_ranking(borda))


def test_equal_votes_keep_the_order_of_the_labels():
    X, y = _random_problem(30, 4, missing=0.0, random_state=0)
    ranker = PairwiseLabelRanker(DummyClassifier(strategy="uniform")).fit(X, y)
    # The uniform strategy predicts a probability of 0.5 for both classes
    assert_array_equal(ranker.predict(X), np.tile([1, 2, 3, 4], (30, 1)))


def test_pair_without_data():
    """A pair never ranked together gets no estimator and votes 0.5 each way."""
    X, y = _random_problem(40, 3, missing=0.0, random_state=0)
    # The first and the last labels are never ranked together
    y[:20, 0] = np.nan
    y[20:, 2] = np.nan
    y[:20, 1:] = rankdata(y[:20, 1:], axis=1)
    y[20:, :2] = rankdata(y[20:, :2], axis=1)
    with pytest.warns(UserWarning, match=r"the pairs of labels \[\(0, 2\)\]"):
        ranker = PairwiseLabelRanker().fit(X, y)
    assert ranker.estimators_[1] is None
    assert_allclose(ranker._predict_pairwise_proba(X)[:, 1], 0.5)


def test_label_never_ranked():
    """A label never ranked gets half a vote from each other label."""
    X, y = _random_problem(30, 3, missing=0.0, random_state=0)
    y = np.column_stack((y, np.full(30, np.nan)))
    with pytest.warns(UserWarning, match=r"\[\(0, 3\), \(1, 3\), \(2, 3\)\]"):
        ranker = PairwiseLabelRanker().fit(X, y)
    assert_allclose(_votes(ranker._predict_pairwise_proba(X), 4)[:, 3], 1.5)


def test_pair_with_one_order():
    """A pair always ordered in the same way is predicted in that order."""
    X, y = _random_problem(30, 3, missing=0.0, random_state=0)
    # The first label is always ahead of the last one
    y[:, [0, 2]] = np.sort(y[:, [0, 2]], axis=1)
    sample_weight = np.ones(30)
    # The only sample with the last label ahead has no weight
    y[0] = [3, 2, 1]
    sample_weight[0] = 0.0
    ranker = PairwiseLabelRanker().fit(X, y, sample_weight)
    assert_allclose(ranker._predict_pairwise_proba(X)[:, 1], 1.0)
    y[:, [0, 2]] = y[:, [2, 0]]
    ranker = PairwiseLabelRanker().fit(X, y, sample_weight)
    assert_allclose(ranker._predict_pairwise_proba(X)[:, 1], 0.0)


def test_uninformative_rankings_are_ignored():
    """Rankings of fewer than two labels do not change the estimators."""
    X, y = _random_problem(40, 4, missing=0.3, random_state=0)
    ranker = PairwiseLabelRanker().fit(X, y)
    rng = np.random.RandomState(1)
    X_extra = rng.normal(size=(6, 2))
    y_extra = np.full((6, 4), np.nan)
    y_extra[:3, 1] = 1.0
    extended = PairwiseLabelRanker().fit(
        np.vstack((X, X_extra)), np.vstack((y, y_extra))
    )
    for estimator, expected in zip(extended.estimators_, ranker.estimators_):
        assert_allclose(estimator.coef_, expected.coef_)
        assert_allclose(estimator.intercept_, expected.intercept_)


def test_rankings_with_ties():
    X, y = _random_problem(20, 3, missing=0.0, random_state=0)
    y[0] = [1, 1, 2]
    with pytest.raises(ValueError, match="Expected rankings without ties in y"):
        PairwiseLabelRanker().fit(X, y)


def test_estimator_without_predict_proba():
    X, y = _random_problem(20, 3, missing=0.0, random_state=0)
    with pytest.raises(ValueError, match="The 'estimator' parameter"):
        PairwiseLabelRanker(LinearSVC()).fit(X, y)


def test_estimator_without_sample_weight():
    X, y = _random_problem(20, 3, missing=0.0, random_state=0)
    with pytest.raises(ValueError, match="does not support sample weights"):
        PairwiseLabelRanker(KNeighborsClassifier()).fit(X, y, np.ones(20))
    # Without sample weights, it is fitted
    PairwiseLabelRanker(KNeighborsClassifier()).fit(X, y)


@pytest.mark.parametrize("missing", [0.0, 0.3])
def test_integer_sample_weight_repeats_samples(missing):
    X, y = _random_problem(30, 4, missing=missing, random_state=0)
    sample_weight = np.random.RandomState(1).randint(0, 3, size=30)
    estimator = LogisticRegression(tol=1e-10)
    weighted = PairwiseLabelRanker(estimator).fit(X, y, sample_weight)
    repeated = PairwiseLabelRanker(estimator).fit(
        np.repeat(X, sample_weight, axis=0), np.repeat(y, sample_weight, axis=0)
    )
    # The solver stops at a tolerance on the gradient, so the models agree up to
    # the precision that it allows
    assert_allclose(
        weighted._predict_pairwise_proba(X),
        repeated._predict_pairwise_proba(X),
        atol=1e-6,
    )


def test_n_jobs_and_dtype():
    """The predictions are the same with several jobs and with float32 data."""
    X, y = _random_problem(40, 4, missing=0.3, random_state=0)
    expected = PairwiseLabelRanker().fit(X, y).predict(X)
    parallel = PairwiseLabelRanker(n_jobs=2).fit(X, y)
    assert_array_equal(parallel.predict(X), expected)
    X32 = X.astype(np.float32)
    assert_array_equal(PairwiseLabelRanker().fit(X32, y).predict(X32), expected)


def test_random_state_of_the_estimator():
    X, y = _random_problem(40, 4, missing=0.3, random_state=0)
    estimator = DecisionTreeClassifier(max_features=1, random_state=0)
    first = PairwiseLabelRanker(estimator).fit(X, y)
    second = PairwiseLabelRanker(estimator, n_jobs=2).fit(X, y)
    assert_allclose(first._predict_pairwise_proba(X), second._predict_pairwise_proba(X))


@pytest.mark.parametrize("n_labels", [2, 5])
def test_learns_rankings(n_labels):
    """The rankings of a linear model of the features are learned well."""
    X, y = _random_problem(300, n_labels, missing=0.0, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ranker = PairwiseLabelRanker().fit(X[:200], y[:200])
    baseline = PairwiseLabelRanker(DummyClassifier()).fit(X[:200], y[:200])
    assert ranker.score(X[200:], y[200:]) > baseline.score(X[200:], y[200:]) + 0.2
