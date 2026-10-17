import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy.sparse import csr_array
from scipy.stats import rankdata
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning

from sklr._aggregation.tests import _oracles as _bucket_oracles
from sklr._aggregation.tests._oracles import random_partial_rankings
from sklr._distributions.tests import _oracles
from sklr._distributions.tests._oracles import random_rankings
from sklr.dummy import DummyLabelRanker, DummyPartialLabelRanker
from sklr.metrics import kendall_tau_score, tau_x_score
from sklr.utils._estimator_checks import parametrize_with_checks

MODELS = ["mallows", "plackett_luce"]

# The examples of the docstrings, with the rankings of four samples of one feature
X_EXAMPLE = [[0], [1], [2], [3]]
Y_EXAMPLE = [[1, 2, 3], [2, 1, 3], [1, 2, 3], [3, 1, 2]]
Y_PARTIAL_EXAMPLE = [[1, 2, 2], [2, 1, 3], [1, 1, 2], [3, 1, 2]]

RANKERS = [
    DummyLabelRanker(),
    DummyLabelRanker(model="plackett_luce"),
    DummyPartialLabelRanker(),
]


def _expected_failed_checks(estimator):
    # The checks of sample weights build a target of classes 1 and 2 by hand,
    # without adapting it to the tags of the estimator, so it is not a valid
    # ranking. The checks of sparse data expect the prediction of an estimator
    # with several outputs to have a single column, while a ranking has one per
    # label
    return {
        "check_sample_weights_pandas_series": "Builds a target that is not a ranking",
        "check_sample_weights_not_an_array": "Builds a target that is not a ranking",
        "check_estimator_sparse_array": "Expects predictions of shape (n, 1)",
        "check_estimator_sparse_matrix": "Expects predictions of shape (n, 1)",
    }


@parametrize_with_checks(RANKERS, expected_failed_checks=_expected_failed_checks)
# The checks fit the Plackett-Luce model to rankings that may agree
@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_estimators(estimator, check):
    check(estimator)


def _random_weights(n_samples, random_state):
    """Draw sample weights, some of them zero."""
    sample_weight = random_state.rand(n_samples)
    sample_weight[random_state.rand(n_samples) < 0.2] = 0
    return sample_weight


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("missing", [0, 0.4])
@pytest.mark.parametrize("weighted", [False, True])
# The Plackett-Luce model may not converge, but the oracle runs the same iterations
@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_label_ranker_oracle(model, missing, weighted):
    # The consensus is estimated by the naive implementations of the models
    rng = np.random.RandomState(0)
    y = random_rankings(30, 5, missing=missing, random_state=rng)
    sample_weight = _random_weights(len(y), rng) if weighted else np.ones(len(y))
    X = rng.rand(len(y), 2)

    ranker = DummyLabelRanker(model=model).fit(X, y, sample_weight=sample_weight)

    if model == "mallows":
        expected, _ = _oracles.estimate_center(y, sample_weight)
    else:
        parameters, _, _ = _oracles.estimate_plackett_luce(
            y, sample_weight, tol=1e-6, max_iter=1000
        )
        expected = rankdata(-parameters, method="ordinal")
    assert_array_equal(ranker.ranking_, expected)
    assert_array_equal(ranker.predict(rng.rand(7, 2)), np.tile(expected, (7, 1)))


@pytest.mark.parametrize("missing", [0, 0.3])
@pytest.mark.parametrize("weighted", [False, True])
def test_partial_label_ranker_oracle(missing, weighted):
    # The consensus is found by the Bucket Pivot Algorithm in exact arithmetic
    rng = np.random.RandomState(0)
    y = random_partial_rankings(30, 5, missing=missing, random_state=rng)
    sample_weight = _random_weights(len(y), rng) if weighted else np.ones(len(y))
    X = rng.rand(len(y), 2)

    ranker = DummyPartialLabelRanker().fit(X, y, sample_weight=sample_weight)

    expected = _bucket_oracles.aggregate(y, sample_weight)
    assert_array_equal(ranker.ranking_, expected)
    assert_array_equal(ranker.predict(rng.rand(7, 2)), np.tile(expected, (7, 1)))


@pytest.mark.parametrize(
    "ranker, y, expected",
    [
        # The Borda totals are 9, 10 and 5, so the consensus differs from the most
        # frequent ranking, [1, 2, 3]
        (DummyLabelRanker(), Y_EXAMPLE, [2, 1, 3]),
        # The maximum likelihood estimate of the parameters is about 0.349, 0.566
        # and 0.086
        (DummyLabelRanker(model="plackett_luce"), Y_EXAMPLE, [2, 1, 3]),
        # The entries of the pair order matrix are 3/8 for the first two labels,
        # 3/4 for the first and the third and 7/8 for the second and the third. The
        # second label is the pivot, the first one joins its bucket, and the third
        # one goes after it
        (DummyPartialLabelRanker(), Y_PARTIAL_EXAMPLE, [1, 1, 2]),
    ],
)
@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_consensus_examples(ranker, y, expected):
    ranker.fit(X_EXAMPLE, y)
    assert_array_equal(ranker.ranking_, expected)
    assert_array_equal(ranker.predict([[1.1], [2.2]]), [expected, expected])


@pytest.mark.parametrize(
    "ranker, expected",
    [
        # Every label gets the same Borda votes, so they keep their order
        (DummyLabelRanker(), [1, 2, 3]),
        # Every parameter is the same, so the labels keep their order
        (DummyLabelRanker(model="plackett_luce"), [1, 2, 3]),
        # No label is ranked together with another one, so all of them are tied
        # in the last bucket
        (DummyPartialLabelRanker(), [1, 1, 1]),
    ],
)
def test_consensus_no_ranked_labels(ranker, expected):
    y = np.full((2, 3), np.nan)
    assert_array_equal(ranker.fit([[0], [1]], y).ranking_, expected)


def test_partial_label_ranker_unranked_label():
    # The last label is never ranked, so it goes alone in the last bucket, after
    # the order of the other two on which the rankings agree
    y = [[1, 2, np.nan], [1, 2, np.nan], [1, 2, np.nan]]
    ranker = DummyPartialLabelRanker().fit([[0], [1], [2]], y)
    assert_array_equal(ranker.ranking_, [1, 2, 3])


@pytest.mark.parametrize("ranker", RANKERS)
@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_consensus_weights(ranker):
    # Integer weights are the same as repeated samples, zero weights the same as
    # removed samples, and scaling the weights does not change the consensus
    rng = np.random.RandomState(0)
    draw = random_partial_rankings if ranker._allow_ties else random_rankings
    y = draw(20, 4, missing=0.2, random_state=rng)
    X = rng.rand(len(y), 2)
    sample_weight = rng.randint(0, 4, size=len(y))

    expected = clone(ranker).fit(X, y, sample_weight=sample_weight).ranking_
    repeated = np.repeat(np.arange(len(y)), sample_weight)
    assert_array_equal(clone(ranker).fit(X[repeated], y[repeated]).ranking_, expected)
    assert_array_equal(
        clone(ranker).fit(X, y, sample_weight=sample_weight * 0.37).ranking_, expected
    )


@pytest.mark.parametrize("ranker", RANKERS)
def test_consensus_weights_none(ranker):
    rng = np.random.RandomState(0)
    y = random_rankings(15, 4, missing=0.2, random_state=rng)
    X = rng.rand(len(y), 2)
    assert_array_equal(
        clone(ranker).fit(X, y).ranking_,
        clone(ranker).fit(X, y, sample_weight=np.ones(len(y))).ranking_,
    )


@pytest.mark.parametrize(
    "ranker",
    [*RANKERS, DummyPartialLabelRanker(strategy="constant", constant=[1, 1, 2])],
)
@pytest.mark.parametrize(
    "sample_weight, message",
    [
        ([1, -1, 1, 1], "Negative values in data passed to `sample_weight`"),
        ([0, 0, 0, 0], "Sample weights must contain at least one non-zero number"),
    ],
)
def test_invalid_weights(ranker, sample_weight, message):
    with pytest.raises(ValueError, match=message):
        ranker.fit(X_EXAMPLE, Y_EXAMPLE, sample_weight=sample_weight)


@pytest.mark.parametrize("ranker", RANKERS)
@pytest.mark.parametrize(
    "X",
    [
        np.zeros((3, 4, 2)),
        np.array([["a", "b"]] * 3, dtype=object),
        csr_array(np.ones((3, 5))),
        [[np.nan, np.inf]] * 3,
    ],
    ids=["3d", "strings", "sparse", "nan"],
)
@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_predict_ignores_X(ranker, X):
    # The samples are neither validated nor used, apart from their number
    ranker.fit(X_EXAMPLE, Y_EXAMPLE)
    assert_array_equal(ranker.predict(X), np.tile(ranker.ranking_, (3, 1)))
    assert_array_equal(clone(ranker).fit(X, Y_EXAMPLE[:3]).predict(X).shape, (3, 3))


@pytest.mark.parametrize("ranker", RANKERS)
@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_fit_target_dtypes(ranker):
    y = np.array(Y_EXAMPLE)
    expected = clone(ranker).fit(X_EXAMPLE, y).ranking_
    for dtype in [np.int32, np.float32, np.float64]:
        ranking = clone(ranker).fit(X_EXAMPLE, y.astype(dtype)).ranking_
        assert ranking.dtype == np.intp
        assert_array_equal(ranking, expected)


def test_label_ranker_rejects_ties():
    with pytest.raises(ValueError, match="Expected rankings without ties in y"):
        DummyLabelRanker().fit(X_EXAMPLE, Y_PARTIAL_EXAMPLE)


def test_label_ranker_max_iter():
    # The rankings agree, so the maximum likelihood estimate of the Plackett-Luce
    # model does not exist and the estimation stops at max_iter
    ranker = DummyLabelRanker(model="plackett_luce", max_iter=5)
    with pytest.warns(ConvergenceWarning, match="max_iter=5"):
        ranker.fit(X_EXAMPLE, [[1, 2, 3]] * 4)
    assert_array_equal(ranker.ranking_, [1, 2, 3])


@pytest.mark.parametrize(
    "ranker, constant",
    [
        (DummyLabelRanker(strategy="constant"), [3, 1, 2]),
        (DummyLabelRanker(strategy="constant"), np.array([3.0, 1.0, 2.0])),
        (DummyPartialLabelRanker(strategy="constant"), [2, 1, 1]),
        (DummyPartialLabelRanker(strategy="constant"), (1, 1, 1)),
    ],
)
def test_constant(ranker, constant):
    ranker.set_params(constant=constant).fit(X_EXAMPLE, Y_EXAMPLE)
    assert ranker.ranking_.dtype == np.intp
    assert_array_equal(ranker.ranking_, constant)
    assert_array_equal(ranker.predict([[5], [6]]), [constant, constant])
    # The parameter is left as given
    assert ranker.constant is constant


@pytest.mark.parametrize("ranker", RANKERS)
def test_consensus_ignores_constant(ranker):
    expected = clone(ranker).fit(X_EXAMPLE, Y_EXAMPLE).ranking_
    ranker = clone(ranker).set_params(constant=[1, 2])
    assert_array_equal(ranker.fit(X_EXAMPLE, Y_EXAMPLE).ranking_, expected)


def test_constant_ignores_weights():
    ranker = DummyLabelRanker(strategy="constant", constant=[3, 1, 2])
    ranker.fit(X_EXAMPLE, Y_EXAMPLE, sample_weight=[0, 1, 2, 3])
    assert_array_equal(ranker.ranking_, [3, 1, 2])


@pytest.mark.parametrize(
    "ranker, constant, message",
    [
        (
            DummyLabelRanker(strategy="constant"),
            None,
            "has to be given when strategy='constant', got constant=None",
        ),
        (
            DummyPartialLabelRanker(strategy="constant"),
            [1, 2],
            r"Expected a constant ranking of 3 labels, as y, got constant=\[1, 2\]",
        ),
        (
            DummyLabelRanker(strategy="constant"),
            [[1, 2, 3]],
            r"Expected a constant ranking of 3 labels, as y, got constant=\[\[1, 2",
        ),
        (
            DummyLabelRanker(strategy="constant"),
            [1, 1, 2],
            r"complete ranking without ties as constant, .* constant=\[1, 1, 2\]",
        ),
        (
            DummyPartialLabelRanker(strategy="constant"),
            [1, np.nan, 2],
            r"Expected a complete ranking as constant, .* got constant=\[1, nan, 2\]",
        ),
        (
            DummyPartialLabelRanker(strategy="constant"),
            [1, 3, 3],
            r"Expected a complete ranking as constant, .* got constant=\[1, 3, 3\]",
        ),
    ],
)
def test_constant_invalid(ranker, constant, message):
    ranker.set_params(constant=constant)
    with pytest.raises(ValueError, match=message):
        ranker.fit(X_EXAMPLE, Y_EXAMPLE)


@pytest.mark.parametrize(
    "ranker, score",
    [(DummyLabelRanker(), kendall_tau_score), (DummyPartialLabelRanker(), tau_x_score)],
)
def test_score(ranker, score):
    ranker.fit(X_EXAMPLE, Y_EXAMPLE)
    y_pred = np.tile(ranker.ranking_, (len(Y_EXAMPLE), 1))
    assert ranker.score(X_EXAMPLE, Y_EXAMPLE) == score(Y_EXAMPLE, y_pred)


@pytest.mark.parametrize("ranker", RANKERS)
@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_n_features_in(ranker):
    ranker.fit(np.zeros((4, 7)), Y_EXAMPLE)
    assert ranker.n_features_in_ == 7
    assert not hasattr(ranker, "feature_names_in_")
