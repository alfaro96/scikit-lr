import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy.stats import rankdata
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted, validate_data

from sklr.base import LabelRankerMixin, PartialLabelRankerMixin, is_partial_label_ranker
from sklr.utils import check_ranking
from sklr.utils._estimator_checks import (
    N_LABELS,
    _enforce_estimator_tags_y,
    _RankingCheck,
    parametrize_with_checks,
)
from sklr.utils._ranking import _validate_data


class _ConsensusRanker(BaseEstimator):
    """Predict for every sample the ranking of the labels by their mean position."""

    def fit(self, X, y):
        allow_ties = is_partial_label_ranker(self)
        X, y = _validate_data(self, X, y, allow_ties=allow_ties)
        # The mean position of a label is taken over the rankings that rank it,
        # and the labels that no ranking ranks go last
        is_ranked = ~np.isnan(y)
        n_ranked = is_ranked.sum(axis=0)
        positions = np.where(is_ranked, y, 0).sum(axis=0)
        mean_positions = np.full(y.shape[1], np.inf)
        np.divide(positions, n_ranked, out=mean_positions, where=n_ranked > 0)
        # The partial ranker ties the labels with the same mean position
        method = "dense" if allow_ties else "ordinal"
        self.ranking_ = rankdata(mean_positions, method=method).astype(np.intp)
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        return np.tile(self.ranking_, (X.shape[0], 1))


class ConsensusLabelRanker(LabelRankerMixin, _ConsensusRanker):
    pass


class ConsensusPartialLabelRanker(PartialLabelRankerMixin, _ConsensusRanker):
    pass


@parametrize_with_checks([ConsensusLabelRanker(), ConsensusPartialLabelRanker()])
def test_estimators(estimator, check):
    check(estimator)


@pytest.mark.parametrize(
    "estimator, expected",
    [
        (ConsensusLabelRanker(), [1, 2, 3, 4]),
        (ConsensusPartialLabelRanker(), [1, 1, 2, 3]),
    ],
)
def test_consensus_ranker_incomplete(estimator, expected):
    # The mean positions of the first three labels are 1.5, 1.5 and 3, from the
    # rankings that rank them, and the last label goes last because no ranking
    # ranks it. The last ranking ranks no label, so it does not change the mean
    # positions. The label ranker breaks the tie between the first two labels
    # in favour of the first one
    X = [[0], [1], [2]]
    y = [[1, 2, np.nan, np.nan], [2, 1, 3, np.nan], [np.nan, np.nan, np.nan, np.nan]]
    ranking = estimator.fit(X, y).predict([[0]])[0]
    assert_array_equal(ranking, expected)


@pytest.mark.parametrize(
    "estimator, expected_class_1",
    [(ConsensusLabelRanker(), [2, 1, 3]), (ConsensusPartialLabelRanker(), [2, 1, 2])],
)
def test_enforce_estimator_tags_y_classes(estimator, expected_class_1):
    y = np.array([0, 1, 2, 1, 0])
    y_enforced = _enforce_estimator_tags_y(None, estimator, y)
    # Each label is ranked by its distance to the class. Labels 0 and 2 are at
    # the same distance from class 1, so they are tied for the partial label
    # ranker, and the tie is broken in favour of label 0 for the label ranker
    expected = [[1, 2, 3], expected_class_1, [3, 2, 1], expected_class_1, [1, 2, 3]]
    assert_array_equal(y_enforced, expected)


@pytest.mark.parametrize(
    "estimator", [ConsensusLabelRanker(), ConsensusPartialLabelRanker()]
)
@pytest.mark.parametrize("shape", [(20,), (20, 2)])
def test_enforce_estimator_tags_y_real(estimator, shape):
    rng = np.random.RandomState(0)
    y = rng.normal(loc=1, scale=5, size=shape)
    y_enforced = _enforce_estimator_tags_y(None, estimator, y)
    assert y_enforced.shape == (shape[0], N_LABELS)
    # Valid rankings, with ties only for the partial label ranker
    check_ranking(y_enforced, allow_ties=is_partial_label_ranker(estimator))
    # A target with several outputs is reduced to its first one
    y_first = y if y.ndim == 1 else y[:, 0]
    assert_array_equal(y_enforced, _enforce_estimator_tags_y(None, estimator, y_first))


def test_enforce_estimator_tags_y_empty():
    y = np.array([], dtype=np.intp)
    y_enforced = _enforce_estimator_tags_y(None, ConsensusLabelRanker(), y)
    assert y_enforced.shape == (0, N_LABELS)


def test_enforce_estimator_tags_y_not_ranker():
    calls = []

    def enforce_estimator_tags_y(estimator, y):
        calls.append((estimator, y))
        return y

    estimator, y = BaseEstimator(), np.array([0, 1, 2])
    assert _enforce_estimator_tags_y(enforce_estimator_tags_y, estimator, y) is y
    assert calls == [(estimator, y)]


def test_parametrize_with_checks_expected_failed_checks():
    def expected_failed_checks(estimator):
        return {"check_fit2d_1sample": "Expected to fail."}

    mark = parametrize_with_checks(
        [ConsensusLabelRanker()], expected_failed_checks=expected_failed_checks
    )
    argnames, params = mark.args
    assert argnames == "estimator, check"
    for param in params:
        _, check = param.values
        assert isinstance(check, _RankingCheck)
        if check.func.__name__ == "check_fit2d_1sample":
            [xfail] = param.marks
            assert xfail.name == "xfail"
            assert xfail.kwargs["reason"] == "Expected to fail."
        else:
            assert not param.marks
