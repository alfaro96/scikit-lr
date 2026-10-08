import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils import get_tags

from sklr.base import (
    LabelRankerMixin,
    PartialLabelRankerMixin,
    is_label_ranker,
    is_partial_label_ranker,
)
from sklr.metrics import kendall_tau_score, tau_x_score


class _FixedRanker(BaseEstimator):
    """Predict the same ranking for every sample."""

    def __init__(self, ranking=(1, 2)):
        self.ranking = ranking

    def predict(self, X):
        return np.tile(self.ranking, (len(X), 1))


class LabelRanker(LabelRankerMixin, _FixedRanker):
    pass


class PartialLabelRanker(PartialLabelRankerMixin, _FixedRanker):
    pass


class Classifier(ClassifierMixin, BaseEstimator):
    pass


class Regressor(RegressorMixin, BaseEstimator):
    pass


@pytest.mark.parametrize(
    "estimator, estimator_type",
    [
        (LabelRanker(), "label_ranker"),
        (PartialLabelRanker(), "partial_label_ranker"),
    ],
)
def test_ranker_tags(estimator, estimator_type):
    tags = get_tags(estimator)
    assert tags.estimator_type == estimator_type
    assert tags.target_tags.required
    assert tags.target_tags.multi_output
    assert not tags.target_tags.single_output
    assert tags.classifier_tags is None
    assert tags.regressor_tags is None


@pytest.mark.parametrize(
    "estimator, expected_label_ranker, expected_partial_label_ranker",
    [
        (LabelRanker(), True, False),
        (PartialLabelRanker(), False, True),
        (Classifier(), False, False),
        (Regressor(), False, False),
        (BaseEstimator(), False, False),
    ],
)
def test_is_ranker(estimator, expected_label_ranker, expected_partial_label_ranker):
    assert is_label_ranker(estimator) is expected_label_ranker
    assert is_partial_label_ranker(estimator) is expected_partial_label_ranker


@pytest.mark.parametrize(
    "estimator, metric",
    [
        (LabelRanker(ranking=[1, 2, 3]), kendall_tau_score),
        (PartialLabelRanker(ranking=[1, 1, 2]), tau_x_score),
    ],
)
def test_ranker_score(estimator, metric):
    X = np.zeros((4, 1))
    y = np.array([[1, 2, 3], [3, 2, 1], [2, 1, 3], [1, 3, 2]])
    sample_weight = [1, 2, 3, 4]
    y_pred = estimator.predict(X)
    assert estimator.score(X, y) == pytest.approx(metric(y, y_pred))
    assert estimator.score(X, y, sample_weight=sample_weight) == pytest.approx(
        metric(y, y_pred, sample_weight=sample_weight)
    )


def test_label_ranker_score_rejects_ties():
    # The score of a label ranker is Kendall's tau, which rejects ties, so it
    # cannot be computed on partial label rankings
    estimator = LabelRanker(ranking=[1, 2, 3])
    with pytest.raises(ValueError, match="Expected rankings without ties in y_true"):
        estimator.score(np.zeros((1, 1)), [[1, 1, 2]])
