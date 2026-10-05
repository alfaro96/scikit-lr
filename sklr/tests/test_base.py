import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils import get_tags

from sklr.base import (
    LabelRankerMixin,
    PartialLabelRankerMixin,
    is_label_ranker,
    is_partial_label_ranker,
)


class LabelRanker(LabelRankerMixin, BaseEstimator):
    pass


class PartialLabelRanker(PartialLabelRankerMixin, BaseEstimator):
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
    # The rankers are neither classifiers nor regressors
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
