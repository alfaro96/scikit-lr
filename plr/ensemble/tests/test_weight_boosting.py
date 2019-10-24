"""Testing for the boost module."""


# =============================================================================
# Imports
# =============================================================================

# Third party
import numpy as np
import pytest

# Local application
from plr.datasets import load_iris
from plr.ensemble import (
    AdaBoostLabelRanker, AdaBoostPartialLabelRanker,
    RandomForestLabelRanker, RandomForestPartialLabelRanker)
from plr.neighbors import KNeighborsLabelRanker, KNeighborsPartialLabelRanker
from plr.tree import DecisionTreeLabelRanker, DecisionTreePartialLabelRanker
from plr.utils import check_random_state


# =============================================================================
# Initialization
# =============================================================================

# Initialize a seed to always obtain the same results
seed = 198075

# The random number generator
random_state = check_random_state(seed)

# Toy example

# Training

# Data
X_train = np.array([
    [-2, -1],
    [-1, -1],
    [-1, -2],
    [1, 1],
    [1, 2],
    [2, 1]])

# Rankings
Y_train = np.array([
    [1, 2, 3],
    [1, 2, 3],
    [1, 2, 3],
    [2, 1, 3],
    [2, 1, 3],
    [2, 1, 3]])

# Test

# Data
X_test = np.array([
    [-1, -1],
    [2, 2],
    [3, 2]])

# Rankings
Y_test = np.array([
    [1, 2, 3],
    [2, 1, 3],
    [2, 1, 3]])

# Load the iris dataset
iris = load_iris()
idx = random_state.permutation(iris.data_lr.shape[0])
(train_size, test_size) = (int(0.8 * idx.shape[0]), int(0.2 * idx.shape[0]))
(train_idx, test_idx) = (idx[:train_size], idx[test_size:])


# =============================================================================
# Testing
# =============================================================================

@pytest.mark.label_ranking_toy
def test_label_ranking_toy():
    """Test Label Ranking on a toy dataset."""
    # Initialize the AdaBoost Label Ranker model and fit it
    adaboost = AdaBoostLabelRanker(random_state=seed)
    clf = adaboost.fit(X_train, Y_train)

    # Assert that the predictions are correct
    np.testing.assert_array_equal(clf.predict(X_test), Y_test)


@pytest.mark.partial_label_ranking_toy
def test_partiallabel_ranking_toy():
    """Test Partial Label Ranking on a toy dataset."""
    # Initialize the AdaBoost Partial Label Ranker model and fit it
    adaboost = AdaBoostPartialLabelRanker(random_state=seed)
    clf = adaboost.fit(X_train, Y_train)

    # Assert that the predictions are correct
    np.testing.assert_array_equal(clf.predict(X_test), Y_test)


@pytest.mark.iris_lr
def test_iris_lr():
    """Test consistency on dataset iris for Label Ranking."""
    # Initialize the AdaBoost Label Ranker model, fit it
    # and assert that a minimum score is achieved
    adaboost = AdaBoostLabelRanker(random_state=seed)
    clf = adaboost.fit(iris.data_lr[train_idx], iris.ranks_lr[train_idx])
    assert clf.score(iris.data_lr[test_idx], iris.ranks_lr[test_idx]) > 0.9

    # Check for distinct random states
    assert (len(set(est.random_state for est in clf.estimators_)) ==
            len(clf.estimators_))


@pytest.mark.iris_plr
def test_iris_plr():
    """Test consistency on dataset iris for Partial Label Ranking."""
    # Initialize the AdaBoost Label Ranker model, fit it
    # and assert that a minimum score is achieved
    adaboost = AdaBoostPartialLabelRanker(random_state=seed)
    clf = adaboost.fit(iris.data_plr[train_idx], iris.ranks_plr[train_idx])
    assert clf.score(iris.data_plr[test_idx], iris.ranks_plr[test_idx]) > 0.9

    # Check for distinct random states
    assert (len(set(est.random_state for est in clf.estimators_)) ==
            len(clf.estimators_))


@pytest.mark.error
def test_error():
    """Test that it gives proper exception on deficient input."""
    # Assert that an error is raised when the learning rate is less
    # than zero or when the learning rate is not an integer or floating
    with pytest.raises(ValueError):
        AdaBoostLabelRanker(learning_rate=-1,
                            random_state=seed).fit(X_train, Y_train)
    with pytest.raises(TypeError):
        AdaBoostLabelRanker(learning_rate="foo",
                            random_state=seed).fit(X_train, Y_train)

    # Assert that an error is raised when the
    # sample weights are not properly formatted
    with pytest.raises(ValueError):
        AdaBoostLabelRanker(random_state=seed).fit(
            X_train, Y_train, sample_weight=np.array([-1]))


def test_base_estimator():
    """Test different base estimators."""
    # Use a Random Forests Label Ranker as base estimator
    AdaBoostLabelRanker(base_estimator=RandomForestLabelRanker(),
                        random_state=seed).fit(X_train, Y_train)

    # Use a Random Forests Partial Label Ranker as base estimator
    AdaBoostPartialLabelRanker(base_estimator=RandomForestPartialLabelRanker(),
                               random_state=seed).fit(X_train, Y_train)


def test_sample_weight_missing():
    """Test that if sample weight is not supported, an error is raised."""
    # K-Neighbors Label Ranker and K-Neighbors Partial Label Ranker
    # does not support sample weighting so that an error is raised
    with pytest.raises(ValueError):
        AdaBoostLabelRanker(base_estimator=KNeighborsLabelRanker(),
                            random_state=seed).fit(X_train, Y_train)

    with pytest.raises(ValueError):
        AdaBoostPartialLabelRanker(base_estimator=(
                                        KNeighborsPartialLabelRanker()),
                                   random_state=seed).fit(X_train, Y_train)


def test_early_termination():
    """Test that early termination is working."""
    # Initialize two datasets, one with early termination
    # because a perfect fit can be achieved an another
    # one because the base estimator is not good enough
    X_perf = np.array([[1, 1], [1, 1]])
    Y_perf = np.array([[1, 2, 3], [1, 2, 3]])
    X_fail = np.array([[1, 1], [1, 1]])
    Y_fail = np.array([[1, 2, 3], [3, 2, 1]])

    # Fit the AdaBoost Label Rankers to both datasets
    clf1 = AdaBoostLabelRanker(random_state=seed).fit(X_perf, Y_perf)
    clf2 = AdaBoostLabelRanker(random_state=seed).fit(X_fail, Y_fail)

    # Assert that the length of the estimators is less than expected.
    # Also, assert that the weights are as expected
    assert len(clf1.estimators_) < clf1.n_estimators
    assert clf1.estimator_weights_[len(clf1.estimators_) - 1] == 1.0
    assert len(clf2.estimators_) < clf2.n_estimators
    assert clf2.estimator_weights_[len(clf2.estimators_) - 1] == 0.0

    # The same for the AdaBoost Partial Label Rankers
    clf1 = AdaBoostPartialLabelRanker(random_state=seed).fit(X_perf, Y_perf)
    clf2 = AdaBoostPartialLabelRanker(random_state=seed).fit(X_fail, Y_fail)

    assert len(clf1.estimators_) < clf1.n_estimators
    assert clf1.estimator_weights_[len(clf1.estimators_) - 1] == 1.0
    assert len(clf2.estimators_) < clf2.n_estimators
    assert clf2.estimator_weights_[len(clf2.estimators_) - 1] == 0.0
