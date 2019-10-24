"""Testing for the forest module."""


# =============================================================================
# Imports
# =============================================================================

# Standard
from itertools import product

# Third party
import numpy as np
import pytest

# Local application
from plr.datasets import load_iris
from plr.ensemble import (
    RandomForestLabelRanker, RandomForestPartialLabelRanker)
from plr.ensemble._forest import _get_n_samples_bootstrap
from plr.utils import check_random_state


# =============================================================================
# Initialization
# =============================================================================

# Initialize a seed to always obtain the same results
seed = 198075

# The random number generator
random_state = check_random_state(seed)

# The criteria
LR_CRITERIA = ["mallows"]
PLR_CRITERIA = ["disagreements", "distance", "entropy"]

# The distances
DISTANCES = ["kendall"]

# The splitter
SPLITTERS = ["binary", "frequency", "width"]

# The forests
FORESTS = [RandomForestLabelRanker, RandomForestPartialLabelRanker]

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
    """Check Label Ranking on a toy dataset."""
    # Initialize the Random Forests Label Ranker
    # using "sqrt" features at each internal node
    forest = RandomForestLabelRanker(n_estimators=10,
                                     max_samples=0.5,
                                     random_state=seed)

    # Fit the forest to the data
    clf = forest.fit(
        X_train, Y_train,
        sample_weight=random_state.randint(
                10, size=(X_train.shape[0])))

    # Assert that the predictions are correct
    np.testing.assert_array_equal(clf.predict(X_test), Y_test)

    # Initialize the Random Forests Label Ranker
    # using one feature at each internal node
    forest = RandomForestLabelRanker(n_estimators=10,
                                     max_samples=0.5,
                                     max_features=1,
                                     random_state=seed)

    # Fit the forest to the data
    clf = forest.fit(
        X_train, Y_train,
        sample_weight=random_state.randint(
                10, size=(X_train.shape[0])))

    # Assert that the predictions are correct
    np.testing.assert_array_equal(clf.predict(X_test), Y_test)


@pytest.mark.partial_label_ranking_toy
def test_partial_label_ranking_toy():
    """Check Partial Label Ranking on a toy dataset."""
    # Initialize the Random Forests Partial Label Ranker
    # using "sqrt" features at each internal node
    forest = RandomForestPartialLabelRanker(n_estimators=10,
                                            max_samples=0.5,
                                            random_state=seed)

    # Fit the forest to the data
    clf = forest.fit(
        X_train, Y_train,
        sample_weight=random_state.randint(
                10, size=(X_train.shape[0])))

    # Assert that the predictions are correct
    np.testing.assert_array_equal(clf.predict(X_test), Y_test)

    # Initialize the Random Forests Partial Label Ranker
    # using one feature at each internal node
    forest = RandomForestPartialLabelRanker(n_estimators=10,
                                            max_samples=0.5,
                                            max_features=1,
                                            random_state=seed)

    # Fit the forest to the data
    clf = forest.fit(
        X_train, Y_train,
        sample_weight=random_state.randint(
                10, size=(X_train.shape[0])))

    # Assert that the predictions are correct
    np.testing.assert_array_equal(clf.predict(X_test), Y_test)


@pytest.mark.iris_lr
@pytest.mark.parametrize(
    "criterion,distance,splitter", product(LR_CRITERIA, DISTANCES, SPLITTERS))
def test_iris_lr(criterion, distance, splitter):
    """Test consistency on dataset iris for Random Forest Label Rankers."""
    # Initialize the Random Forests Label Ranker using all features
    forest = RandomForestLabelRanker(n_estimators=10,
                                     criterion=criterion,
                                     distance=distance,
                                     splitter=splitter,
                                     random_state=seed)

    # Fit the Random Forests Label Ranker
    clf = forest.fit(iris.data_lr[train_idx], iris.ranks_lr[train_idx])

    # Assert that a minimum score is achieved
    assert clf.score(iris.data_lr[test_idx], iris.ranks_lr[test_idx]) > 0.9

    # Initialize the Random Forests Label Ranker using two features
    model = RandomForestLabelRanker(n_estimators=10,
                                    criterion=criterion,
                                    distance=distance,
                                    splitter=splitter,
                                    max_features=2,
                                    random_state=seed)

    # Fit the Random Forests Label Ranker
    clf = model.fit(iris.data_lr[train_idx], iris.ranks_lr[train_idx])

    # Assert that a minimum score is achieved
    assert clf.score(iris.data_lr[test_idx], iris.ranks_lr[test_idx]) > 0.5

    # Also, without bootstrapping
    forest = RandomForestLabelRanker(n_estimators=10,
                                     bootstrap=False,
                                     criterion=criterion,
                                     distance=distance,
                                     splitter=splitter,
                                     random_state=seed)

    # Fit the Random Forests Label Ranker
    clf = forest.fit(iris.data_lr[train_idx], iris.ranks_lr[train_idx])

    # Assert that a minimum score is achieved
    assert clf.score(iris.data_lr[test_idx], iris.ranks_lr[test_idx]) > 0.9


@pytest.mark.iris_plr
@pytest.mark.parametrize(
    "criterion,splitter", product(PLR_CRITERIA, SPLITTERS))
def test_iris_plr(criterion, splitter):
    """Test consistency on dataset iris for
    Random Forest Partial Label Rankers."""
    # Initialize the Random Forests Partial Label Ranker using all features
    forest = RandomForestPartialLabelRanker(n_estimators=10,
                                            criterion=criterion,
                                            splitter=splitter,
                                            random_state=seed)

    # Fit the Random Forests Partial Label Ranker
    clf = forest.fit(iris.data_plr[train_idx], iris.ranks_plr[train_idx])

    # Assert that a minimum score is achieved
    assert clf.score(iris.data_plr[test_idx], iris.ranks_plr[test_idx]) > 0.9

    # Initialize the Random Forests Label Ranker using two features
    model = RandomForestPartialLabelRanker(n_estimators=10,
                                           criterion=criterion,
                                           splitter=splitter,
                                           max_features=2,
                                           random_state=seed)

    # Fit the Random Forests Label Ranker
    clf = model.fit(iris.data_plr[train_idx], iris.ranks_plr[train_idx])

    # Assert that a minimum score is achieved
    assert clf.score(iris.data_plr[test_idx], iris.ranks_plr[test_idx]) > 0.5

    # Also, without bootstrapping
    forest = RandomForestPartialLabelRanker(n_estimators=10,
                                            bootstrap=False,
                                            criterion=criterion,
                                            splitter=splitter,
                                            random_state=seed)

    # Fit the Random Forests Partial Label Ranker
    clf = forest.fit(iris.data_plr[train_idx], iris.ranks_plr[train_idx])

    # Assert that a minimum score is achieved
    assert clf.score(iris.data_plr[test_idx], iris.ranks_plr[test_idx]) > 0.9


@pytest.mark.max_samples_exceptions
def test_max_samples_exceptions():
    """Test invalid max_samples values."""
    # Test max_samples
    with pytest.raises(ValueError):
        _get_n_samples_bootstrap(100, -1)
    with pytest.raises(ValueError):
        _get_n_samples_bootstrap(100, 0.0)
    with pytest.raises(ValueError):
        _get_n_samples_bootstrap(100, 1.0)
    with pytest.raises(ValueError):
        _get_n_samples_bootstrap(100, 1000)
    with pytest.raises(TypeError):
        _get_n_samples_bootstrap(100, "foo")

    # Properly formatted samples
    assert _get_n_samples_bootstrap(100, None) == 100
    assert _get_n_samples_bootstrap(100, 50) == 50
    assert _get_n_samples_bootstrap(100, 0.5) == 50
