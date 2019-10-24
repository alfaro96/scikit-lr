"""Testing for the bagging ensemble module."""


# =============================================================================
# Imports
# =============================================================================

# Third party
import numpy as np
import pytest

# Local application
from plr.datasets import load_wine
from plr.ensemble import BaggingLabelRanker, BaggingPartialLabelRanker
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

# Load the wine dataset
wine = load_wine()
idx = random_state.permutation(wine.data_lr.shape[0])
(train_size, test_size) = (int(0.8 * idx.shape[0]), int(0.2 * idx.shape[0]))
(train_idx, test_idx) = (idx[:train_size], idx[test_size:])


# =============================================================================
# Testing
# =============================================================================

@pytest.mark.bootstrap_samples
def test_bootstrap_samples():
    """Test that bootstrapping samples generate non-perfect base estimators."""
    # Initialize the base estimator and the ensemble (without bootstraping)
    estimator = DecisionTreePartialLabelRanker(random_state=seed)
    ensemble = BaggingPartialLabelRanker(base_estimator=(
                                            DecisionTreePartialLabelRanker()),
                                         bootstrap=False,
                                         random_state=seed)

    # Fit the base estimator and the ensemble to the data
    clf1 = estimator.fit(wine.data_plr, wine.ranks_plr)
    clf2 = ensemble.fit(wine.data_plr, wine.ranks_plr)

    # Without bootstrap, all trees are perfect on the training set
    assert (clf1.score(wine.data_plr, wine.ranks_plr) ==
            clf2.score(wine.data_plr, wine.ranks_plr))

    # Initialize the ensemble (with bootstraping)
    ensemble = BaggingPartialLabelRanker(base_estimator=(
                                            DecisionTreePartialLabelRanker()),
                                         bootstrap=True,
                                         random_state=seed)

    # Fit the ensemble to the data
    clf2 = ensemble.fit(wine.data_plr, wine.ranks_lr)

    # With bootstrap, trees are no longer perfect on the training set
    assert (estimator.score(wine.data_plr, wine.ranks_plr) >
            ensemble.score(wine.data_plr, wine.ranks_plr))


@pytest.mark.bootstrap_features
def test_bootstrap_features():
    """Test that bootstrapping features may generate duplicate features."""
    # Initialize the ensemble (without features bootstraping)
    ensemble = BaggingPartialLabelRanker(base_estimator=(
                                            DecisionTreePartialLabelRanker()),
                                         bootstrap_features=False,
                                         random_state=seed)

    # Fit the ensemble to the data
    clf = ensemble.fit(wine.data_plr, wine.ranks_plr)

    # Assert that the features employed by each estimator is different
    assert all(map(lambda features: (wine.data_plr.shape[1] ==
                                     np.unique(features).shape[0]),
                   clf.estimators_features_))

    # Initialize the ensemble (with features bootstraping)
    ensemble = BaggingPartialLabelRanker(base_estimator=(
                                            DecisionTreePartialLabelRanker()),
                                         bootstrap_features=True,
                                         random_state=seed)

    # Fit the ensemble to the data
    clf = ensemble.fit(wine.data_plr, wine.ranks_plr)

    # Assert that the features employed by each estimator contains duplicates
    assert all(map(lambda features: (wine.data_plr.shape[1] >
                                     np.unique(features).shape[0]),
                   clf.estimators_features_))


@pytest.mark.single_estimator
def test_single_estimator():
    """Test singleton ensembles."""
    # Initialize the base estimator and the ensemble
    estimator = KNeighborsPartialLabelRanker()
    ensemble = BaggingPartialLabelRanker(base_estimator=(
                                            KNeighborsPartialLabelRanker()),
                                         n_estimators=1,
                                         bootstrap=False,
                                         bootstrap_features=False,
                                         random_state=seed)

    # Fit the base estimator and the ensemble to the data
    clf1 = estimator.fit(wine.data_plr, wine.ranks_plr)
    clf2 = ensemble.fit(wine.data_plr, wine.ranks_plr)

    # Assert that the predictions are the same
    np.testing.assert_array_equal(
        clf1.predict(wine.data_plr),
        clf2.predict(wine.data_plr))


@pytest.mark.error
def test_error():
    """Test that it gives proper exception on deficient input."""
    # Initialize the data and the rankings
    (X, Y) = (wine.data_lr, wine.ranks_lr)

    # Test max_samples
    with pytest.raises(ValueError):
        BaggingLabelRanker(max_samples=-1).fit(X, Y)
    with pytest.raises(ValueError):
        BaggingLabelRanker(max_samples=0.0).fit(X, Y)
    with pytest.raises(ValueError):
        BaggingLabelRanker(max_samples=2.0).fit(X, Y)
    with pytest.raises(ValueError):
        BaggingLabelRanker(max_samples=1000).fit(X, Y)
    with pytest.raises(TypeError):
        BaggingLabelRanker(max_samples="foo").fit(X, Y)

    # Test max_features
    with pytest.raises(ValueError):
        BaggingLabelRanker(max_features=-1).fit(X, Y)
    with pytest.raises(ValueError):
        BaggingLabelRanker(max_features=0.0).fit(X, Y)
    with pytest.raises(ValueError):
        BaggingLabelRanker(max_features=2.0).fit(X, Y)
    with pytest.raises(ValueError):
        BaggingLabelRanker(max_features=20).fit(X, Y)
    with pytest.raises(TypeError):
        BaggingLabelRanker(max_features="foobar").fit(X, Y)


@pytest.mark.base_estimator
def test_base_estimator():
    """Test base_estimator and its default values."""
    # Initialize the ensemble (None as base_estimator), fit it to the data
    # and assert that the base estimator is DecisionTreeLabelRanker
    ensemble = BaggingLabelRanker(base_estimator=None,
                                  random_state=seed)
    clf = ensemble.fit(wine.data_lr, wine.ranks_lr)
    assert isinstance(clf.base_estimator_, DecisionTreeLabelRanker)

    # Initialize the ensemble (DecisionTreeLabelRanker as base_estimator),
    # fit it to the data and assert that the base estimator is
    # DecisionTreeLabelRanker
    ensemble = BaggingLabelRanker(base_estimator=DecisionTreeLabelRanker(),
                                  random_state=seed)
    clf = ensemble.fit(wine.data_lr, wine.ranks_lr)
    assert isinstance(clf.base_estimator_, DecisionTreeLabelRanker)

    # Initialize the ensemble (KNeighborsLabelRanker as base_estimator),
    # fit it to the data and assert that the base estimator is
    # KNeighborsLabelRanker
    ensemble = BaggingLabelRanker(base_estimator=KNeighborsLabelRanker(),
                                  random_state=seed)
    clf = ensemble.fit(wine.data_lr, wine.ranks_lr)
    assert isinstance(clf.base_estimator_, KNeighborsLabelRanker)

    # Initialize the ensemble (None as base_estimator), fit it to the data
    # and assert that the base estimator is DecisionTreePartialLabelRanker
    ensemble = BaggingPartialLabelRanker(base_estimator=None,
                                         random_state=seed)
    clf = ensemble.fit(wine.data_plr, wine.ranks_plr)
    assert isinstance(clf.base_estimator_, DecisionTreePartialLabelRanker)

    # Initialize the ensemble (DecisionTreePartialLabelRanker as
    # base_estimator), fit it to the data and assert that the base
    # estimator is DecisionTreePartialLabelRanker
    ensemble = BaggingPartialLabelRanker(base_estimator=(
                                            DecisionTreePartialLabelRanker()),
                                         random_state=seed)
    clf = ensemble.fit(wine.data_plr, wine.ranks_plr)
    assert isinstance(clf.base_estimator_, DecisionTreePartialLabelRanker)

    # Initialize the ensemble (KNeighborsPartialLabelRanker as base_estimator),
    # fit it to the data and assert that the base estimator is
    # KNeighborsPartialLabelRanker
    ensemble = BaggingPartialLabelRanker(base_estimator=(
                                            KNeighborsPartialLabelRanker()),
                                         random_state=seed)
    clf = ensemble.fit(wine.data_plr, wine.ranks_plr)
    assert isinstance(clf.base_estimator_, KNeighborsPartialLabelRanker)


@pytest.mark.bagging_sample_weight_unsupported_but_passed
def test_bagging_sample_weight_unsupported_but_passed():
    """Test that error is raised when passing sample
    weights to an unsupported estimator."""
    # Initialize the ensemble
    ensemble = BaggingLabelRanker(base_estimator=KNeighborsLabelRanker(),
                                  random_state=seed)

    # Assert than an error is raised when passing sample weights
    # to a base estimator that does not support sample weighting
    with pytest.raises(ValueError):
        ensemble.fit(
            wine.data_lr, wine.ranks_lr,
            sample_weight=random_state.randint(
                10, size=(wine.data_lr.shape[0])))

    # For the sake of coverage, test that no exception is
    # raised when sample weights are used with supported estimator

    # Initialize and fit the ensemble
    ensemble = BaggingLabelRanker(random_state=seed)
    ensemble.fit(
        wine.data_lr, wine.ranks_lr,
        sample_weight=random_state.randint(
                10, size=(wine.data_lr.shape[0])))


@pytest.mark.max_samples_consistency
def test_max_samples_consistency():
    """Test to make sure validated max_samples and original max_samples
    are identical when valid integer max_samples supplied by user."""
    # Initialize the maximum number of samples
    max_samples = 100

    # Initialize the ensemble
    ensemble = BaggingLabelRanker(KNeighborsLabelRanker(),
                                  max_samples=max_samples,
                                  max_features=0.5,
                                  random_state=seed)

    # Fit the ensemble to the data
    clf = ensemble.fit(wine.data_lr, wine.ranks_lr)

    # Assert that the maximum number of samples is the same
    assert clf._max_samples == max_samples
