"""Testing for base classes of all estimators."""


# =============================================================================
# Imports
# =============================================================================

# Third party
import numpy as np
import pytest

# Local application
from plr.base import BaseEstimator
from plr.base import is_label_ranker, is_partial_label_ranker
from plr.datasets import load_iris
from plr.neighbors import KNeighborsLabelRanker, KNeighborsPartialLabelRanker


# =============================================================================
# Initialization
# =============================================================================

# Initialize a seed to always obtain the same results
seed = 198075

# Initialize the random state generator
random_state = np.random.RandomState(seed)


# =============================================================================
# Classes
# =============================================================================

# =============================================================================
# P
# =============================================================================
class P(BaseEstimator):
    """P."""
    pass


# =============================================================================
# K
# =============================================================================
class K(BaseEstimator):
    """K."""

    def __init__(self, c=None, d=None):
        """Constructor."""
        # Initialize the hyperparameters
        self.c = c
        self.d = d


# =============================================================================
# T
# =============================================================================
class T(BaseEstimator):
    """T."""

    def __init__(self, a=None, b=None):
        """Constructor."""
        # Initialize the hyperparameters
        self.a = a
        self.b = b


# =============================================================================
# Testing
# =============================================================================

@pytest.mark.get_hyperparams
def test_get_hyperparams():
    """Test the get_hyperparams method."""
    # Initialize an estimator whose hyperparameters are estimators
    test = T(a=K(), b=K())

    # Assert that the hyperparameters are properly obtained according
    # to whether those of the sub-objects must be also obtained
    assert "a__d" in test.get_hyperparams(deep=True)
    assert "a__d" not in test.get_hyperparams(deep=False)

    # Assert that an exception is raised when the estimator does not set
    # their hyperparameters in the signature of their __init__ constructor
    with pytest.raises(RuntimeError):
        P().get_hyperparams(deep=True)


@pytest.mark.set_hyperparams
def test_set_hyperparams():
    """Test the set_hyperparams method."""
    # Initialize an estimator whose hyperparameters are estimators
    test = T(a=K(), b=K())

    # Set zero hyperparameters
    test.set_hyperparams()

    # Set an hyperparameter into the estimator and
    # an inner hyperparameters into the sub-estimators
    test.set_hyperparams(b=None)
    test.set_hyperparams(a__d=2)

    # Assert the hyperparameters has been properly set
    assert test.b is None
    assert test.a.d == 2

    # Assert that an error is raised when
    # the hyperparameter is not in the estimator
    with pytest.raises(ValueError):
        test.set_hyperparams(c=None)


def test_score_sample_weight():
    """Test the score method."""
    # Initialize the estimators
    # (LabelRankerMixin and PartialLabelRankingMixin)
    estimators = [
        KNeighborsLabelRanker(),
        KNeighborsPartialLabelRanker()
    ]

    # Initialize the datasets
    datasets = load_iris()
    datasets = [
        (datasets.data_lr, datasets.ranks_lr),
        (datasets.data_plr, datasets.ranks_plr)
    ]

    # Check that the score obtained without
    # and with sample weighting is different
    for (estimator, dataset) in zip(estimators, datasets):
        # Fit the estimator
        estimator.fit(X=dataset[0], Y=dataset[1])
        # Generate random sample weights
        sample_weight = random_state.randint(1, 10, size=dataset[0].shape[0])
        # Obtain the score without and with sample weighting
        score = estimator.score(
            X=dataset[0], Y=dataset[1])
        weighted_score = estimator.score(
            X=dataset[0], Y=dataset[1], sample_weight=sample_weight)
        # Check that the score without and with sample weighting is different
        assert score != weighted_score


@pytest.mark.is_label_ranker
def test_is_label_ranker():
    """Test the is_label_ranker method."""
    assert is_label_ranker(KNeighborsLabelRanker())
    assert not is_label_ranker(KNeighborsPartialLabelRanker())


@pytest.mark.is_partial_label_ranker
def test_is_partial_label_ranker():
    """Test the is_partial_label_ranker method."""
    assert is_partial_label_ranker(KNeighborsPartialLabelRanker())
    assert not is_partial_label_ranker(KNeighborsLabelRanker())
