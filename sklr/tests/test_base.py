"""Testing for base classes of all estimators."""


# =============================================================================
# Imports
# =============================================================================

# Third party
import numpy as np
import pytest

# Local application
from sklr.base import BaseEstimator
from sklr.base import is_label_ranker, is_partial_label_ranker
from sklr.datasets import load_iris
from sklr.tree import DecisionTreeLabelRanker, DecisionTreePartialLabelRanker


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
        self.c = c
        self.d = d


# =============================================================================
# T
# =============================================================================
class T(BaseEstimator):
    """T."""

    def __init__(self, a=None, b=None):
        """Constructor."""
        self.a = a
        self.b = b


# =============================================================================
# Initialization
# =============================================================================

# Initialize a seed to always obtain the same results. It
# has been obtained from the service https://www.random.org,
# which seems to obtain true random numbers
seed = 198075

# Initialize the random state generator using the seed.
# This will guarantee that the tests are reproducible
# by anyone even if they are run in other conditions
random_state = np.random.RandomState(seed)


# =============================================================================
# Testing
# =============================================================================

@pytest.mark.get_hyperparams
def test_get_hyperparams():
    """Test the get_hyperparams method."""
    # Initialize an estimator whose hyperparameters are estimators
    # to ensure that the nested hyperparameters are properly got
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
    # Initialize an estimator whose hyperparameters are
    # estimators to ensure that the nested hyperparameters
    # are properly set in the corresponding objects
    test = T(a=K(), b=K())

    # Set an hyperparameter (except in the first call) into the
    # estimator and an inner hyperparameters into the sub-estimators
    test.set_hyperparams()
    test.set_hyperparams(b=None)
    test.set_hyperparams(a__d=2)

    # Assert the hyperparameters has been properly set for both,
    # the estimator and the nested objects that are estimators
    assert test.b is None
    assert test.a.d == 2

    # Assert that an error is raised when the
    # hyperparameter does not belong to the estimator
    with pytest.raises(ValueError):
        test.set_hyperparams(c=None)


def test_score():
    """Test the score method."""
    # Initialize the iris dataset, which will be employed
    # for testing the score method for the Label Ranking
    # problem and the Partial Label Ranking problem, with
    # and without sample weighting to ensure that the
    # score without sample weighting is different than with it
    iris = load_iris()

    # Initialize estimators for the Label Ranking problem
    # and the Partial Label Ranking problem, that is, a
    # Label Ranker and a Partial Label Ranker
    estimators = [
        DecisionTreeLabelRanker(random_state=seed, max_depth=3),
        DecisionTreePartialLabelRanker(random_state=seed, max_depth=3)
    ]

    # Initialize the data and the rankings of the
    # iris dataset for the Label Ranking problem
    # and the Partial Label Ranking problem
    datasets = [
        (iris.data_lr, iris.ranks_lr),
        (iris.data_plr, iris.ranks_plr)
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
        # Assert that the score without and with sample weighting is different
        assert score != weighted_score


@pytest.mark.is_label_ranker
def test_is_label_ranker():
    """Test the is_label_ranker method."""
    # Assert that an estimator of the class
    # DecisionTreeLabelRanker is a Label Ranker
    assert is_label_ranker(
        DecisionTreeLabelRanker(random_state=seed))

    # Assert that an estimator of the class
    # DecisionTreePartialLabelRanker
    # is not a Label Ranker
    assert not is_label_ranker(
        DecisionTreePartialLabelRanker(random_state=seed))


@pytest.mark.is_partial_label_ranker
def test_is_partial_label_ranker():
    """Test the is_partial_label_ranker method."""
    # Assert that an estimator of the class
    # DecisionTreePartialLabelRanker
    # is a Partial Label Ranker
    assert is_partial_label_ranker(
        DecisionTreePartialLabelRanker(random_state=seed))

    # Assert that an estimator of the class
    # DecisionTreeLabelRanker is not a
    # Partial Label Ranker
    assert not is_partial_label_ranker(
        DecisionTreeLabelRanker(random_state=seed))
