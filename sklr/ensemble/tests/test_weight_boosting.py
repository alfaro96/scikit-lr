"""Testing for the boost module."""


# =============================================================================
# Imports
# =============================================================================

# Third party
import numpy as np
import pytest

# Local application
from sklr.datasets import load_iris
from sklr.ensemble import AdaBoostLabelRanker, RandomForestLabelRanker
from sklr.neighbors import KNeighborsLabelRanker
from sklr.tree import DecisionTreeLabelRanker
from sklr.utils import check_random_state


# =============================================================================
# Initialization
# =============================================================================

# The seed and the random state generator
# to always obtain the same results and,
# so, ensure that the tests carried out
# are always the same
seed = 198075
random_state = check_random_state(seed)

# The following variables are required for some
# test methods to work. Even if they will not be
# used in all the tests, they are globally declared
# to avoid that they are defined along several methods.
# The extra memory overhead should not be an issue

# Not allowed base estimators
LR_DEFFICIENT_BASE_ESTIMATORS = [KNeighborsLabelRanker]
DEFFICIENT_BASE_ESTIMATORS = [*LR_DEFFICIENT_BASE_ESTIMATORS]

# The boosting rankers
LR_BOOSTING = [AdaBoostLabelRanker]
BOOSTING = [*LR_BOOSTING]

# A toy example to check that the forest rankers
# are properly working. In fact, initialize two datasets,
# one with training data and rankings and another one with
# test data and rankings

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


# =============================================================================
# Testing
# =============================================================================

@pytest.mark.toy_example
@pytest.mark.parametrize("AdaBoostRanker", BOOSTING)
def test_toy_example(AdaBoostRanker):
    """Test the boosting rankers on a toy dataset."""
    # Initialize the boosting ranker using an small number
    # of estimators to not introduce too much memory overhead
    model = AdaBoostRanker(n_estimators=10, random_state=seed)

    # Fit the boosting ranker
    # to the training dataset
    clf = model.fit(X_train, Y_train)

    # Obtain the predictions of the boosting ranker
    Y_pred = clf.predict(X_test)

    # Assert that the predictions are correct
    np.testing.assert_array_equal(Y_pred, Y_test)


@pytest.mark.iris
@pytest.mark.parametrize("AdaBoostRanker", BOOSTING)
def test_iris(AdaBoostRanker):
    """Test the boosting rankers on iris
    dataset for the sake of coverage."""
    # Initialize the boosting ranker using an small number
    # of estimators to not introduce too much memory overhead
    model = AdaBoostRanker(n_estimators=10, random_state=seed)

    # Initialize the iris dataset
    iris = load_iris()
    (X, Y) = (iris.data_lr, iris.ranks_lr)

    # Fit the boosting ranker
    # to the training dataset
    clf = model.fit(X, Y)

    # Assert that the score is less than expected
    # because the base estimator is too simple
    assert clf.score(X, Y) < 0.95


@pytest.mark.error
@pytest.mark.parametrize("AdaBoostRanker", BOOSTING)
def test_error(AdaBoostRanker):
    """Test that it gives proper exception on deficient input."""
    # Initialize the boosting ranker using an small number
    # of estimators to not introduce too much memory overhead
    model = AdaBoostRanker(random_state=seed)

    # Assert that an error is raised when
    # the learning rate is less than zero
    with pytest.raises(ValueError):
        model.set_hyperparams(learning_rate=-1).fit(X_train, Y_train)

    # Assert that an error is raised when the learning
    # rate is not an integer or floating data type
    with pytest.raises(TypeError):
        model.set_hyperparams(learning_rate="foo").fit(X_train, Y_train)


@pytest.mark.sample_weight_missing
@pytest.mark.parametrize(
    "DefficientBaseEstimator,AdaBoostRanker",
    zip(DEFFICIENT_BASE_ESTIMATORS, BOOSTING))
def test_sample_weight_missing(DefficientBaseEstimator, AdaBoostRanker):
    """Test that if sample weight is not supported, an error is raised."""
    # Initialize the boosting ranker
    # with deficient base estimator
    model = AdaBoostRanker(base_estimator=DefficientBaseEstimator(),
                           random_state=seed)

    # Assert that an exception is raised when trying
    # to fit a boosting estimator whose base estimator
    # does not support sample weighting
    with pytest.raises(ValueError):
        model.fit(X_train, Y_train)


@pytest.mark.parametrize("AdaBoostRanker", BOOSTING)
def test_early_termination(AdaBoostRanker):
    """Test that early termination is working."""
    # Initialize two datasets, one with early termination
    # because a perfect fit can be achieved an another
    # one because the base estimator is not good enough
    X_perf = np.array([[1, 1], [1, 1]])
    Y_perf = np.array([[1, 2, 3], [1, 2, 3]])
    X_fail = np.array([[1, 1], [1, 1]])
    Y_fail = np.array([[1, 2, 3], [3, 2, 1]])

    # Initialize the boosting rankers
    # (using an small number of estimators to
    # not introduce too much overhead)
    model1 = AdaBoostRanker(n_estimators=10, random_state=seed)
    model2 = AdaBoostRanker(n_estimators=10, random_state=seed)

    # Fit the boosting rankers, one to the dataset with
    # perfect fit and another one with a fail fit
    clf1 = model1.fit(X_perf, Y_perf)
    clf2 = model2.fit(X_fail, Y_fail)

    # Assert that the length of the
    # estimators is less than expected
    assert len(clf1.estimators_) < clf1.n_estimators
    assert len(clf2.estimators_) < clf2.n_estimators

    # Assert that the weights are as expected (one
    # for the perfect fit and zero for the fail fit)
    assert clf1.estimator_weights_[len(clf1.estimators_) - 1] == 1.0
    assert clf2.estimator_weights_[len(clf2.estimators_) - 1] == 0.0
