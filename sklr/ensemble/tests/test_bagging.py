"""Testing for the bagging ensemble module."""


# =============================================================================
# Imports
# =============================================================================

# Third party
import numpy as np
import pytest

# Local application
from sklr.ensemble import BaggingLabelRanker, BaggingPartialLabelRanker
from sklr.neighbors import KNeighborsLabelRanker, KNeighborsPartialLabelRanker
from sklr.tree import DecisionTreeLabelRanker, DecisionTreePartialLabelRanker
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

# The default estimators
LR_DEFAULT_ESTIMATOR = DecisionTreeLabelRanker
PLR_DEFAULT_ESTIMATOR = DecisionTreePartialLabelRanker
DEFAULT_ESTIMATORS = [LR_DEFAULT_ESTIMATOR, PLR_DEFAULT_ESTIMATOR]

# The base estimators
LR_BASE_ESTIMATORS = [KNeighborsLabelRanker]
PLR_BASE_ESTIMATORS = [KNeighborsPartialLabelRanker]
BASE_ESTIMATORS = [*LR_BASE_ESTIMATORS, *PLR_BASE_ESTIMATORS]

# The bagging rankers
LR_BAGGING = [BaggingLabelRanker]
PLR_BAGGING = [BaggingPartialLabelRanker]
BAGGING = [*LR_BAGGING, *PLR_BAGGING]

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
@pytest.mark.parametrize("BaggingRanker", BAGGING)
def test_toy_example(BaggingRanker):
    """Test the bagging rankers on a toy dataset."""
    # Initialize the bagging ranker
    # using its default hyperparameters
    model = BaggingRanker(random_state=seed)

    # Fit the bagging ranker
    # to the training dataset
    clf = model.fit(X_train, Y_train)

    # Obtain the predictions of the bagging ranker
    Y_pred = clf.predict(X_test)

    # Assert that the predictions are correct
    np.testing.assert_array_equal(Y_pred, Y_test)

    # For the sake of fully-tested code, apply
    # the same procedure without boostrapping
    clf = model.set_params(bootstrap=False).fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    np.testing.assert_array_equal(Y_pred, Y_test)


@pytest.mark.error
@pytest.mark.parametrize("BaggingRanker", BAGGING)
def test_error(BaggingRanker):
    """Test that it gives proper exception on deficient input."""
    # Initialize the bagging ranker
    # using its default hyperparameters
    model = BaggingRanker(random_state=seed)

    # Assert that an error is raised when the maximum number
    # of samples is an integer type less or equal than zero
    with pytest.raises(ValueError):
        model.set_params(max_samples=0).fit(X_train, Y_train)

    # Assert that an error is raised when the maximum number of
    # samples is an integer type greater than the number of samples
    with pytest.raises(ValueError):
        model.set_params(max_samples=100).fit(X_train, Y_train)

    # Assert that an error is raised when the maximum number
    # of samples is a floating type less than or equal zero
    with pytest.raises(ValueError):
        model.set_params(max_samples=0.0).fit(X_train, Y_train)

    # Assert that an error is raised when the maximum number
    # of samples is a floating type greater than one
    with pytest.raises(ValueError):
        model.set_params(max_samples=2.0).fit(X_train, Y_train)

    # Asssert that an error is raised when the maximum
    # number of samples in not an integer or floating type
    with pytest.raises(TypeError):
        model.set_params(max_samples="foo").fit(X_train, Y_train)

    # The same procedure with the maximum number of features

    # Initialize the bagging ranker using an small number of
    # estimators to not introduce too much memory overhead
    model = BaggingRanker(n_estimators=10, random_state=seed)

    # Assert that an error is raised when the maximum number
    # of features is an integer type less or equal than zero
    with pytest.raises(ValueError):
        model.set_params(max_features=0).fit(X_train, Y_train)

    # Assert that an error is raised when the maximum number of
    # features is an integer type greater than the number of features
    with pytest.raises(ValueError):
        model.set_params(max_features=10).fit(X_train, Y_train)

    # Assert that an error is raised when the maximum number
    # of features is a floating type less than or equal zero
    with pytest.raises(ValueError):
        model.set_params(max_features=0.0).fit(X_train, Y_train)

    # Assert that an error is raised when the maximum number
    # of features is a floating type greater than one
    with pytest.raises(ValueError):
        model.set_params(max_features=2.0).fit(X_train, Y_train)

    # Asssert that an error is raised when the maximum
    # number of features in not an integer or floating type
    with pytest.raises(TypeError):
        model.set_params(max_features="foo").fit(X_train, Y_train)


@pytest.mark.base_estimator
@pytest.mark.parametrize(
    "BaseEstimator,DefaultEstimator,BaggingRanker",
    zip(BASE_ESTIMATORS, DEFAULT_ESTIMATORS, BAGGING))
def test_base_estimator(BaseEstimator, DefaultEstimator, BaggingRanker):
    """Test base_estimator and its default values."""
    # Initialize the bagging ranker using the None as base estimator
    # and assert that the base estimator is the default estimator
    model = BaggingRanker(base_estimator=None, random_state=seed)
    clf = model.fit(X_train, Y_train)
    assert isinstance(clf.base_estimator_, DefaultEstimator)

    # Initialize the bagging ranker using the default estimator as base
    # estimator and assert that the base estimator is the default estimator
    model = BaggingRanker(base_estimator=DefaultEstimator(), random_state=seed)
    clf = model.fit(X_train, Y_train)
    assert isinstance(clf.base_estimator_, DefaultEstimator)

    # Initialize the bagging ranker explictly given the
    # base estimator and assert that it is properly set
    model = BaggingRanker(base_estimator=BaseEstimator(), random_state=seed)
    clf = model.fit(X_train, Y_train)
    assert isinstance(clf.base_estimator_, BaseEstimator)


@pytest.mark.bagging_sample_weight_unsupported_but_passed
@pytest.mark.parametrize(
    "BaseEstimator,BaggingRanker", zip(BASE_ESTIMATORS, BAGGING))
def test_bagging_sample_weight_unsupported_but_passed(BaseEstimator,
                                                      BaggingRanker):
    """Test that error is raised when passing
    sample weights to an unsupported estimator."""
    # Initialize the bagging ranker, setting the base
    # estimator that does not support sample weighting
    model = BaggingRanker(base_estimator=BaseEstimator(),
                          random_state=seed)

    # Initialize a sample weight that will be
    # used to fit the different bagging rankers
    sample_weight = random_state.randint(10, size=(X_train.shape[0]))

    # Assert than an error is raised when passing sample weights
    # to a base estimator that does not support sample weighting
    with pytest.raises(ValueError):
        model.fit(X_train, Y_train, sample_weight)

    # For the sake of coverage, test that no error is raised
    # when sample weights are used with supported estimator
    model = BaggingRanker(n_estimators=10, random_state=seed)
    model.fit(X_train, Y_train, sample_weight)


@pytest.mark.max_samples_consistency
@pytest.mark.parametrize("BaggingRanker", BAGGING)
def test_max_samples_consistency(BaggingRanker):
    """Test to make sure validated max_samples and original max_samples
    are identical when valid integer max_samples supplied by user."""
    # Initialize the maximum number
    # of samples to be tested
    max_samples = 3

    # Initialize the bagging ranker explictly
    # given the maximum number of samples
    model = BaggingRanker(max_samples=max_samples,
                          random_state=seed)

    # Fit the bagging ranker
    # to the training dataset
    clf = model.fit(X_train, Y_train)

    # Assert that the maximum number of samples stored in the
    # model and the given maximum number of samples is the same
    assert clf._max_samples == max_samples
