"""Testing for the tree module."""


# =============================================================================
# Imports
# =============================================================================

# Standard
from itertools import product, chain

# Third party
import numpy as np
import pytest

# Local application
from sklr.exceptions import NotFittedError
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

# The criteria for the Label Ranking problem
# and the Partial Label Ranking problem
LR_CRITERIA = ["mallows"]
PLR_CRITERIA = ["disagreements", "distance", "entropy"]
CRITERIA = [*LR_CRITERIA, *PLR_CRITERIA]

# The distances required
# for the Mallows criterion
DISTANCES = ["kendall"]

# The splitters that can be used to split
# an internal node of the decision tree
SPLITTERS = ["binary", "frequency", "width"]

# The decision trees rankers
LR_TREES = [DecisionTreeLabelRanker]
PLR_TREES = [DecisionTreePartialLabelRanker]
TREES = [*LR_TREES, *PLR_TREES]

# The possible combinations of decision tree
# rankers, criteria, splitters and distance
COMBINATIONS_LR = product(LR_TREES, LR_CRITERIA, SPLITTERS, DISTANCES)
COMBINATIONS_PLR = product(PLR_TREES, PLR_CRITERIA, SPLITTERS, DISTANCES)
COMBINATIONS = list(chain(COMBINATIONS_LR, COMBINATIONS_PLR))

# A toy example to check that the decision tree rankers
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
@pytest.mark.parametrize(
    "DecisionTreeRanker,criterion,splitter,distance", COMBINATIONS)
def test_toy_example(DecisionTreeRanker, criterion, splitter, distance):
    """Test the decision tree rankers on a toy dataset."""
    # Initialize the decision tree ranker using the given
    # criterion, splitter and (if corresponds) distance
    if DecisionTreeRanker is DecisionTreeLabelRanker:
        model = DecisionTreeRanker(criterion, distance, splitter,
                                   random_state=seed)
    else:
        model = DecisionTreeRanker(criterion, splitter,
                                   random_state=seed)

    # Fit the decision tree ranker
    # to the training dataset
    clf = model.fit(X_train, Y_train)

    # Obtain the predictions of the decision tree ranker
    Y_pred = clf.predict(X_test)

    # Assert that the predictions are correct
    np.testing.assert_array_equal(Y_pred, Y_test)

    # Now, apply the same procedure but only
    # using one feature. By this way, it is
    # ensured that the full code is tested
    model = model.set_hyperparams(max_features=1)
    clf = model.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    np.testing.assert_array_equal(Y_pred, Y_test)


@pytest.mark.weighted_toy_example
@pytest.mark.parametrize(
    "DecisionTreeRanker,criterion,splitter,distance", COMBINATIONS)
def test_weighted_toy_example(DecisionTreeRanker,
                              criterion, splitter, distance):
    """Test the decision tree rankers on a weighted toy dataset."""
    # Initialize the decision tree ranker using the given
    # criterion, splitter and (if corresponds) distance
    if DecisionTreeRanker is DecisionTreeLabelRanker:
        model = DecisionTreeRanker(criterion, distance, splitter,
                                   random_state=seed)
    else:
        model = DecisionTreeRanker(criterion, splitter,
                                   random_state=seed)

    # Initilize a sample weight
    # with uniform weighting
    sample_weight = np.ones(X_train.shape[0])

    # Fit the decision tree ranker to the training
    # dataset using an uniform sample weighting
    clf = model.fit(X_train, Y_train, sample_weight)

    # Obtain the predictions of the decision tree ranker
    Y_pred = clf.predict(X_test)

    # Now, apply the same procedure but only using
    # one feature and a half of the sample weight
    sample_weight *= 0.5
    model = model.set_hyperparams(max_features=1)
    clf = model.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    np.testing.assert_array_equal(Y_pred, Y_test)


@pytest.mark.pure_set
@pytest.mark.parametrize("DecisionTreeRanker", TREES)
def test_pure_set(DecisionTreeRanker):
    """Test when Y is pure."""
    # Initialize the decision tree ranker
    # using the default hyperparameters
    model = DecisionTreeRanker(random_state=seed)

    # Fit the decision tree ranker
    # to the training dataset
    clf = model.fit(X_train[:3], Y_train[:3])

    # Obtain the predictions of the decision tree ranker
    Y_pred = clf.predict(X_test)

    # Assert that the predictions are correct, that
    # is, that the same prediction is always returned
    np.testing.assert_array_equal(Y_pred, Y_train[:3])


@pytest.mark.max_features
@pytest.mark.parametrize("DecisionTreeRanker", TREES)
def test_max_features(DecisionTreeRanker):
    """Test the max_features hyperparameter."""
    # Initialize the decision tree ranker
    # using the default hyperparameters
    model = DecisionTreeRanker(random_state=seed)

    # Assert "auto" maximum number of features
    model = model.set_hyperparams(max_features="auto")
    clf = model.fit(X_train, Y_train)
    assert clf.max_features_ == int(np.sqrt(X_train.shape[1]))

    # Assert "sqrt" maximum number of features
    model = model.set_hyperparams(max_features="sqrt")
    clf = model.fit(X_train, Y_train)
    assert clf.max_features_ == int(np.sqrt(X_train.shape[1]))

    # Assert "log2" maximum number of features
    model = model.set_hyperparams(max_features="log2")
    clf = model.fit(X_train, Y_train)
    assert clf.max_features_ == int(np.sqrt(X_train.shape[1]))

    # Assert one as maximum number of features
    model = model.set_hyperparams(max_features=1)
    clf = model.fit(X_train, Y_train)
    assert clf.max_features_ == 1

    # Assert a half of the features as maximum number of features
    model = model.set_hyperparams(max_features=0.5)
    clf = model.fit(X_train, Y_train)
    assert clf.max_features_ == int(0.5 * X_train.shape[1])

    # Assert "None" maximum number of features
    model = model.set_hyperparams(max_features=None)
    clf = model.fit(X_train, Y_train)
    assert clf.max_features_ == X_train.shape[1]

    # Assert that an error is raised when the
    # maximum number of features is greater than
    # the number of features on the dataset
    with pytest.raises(ValueError):
        model.set_hyperparams(max_features=10).fit(X_train, Y_train)
    with pytest.raises(ValueError):
        model.set_hyperparams(max_features=1.5).fit(X_train, Y_train)

    # Assert that an error is raised when the maximum
    # number of features is less than or equal zero
    with pytest.raises(ValueError):
        model.set_hyperparams(max_features=-10).fit(X_train, Y_train)
    with pytest.raises(ValueError):
        model.set_hyperparams(max_features=-1.5).fit(X_train, Y_train)

    # Assert that an error is raised when the maximum
    # number of features is not integer or floating type
    with pytest.raises(ValueError):
        model.set_hyperparams(max_features="foo").fit(X_train, Y_train)


@pytest.mark.max_depth
@pytest.mark.parametrize("DecisionTreeRanker", TREES)
def test_max_depth(DecisionTreeRanker):
    """Test max_depth hyperparameter."""
    # Initialize the decision tree ranker
    # using the default hyperparameters
    model = DecisionTreeRanker(random_state=seed)

    # Assert that a leaf node is created when
    # the maximum depth of the tree is zero
    model = model.set_hyperparams(max_depth=0)
    clf = model.fit(X_train, Y_train)
    assert clf.get_depth() == 0
    assert clf.get_n_internal() == 0
    assert clf.get_n_leaves() == 1
    assert clf.get_n_nodes() == 1

    # Assert that a one level tree is created
    # when the maximum depth of the tree is one
    model = model.set_hyperparams(max_depth=1)
    clf = model.fit(X_train, Y_train)
    assert clf.get_depth() == 1
    assert clf.get_n_internal() == 1
    assert clf.get_n_leaves() == 2
    assert clf.get_n_nodes() == 3

    # Assert that an error is raised when the
    # maximum depth of the tree is less than zero
    with pytest.raises(ValueError):
        model = model.set_hyperparams(max_depth=-1).fit(X_train, Y_train)


@pytest.mark.min_samples_split
@pytest.mark.parametrize("DecisionTreeRanker", TREES)
def test_min_samples_split(DecisionTreeRanker):
    """Test min_samples_split hyperparameter."""
    # Initialize the decision tree ranker
    # using the default hyperparameters
    model = DecisionTreeRanker(random_state=seed)

    # Assert that a one level tree is created when the
    # minimum number of samples to split an internal node is
    # equal than the number of samples on the training dataset
    model = model.set_hyperparams(min_samples_split=X_train.shape[0])
    clf = model.fit(X_train, Y_train)
    assert clf.get_depth() == 1
    assert clf.get_n_internal() == 1
    assert clf.get_n_leaves() == 2
    assert clf.get_n_nodes() == 3

    # Assert the same than the above test but using
    # a floating value instead of an integer one
    model = model.set_hyperparams(min_samples_split=1.0)
    clf = model.fit(X_train, Y_train)
    assert clf.get_depth() == 1
    assert clf.get_n_internal() == 1
    assert clf.get_n_leaves() == 2
    assert clf.get_n_nodes() == 3

    # Assert that an error is raised when the minimum number of
    # samples to split an internal node is an integer less than two
    with pytest.raises(ValueError):
        model.set_hyperparams(min_samples_split=1).fit(X_train, Y_train)

    # Assert that an error is raised when the minimum number of samples
    # to split an internal node is a floating number greater than one
    with pytest.raises(ValueError):
        model.set_hyperparams(min_samples_split=1.5).fit(X_train, Y_train)


@pytest.mark.max_splits
@pytest.mark.parametrize("DecisionTreeRanker", TREES)
def test_max_splits(DecisionTreeRanker):
    """Test max_splits hyperparameter."""
    # Initialize the decision tree ranker
    # using a maximum number of splits of five
    model = DecisionTreeRanker(max_splits=5, random_state=seed)

    # Assert that the maximum number of splits is
    # properly set to two for the binary splitter
    model = model.set_hyperparams(splitter="binary")
    clf = model.fit(X_train, Y_train)
    assert len(clf.tree_.children) == 2

    # Assert that the maximum number of splits is
    # properly set to two for the frequency splitter
    model = model.set_hyperparams(splitter="frequency")
    clf = model.fit(X_train, Y_train)
    assert len(clf.tree_.children) > 2

    # Assert that the maximum number of splits is
    # properly set to two for the width splitter
    model = model.set_hyperparams(splitter="width")
    clf = model.fit(X_train, Y_train)
    assert len(clf.tree_.children) > 2

    # Assert that an error is raised when the
    # maximum number of samples is less than one
    with pytest.raises(ValueError):
        model.set_hyperparams(max_splits=-1).fit(X_train, Y_train)


@pytest.mark.error
@pytest.mark.parametrize("DecisionTreeRanker", TREES)
def test_error(DecisionTreeRanker):
    """Test that it gives proper error on deficient input."""
    # Initialize the decision tree ranker
    # using the default hyperparameters
    model = DecisionTreeRanker(random_state=seed)

    # The following tests are just required
    # for one of the estimators since they are
    # common for all of them

    # Assert that an error is raised when
    # the sample weights is not a 1-D array
    with pytest.raises(ValueError):
        model.fit(X_train, Y_train, sample_weight=X_test)

    # Assert that an error is raised when
    # using an estimator before fitting
    with pytest.raises(NotFittedError):
        model.predict(X_test)

    # Assert that an error is raised when
    # the number of features per sample of
    # the test dataset is different than the
    # number of samples on the training dataset
    with pytest.raises(ValueError):
        model.fit(X_train, Y_train).predict(X_test[:, 1:])


def check_equal_trees(tree1, tree2):
    """Test that the trees are the same."""
    # Assert that both roots of the trees are the same
    # (that is, both internal nodes or both leaf nodes)
    assert tree1.root.node == tree2.root.node

    # Assert that the impurity, the consensus
    # ranking, the count and the precedences
    # matrix of the root of the tree are the same

    # Impurity
    np.testing.assert_almost_equal(
        tree1.root.impurity, tree2.root.impurity)

    # Consensus
    np.testing.assert_array_almost_equal(
        tree1.root.consensus, tree2.root.consensus)

    # Count
    try:
        np.testing.assert_array_almost_equal(
            tree1.root.count, tree2.root.count)
    except TypeError:
        assert tree1.root.count is None and tree1.root.count is None

    # Precedences matrix
    try:
        np.testing.assert_array_almost_equal(
            tree1.root.precedences_matrix, tree2.root.precedences_matrix)
    except TypeError:
        assert (tree1.root.precedences_matrix is None and
                tree2.root.precedences_matrix is None)

    # Assert that the feature used for splitting and the
    # thresholds values are the same for internal nodes
    if tree1.root.node == tree2.root.node == 0:
        # Assert that the feature used for splitting
        # the internal node is the same for both trees
        assert tree1.root.feature == tree2.root.feature
        # Assert that the thresholds values for the
        # feature in the internal nodes are the same
        np.testing.assert_array_almost_equal(
            tree1.root.thresholds, tree2.root.thresholds)
        # Apply the process recursively for each child
        for (child1, child2) in zip(tree1.children, tree2.children):
            check_equal_trees(child1, child2)


@pytest.mark.sample_weight
@pytest.mark.parametrize(
    "DecisionTreeRanker,criterion,splitter,distance", COMBINATIONS)
def test_sample_weight(DecisionTreeRanker, criterion, splitter, distance):
    """Test that sample weighting is the
    same that having duplicated instances."""
    # Initialize the decision tree rankers using the given
    # criterion, splitter and (if corresponds) distance
    if DecisionTreeRanker is DecisionTreeLabelRanker:
        model = DecisionTreeRanker(criterion, distance, splitter,
                                   random_state=seed)
        model2 = DecisionTreeRanker(criterion, distance, splitter,
                                    random_state=seed)
    else:
        model = DecisionTreeRanker(criterion, splitter,
                                   random_state=seed)
        model2 = DecisionTreeRanker(criterion, splitter,
                                    random_state=seed)

    # Initialize the duplicates instances
    # and the associated sample weighting
    n_samples = X_train.shape[0]
    duplicates = random_state.randint(0, n_samples, n_samples // 2)
    sample_weight = np.bincount(duplicates, minlength=n_samples)

    # Fit the decision tree ranker to the training
    # dataset using the duplicated instances and also
    # the decision tree ranker using sample weighting
    clf = model.fit(X_train[duplicates], Y_train[duplicates])
    clf2 = model2.fit(X_train, Y_train, sample_weight)

    # Assert that the decision tree obtained
    # with duplicated instances ia the same
    # than the decision tree obtained
    # with sample weighting
    check_equal_trees(clf.tree_, clf2.tree_)
