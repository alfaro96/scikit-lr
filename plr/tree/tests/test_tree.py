"""Testing for the tree module."""


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
from plr.exceptions import NotFittedError
from plr.tree import DecisionTreeLabelRanker, DecisionTreePartialLabelRanker
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

# The trees
TREES = [DecisionTreeLabelRanker, DecisionTreePartialLabelRanker]

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


# =============================================================================
# Testing
# =============================================================================

@pytest.mark.label_ranking_toy
@pytest.mark.parametrize(
    "criterion,distance,splitter", product(LR_CRITERIA, DISTANCES, SPLITTERS))
def test_label_ranking_toy(criterion, distance, splitter):
    """Test Label Ranking on a toy dataset."""
    # Initialize the Label Ranking tree using all features
    model = DecisionTreeLabelRanker(criterion, distance, splitter,
                                    random_state=seed)

    # Fit the Label Ranking tree using all features to obtain the classifier
    clf = model.fit(X_train, Y_train)

    # Assert that the predictions are correct
    np.testing.assert_array_equal(clf.predict(X_test), Y_test)

    # Initialize the Label Ranking tree using one feature
    model = DecisionTreeLabelRanker(criterion, distance, splitter,
                                    max_features=1, random_state=seed)

    # Fit the Label Ranking tree using one feature to obtain the classifier
    clf = model.fit(X_train, Y_train)

    # Assert that the predictions are correct
    np.testing.assert_array_equal(clf.predict(X_test), Y_test)


@pytest.mark.weighted_label_ranking_toy
@pytest.mark.parametrize(
    "criterion,distance,splitter", product(LR_CRITERIA, DISTANCES, SPLITTERS))
def test_weighted_label_ranking_toy(criterion, distance, splitter):
    """Test Label Ranking on a weighted toy dataset."""
    # Using a sample weight with all ones
    sample_weight = np.ones(X_train.shape[0])

    # Initialize the Label Ranking tree
    model = DecisionTreeLabelRanker(criterion, distance, splitter,
                                    random_state=seed)

    # Fit the Label Ranking tree using
    # weighted samples to obtain the classifier
    clf = model.fit(X_train, Y_train, sample_weight)

    # Assert that the predictions are correct
    np.testing.assert_array_equal(clf.predict(X_test), Y_test)

    # Using a sample weight with all a half
    sample_weight = np.ones(X_train.shape[0]) * 0.5

    # Initialize the Label Ranking tree
    model = DecisionTreeLabelRanker(criterion, distance, splitter,
                                    max_features=1, random_state=seed)

    # Fit the Label Ranking model using
    # weighted samples to obtain the classifier
    clf = model.fit(X_train, Y_train, sample_weight)

    # Assert that the predictions are correct
    np.testing.assert_array_equal(clf.predict(X_test), Y_test)


@pytest.mark.partial_label_ranking_toy
@pytest.mark.parametrize(
    "criterion,splitter", product(PLR_CRITERIA, SPLITTERS))
def test_partial_label_ranking_toy(criterion, splitter):
    """Test Partial Label Ranking on a toy dataset."""
    # Initialize the Partial Label Ranking tree using all features
    model = DecisionTreePartialLabelRanker(criterion, splitter,
                                           random_state=seed)

    # Fit the Partial Label Ranking tree using
    # all features to obtain the classifier
    clf = model.fit(X_train, Y_train)

    # Assert that the predictions are correct
    np.testing.assert_array_equal(clf.predict(X_test), Y_test)

    # Initialize the Partial Label Ranking tree using one feature
    model = DecisionTreePartialLabelRanker(criterion, splitter,
                                           max_features=1, random_state=seed)

    # Fit the Partial Label Ranking tree using
    # one feature to obtain the classifier
    clf = model.fit(X_train, Y_train)

    # Assert that the predictions are correct
    np.testing.assert_array_equal(clf.predict(X_test), Y_test)


@pytest.mark.weighted_partial_label_ranking_toy
@pytest.mark.parametrize(
    "criterion,splitter", product(PLR_CRITERIA, SPLITTERS))
def test_weighted_partial_label_ranking_toy(criterion, splitter):
    """Test Partial Label Ranking on a weighted toy dataset."""
    # Using a sample weight with all ones
    sample_weight = np.ones(X_train.shape[0])

    # Initialize the Partial Label Ranking tree
    model = DecisionTreePartialLabelRanker(criterion, splitter,
                                           random_state=seed)

    # Fit the Partial Label Ranking tree using
    # weighted samples to obtain the classifier
    clf = model.fit(X_train, Y_train, sample_weight)

    # Assert that the predictions are correct
    np.testing.assert_array_equal(clf.predict(X_test), Y_test)

    # Using a sample weight with all a half
    sample_weight = np.ones(X_train.shape[0]) * 0.5

    # Initialize the Partial Label Ranking tree
    model = DecisionTreePartialLabelRanker(criterion, splitter,
                                           max_features=1, random_state=seed)

    # Fit the Partial Label Ranking tree using
    # weighted samples to obtain the classifier
    clf = model.fit(X_train, Y_train, sample_weight)

    # Assert that the predictions are correct
    np.testing.assert_array_equal(clf.predict(X_test), Y_test)


@pytest.mark.iris_lr
@pytest.mark.parametrize(
    "criterion,distance,splitter", product(LR_CRITERIA, DISTANCES, SPLITTERS))
def test_iris_lr(criterion, distance, splitter):
    """Test consistency on dataset iris for Label Rankers."""
    # Load the iris dataset
    iris = load_iris()

    # Obtain training and test indexes
    idx = random_state.permutation(iris.data_lr.shape[0])
    train_size, test_size = int(0.8 * idx.shape[0]), int(0.2 * idx.shape[0])
    train_idx, test_idx = idx[:train_size], idx[test_size:]

    # Initialize the Label Ranking tree using all features
    model = DecisionTreeLabelRanker(criterion, distance, splitter,
                                    random_state=seed)

    # Fit the Label Ranking tree with all features to obtain the classifier
    clf = model.fit(iris.data_lr[train_idx], iris.ranks_lr[train_idx])

    # Assert that a minimum score is achieved
    assert clf.score(iris.data_lr[test_idx], iris.ranks_lr[test_idx]) > 0.9

    # Initialize the Label Ranking tree using two features
    model = DecisionTreeLabelRanker(criterion, distance, splitter,
                                    max_features=2, random_state=seed)

    # Fit the Label Ranking tree using two features to obtain the classifier
    clf = model.fit(iris.data_lr[train_idx], iris.ranks_lr[train_idx])

    # Assert that a minimum score is achieved
    assert clf.score(iris.data_lr[test_idx], iris.ranks_lr[test_idx]) > 0.5


@pytest.mark.iris_plr
@pytest.mark.parametrize(
    "criterion,splitter", product(PLR_CRITERIA, SPLITTERS))
def test_iris_plr(criterion, splitter):
    """Test consistency on dataset iris for Partial Label Rankers."""
    # Load the iris dataset
    iris = load_iris()

    # Obtain training and test indexes
    idx = random_state.permutation(iris.data_plr.shape[0])
    train_size, test_size = int(0.8 * idx.shape[0]), int(0.2 * idx.shape[0])
    train_idx, test_idx = idx[:train_size], idx[test_size:]

    # Initialize the Partial Label Ranking tree using all features
    model = DecisionTreePartialLabelRanker(criterion, splitter,
                                           random_state=seed)

    # Fit the Partial Label Ranking tree using
    # all features to obtain the classifier
    clf = model.fit(iris.data_plr[train_idx], iris.ranks_plr[train_idx])

    # Assert that a minimum score is achieved
    assert clf.score(iris.data_plr[test_idx], iris.ranks_plr[test_idx]) > 0.9

    # Initialize the Partial Label Ranking tree using two features
    model = DecisionTreePartialLabelRanker(criterion, splitter,
                                           max_features=2, random_state=seed)

    # Fit the Partial Label Ranking tree using
    # two features to obtain the classifier
    clf = model.fit(iris.data_plr[train_idx], iris.ranks_plr[train_idx])

    # Assert that a minimum score is achieved
    assert clf.score(iris.data_plr[test_idx], iris.ranks_plr[test_idx]) > 0.5


@pytest.mark.pure_set
@pytest.mark.parametrize("tree", TREES)
def test_pure_set(tree):
    """Test when Y is pure."""
    # Initialize the pure dataset
    X = np.array([[-2, -1], [-1, -1], [-1, -2]])
    Y = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])

    # Initialize the tree
    model = tree(random_state=seed)

    # Fit the tree to obtain the classifier
    clf = model.fit(X, Y)

    # Assert that the predictions are correct
    np.testing.assert_array_equal(clf.predict(X), Y)


@pytest.mark.max_features
@pytest.mark.parametrize("tree", TREES)
def test_max_features(tree):
    """Test the max_features hyperparameter."""
    # Assert "auto" maximum number of features
    model = tree(max_features="auto", random_state=seed)
    clf = model.fit(X_train, Y_train)
    assert clf.max_features_ == int(np.sqrt(X_train.shape[1]))

    # Assert "sqrt" maximum number of features
    model = tree(max_features="sqrt", random_state=seed)
    clf = model.fit(X_train, Y_train)
    assert clf.max_features_ == int(np.sqrt(X_train.shape[1]))

    # Assert "log2" maximum number of features
    model = tree(max_features="log2", random_state=seed)
    clf = model.fit(X_train, Y_train)
    assert clf.max_features_ == int(np.log2(X_train.shape[1]))

    # Assert one as maximum number of features
    model = tree(max_features=1, random_state=seed)
    clf = model.fit(X_train, Y_train)
    assert clf.max_features_ == 1

    # Assert three as maximum number of features
    model = tree(max_features=2, random_state=seed)
    clf = model.fit(X_train, Y_train)
    assert clf.max_features_ == 2

    # Assert a half of the features as maximum number of features
    model = tree(max_features=0.5, random_state=seed)
    clf = model.fit(X_train, Y_train)
    assert clf.max_features_ == int(0.5 * X_train.shape[1])

    # Assert "None" maximum number of features
    model = tree(max_features=None, random_state=seed)
    clf = model.fit(X_train, Y_train)
    assert clf.max_features_ == X_train.shape[1]

    # Assert that an error is raised when the maximum number of
    # features is greater than the number of features on the dataset
    with pytest.raises(ValueError):
        tree(max_features=10, random_state=seed).fit(X_train, Y_train)
    with pytest.raises(ValueError):
        tree(max_features=1.5, random_state=seed).fit(X_train, Y_train)

    # Assert that an error is raised when the maximum
    # number of features is less or equal than zero
    with pytest.raises(ValueError):
        tree(max_features=-1, random_state=seed).fit(X_train, Y_train)
    with pytest.raises(ValueError):
        tree(max_features=0.0, random_state=seed).fit(X_train, Y_train)

    # Assert that an error is raised when the maximum
    # number of features is not integer or float
    with pytest.raises(ValueError):
        tree(max_features="foo", random_state=seed).fit(X_train, Y_train)


@pytest.mark.max_depth
@pytest.mark.parametrize("tree", TREES)
def test_max_depth(tree):
    """Test max_depth hyperparameter."""
    # Assert that a leaf node is created when the
    # maximum depth of the tree is zero
    model = tree(max_depth=0, random_state=seed)
    clf = model.fit(X_train, Y_train)
    assert clf.get_depth() == clf.get_n_internal() == 0
    assert clf.get_n_leaves() == clf.get_n_nodes() == 1

    # Assert that a one level tree is created
    # when the maximum depth of the tree is one
    model = tree(max_depth=1, random_state=seed)
    clf = model.fit(X_train, Y_train)
    assert clf.get_depth() == clf.get_n_internal() == 1
    assert clf.get_n_leaves() == 2
    assert clf.get_n_nodes() == 3

    # Assert that an error is raised when the
    # maximum depth of the tree is less than zero
    with pytest.raises(ValueError):
        tree(max_depth=-1, random_state=seed).fit(X_train, Y_train)


@pytest.mark.min_samples_split
@pytest.mark.parametrize("tree", TREES)
def test_min_samples_split(tree):
    """Test min_samples_split hyperparameter."""
    # Assert that a one level tree is created when the
    # minimum number of samples to split an internal node is
    # equal than the number of samples on the training dataset
    model = tree(min_samples_split=X_train.shape[0], random_state=seed)
    clf = model.fit(X_train, Y_train)
    assert clf.get_depth() == clf.get_n_internal() == 1
    assert clf.get_n_leaves() == 2
    assert clf.get_n_nodes() == 3

    model = tree(min_samples_split=1.0, random_state=seed)
    clf = model.fit(X_train, Y_train)
    assert clf.get_depth() == clf.get_n_internal() == 1
    assert clf.get_n_leaves() == 2
    assert clf.get_n_nodes() == 3

    # Assert that an error is raised when the minimum number of samples
    # to split an internal node is an integer less than two
    with pytest.raises(ValueError):
        tree(min_samples_split=1, random_state=seed).fit(X_train, Y_train)

    # Assert that an error is raised when the minimum number of samples
    # to split an internal node is a floating number greater than one
    with pytest.raises(ValueError):
        tree(min_samples_split=1.5, random_state=seed).fit(X_train, Y_train)


@pytest.mark.max_splits
@pytest.mark.parametrize("tree", TREES)
def test_max_splits(tree):
    """Test max_splits hyperparameter."""
    # Assert that an error is raised when the maximum
    # number of samples is less than one
    with pytest.raises(ValueError):
        tree(max_splits=-1, random_state=seed).fit(X_train, Y_train)


@pytest.mark.error
@pytest.mark.parametrize("tree", TREES)
def test_error(tree):
    """Test that it gives proper error on deficient input."""
    # Assert that an error is raised when
    # trying to use the model before fitting it
    with pytest.raises(NotFittedError):
        tree(random_state=seed).get_depth()
    with pytest.raises(NotFittedError):
        tree(random_state=seed).get_n_internal()
    with pytest.raises(NotFittedError):
        tree(random_state=seed).get_n_leaves()
    with pytest.raises(NotFittedError):
        tree(random_state=seed).get_n_nodes()
    with pytest.raises(NotFittedError):
        tree(random_state=seed).predict(X_test)

    # Assert that an error is raised when the
    # number of features per sample is incorrect
    with pytest.raises(ValueError):
        tree(random_state=seed).fit(X_train, Y_train).predict(X_test[:, 1:])


def check_equal_trees(tree1, tree2):
    """Test that the trees are the same."""
    # Assert that the root of the trees are the same
    assert tree1.root.node == tree2.root.node

    # Assert that the impurity, the consensus ranking,
    # the count and the precedences matrix of the root
    # of the tree are the same
    np.testing.assert_almost_equal(
        tree1.root.impurity, tree2.root.impurity)
    np.testing.assert_array_almost_equal(
        tree1.root.consensus, tree2.root.consensus)
    try:
        np.testing.assert_array_almost_equal(
            tree1.root.count, tree2.root.count)
    except TypeError:
        assert tree1.root.count is None
        assert tree1.root.count is None
    try:
        np.testing.assert_array_almost_equal(
            tree1.root.precedences_matrix, tree2.root.precedences_matrix)
    except TypeError:
        assert tree1.root.precedences_matrix is None
        assert tree2.root.precedences_matrix is None

    # Assert that the feature used for splitting and the
    # thresholds values are the same for internal nodes
    # and apply the checking process recursively for each child
    if tree1.root.node == tree2.root.node == 0:
        assert tree1.root.feature == tree2.root.feature
        np.testing.assert_array_almost_equal(
            tree1.root.thresholds, tree2.root.thresholds)
        for (child1, child2) in zip(tree1.children, tree2.children):
            check_equal_trees(child1, child2)


@pytest.mark.sample_weight_lr
@pytest.mark.parametrize(
    "criterion,distance,splitter", product(LR_CRITERIA, DISTANCES, SPLITTERS))
def test_sample_weight_lr(criterion, distance, splitter):
    """Test sample weighting for Label Ranking."""
    # Load the iris dataset
    iris = load_iris()

    # Initialize the duplicates instances
    duplicates = random_state.randint(
        0, iris.data_lr.shape[0], iris.data_lr.shape[0] // 2)

    # Initialize the Label tree model that will
    # be fitted from the duplicates instances
    model = DecisionTreeLabelRanker(criterion, distance, splitter,
                                    random_state=seed)

    # Fit the Label Ranker tree to obtain the
    # classifier using the duplicates instances
    clf = model.fit(iris.data_lr[duplicates], iris.ranks_lr[duplicates])

    # Initialize the sample weight from the duplicates instances
    sample_weight = np.bincount(duplicates, minlength=iris.data_lr.shape[0])

    # Initialize the Label Ranker tree that will
    # be fitted from the sample weight
    model2 = DecisionTreeLabelRanker(criterion, distance, splitter,
                                     random_state=seed)

    # Fit the Label Ranker tree to obtain the
    # classifier using the sample weight
    clf2 = model2.fit(iris.data_lr, iris.ranks_lr, sample_weight)

    # Assert that sample weighting is the same as having duplicates
    check_equal_trees(clf.tree_, clf2.tree_)


@pytest.mark.sample_weight_plr
@pytest.mark.parametrize(
    "criterion,splitter", product(PLR_CRITERIA, SPLITTERS))
def test_sample_weight_plr(criterion, splitter):
    """Test sample weighting for Partial Label Ranking."""
    # Load the iris dataset
    iris = load_iris()

    # Initialize the duplicates instances
    duplicates = random_state.randint(
        0, iris.data_plr.shape[0], iris.data_plr.shape[0] // 2)

    # Initialize the Partial Label Ranker tree that
    # will be fitted from the duplicates instances
    model = DecisionTreePartialLabelRanker(criterion, splitter,
                                           random_state=seed)

    # Fit the Partial Label Ranker tree to obtain
    # the classifier using the duplicates instances
    clf = model.fit(iris.data_plr[duplicates], iris.ranks_plr[duplicates])

    # Initialize the sample weight from the duplicates instances
    sample_weight = np.bincount(duplicates, minlength=iris.data_plr.shape[0])

    # Initialize the Partial Label Ranker tree
    # that will be fitted from the sample weight
    model2 = DecisionTreePartialLabelRanker(criterion, splitter,
                                            random_state=seed)

    # Fit the Partial Label Ranker tree to obtain
    # the classifier using the sample weight
    clf2 = model2.fit(iris.data_plr, iris.ranks_plr, sample_weight)

    # Assert that sample weighting is the same as having duplicates
    check_equal_trees(clf.tree_, clf2.tree_)


@pytest.mark.sample_weight_invalid
@pytest.mark.parametrize("tree", TREES)
def test_sample_weight_invalid(tree):
    """Test that invalid sample weighting raises errors."""
    # Assert that an error is raised when the
    # sample weights is not a 1-D array
    with pytest.raises(ValueError):
        sample_weight = np.array(0)
        tree(random_state=seed).fit(X_train, Y_train, sample_weight)

    with pytest.raises(ValueError):
        sample_weight = np.ones(X_train.shape[0])[:, None]
        tree(random_state=seed).fit(X_train, Y_train, sample_weight)

    # Assert that an error is raised when the
    # sample weights have wrong number of samples
    with pytest.raises(ValueError):
        sample_weight = np.ones(X_train.shape[0] - 1)
        tree(random_state=seed).fit(X_train, Y_train, sample_weight)

    with pytest.raises(ValueError):
        sample_weight = np.ones(X_train.shape[0] + 1)
        tree(random_state=seed).fit(X_train, Y_train, sample_weight)
