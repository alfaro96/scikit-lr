import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.sparse import csc_array, csr_array
from sklearn.exceptions import ConvergenceWarning
from sklearn.tree import export_graphviz

from sklr._distributions.tests import _oracles
from sklr._distributions.tests._oracles import random_rankings
from sklr.tree import DecisionTreeLabelRanker
from sklr.utils._estimator_checks import parametrize_with_checks

# The threshold below which scikit-learn takes an impurity as zero, and an
# impurity decrease as negative
EPSILON = np.finfo(np.float64).eps


def _expected_failed_checks(estimator):
    # The checks of sample weights build a target of classes 1 and 2 by hand,
    # without adapting it to the tags of the estimator, so it is not a valid
    # ranking. The checks of sparse data expect the prediction of an estimator
    # with several outputs to have a single column, while a ranking has one per
    # label
    return {
        "check_sample_weights_pandas_series": "Builds a target that is not a ranking",
        "check_sample_weights_not_an_array": "Builds a target that is not a ranking",
        "check_estimator_sparse_array": "Expects predictions of shape (n, 1)",
        "check_estimator_sparse_matrix": "Expects predictions of shape (n, 1)",
    }


@parametrize_with_checks(
    [
        DecisionTreeLabelRanker(),
        DecisionTreeLabelRanker(splitter="random", max_depth=3),
        DecisionTreeLabelRanker(max_leaf_nodes=4, ccp_alpha=0.01),
    ],
    expected_failed_checks=_expected_failed_checks,
)
def test_estimators(estimator, check):
    check(estimator)


def _impurity(y, sample_weight):
    """Compute the impurity of a node as the docstring describes, label by label."""
    is_ranked = ~np.all(np.isnan(y), axis=0)
    y = y[:, is_ranked]
    n_labels = y.shape[1]
    consistent = all(
        len(
            {
                row[i] < row[j]
                for row in y
                if not np.isnan(row[i]) and not np.isnan(row[j])
            }
        )
        < 2
        for i in range(n_labels)
        for j in range(i + 1, n_labels)
    )
    if n_labels <= 1 or consistent:
        return 0.0
    center, _ = _oracles.estimate_center(y, sample_weight)
    return 1 / (1 + _oracles.estimate_spread(y, center, sample_weight))


def _center(y, sample_weight):
    """Estimate the center of a node on its ranked labels, ``nan`` for the others."""
    is_ranked = ~np.all(np.isnan(y), axis=0)
    value = np.full(y.shape[1], np.nan)
    if np.any(is_ranked):
        value[is_ranked], _ = _oracles.estimate_center(y[:, is_ranked], sample_weight)
    return value


def _check_node(ranker, X, y, sample_weight, node, rows, parent_value):
    """Check a node of a fitted tree, and its descendants, against naive estimates.

    The impurity of the node, whether it is a leaf and its ranking are computed
    from its training samples `rows`. Its split must be among the best ones found
    by trying every threshold between consecutive values of every feature.
    """
    tree = ranker.tree_
    total_weight = sample_weight.sum()
    impurity = _impurity(y[rows], sample_weight[rows])
    assert_allclose(tree.impurity[node], impurity, rtol=1e-9, atol=1e-12)

    value = _center(y[rows], sample_weight[rows])
    if parent_value is None:
        value, _ = _oracles.estimate_center(value[np.newaxis], [1])
    else:
        value = _oracles.complete_ranking(value, parent_value)
    assert_array_equal(tree.value[node, :, 0], value)

    weight = sample_weight[rows].sum()
    splits = []
    for feature in range(X.shape[1]):
        values = np.unique(X[rows, feature])
        for threshold in values[:-1]:
            left = rows[X[rows, feature] <= threshold]
            right = rows[X[rows, feature] > threshold]
            proxy = -(
                sample_weight[left].sum() * _impurity(y[left], sample_weight[left])
                + sample_weight[right].sum() * _impurity(y[right], sample_weight[right])
            )
            splits.append((proxy, feature, left, right))

    if len(rows) < 2 or impurity <= EPSILON or not splits:
        assert tree.children_left[node] == -1
        return
    best_proxy = max(proxy for proxy, *_ in splits)
    best_improvement = (impurity * weight + best_proxy) / total_weight
    if best_improvement + EPSILON < 0:
        assert tree.children_left[node] == -1
        return

    assert tree.children_left[node] != -1
    feature = tree.feature[node]
    left = rows[X[rows, feature] <= tree.threshold[node]]
    right = rows[X[rows, feature] > tree.threshold[node]]
    chosen = [
        proxy
        for proxy, split_feature, split_left, _ in splits
        if split_feature == feature and np.array_equal(split_left, left)
    ]
    assert len(chosen) == 1
    assert chosen[0] >= best_proxy - 1e-9 * abs(best_proxy)
    for child, child_rows in [
        (tree.children_left[node], left),
        (tree.children_right[node], right),
    ]:
        _check_node(ranker, X, y, sample_weight, child, child_rows, value)


# With the seed 6 and complete rankings, a node becomes a leaf because its best
# split increases the impurity
@pytest.mark.parametrize("seed", [0, 6])
@pytest.mark.parametrize("missing", [0, 0.3, 0.6])
@pytest.mark.parametrize("weighted", [False, True])
def test_tree_oracle(seed, missing, weighted):
    # Every node is checked against the naive estimates of the impurity and the
    # ranking, and against an exhaustive search of the best split. The weights are
    # integers, so that the Borda counts are exact and their ties do not depend on
    # rounding
    rng = np.random.RandomState(seed)
    n_samples, n_labels = 25, 4
    X = rng.rand(n_samples, 2).astype(np.float32)
    y = random_rankings(n_samples, n_labels, missing=missing, random_state=rng)
    sample_weight = (
        rng.randint(1, 4, n_samples).astype(np.float64)
        if weighted
        else np.ones(n_samples)
    )

    ranker = DecisionTreeLabelRanker(random_state=0)
    ranker.fit(X, y, sample_weight=sample_weight if weighted else None)

    _check_node(ranker, X, y, sample_weight, 0, np.arange(n_samples), None)
    leaves = ranker.apply(X)
    assert_array_equal(ranker.predict(X), ranker.tree_.value[leaves, :, 0])


def test_predict_example():
    # The first and fourth samples have the same features, so they end in a leaf
    # whose Borda counts are 4, 5 and 3. The second and third samples are split
    # from them and from each other by the first feature
    X = [[0, 1, 0], [1, 0, 1], [1, 1, 0], [0, 1, 0]]
    y = [[1, 2, 3], [2, 1, 3], [1, 2, 3], [3, 1, 2]]
    ranker = DecisionTreeLabelRanker(random_state=0).fit(X, y)
    assert_array_equal(ranker.predict(X), [[2, 1, 3], [2, 1, 3], [1, 2, 3], [2, 1, 3]])
    assert ranker.predict(X).dtype == np.intp


@pytest.mark.parametrize(
    "y",
    [
        # Equal complete rankings
        [[1, 2, 3], [1, 2, 3], [1, 2, 3]],
        # Incomplete rankings that agree with a single complete ranking
        [[1, 2, np.nan], [np.nan, 1, 2], [1, np.nan, 2]],
        # Incomplete rankings that order every pair of labels in the same way,
        # but in a cycle that no complete ranking agrees with
        [[1, 2, np.nan], [np.nan, 1, 2], [2, np.nan, 1]],
        # Rankings of a single label or of none
        [[1, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, 1, np.nan]],
    ],
)
def test_consistent_rankings_make_a_leaf(y):
    ranker = DecisionTreeLabelRanker().fit([[0], [1], [2]], y)
    assert ranker.tree_.node_count == 1
    assert ranker.tree_.impurity[0] == 0


def test_pure_children_with_fractional_weights():
    # The rankings of each child are counted, not weighed, to decide whether they
    # are consistent. The node adds up the weights of the first three samples in
    # their order, and the left child in the order of the feature, which leaves a
    # positive rounding error in the weights of the right child, found by
    # subtraction, as if one of its rankings ordered the labels as the left child
    X = [[2], [0], [1], [3], [4]]
    y = [[1, 2, 3], [1, 2, 3], [1, 2, 3], [3, 2, 1], [3, 2, 1]]
    sample_weight = [0.1, 0.2, 0.3, 1, 1]
    ranker = DecisionTreeLabelRanker().fit(X, y, sample_weight=sample_weight)
    assert ranker.tree_.node_count == 3
    assert_array_equal(ranker.tree_.impurity[1:], 0)


def test_spread_zero_impurity_one():
    # A ranking and its reverse with equal weights have a mean Kendall distance of
    # n(n - 1)/4 to any center, the expected one of the uniform distribution, so
    # the spread is zero and the impurity one. The features cannot split them
    ranker = DecisionTreeLabelRanker().fit([[0], [0]], [[1, 2, 3], [3, 2, 1]])
    assert ranker.tree_.node_count == 1
    assert ranker.tree_.impurity[0] == 1


def test_node_completed_from_parent():
    # The left leaf only ranks the first two labels, and the third label goes to
    # its closest extension to the ranking of the root, which puts the third label
    # first. Ranking it ahead of both labels disagrees with the root on no pair
    X = [[0], [0], [1], [1]]
    y = [
        [1, 2, np.nan],
        [1, 2, np.nan],
        [3, 2, 1],
        [3, 2, 1],
    ]
    ranker = DecisionTreeLabelRanker().fit(X, y)
    assert_array_equal(ranker.tree_.value[0, :, 0], [2, 3, 1])
    assert_array_equal(ranker.predict([[0], [1]]), [[2, 3, 1], [3, 2, 1]])


def test_label_never_ranked():
    # The third label is never ranked, so the ranking of the root goes to its
    # closest extension to its own Borda count, which puts that label in the middle
    X = [[0], [0], [1]]
    y = [[1, 2, np.nan], [1, 2, np.nan], [1, 2, np.nan]]
    ranker = DecisionTreeLabelRanker().fit(X, y)
    assert_array_equal(ranker.predict(X), [[1, 3, 2]] * 3)


def test_sample_weight_integer_equals_repeated_samples():
    rng = np.random.RandomState(0)
    X = rng.rand(30, 3)
    y = random_rankings(30, 4, missing=0.3, random_state=rng)
    sample_weight = rng.randint(0, 4, len(X))

    weighted = DecisionTreeLabelRanker(random_state=0)
    weighted.fit(X, y, sample_weight=sample_weight)
    repeated = DecisionTreeLabelRanker(random_state=0)
    repeated.fit(
        np.repeat(X, sample_weight, axis=0), np.repeat(y, sample_weight, axis=0)
    )

    assert_array_equal(weighted.tree_.feature, repeated.tree_.feature)
    assert_array_equal(weighted.tree_.threshold, repeated.tree_.threshold)
    assert_allclose(weighted.tree_.impurity, repeated.tree_.impurity)
    assert_array_equal(weighted.tree_.value, repeated.tree_.value)


def test_sparse_int64_indices():
    X = csc_array(np.eye(3))
    X.indices = X.indices.astype(np.int64)
    X.indptr = X.indptr.astype(np.int64)
    with pytest.raises(ValueError, match=r"No support for np\.int64 index"):
        DecisionTreeLabelRanker().fit(X, [[1, 2], [2, 1], [1, 2]])


def test_negative_sample_weight():
    with pytest.raises(ValueError, match="Negative values in data passed"):
        DecisionTreeLabelRanker().fit([[0], [1]], [[1, 2], [2, 1]], [1, -1])


def test_convergence_warning():
    # With a single iteration, Algorithm 1 cannot confirm the center of a node
    # whose incomplete rankings change it
    rng = np.random.RandomState(1)
    X = rng.rand(30, 2)
    y = random_rankings(30, 5, missing=0.5, random_state=rng)
    match = r"of \d+ nodes: .*max_iter=1 iterations"
    with pytest.warns(ConvergenceWarning, match=match):
        DecisionTreeLabelRanker(max_iter=1, random_state=0).fit(X, y)


@pytest.mark.parametrize("splitter", ["best", "random"])
def test_random_state(splitter):
    rng = np.random.RandomState(0)
    X = rng.rand(40, 4)
    y = random_rankings(40, 4, missing=0.2, random_state=rng)
    trees = [
        DecisionTreeLabelRanker(splitter=splitter, max_features=2, random_state=1)
        .fit(X, y)
        .tree_
        for _ in range(2)
    ]
    assert_array_equal(trees[0].feature, trees[1].feature)
    assert_array_equal(trees[0].threshold, trees[1].threshold)


def test_cost_complexity_pruning():
    # The impurities are finite, so the pruning path is defined, and pruning with
    # its largest effective alpha leaves the root alone
    rng = np.random.RandomState(0)
    X = rng.rand(40, 3)
    y = random_rankings(40, 4, missing=0.2, random_state=rng)
    ranker = DecisionTreeLabelRanker(random_state=0)
    path = ranker.cost_complexity_pruning_path(X, y)
    assert np.all(np.isfinite(path.ccp_alphas))
    assert np.all(np.diff(path.ccp_alphas) >= 0)
    pruned = DecisionTreeLabelRanker(random_state=0, ccp_alpha=path.ccp_alphas[-1])
    assert pruned.fit(X, y).tree_.node_count == 1


@pytest.mark.parametrize("sparse_container", [csc_array, csr_array])
def test_sparse_equals_dense(sparse_container):
    # The checks of sparse data cannot check the predictions of a ranker
    rng = np.random.RandomState(0)
    X = rng.rand(30, 3)
    X[X < 0.5] = 0
    y = random_rankings(30, 4, missing=0.2, random_state=rng)
    dense = DecisionTreeLabelRanker(random_state=0).fit(X, y)
    sparse = DecisionTreeLabelRanker(random_state=0).fit(sparse_container(X), y)
    assert_array_equal(dense.tree_.feature, sparse.tree_.feature)
    assert_array_equal(dense.tree_.threshold, sparse.tree_.threshold)
    assert_array_equal(dense.predict(X), sparse.predict(sparse_container(X)))


def test_export_graphviz():
    ranker = DecisionTreeLabelRanker().fit([[0], [1]], [[1, 2], [2, 1]])
    assert "mallows" in export_graphviz(ranker)
