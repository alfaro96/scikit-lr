import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.sparse import csc_array, csr_array
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.tree import export_graphviz

from sklr._aggregation.tests import _oracles as _bucket_oracles
from sklr._aggregation.tests._oracles import random_partial_rankings
from sklr._distributions.tests import _oracles
from sklr._distributions.tests._oracles import random_rankings
from sklr.tree import DecisionTreeLabelRanker, DecisionTreePartialLabelRanker
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
        DecisionTreePartialLabelRanker(),
        DecisionTreePartialLabelRanker(criterion="disagreements", splitter="random"),
        DecisionTreePartialLabelRanker(criterion="distance", max_depth=3),
        DecisionTreePartialLabelRanker(
            criterion="gini", max_leaf_nodes=4, ccp_alpha=0.01
        ),
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


def _value(y, sample_weight, parent_value):
    """Compute the ranking of a node, completed from that of its parent."""
    value = _center(y, sample_weight)
    if parent_value is None:
        value, _ = _oracles.estimate_center(value[np.newaxis], [1])
    else:
        value = _oracles.complete_ranking(value, parent_value)
    return value


def _check_node(
    ranker, X, y, sample_weight, node, rows, parent_value, impurity_of, value_of
):
    """Check a node of a fitted tree, and its descendants, against naive estimates.

    The impurity of the node, computed by `impurity_of` from its rankings and
    weights, whether it is a leaf and its value, computed by `value_of` from them
    and from the value of its parent, are checked from its training samples `rows`.
    Its split must be among the best ones found by trying every threshold between
    consecutive values of every feature.
    """
    tree = ranker.tree_
    total_weight = sample_weight.sum()
    impurity = impurity_of(y[rows], sample_weight[rows])
    assert_allclose(tree.impurity[node], impurity, rtol=1e-9, atol=1e-12)

    value = value_of(y[rows], sample_weight[rows], parent_value)
    assert_array_equal(tree.value[node, :, 0], value)

    weight = sample_weight[rows].sum()
    splits = []
    for feature in range(X.shape[1]):
        values = np.unique(X[rows, feature])
        for threshold in values[:-1]:
            left = rows[X[rows, feature] <= threshold]
            right = rows[X[rows, feature] > threshold]
            proxy = -(
                sample_weight[left].sum() * impurity_of(y[left], sample_weight[left])
                + sample_weight[right].sum()
                * impurity_of(y[right], sample_weight[right])
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
        _check_node(
            ranker,
            X,
            y,
            sample_weight,
            child,
            child_rows,
            value,
            impurity_of,
            value_of,
        )


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

    _check_node(
        ranker,
        X,
        y,
        sample_weight,
        0,
        np.arange(n_samples),
        None,
        _impurity,
        _value,
    )
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


def _random_y(ranker, n_samples, n_labels, *, missing, random_state):
    """Draw rankings for a label ranker, or partial rankings for the others."""
    if isinstance(ranker, DecisionTreeLabelRanker):
        return random_rankings(
            n_samples, n_labels, missing=missing, random_state=random_state
        )
    return random_partial_rankings(
        n_samples, n_labels, missing=missing, random_state=random_state
    )


RANKERS = [
    DecisionTreeLabelRanker(random_state=0),
    DecisionTreePartialLabelRanker(criterion="disagreements", random_state=0),
    DecisionTreePartialLabelRanker(criterion="distance", random_state=0),
    DecisionTreePartialLabelRanker(criterion="entropy", random_state=0),
    DecisionTreePartialLabelRanker(criterion="gini", random_state=0),
]


@pytest.mark.parametrize("ranker", RANKERS)
def test_sample_weight_integer_equals_repeated_samples(ranker):
    rng = np.random.RandomState(0)
    X = rng.rand(30, 3)
    y = _random_y(ranker, 30, 4, missing=0.3, random_state=rng)
    sample_weight = rng.randint(0, 4, len(X))

    weighted = clone(ranker).fit(X, y, sample_weight=sample_weight)
    repeated = clone(ranker).fit(
        np.repeat(X, sample_weight, axis=0), np.repeat(y, sample_weight, axis=0)
    )

    assert_array_equal(weighted.tree_.feature, repeated.tree_.feature)
    assert_array_equal(weighted.tree_.threshold, repeated.tree_.threshold)
    assert_allclose(weighted.tree_.impurity, repeated.tree_.impurity)
    assert_array_equal(weighted.tree_.value, repeated.tree_.value)


@pytest.mark.parametrize("ranker", RANKERS)
def test_sample_weight_ones_and_scale(ranker):
    # Weights of one are no weights, and scaling the weights by a power of two
    # scales the weights of every node exactly, so the trees are the same
    rng = np.random.RandomState(0)
    X = rng.rand(30, 3)
    y = _random_y(ranker, 30, 4, missing=0.3, random_state=rng)
    sample_weight = rng.rand(len(X))

    unweighted = clone(ranker).fit(X, y)
    ones = clone(ranker).fit(X, y, sample_weight=np.ones(len(X)))
    weighted = clone(ranker).fit(X, y, sample_weight=sample_weight)
    scaled = clone(ranker).fit(X, y, sample_weight=4 * sample_weight)

    for first, second in [(unweighted, ones), (weighted, scaled)]:
        assert_array_equal(first.tree_.feature, second.tree_.feature)
        assert_array_equal(first.tree_.threshold, second.tree_.threshold)
        assert_array_equal(first.tree_.impurity, second.tree_.impurity)
        assert_array_equal(first.tree_.value, second.tree_.value)


@pytest.mark.parametrize("ranker", RANKERS)
def test_sparse_int64_indices(ranker):
    X = csc_array(np.eye(3))
    X.indices = X.indices.astype(np.int64)
    X.indptr = X.indptr.astype(np.int64)
    with pytest.raises(ValueError, match=r"No support for np\.int64 index"):
        clone(ranker).fit(X, [[1, 2], [2, 1], [1, 2]])


@pytest.mark.parametrize("ranker", RANKERS)
def test_negative_sample_weight(ranker):
    with pytest.raises(ValueError, match="Negative values in data passed"):
        clone(ranker).fit([[0], [1]], [[1, 2], [2, 1]], [1, -1])


def test_convergence_warning():
    # With a single iteration, Algorithm 1 cannot confirm the center of a node
    # whose incomplete rankings change it
    rng = np.random.RandomState(1)
    X = rng.rand(30, 2)
    y = random_rankings(30, 5, missing=0.5, random_state=rng)
    match = r"of \d+ nodes: .*max_iter=1 iterations"
    with pytest.warns(ConvergenceWarning, match=match):
        DecisionTreeLabelRanker(max_iter=1, random_state=0).fit(X, y)


@pytest.mark.parametrize("ranker", RANKERS)
@pytest.mark.parametrize("splitter", ["best", "random"])
def test_random_state(ranker, splitter):
    rng = np.random.RandomState(0)
    X = rng.rand(40, 4)
    y = _random_y(ranker, 40, 4, missing=0.2, random_state=rng)
    trees = [
        clone(ranker)
        .set_params(splitter=splitter, max_features=2, random_state=1)
        .fit(X, y)
        .tree_
        for _ in range(2)
    ]
    assert_array_equal(trees[0].feature, trees[1].feature)
    assert_array_equal(trees[0].threshold, trees[1].threshold)


@pytest.mark.parametrize("ranker", RANKERS)
def test_cost_complexity_pruning(ranker):
    # The impurities are finite, so the pruning path is defined, and pruning with
    # its largest effective alpha leaves the root alone
    rng = np.random.RandomState(0)
    X = rng.rand(40, 3)
    y = _random_y(ranker, 40, 4, missing=0.2, random_state=rng)
    path = clone(ranker).cost_complexity_pruning_path(X, y)
    assert np.all(np.isfinite(path.ccp_alphas))
    assert np.all(np.diff(path.ccp_alphas) >= 0)
    pruned = clone(ranker).set_params(ccp_alpha=path.ccp_alphas[-1])
    assert pruned.fit(X, y).tree_.node_count == 1


@pytest.mark.parametrize("ranker", RANKERS)
@pytest.mark.parametrize("sparse_container", [csc_array, csr_array])
def test_sparse_equals_dense(ranker, sparse_container):
    # The checks of sparse data cannot check the predictions of a ranker
    rng = np.random.RandomState(0)
    X = rng.rand(30, 3)
    X[X < 0.5] = 0
    y = _random_y(ranker, 30, 4, missing=0.2, random_state=rng)
    dense = clone(ranker).fit(X, y)
    sparse = clone(ranker).fit(sparse_container(X), y)
    assert_array_equal(dense.tree_.feature, sparse.tree_.feature)
    assert_array_equal(dense.tree_.threshold, sparse.tree_.threshold)
    assert_array_equal(dense.predict(X), sparse.predict(sparse_container(X)))


@pytest.mark.parametrize("ranker", RANKERS)
def test_export_graphviz(ranker):
    ranker = clone(ranker).fit([[0], [1]], [[1, 2], [2, 1]])
    assert ranker.criterion in export_graphviz(ranker)


def _pairwise_impurity(y, sample_weight, criterion):
    """Compute the impurity of a partial label ranking node, ranking by ranking.

    The relation of each pair in a ranking or in the consensus is ``1``, ``0`` or
    ``-1`` when it puts the first label ahead, ties them or puts the second one
    ahead. The distance of a ranking to the consensus on a pair is the absolute
    difference of their relations, and the consensus is that of the exact
    implementation of the Bucket Pivot Algorithm.
    """
    n_labels = y.shape[1]
    pairs = {}
    for i in range(n_labels):
        for j in range(i + 1, n_labels):
            pairs[i, j] = [
                (np.sign(row[j] - row[i]), weight)
                for row, weight in zip(y, sample_weight)
                if not np.isnan(row[i]) and not np.isnan(row[j])
            ]
    if all(len({relation for relation, _ in pair}) < 2 for pair in pairs.values()):
        return 0.0
    consensus = _bucket_oracles.aggregate(y, sample_weight)

    impurity = 0.0
    for (i, j), pair in pairs.items():
        if not pair:
            continue
        total = sum(weight for _, weight in pair)
        expected = np.sign(consensus[j] - consensus[i])
        if criterion == "disagreements":
            impurity += sum(w for r, w in pair if r != expected) / total
        elif criterion == "distance":
            impurity += sum(w * abs(r - expected) for r, w in pair) / total
        else:
            fractions = [
                sum(w for r, w in pair if r == relation) / total
                for relation in [1, 0, -1]
            ]
            if criterion == "entropy":
                pair_impurity = -sum(q * np.log2(q) for q in fractions if q > 0)
            else:
                pair_impurity = 1 - sum(q**2 for q in fractions)
            impurity += pair_impurity / (n_labels * (n_labels - 1) / 2)
    return impurity


@pytest.mark.parametrize("criterion", ["disagreements", "distance", "entropy", "gini"])
@pytest.mark.parametrize("missing", [0, 0.3, 0.6])
@pytest.mark.parametrize("weighted", [False, True])
def test_partial_label_ranking_tree_oracle(criterion, missing, weighted):
    # Every node is checked against the naive impurity, its consensus in exact
    # arithmetic and an exhaustive search of the best split, as in the label
    # ranking tree. The weights are integers, so that the entries of the pair order
    # matrices are exact
    rng = np.random.RandomState(0)
    n_samples, n_labels = 25, 4
    X = rng.rand(n_samples, 2).astype(np.float32)
    y = random_partial_rankings(n_samples, n_labels, missing=missing, random_state=rng)
    sample_weight = (
        rng.randint(1, 4, n_samples).astype(np.float64)
        if weighted
        else np.ones(n_samples)
    )

    ranker = DecisionTreePartialLabelRanker(criterion=criterion, random_state=0)
    ranker.fit(X, y, sample_weight=sample_weight if weighted else None)

    _check_node(
        ranker,
        X,
        y,
        sample_weight,
        0,
        np.arange(n_samples),
        None,
        lambda y, sample_weight: _pairwise_impurity(y, sample_weight, criterion),
        lambda y, sample_weight, _: _bucket_oracles.aggregate(y, sample_weight),
    )
    leaves = ranker.apply(X)
    assert_array_equal(ranker.predict(X), ranker.tree_.value[leaves, :, 0])


@pytest.mark.parametrize(
    "criterion, expected",
    [
        # The consensus of a ranking, its reverse and a ranking that ties both
        # labels ties them, which only the reversed rankings disagree with, at a
        # distance of one each
        ("disagreements", 2 / 3),
        ("distance", 2 / 3),
        ("entropy", np.log2(3)),
        ("gini", 2 / 3),
    ],
)
def test_partial_label_ranking_impurity(criterion, expected):
    ranker = DecisionTreePartialLabelRanker(criterion=criterion)
    ranker.fit([[0], [0], [0]], [[1, 2], [2, 1], [1, 1]])
    assert ranker.tree_.node_count == 1
    assert_allclose(ranker.tree_.impurity[0], expected)
    assert_array_equal(ranker.tree_.value[0, :, 0], [1, 1])


@pytest.mark.parametrize("criterion", ["disagreements", "distance", "entropy", "gini"])
@pytest.mark.parametrize(
    "y",
    [
        # Equal partial rankings
        [[1, 1, 2], [1, 1, 2], [1, 1, 2]],
        # Incomplete partial rankings that agree with a single bucket order
        [[1, 1, np.nan], [np.nan, 1, 2], [1, np.nan, 2]],
        # Incomplete rankings that order every pair of labels in the same way,
        # but in a cycle that no bucket order agrees with
        [[1, 2, np.nan], [np.nan, 1, 2], [2, np.nan, 1]],
        # Rankings of a single label or of none
        [[1, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, 1, np.nan]],
    ],
)
def test_partial_label_ranking_consistent_rankings_make_a_leaf(criterion, y):
    ranker = DecisionTreePartialLabelRanker(criterion=criterion)
    ranker.fit([[0], [1], [2]], y)
    assert ranker.tree_.node_count == 1
    assert ranker.tree_.impurity[0] == 0


def test_partial_label_ranking_labels_without_information_tied_last():
    # The leaves order the first two labels in opposite ways. The third label is
    # only ranked alone, in the right leaf, and the fourth one is never ranked, so
    # both are tied in a last bucket in each leaf
    X = [[0], [0], [1], [1]]
    y = [
        [2, 1, np.nan, np.nan],
        [2, 1, np.nan, np.nan],
        [1, 2, np.nan, np.nan],
        [np.nan, np.nan, 1, np.nan],
    ]
    ranker = DecisionTreePartialLabelRanker().fit(X, y)
    assert ranker.tree_.node_count == 3
    assert_array_equal(ranker.predict([[0], [1]]), [[2, 1, 3, 3], [1, 2, 3, 3]])


def test_partial_label_ranking_accepts_label_rankings():
    # A ranking without ties is a bucket order with one label in each bucket
    rng = np.random.RandomState(0)
    X = rng.rand(20, 2)
    y = random_rankings(20, 4, missing=0, random_state=rng)
    ranker = DecisionTreePartialLabelRanker(random_state=0).fit(X, y)
    assert_array_equal(ranker.predict(X), y)
    assert ranker.predict(X).dtype == np.intp


def test_partial_label_ranking_right_child_without_rounding_error():
    # The node adds up the weights of the first three samples in their order, and
    # the left child in the order of the feature, which leaves a positive rounding
    # error in the weight of the rankings of the right child that tie the third label
    # with the others, although none of its rankings ranks it. Taken as information,
    # it would make that label the pivot of the consensus of the right child, which
    # would then tie the first two labels, and their disagreements would be one
    # instead of the fraction of rankings that reverse them
    X = [[2], [0], [1], [3], [4], [5], [6], [7]]
    y = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 2, np.nan],
        [1, 2, np.nan],
        [1, 2, np.nan],
        [1, 2, np.nan],
        [2, 1, np.nan],
    ]
    sample_weight = [0.1, 0.2, 0.3, 1, 1, 1, 1, 1]
    ranker = DecisionTreePartialLabelRanker(criterion="disagreements", max_depth=1)
    ranker.fit(X, y, sample_weight=sample_weight)
    assert_array_equal(ranker.apply([[2], [3]]), [1, 2])
    assert_allclose(ranker.tree_.impurity[2], 1 / 5)
