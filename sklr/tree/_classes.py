"""Decision tree label ranker."""

from numbers import Integral
from typing import Literal, Self

import numpy as np
from numpy.random import RandomState
from numpy.typing import ArrayLike
from scipy.sparse import issparse
from sklearn.utils.validation import check_is_fitted

from sklr._distributions._convergence import warn_if_not_converged
from sklr._distributions._extension import complete_rankings
from sklr._distributions.mallows import estimate_center
from sklr.base import LabelRankerMixin
from sklr.tree._criterion import MallowsCriterion
from sklr.utils._ranking import _validate_data
from sklr.utils._sklearn_compat import (
    BaseDecisionTree,
    Interval,
    StrOptions,
    _check_sample_weight,
    _fit_context,
)

CRITERIA = {"mallows": MallowsCriterion}


def _complete_node_values(tree):
    """Complete the ranking of each node from those of its ancestors.

    The value of each node ranks only the labels ranked in its samples. As in
    Section 5.2 of [cheng_decision_2009], it is completed with its closest
    extension to the completed ranking of its parent, which is the ranking of the
    closest ancestor that ranks more labels. The ranking of the root, which only
    misses the labels never ranked in the training set, is completed with its
    closest extension to its own Borda count.
    """
    value = tree.value[:, :, 0]
    parent = np.empty(tree.node_count, dtype=np.intp)
    is_split = tree.children_left >= 0
    parent[tree.children_left[is_split]] = np.flatnonzero(is_split)
    parent[tree.children_right[is_split]] = np.flatnonzero(is_split)
    # The builders add each node before its children, so the parent of a node is
    # completed before the node
    value[0] = estimate_center(value[:1])
    for node in range(1, tree.node_count):
        value[node] = complete_rankings(
            value[node : node + 1], value[parent[node]].astype(np.intp)
        )[0]


class DecisionTreeLabelRanker(LabelRankerMixin, BaseDecisionTree):
    """A decision tree label ranker.

    The tree is grown as in [cheng_decision_2009]_: each split is the threshold on
    a feature that best separates the rankings of the samples of a node, measured
    by the spread of the Mallows model of the rankings of each child, and each
    leaf predicts the center ranking of that model. The training rankings may be
    incomplete.

    Read more in the :ref:`User Guide <tree>`.

    Parameters
    ----------
    criterion : {"mallows"}, default="mallows"
        The function to measure the quality of a split. With ``"mallows"``, the
        impurity of a node is :math:`1 / (1 + \\hat{\\theta})`, where
        :math:`\\hat{\\theta}` is the estimated spread of the Mallows model of its
        rankings, and it is zero when its rankings are consistent. See the Notes for
        details.

    splitter : {"best", "random"}, default="best"
        The strategy used to choose the split at each node. Supported strategies
        are ``"best"`` to choose the best split and ``"random"`` to choose the best
        random split.

    max_depth : int, default=None
        The maximum depth of the tree. If ``None``, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        `min_samples_split` samples.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node. If int,
        then consider `min_samples_split` as the minimum number. If float, then
        `min_samples_split` is a fraction and ``ceil(min_samples_split *
        n_samples)`` are the minimum number of samples for each split.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node. A split point
        at any depth will only be considered if it leaves at least
        `min_samples_leaf` training samples in each of the left and right
        branches. If int, then consider `min_samples_leaf` as the minimum number.
        If float, then `min_samples_leaf` is a fraction and ``ceil(min_samples_leaf
        * n_samples)`` are the minimum number of samples for each node.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all the input
        samples) required to be at a leaf node. Samples have equal weight when
        `sample_weight` is not provided.

    max_features : int, float, {"sqrt", "log2"} or None, default=None
        The number of features to consider when looking for the best split. If
        int, then consider `max_features` features at each split. If float, then
        `max_features` is a fraction and ``max(1, int(max_features *
        n_features_in_))`` features are considered at each split. If ``"sqrt"``,
        then ``max_features=sqrt(n_features)``. If ``"log2"``, then
        ``max_features=log2(n_features)``. If ``None``, then
        ``max_features=n_features``. The search for a split does not stop until at
        least one valid partition of the node samples is found, even if it
        requires to effectively inspect more than `max_features` features.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator. The features are always randomly
        permuted at each split, even if `splitter` is set to ``"best"``. When
        ``max_features < n_features``, the algorithm will select `max_features` at
        random at each split before finding the best split among them. But the
        best found split may vary across different runs, even if ``max_features ==
        n_features``. That is the case if the improvement of the criterion is
        identical for several splits and one split has to be selected at random.
        To obtain a deterministic behaviour during fitting, `random_state` has to
        be fixed to an integer. See :term:`Glossary <random_state>` for details.

    max_leaf_nodes : int, default=None
        Grow a tree with `max_leaf_nodes` in best-first fashion. Best nodes are
        defined as relative reduction in impurity. If ``None`` then unlimited
        number of leaf nodes.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value. See the Notes for the weighted
        impurity decrease.

    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The subtree
        with the largest cost complexity that is smaller than `ccp_alpha` will be
        chosen. By default, no pruning is performed. See
        :ref:`sklearn:minimal_cost_complexity_pruning` for details.

    max_iter : int, default=100
        Maximum number of iterations of Algorithm 1 of [cheng_decision_2009]_,
        which estimates the center ranking of the Mallows model of each node and
        only iterates when some of its rankings are incomplete. A
        :class:`~sklearn.exceptions.ConvergenceWarning` is issued if the center
        ranking of some node does not converge.

    Attributes
    ----------
    max_features_ : int
        The inferred value of `max_features`.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    n_outputs_ : int
        The number of labels when :meth:`fit` is performed.

    feature_importances_ : ndarray of shape (n_features,)
        The impurity-based feature importances. The higher, the more important the
        feature. The importance of a feature is computed as the (normalized) total
        reduction of the criterion brought by that feature.

    tree_ : Tree instance
        The underlying Tree object, whose `value` attribute holds the ranking of
        each node, with shape ``(node_count, n_labels, 1)``. Please refer to
        ``help(sklearn.tree._tree.Tree)`` for attributes of Tree object and
        :ref:`sklearn:sphx_glr_auto_examples_tree_plot_unveil_tree_structure.py`
        in the examples of scikit-learn for basic usage of these attributes.

    See Also
    --------
    sklearn.tree.DecisionTreeRegressor : A decision tree regressor.
    sklr.neighbors.KNeighborsLabelRanker : Label ranker based on the k-nearest
        neighbors.

    Notes
    -----
    The center ranking :math:`\\pi` and the spread :math:`\\theta \\geq 0` of the
    Mallows model of the rankings of a node :math:`t` are estimated as in
    [cheng_decision_2009]_, by Algorithm 1 and by maximum likelihood, on the labels
    ranked by some of those rankings, which gives :math:`\\hat{\\pi}_t` and
    :math:`\\hat{\\theta}_t`. The impurity of the node is

    .. math::

        I_t = \\frac{1}{1 + \\hat{\\theta}_t},

    which is zero when :math:`\\hat{\\theta}_t` is infinite, the case of rankings that
    agree with :math:`\\hat{\\pi}_t`, and one when :math:`\\hat{\\theta}_t` is zero, the
    case of rankings whose mean Kendall distance to :math:`\\hat{\\pi}_t` is not below
    that of rankings drawn uniformly at random. It is also zero when the rankings of the
    node are consistent, that is, when no two of them order a pair of labels
    differently, so that the node becomes a leaf as in [cheng_decision_2009]_.
    Incomplete rankings may be consistent without agreeing with any center ranking, such
    as those that put :math:`y_1` ahead of :math:`y_2`, :math:`y_2` ahead of :math:`y_3`
    and :math:`y_3` ahead of :math:`y_1`.

    [cheng_decision_2009]_ choose the split that maximizes the weighted mean of
    the spreads of the children instead, which is infinite whenever a child has a
    single sample. The tree chooses the split that minimizes the weighted mean of
    their impurities, as the trees of scikit-learn do. The weighted impurity
    decrease of a split of a node :math:`t` with :math:`N_t` of the :math:`N`
    samples into children :math:`t_L` and :math:`t_R` with :math:`N_{t_L}` and
    :math:`N_{t_R}` samples is

    .. math::

        \\frac{N_t}{N} \\left(I_t - \\frac{N_{t_L}}{N_t} I_{t_L}
        - \\frac{N_{t_R}}{N_t} I_{t_R}\\right),

    where all the numbers of samples are weighted if `sample_weight` is passed,
    and a node becomes a leaf when that of its best split is below
    `min_impurity_decrease`. With the default value, this happens when even the
    best split increases the impurity, which [cheng_decision_2009]_ do not
    consider.

    The ranking of each node orders the labels ranked by its samples as their
    center ranking does, and it is completed as in Section 5.2 of
    [cheng_decision_2009]_, with its closest extension to the ranking of its
    parent, so the other labels are ordered as in the closest ancestor that ranks
    them. The ranking of the root, which only misses the labels never ranked in
    the training set, is completed with its closest extension to its own Borda
    count.

    The default values for the parameters controlling the size of the trees (e.g.
    `max_depth`, `min_samples_leaf`, etc.) lead to fully grown and unpruned trees
    which can potentially be very large on some data sets. To reduce memory
    consumption, the complexity and size of the trees should be controlled by
    setting those parameter values.

    References
    ----------
    .. [cheng_decision_2009] W. Cheng, J. Hühn and E. Hüllermeier, "Decision tree
       and instance-based learning for label ranking", in Proceedings of the 26th
       International Conference on Machine Learning, Montreal, Canada,
       pp. 161-168, 2009.

    Examples
    --------
    >>> from sklr.tree import DecisionTreeLabelRanker
    >>> X = [[0, 1, 0], [1, 0, 1], [1, 1, 0], [0, 1, 0]]
    >>> y = [[1, 2, 3], [2, 1, 3], [1, 2, 3], [3, 1, 2]]
    >>> ranker = DecisionTreeLabelRanker(random_state=0).fit(X, y)
    >>> ranker.predict([[0, 1, 0], [1, 1, 0]])
    array([[2, 1, 3],
           [1, 2, 3]])
    """

    _parameter_constraints: dict = {
        **BaseDecisionTree._parameter_constraints,
        "criterion": [StrOptions(set(CRITERIA))],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
    }
    _parameter_constraints.pop("monotonic_cst")

    def __init__(
        self,
        *,
        criterion: Literal["mallows"] = "mallows",
        splitter: Literal["best", "random"] = "best",
        max_depth: int | None = None,
        min_samples_split: float = 2,
        min_samples_leaf: float = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features: float | Literal["sqrt", "log2"] | None = None,
        random_state: int | RandomState | None = None,
        max_leaf_nodes: int | None = None,
        min_impurity_decrease: float = 0.0,
        ccp_alpha: float = 0.0,
        max_iter: int = 100,
    ) -> None:
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            random_state=random_state,
            min_impurity_decrease=min_impurity_decrease,
            ccp_alpha=ccp_alpha,
        )
        self.max_iter = max_iter

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        # As in the trees of scikit-learn, missing values are only accepted in
        # dense arrays, but the tag is set for the common checks
        tags.input_tags.allow_nan = True
        return tags

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(
        self, X: ArrayLike, y: ArrayLike, sample_weight: ArrayLike | None = None
    ) -> Self:
        """Build a decision tree label ranker from the training set.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided to a sparse
            ``csc_matrix``.

        y : array-like of shape (n_samples, n_labels)
            Training rankings, as described in :ref:`ranking_representation`,
            possibly incomplete but without ties.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights, which must be non-negative. If ``None``, then samples
            are equally weighted. Samples with zero weight are left out of the
            nodes.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X, y = _validate_data(
            self, X, y, dtype=np.float32, accept_sparse="csc", ensure_all_finite=False
        )
        missing_values_in_feature_mask = self._compute_missing_values_in_feature_mask(X)
        if issparse(X):
            X.sort_indices()
            if X.indices.dtype != np.intc or X.indptr.dtype != np.intc:
                raise ValueError("No support for np.int64 index based sparse matrices")
        if sample_weight is not None:
            sample_weight = _check_sample_weight(
                sample_weight, X, dtype=np.float64, ensure_non_negative=True
            )

        # BaseDecisionTree._fit only takes the criterion from the attribute, and
        # only knows the names of those of scikit-learn
        # np.unique does not take two nan as equal, so the unranked labels are
        # replaced by zero, which is no position, to find the distinct rankings
        _, first, pattern_of = np.unique(
            np.nan_to_num(y, nan=0), axis=0, return_index=True, return_inverse=True
        )
        criterion = self.criterion
        self.criterion = CRITERIA[criterion](
            y.shape[1], X.shape[0], self.max_iter, y[first], pattern_of.reshape(-1)
        )
        try:
            self._fit(
                X,
                y,
                sample_weight=sample_weight,
                check_input=False,
                missing_values_in_feature_mask=missing_values_in_feature_mask,
            )
            n_values = self.criterion.n_values
            n_not_converged = self.criterion.n_not_converged
        finally:
            self.criterion = criterion

        warn_if_not_converged(
            np.arange(n_values) >= n_not_converged,
            "center ranking",
            f"it kept changing after max_iter={self.max_iter} iterations or started "
            "to repeat itself",
            groups="nodes",
        )
        _complete_node_values(self.tree_)
        return self

    def predict(self, X: ArrayLike, check_input: bool = True) -> np.ndarray:
        """Predict the rankings for the provided data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided to a sparse
            ``csr_matrix``.

        check_input : bool, default=True
            Allow to bypass several input checking. Don't use this parameter unless
            you know what you're doing.

        Returns
        -------
        y : ndarray of shape (n_samples, n_labels)
            Predicted rankings, complete and without ties, as the positions of the
            labels.
        """
        check_is_fitted(self)
        X = self._validate_X_predict(X, check_input)
        return self.tree_.predict(X)[:, :, 0].astype(np.intp)
