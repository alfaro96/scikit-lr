.. _tree:

Decision trees
==============

.. currentmodule:: sklr.tree

:class:`DecisionTreeLabelRanker` implements the label ranking trees of
[cheng_decision_2009]_. Like the decision trees of scikit-learn, it splits the
feature space recursively with thresholds on the features, but each node fits a
Mallows model to the rankings of its training samples, which may be incomplete,
and each leaf predicts the center ranking of its model:

>>> import numpy as np
>>> from sklr.tree import DecisionTreeLabelRanker
>>> X = [[0], [1], [2], [3], [4], [5]]
>>> y = [[1, 2, 3], [1, 3, 2], [np.nan, 1, 2], [3, 2, 1], [2, np.nan, 1], [3, 1, 2]]
>>> ranker = DecisionTreeLabelRanker(random_state=0).fit(X, y)
>>> ranker.predict([[0.4], [2.6], [4.2]])
array([[1, 2, 3],
       [3, 2, 1],
       [3, 1, 2]])

The rankings follow the representation described in :ref:`ranking_representation`:
each row holds the position of each label, and ``np.nan`` marks the labels that a
training ranking does not rank. The training rankings may be incomplete, but they
must not have ties. The predicted rankings are always complete and without ties.

:class:`DecisionTreeLabelRanker` is built on the trees of scikit-learn, so it
shares their parameters, except ``monotonic_cst``, and their methods, such as
:meth:`~DecisionTreeLabelRanker.apply`,
:meth:`~DecisionTreeLabelRanker.decision_path`,
:meth:`~DecisionTreeLabelRanker.get_depth`,
:meth:`~DecisionTreeLabelRanker.get_n_leaves` and
:meth:`~DecisionTreeLabelRanker.cost_complexity_pruning_path`. Its fitted tree is
stored in the same structure, the :attr:`~DecisionTreeLabelRanker.tree_`
attribute, whose ``value`` holds the ranking of each node. The
:ref:`decision trees <sklearn:tree>` section of the scikit-learn user guide
describes them in detail. As there, the features may have missing values, encoded
as ``np.nan``, unless they are given as a sparse matrix; see
:ref:`sklearn:tree_missing_value_support`.

Tree induction
--------------

As in Section 5.1 of [cheng_decision_2009]_, the tree is grown by binary splits of the
form :math:`x_j \leq a`, where :math:`x_j` is the :math:`j`-th feature of an instance
:math:`\boldsymbol{x}` and :math:`a` a threshold, and the training samples of each node
are split between its children. The ``splitter`` parameter chooses how the thresholds
are searched: ``"best"`` evaluates every threshold between consecutive values of each
feature, and ``"random"`` a single random threshold for each feature. ``max_features``
sets the number of features drawn at random for each split, and ``random_state`` fixes
these random choices.

The quality of a split is measured by the Mallows model of the rankings of each child.
The Mallows model is a distance-based model whose mode is a center ranking :math:`\pi`:
the probability of a ranking decreases with its Kendall distance to :math:`\pi`, at a
rate given by a spread parameter :math:`\theta \geq 0`. As in [cheng_decision_2009]_,
the model of a node :math:`t` is estimated only on the labels that some of its rankings
rank, which gives the estimates :math:`\hat{\pi}_t` and :math:`\hat{\theta}_t`. The
center ranking is estimated by Algorithm 1 of [cheng_decision_2009]_, the generalized
Borda count with the completion of the incomplete rankings that :ref:`neighbors`
describes, and the spread by maximum likelihood from the mean Kendall distance of the
completed rankings to the center. The spread is zero when that distance is not below
that of rankings drawn uniformly at random, and infinite when it is zero.

Impurity
~~~~~~~~

[cheng_decision_2009]_ choose the split of a node :math:`t` with :math:`N_t` samples
into a left child :math:`t_L` and a right child :math:`t_R`, with :math:`N_{t_L}` and
:math:`N_{t_R}` samples, that maximizes the weighted mean of the spreads of the
children, equation (6) of the paper:

.. math::

    \frac{N_{t_L} \hat{\theta}_{t_L} + N_{t_R} \hat{\theta}_{t_R}}{N_t}.

However, a child with a single sample has an infinite spread, so this measure would
be infinite for any split that isolates a sample. Instead,
:class:`DecisionTreeLabelRanker` gives each node :math:`t` the impurity

.. math::

    I_t = \frac{1}{1 + \hat{\theta}_t},

which, like the variance in the regression trees that [cheng_decision_2009]_ take as a
model, is smaller when the rankings are more concentrated around the center ranking. It
is zero when :math:`\hat{\theta}_t` is infinite, the case of rankings that agree with
the center ranking, and one when :math:`\hat{\theta}_t` is zero. Then, as the trees of
scikit-learn do, the tree chooses the split that minimizes the weighted mean of the
impurities of the children, whose weighted impurity decrease is

.. math::

    \frac{N_t}{N} \left(I_t - \frac{N_{t_L}}{N_t} I_{t_L}
    - \frac{N_{t_R}}{N_t} I_{t_R}\right),

where :math:`N` is the number of training samples, all the numbers of samples being
weighted if ``sample_weight`` is passed. This criterion and equation (6) do not always
prefer the same split.

The impurity of a node is also zero when its rankings are consistent, that is, when no
two of them order a pair of labels differently, which is the first rule of
[cheng_decision_2009]_ to stop the growth. With complete rankings, consistent rankings
are equal and their spread is infinite, but incomplete rankings may be consistent
without agreeing with any center ranking, such as those that put :math:`y_1` ahead of
:math:`y_2`, :math:`y_2` ahead of :math:`y_3` and :math:`y_3` ahead of :math:`y_1`. A
node with zero impurity is a leaf.

The other rule of [cheng_decision_2009]_ stops the growth when "the number of
labels in a node" is below :math:`2|\mathcal{Y}| = 2n`, where :math:`\mathcal{Y}`
is the set of the :math:`n` labels, which admits several readings. It is not
applied by default: as in scikit-learn, a node is split if it has at least
``min_samples_split`` samples, two by default. Reading it as a node with fewer than
:math:`2n` samples, it amounts to setting ``min_samples_split`` to :math:`2n`, which
is what the reproduction of the experiments of the paper does. Besides, a node
becomes a leaf when the weighted impurity decrease of its best split is below
``min_impurity_decrease``. With the default value, zero, this happens when even the
best split increases the impurity, a case that [cheng_decision_2009]_ do not
consider.

Incomplete rankings
-------------------

When the rankings of a node are incomplete, some labels may not be ranked by any of
its samples. As the model of the node is estimated only on the labels that its
rankings rank, its center ranking leaves the other labels out. As in Section 5.2 of
[cheng_decision_2009]_, the ranking of the node is completed with its closest
extension in Kendall distance to the ranking of its parent, found as in
Proposition 1 of [cheng_decision_2009]_, that is, the ranking of all the labels
that keeps the order of the center ranking and is closest to the ranking of the
parent. The rankings are completed from the root to the leaves, so the ranking of
the parent is already complete, and the other labels are ordered as in the closest
ancestor that ranks them. The ranking of the root, which only misses the labels
never ranked in the training set, is completed with its closest extension to its
own Borda count.

In the following tree, no sample of the right leaf ranks the first label, which
takes its place from the root:

>>> X = [[0], [1], [2], [3]]
>>> y = [[1, 2, 3], [1, 2, 3], [np.nan, 2, 1], [np.nan, 2, 1]]
>>> ranker = DecisionTreeLabelRanker().fit(X, y)
>>> ranker.tree_.value[:, :, 0]
array([[1., 2., 3.],
       [1., 2., 3.],
       [1., 3., 2.]])
>>> ranker.predict([[0], [3]])
array([[1, 2, 3],
       [1, 3, 2]])

Sample weights
--------------

The ``sample_weight`` parameter of :meth:`~DecisionTreeLabelRanker.fit` weighs the
rankings in the estimation of the models and the numbers of samples of the impurity
decrease and of ``min_weight_fraction_leaf``. The weights must be non-negative.
The samples with zero weight do not reach any node, so they are left out of the
models, the splits and the numbers of samples of the nodes, but they still count in
the number of training samples from which ``min_samples_split`` and
``min_samples_leaf`` are found when they are fractions.

Tree size and pruning
---------------------

The default parameters grow the tree until no leaf can be split, because it is
pure, has a single sample or only constant features, or because every split
increases its impurity, so the tree may be very large and overfit the training
rankings. As in scikit-learn, its size is limited
before the growth by ``max_depth``, ``min_samples_split``, ``min_samples_leaf``,
``min_weight_fraction_leaf``, ``max_leaf_nodes``, which grows the tree best first,
and ``min_impurity_decrease``, and after the growth by ``ccp_alpha``, the parameter
of :ref:`minimal cost-complexity pruning <sklearn:minimal_cost_complexity_pruning>`.
[cheng_decision_2009]_ do not prune the trees. As the impurity is between zero and
one, the impurity decrease and the cost complexity are always finite.
:meth:`~DecisionTreeLabelRanker.cost_complexity_pruning_path` gives the values of
``ccp_alpha`` at which the pruned tree changes:

>>> from scipy.stats import rankdata
>>> rng = np.random.RandomState(0)
>>> X = rng.normal(size=(100, 3))
>>> y = rankdata(-X @ rng.normal(size=(3, 4)), axis=1)
>>> ranker = DecisionTreeLabelRanker(random_state=0).fit(X, y)
>>> ranker.get_depth(), int(ranker.get_n_leaves())
(9, 37)
>>> path = ranker.cost_complexity_pruning_path(X, y)
>>> ranker = DecisionTreeLabelRanker(ccp_alpha=path.ccp_alphas[-5], random_state=0)
>>> int(ranker.fit(X, y).get_n_leaves())
5

Convergence
-----------

Algorithm 1 of [cheng_decision_2009]_ only iterates when some ranking of a node is
incomplete, and ``max_iter`` bounds the number of iterations for each node, ``100``
by default. If the center ranking of some node still changes after ``max_iter``
iterations or comes back to an earlier center, which the algorithm would repeat
forever, :meth:`~DecisionTreeLabelRanker.fit` issues a single
:class:`~sklearn.exceptions.ConvergenceWarning` and keeps the center reached.

Exporting the tree
------------------

:func:`sklearn.tree.export_graphviz` exports the tree in the DOT format of
`Graphviz <https://graphviz.org>`_. Each node shows its split, its impurity under
the name of the criterion, its number of samples and its ranking, with the position
of each label in a row:

>>> from sklearn.tree import export_graphviz
>>> X = [[0], [1], [2], [3]]
>>> y = [[1, 2, 3], [1, 2, 3], [np.nan, 2, 1], [np.nan, 2, 1]]
>>> ranker = DecisionTreeLabelRanker().fit(X, y)
>>> print(export_graphviz(ranker, feature_names=["x"]))
digraph Tree {
node [shape=box, fontname="helvetica"] ;
edge [fontname="helvetica"] ;
0 [label="x <= 1.5\nmallows = 0.425\nsamples = 4\nvalue = [[1.0]\n[2.0]\n[3.0]]"] ;
1 [label="mallows = 0.0\nsamples = 2\nvalue = [[1.0]\n[2.0]\n[3.0]]"] ;
0 -> 1 [labeldistance=2.5, labelangle=45, headlabel="True"] ;
2 [label="mallows = 0.0\nsamples = 2\nvalue = [[1.0]\n[3.0]\n[2.0]]"] ;
0 -> 2 [labeldistance=2.5, labelangle=-45, headlabel="False"] ;
}

:func:`sklearn.tree.plot_tree` and :func:`sklearn.tree.export_text` only accept
the trees of scikit-learn, so they cannot draw a :class:`DecisionTreeLabelRanker`.

Practical use
-------------

The splits only depend on the order of the values of each feature, so, unlike the
nearest neighbors, the tree does not need the features to be scaled. The
parameters that limit the size of the tree are chosen with
:class:`~sklearn.model_selection.GridSearchCV` and the scorer of Kendall's
:math:`\tau` from :func:`sklr.metrics.get_scorer`, as described in
:ref:`model_evaluation`:

>>> from sklearn.model_selection import GridSearchCV
>>> from sklr.metrics import get_scorer
>>> X = rng.normal(size=(200, 3))
>>> utilities = np.sin(2 * X) @ rng.normal(size=(3, 4))
>>> y = rankdata(-(utilities + 0.5 * rng.normal(size=(200, 4))), axis=1)
>>> param_grid = {"max_depth": [2, 4, 6, 8, None]}
>>> search = GridSearchCV(
...     DecisionTreeLabelRanker(random_state=0),
...     param_grid,
...     scoring=get_scorer("kendall_tau"),
... )
>>> search.fit(X, y).best_params_
{'max_depth': 4}
>>> search.cv_results_["mean_test_score"].round(2)
array([0.56, 0.61, 0.6 , 0.6 , 0.57])

The rankings are noisy, so the fully grown tree, the last one, fits the noise and
scores worse than a tree of depth four.

The scorers only compare complete rankings, so cross-validation needs complete
rankings in the validation folds. The example
:ref:`sphx_glr_auto_examples_tree_plot_label_ranking_tree.py` shows how the
maximum depth controls the fit, also with incomplete training rankings.

Complexity
----------

With complete rankings, the tree keeps the Borda scores of the labels and the
weight of the rankings that put each label ahead of each other one in each child,
and updates them as the samples move between the children, as the criteria of
scikit-learn do with their sums. For a node with :math:`N_t` samples and :math:`n`
labels, moving a sample takes :math:`O(n^2)` time, and so does the impurity of each
candidate split, apart from solving for the spread, so the ``"best"`` splitter takes
:math:`O(N_t n^2)` time per feature and node.

With incomplete rankings, the closest extension of each ranking depends on the
center ranking, so the impurity of each child is computed again for each candidate
split. Equal rankings have equal extensions, so the tree estimates the model from
the distinct rankings of each child, each weighted by the total weight of its
samples, and only updates those weights as the samples move between the children.
With :math:`R_t` distinct rankings in a node, the consistency test and the
estimation of the spread take :math:`O(R_t n^2)` time, and each iteration of
Algorithm 1 at most :math:`O(R_t n^2)` more. The ``"best"`` splitter evaluates up to
:math:`N_t - 1` thresholds for each feature, which takes :math:`O(N_t R_t n^2)` time
per feature and node, while the ``"random"`` splitter evaluates one, which takes
:math:`O(N_t + R_t n^2)`. :math:`R_t` is at most :math:`N_t`, and much smaller with
few labels, since there are few possible rankings. Limiting the size of the tree,
the number of features with ``max_features``, or using the ``"random"`` splitter
reduce the time of :meth:`~DecisionTreeLabelRanker.fit`. Prediction only follows the
path of each sample to its leaf, as in scikit-learn.
