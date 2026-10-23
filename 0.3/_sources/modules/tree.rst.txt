.. _tree:

Decision trees
==============

.. currentmodule:: sklr.tree

:class:`DecisionTreeLabelRanker` implements the label ranking trees of
[cheng_decision_2009]_, and :class:`DecisionTreePartialLabelRanker` the partial
label ranking trees of [alfaro_learning_2021]_. Like the decision trees of
scikit-learn, they split the feature space recursively with thresholds on the
features, but each node measures how much the rankings of its training samples
disagree, and each leaf predicts a consensus of them.

Both trees are built on the trees of scikit-learn, so they share their parameters,
except ``monotonic_cst``, and their methods, such as
:meth:`~DecisionTreeLabelRanker.apply`,
:meth:`~DecisionTreeLabelRanker.decision_path`,
:meth:`~DecisionTreeLabelRanker.get_depth`,
:meth:`~DecisionTreeLabelRanker.get_n_leaves` and
:meth:`~DecisionTreeLabelRanker.cost_complexity_pruning_path`. Their fitted tree is
stored in the same structure, the :attr:`~DecisionTreeLabelRanker.tree_`
attribute, whose ``value`` holds the ranking of each node. The
:ref:`decision trees <sklearn:tree>` section of the scikit-learn user guide
describes them in detail. As there, the features may have missing values, encoded
as ``np.nan``, unless they are given as a sparse matrix; see
:ref:`sklearn:tree_missing_value_support`.

The rankings follow the representation described in :ref:`ranking_representation`:
each row holds the position of each label, and ``np.nan`` marks the labels that a
training ranking does not rank.

Tree induction
--------------

As in Section 5.1 of [cheng_decision_2009]_ and Section 3.2 of
[alfaro_learning_2021]_, the tree is grown by binary splits of the form
:math:`x_j \leq a`, where :math:`x_j` is the :math:`j`-th feature of an instance
:math:`\boldsymbol{x}` and :math:`a` a threshold, and the training samples of each node
are split between its children. The ``splitter`` parameter chooses how the thresholds
are searched: ``"best"`` evaluates every threshold between consecutive values of each
feature, and ``"random"`` a single random threshold for each feature. ``max_features``
sets the number of features drawn at random for each split, and ``random_state`` fixes
these random choices.

Each node :math:`t` has an impurity :math:`I_t`, computed from its rankings by the
``criterion`` of the tree as the next sections describe. As the trees of
scikit-learn do, the tree chooses the split of a node :math:`t` with :math:`N_t`
samples into a left child :math:`t_L` and a right child :math:`t_R`, with
:math:`N_{t_L}` and :math:`N_{t_R}` samples, that minimizes the weighted mean of the
impurities of the children, whose weighted impurity decrease is

.. math::

    \frac{N_t}{N} \left(I_t - \frac{N_{t_L}}{N_t} I_{t_L}
    - \frac{N_{t_R}}{N_t} I_{t_R}\right),

where :math:`N` is the number of training samples, all the numbers of samples being
weighted if ``sample_weight`` is passed.

The impurity of a node is zero when its rankings are consistent, that is, when no two
of them order a pair of labels differently, which is the first rule of both papers to
stop the growth. With complete rankings, consistent rankings are equal, but
incomplete rankings may be consistent without agreeing with any consensus, such as
those that put :math:`y_1` ahead of :math:`y_2`, :math:`y_2` ahead of :math:`y_3` and
:math:`y_3` ahead of :math:`y_1`. A node with zero impurity is a leaf. Besides, a node
becomes a leaf when the weighted impurity decrease of its best split is below
``min_impurity_decrease``. With the default value, zero, this happens when even the
best split increases the impurity, a case that neither paper considers.

.. _label_ranking_tree:

Label ranking
-------------

:class:`DecisionTreeLabelRanker` fits a Mallows model to the rankings of each node,
which may be incomplete but must not have ties, and each leaf predicts the center
ranking of its model:

>>> import numpy as np
>>> from sklr.tree import DecisionTreeLabelRanker
>>> X = [[0], [1], [2], [3], [4], [5]]
>>> y = [[1, 2, 3], [1, 3, 2], [np.nan, 1, 2], [3, 2, 1], [2, np.nan, 1], [3, 1, 2]]
>>> ranker = DecisionTreeLabelRanker(random_state=0).fit(X, y)
>>> ranker.predict([[0.4], [2.6], [4.2]])
array([[1, 2, 3],
       [3, 2, 1],
       [3, 1, 2]])

The predicted rankings are always complete and without ties.

Mallows impurity
~~~~~~~~~~~~~~~~

The Mallows model is a distance-based model whose mode is a center ranking :math:`\pi`:
the probability of a ranking decreases with its Kendall distance to :math:`\pi`, at a
rate given by a spread parameter :math:`\theta \geq 0`. As in [cheng_decision_2009]_,
the model of a node :math:`t` is estimated only on the labels that some of its rankings
rank, which gives the estimates :math:`\hat{\pi}_t` and :math:`\hat{\theta}_t`. The
center ranking is estimated by Algorithm 1 of [cheng_decision_2009]_, the generalized
Borda count with the completion of the incomplete rankings that
:ref:`nearest neighbors <mallows_neighbors>` describes, and the spread by maximum
likelihood from the mean Kendall distance of the completed rankings to the center. The
spread is zero when that distance is not below that of rankings drawn uniformly at
random, and infinite when it is zero.

[cheng_decision_2009]_ choose the split that maximizes the weighted mean of the
spreads of the children, equation (6) of the paper:

.. math::

    \frac{N_{t_L} \hat{\theta}_{t_L} + N_{t_R} \hat{\theta}_{t_R}}{N_t}.

However, a child with a single sample has an infinite spread, so this measure would
be infinite for any split that isolates a sample. Instead, the ``"mallows"``
criterion gives each node :math:`t` the impurity

.. math::

    I_t = \frac{1}{1 + \hat{\theta}_t},

which, like the variance in the regression trees that [cheng_decision_2009]_ take as a
model, is smaller when the rankings are more concentrated around the center ranking. It
is zero when :math:`\hat{\theta}_t` is infinite, the case of rankings that agree with
the center ranking, and one when :math:`\hat{\theta}_t` is zero. The tree minimizes the
weighted mean of the impurities of the children, which does not always prefer the same
split as equation (6).

The other rule of [cheng_decision_2009]_ stops the growth when "the number of
labels in a node" is below :math:`2|\mathcal{Y}| = 2n`, where :math:`\mathcal{Y}`
is the set of the :math:`n` labels, which admits several readings. It is not
applied by default: as in scikit-learn, a node is split if it has at least
``min_samples_split`` samples, two by default. Reading it as a node with fewer than
:math:`2n` samples, as [alfaro_learning_2021]_ do, it amounts to setting
``min_samples_split`` to :math:`2n`, which is what the reproduction of the
experiments of the paper does.

Incomplete rankings
~~~~~~~~~~~~~~~~~~~

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

Convergence
~~~~~~~~~~~

Algorithm 1 of [cheng_decision_2009]_ only iterates when some ranking of a node is
incomplete, and ``max_iter`` bounds the number of iterations for each node, ``100``
by default. If the center ranking of some node still changes after ``max_iter``
iterations or comes back to an earlier center, which the algorithm would repeat
forever, :meth:`~DecisionTreeLabelRanker.fit` issues a single
:class:`~sklearn.exceptions.ConvergenceWarning` and keeps the center reached.

Complexity
~~~~~~~~~~

For a node with :math:`N_t` samples and :math:`n` labels, the tree keeps the weight
of the rankings that put each label ahead of each other one in each child, and
updates it as the samples move between the children, as the criteria of
scikit-learn do with their sums, which takes :math:`O(n^2)` time per sample. With
complete rankings, the Borda scores and the distances to the center follow from
those weights, so the impurity of each candidate split also takes :math:`O(n^2)`
time, apart from solving for the spread, and the ``"best"`` splitter takes
:math:`O(N_t n^2)` time per feature and node.

With incomplete rankings, the closest extension of each ranking depends on the
center ranking, so the impurity of each child is computed again for each candidate
split. Equal rankings have equal extensions, so the tree estimates the model from
the distinct rankings of each child, each weighted by the total weight of its
samples, which it also updates as the samples move between the children. With
:math:`R_t` distinct rankings in a node, the estimation of the spread takes
:math:`O(R_t n^2)` time, and each iteration of Algorithm 1 at most
:math:`O(R_t n^2)` more. The ``"best"`` splitter evaluates up to :math:`N_t - 1`
thresholds for each feature, which takes :math:`O(N_t R_t n^2)` time per feature and
node, while the ``"random"`` splitter evaluates one, which takes
:math:`O(N_t n^2 + R_t n^2)`. :math:`R_t` is at most :math:`N_t`, and much smaller
with few labels, since there are few possible rankings.

.. _partial_label_ranking_tree:

Partial label ranking
---------------------

:class:`DecisionTreePartialLabelRanker` implements the partial label ranking trees of
[alfaro_learning_2021]_. The training rankings may have ties and be incomplete, and
each leaf predicts a bucket order, an ordered partition of the labels into buckets of
tied labels:

>>> from sklr.tree import DecisionTreePartialLabelRanker
>>> X = [[0], [1], [2], [3], [4], [5]]
>>> y = [
...     [1, 2, 2, np.nan],
...     [1, 1, 2, 3],
...     [2, 1, 1, 3],
...     [np.nan, 1, 1, 2],
...     [3, 2, 1, np.nan],
...     [3, 2, 1, 1],
... ]
>>> ranker = DecisionTreePartialLabelRanker(min_samples_leaf=2, random_state=0)
>>> ranker.fit(X, y).predict([[0.4], [2.6], [4.6]])
array([[1, 1, 2, 3],
       [2, 1, 1, 3],
       [3, 2, 1, 1]])

[alfaro_learning_2021]_ keep the rule of the label ranking trees that stops the
growth on nodes with fewer than :math:`2n` samples, which amounts to setting
``min_samples_split`` to :math:`2n`, as explained in :ref:`label_ranking_tree`.

Impurity
~~~~~~~~

The impurities of [alfaro_learning_2021]_ measure the disagreement of the rankings of
a node on each pair of labels. For a node :math:`t` and a pair of labels :math:`y_i`
and :math:`y_j` with :math:`i < j`, let :math:`p_{ij}`, :math:`e_{ij}` and
:math:`p_{ji}` be the weights of the rankings of the node that put :math:`y_i` ahead of
:math:`y_j`, tie them and put :math:`y_j` ahead of :math:`y_i`, and
:math:`s_{ij} = p_{ij} + e_{ij} + p_{ji}`, the weight of those that rank both labels.
The ``criterion`` parameter chooses among four impurities, where the sums run over
the pairs with :math:`s_{ij} > 0`:

* ``"disagreements"``, equation (5) of the paper, adds up the fraction of the
  rankings that do not order each pair as the consensus bucket order
  :math:`\hat\pi_t` of the node, described below:

  .. math::

      I_t = \sum_{i < j} \frac{\rho_{ij}}{s_{ij}},

  where :math:`\rho_{ij}` is :math:`e_{ij} + p_{ji}`, :math:`p_{ij} + p_{ji}` or
  :math:`p_{ij} + e_{ij}` when :math:`\hat\pi_t` puts :math:`y_i` ahead of
  :math:`y_j`, ties them or puts :math:`y_j` ahead of :math:`y_i`.
* ``"distance"``, equation (6), counts twice the rankings that reverse the pair with
  respect to :math:`\hat\pi_t`, with :math:`\gamma_{ij}` equal to
  :math:`e_{ij} + 2 p_{ji}`, :math:`p_{ij} + p_{ji}` or :math:`2 p_{ij} + e_{ij}` in the
  same cases:

  .. math::

      I_t = \sum_{i < j} \frac{\gamma_{ij}}{s_{ij}}.

  It adds up, over the pairs, the mean distance between the bucket matrices of the
  rankings that rank the pair and that of :math:`\hat\pi_t`. When the rankings are
  complete and neither they nor :math:`\hat\pi_t` have ties, it is twice their mean
  Kendall distance to :math:`\hat\pi_t`.
* ``"entropy"``, equations (7) and (8), the default, averages the entropy in bits of
  the relations of each pair, with :math:`0 \log_2 0 = 0`:

  .. math::

      I_t = \frac{2}{n (n - 1)} \sum_{i < j}
      H\left(\frac{p_{ij}}{s_{ij}}, \frac{e_{ij}}{s_{ij}}, \frac{p_{ji}}{s_{ij}}\right),
      \qquad H(q_1, q_2, q_3) = -\sum_{r=1}^{3} q_r \log_2 q_r.

* ``"gini"``, equations (9) and (10), averages their Gini index:

  .. math::

      I_t = \frac{2}{n (n - 1)} \sum_{i < j}
      G\left(\frac{p_{ij}}{s_{ij}}, \frac{e_{ij}}{s_{ij}}, \frac{p_{ji}}{s_{ij}}\right),
      \qquad G(q_1, q_2, q_3) = 1 - \sum_{r=1}^{3} q_r^2.

  Equation (9) of [alfaro_learning_2021]_ averages :math:`1 - G` instead, which grows
  as the rankings agree, unlike the other impurities, so it is read as an erratum.

The pairs that no ranking of the node ranks add nothing, and the factor of the
entropy and the Gini index stays :math:`2 / (n (n - 1))`, so the impurity of a node
does not grow because some pairs are missing. ``"entropy"`` and ``"gini"`` are zero
on consistent rankings by their definition, but ``"disagreements"`` and
``"distance"`` would be positive on incomplete consistent rankings that no bucket
order agrees with, such as the cycle above. The impurity of a consistent node is
zero for every criterion, as explained in the section on tree induction.

In the following node, the rankings order the first two labels in three different
ways, the last two in two ways and the first and the last in a single way:

>>> X = [[0], [0], [1], [1]]
>>> y = [[1, 2, 2], [1, 1, 2], [2, 1, 3], [2, 1, 3]]
>>> for criterion in ["disagreements", "distance", "entropy", "gini"]:
...     ranker = DecisionTreePartialLabelRanker(criterion=criterion).fit(X, y)
...     print(criterion, ranker.tree_.impurity.round(3))
disagreements [1. 1. 0.]
distance [1. 1. 0.]
entropy [0.77  0.667 0.   ]
gini [0.333 0.333 0.   ]

Consensus bucket order
~~~~~~~~~~~~~~~~~~~~~~

The consensus bucket order :math:`\hat\pi_t` of a node is found as in
:ref:`nearest neighbors <partial_label_ranking_neighbors>`, by the Bucket Pivot
Algorithm :math:`\mathrm{LIA}_G^{\mathrm{MP2}}` of [aledo_utopia_2017]_ applied to the
pair order matrix of the rankings of the node, with their weights. It is the
``value`` of the node, and the prediction of a leaf. As there, only the labels that
some ranking of the node ranks together with another label are ordered, and the
others are tied in a last bucket, as explained in :ref:`labels_without_information`.
Unlike the label ranking tree, the bucket order of a node is not completed from that
of its parent, which [alfaro_learning_2021]_ do not describe, so that a leaf and
the nearest neighbors place those labels in the same way.

Complexity
~~~~~~~~~~

For a node with :math:`N_t` samples and :math:`n` labels, the tree keeps the weights
:math:`p_{ij}`, :math:`e_{ij}` and :math:`p_{ji}` of each pair of labels in each child,
and updates them as the samples move between the children, which takes :math:`O(n^2)`
time per sample. ``"entropy"`` and ``"gini"`` compute the impurity of each candidate
split from those weights in :math:`O(n^2)` time, so the ``"best"`` splitter takes
:math:`O(N_t n^2)` time per feature and node. ``"disagreements"`` and ``"distance"``
also need the consensus bucket order of each child for each candidate split, which
the Bucket Pivot Algorithm finds in :math:`O(n^2)` time too, as
[alfaro_learning_2021]_ point out, so the four criteria have the same complexity.
However, the Bucket Pivot Algorithm has a larger constant, so they are slower when
many children are not consistent, up to an order of magnitude with many labels and
rankings that disagree a lot, while the children whose rankings are consistent skip
it.

.. _tree_sample_weights:

Sample weights
--------------

The ``sample_weight`` parameter of :meth:`~DecisionTreeLabelRanker.fit` weighs the
rankings in the estimation of the impurities and of the consensus, and the numbers of
samples of the impurity decrease and of ``min_weight_fraction_leaf``. The weights must
be non-negative. The samples with zero weight do not reach any node, so they are left
out of the impurities, the consensus, the splits and the numbers of samples of the
nodes, but they still count in the number of training samples from which
``min_samples_split`` and ``min_samples_leaf`` are found when they are fractions.

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
Neither paper prunes the trees. As the impurities are bounded, the impurity decrease
and the cost complexity are always finite.
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
the trees of scikit-learn, so they cannot draw the trees of scikit-lr.

Practical use
-------------

The splits only depend on the order of the values of each feature, so, unlike the
nearest neighbors, the trees do not need the features to be scaled. The parameters
that limit the size of the tree are chosen by cross-validation, as shown in
:ref:`getting_started`, with the ``"kendall_tau"`` scorer for label ranking and the
``"tau_x"`` scorer for partial label ranking, described in :ref:`model_evaluation`.
Limiting the size of the tree, the number of features with ``max_features``, or using
the ``"random"`` splitter reduce the time of :meth:`~DecisionTreeLabelRanker.fit`, and
prediction only follows the path of each sample to its leaf, as in scikit-learn. The
examples :ref:`sphx_glr_auto_examples_tree_plot_label_ranking_tree.py` and
:ref:`sphx_glr_auto_examples_tree_plot_partial_label_ranking_tree.py` show how the
maximum depth controls the fit, also with incomplete training rankings, and the
second one compares the criteria of partial label ranking.
