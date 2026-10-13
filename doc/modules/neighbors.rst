.. _neighbors:

Nearest neighbors
=================

.. currentmodule:: sklr.neighbors

:class:`KNeighborsLabelRanker` implements the instance-based label rankers of
[cheng_decision_2009]_ and [cheng_label_2010]_. They assume that the probability
distribution of the rankings is approximately constant around each sample, so the
rankings of the :math:`k` training samples nearest to it are taken as a sample of a
probability model of the rankings. The parameters of the model are estimated from
these rankings, which may be incomplete, and the predicted ranking is the mode of
the model, its most probable ranking:

>>> import numpy as np
>>> from sklr.neighbors import KNeighborsLabelRanker
>>> X = [[0], [1], [2], [3], [4], [5]]
>>> y = [[1, 2, 3], [2, 1, 3], [1, 3, 2], [np.nan, 1, 2], [2, np.nan, 1], [3, 2, 1]]
>>> ranker = KNeighborsLabelRanker(n_neighbors=3).fit(X, y)
>>> ranker.predict([[0.4], [2.6], [4.2]])
array([[1, 2, 3],
       [1, 3, 2],
       [3, 2, 1]])

The rankings follow the representation described in :ref:`ranking_representation`:
each row holds the position of each label, and ``np.nan`` marks the labels that a
training ranking does not rank. The training rankings may be incomplete, even
without any ranked label, but they must not have ties. The predicted rankings are
always complete and without ties.

Neighbor search
---------------

The nearest neighbors are found by :class:`~sklearn.neighbors.NearestNeighbors` of
scikit-learn, which receives the parameters ``n_neighbors``, ``algorithm``,
``leaf_size``, ``metric``, ``p``, ``metric_params`` and ``n_jobs``. Therefore, all
the metrics and search algorithms of scikit-learn are available, including
precomputed distances with ``metric="precomputed"``, and so are the
:meth:`~KNeighborsLabelRanker.kneighbors` and
:meth:`~KNeighborsLabelRanker.kneighbors_graph` methods. The
:ref:`nearest neighbors <sklearn:neighbors>` section of the scikit-learn user guide
describes the search algorithms and how to choose among them.

As with the other nearest neighbors estimators, :meth:`~KNeighborsLabelRanker.fit`
only stores the training data and builds the search structure, and the work is
done by :meth:`~KNeighborsLabelRanker.predict`. Given ``None`` instead of the test
samples, :meth:`~KNeighborsLabelRanker.predict` predicts the ranking of each
training sample from its neighbors, without the sample itself. If the
:math:`k`-th and the :math:`(k + 1)`-th nearest neighbors of a sample are at the
same distance but have different rankings, the prediction depends on the order of
the training data.

Probability models
------------------

The ``model`` parameter chooses the probability model of the rankings of the
neighbors, ``"mallows"`` for the Mallows model and ``"plackett_luce"`` for the
Plackett-Luce model. In both cases, ties between labels in the predicted ranking go
to the label with the smaller index.

Mallows model
~~~~~~~~~~~~~

The Mallows model is a distance-based model whose mode is a center ranking
:math:`\pi`: the probability of a ranking decreases with its distance to
:math:`\pi`, at a rate given by a spread parameter :math:`\theta \geq 0`
[cheng_label_2010]_. With ``model="mallows"``, the predicted ranking is the
estimate of :math:`\pi`, found as in Algorithm 1 of [cheng_decision_2009]_.

When the rankings of the neighbors are complete, :math:`\pi` is found by the
weighted Borda count [ho_decision_1994]_. Each ranking of :math:`n` labels gives
:math:`n` votes to its first label, :math:`n - 1` to the second one and so on, each
vote is multiplied by the weight of the neighbor, and the labels are ranked by
decreasing total votes. When some rankings are incomplete, the initial center is
found by the generalized Borda count of Proposition 2 of [cheng_decision_2009]_: in
a ranking of :math:`m \leq n` of the :math:`n` labels, the label at position
:math:`i` receives :math:`(m - i + 1)(n + 1) / (m + 1)` votes and each unranked
label receives :math:`(n + 1) / 2`, its expected votes when the ranking is
completed uniformly at random. Then, each incomplete ranking is completed with its
closest extension to the center in Kendall distance, found as in Proposition 1 of
[cheng_decision_2009]_, and the center is found again by the Borda count of the
completed rankings. These two steps are repeated until the center no longer
changes.

[cheng_decision_2009]_ also estimate :math:`\theta` as a measure of the reliability
of the prediction, which :class:`KNeighborsLabelRanker` does not compute.

Plackett-Luce model
~~~~~~~~~~~~~~~~~~~

The Plackett-Luce model has a positive parameter :math:`v_j` for each label
:math:`j`. If a ranking ranks :math:`m` labels and :math:`\pi(i)` denotes the label
at position :math:`i`, its probability is

.. math::

    P(\pi \mid v) = \prod_{i=1}^{m} \frac{v_{\pi(i)}}{\sum_{l=i}^{m} v_{\pi(l)}},

which has the same form for complete and incomplete rankings [cheng_label_2010]_.
With ``model="plackett_luce"``, the parameters are the maximum likelihood estimate
from the rankings of the neighbors, which maximizes
:math:`\sum_i w_i \log P(\sigma_i \mid v)` for the rankings :math:`\sigma_i` and
the weights :math:`w_i` of the neighbors, the log-likelihood of
[cheng_label_2010]_ when the weights are uniform. It is found by the MM algorithm
of [hunter_mm_2004]_, which starts from equal parameters, scales them to add up to
one after each iteration and stops when no parameter changes by more than ``tol``.
The predicted ranking, the mode of the model, sorts the labels by decreasing
parameter [cheng_label_2010]_:

>>> ranker = KNeighborsLabelRanker(n_neighbors=3, model="plackett_luce").fit(X, y)
>>> ranker.predict([[0.4], [2.6], [4.2]])
array([[1, 2, 3],
       [1, 3, 2],
       [3, 1, 2]])

A label that no neighbor with a positive weight ranks above another label gets a
zero parameter, so it goes after the other labels, and all the labels get the same
parameter, which predicts them in the order of their indices, if no such neighbor
ranks two labels.

Weights
-------

The ``weights`` parameter weighs the rankings of the neighbors of each sample in
the estimation of the model:

* ``"uniform"``, the default, gives all the neighbors the same weight.
* ``"distance"`` weighs each neighbor by the inverse of its distance to the sample,
  as the nearest neighbors estimators of scikit-learn do. If some neighbors are at
  zero distance, only they are weighted.
* ``"dudani"`` follows equation (1) of [dudani_distance_1976]_, which
  [cheng_decision_2009]_ use. If :math:`d_{(1)} \leq \dots \leq d_{(k)}` are the
  distances of the neighbors to the sample, the :math:`i`-th neighbor weighs

  .. math::

      w_i = \frac{d_{(k)} - d_{(i)}}{d_{(k)} - d_{(1)}},

  from ``1`` for the nearest neighbor to ``0`` for the farthest one, or ``1`` if
  :math:`d_{(k)} = d_{(1)}`.
* A callable receives the array of the distances of the neighbors of the samples
  and returns an array of the same shape with non-negative weights.

>>> ranker = KNeighborsLabelRanker(n_neighbors=3, weights="distance").fit(X, y)
>>> ranker.predict([[0.4], [2.6], [4.2]])
array([[1, 2, 3],
       [1, 2, 3],
       [3, 2, 1]])

With ``model="mallows"``, the weight of each neighbor is also multiplied by the
fraction :math:`|Y(\sigma_i)| / n` of the :math:`n` labels that its ranking
:math:`\sigma_i` ranks, as in [cheng_decision_2009]_: a ranking completed by the
estimation is less reliable than an observed one. The Plackett-Luce model does not
complete the rankings, so its weights are not multiplied by that fraction.

The weights of all the neighbors of a sample may be zero, for instance with
``weights="dudani"`` when the neighbors that are nearer than the farthest one rank
no label. Then, with ``model="mallows"``, the neighbors are weighted by the
fraction of the labels that they rank alone, and if it is also zero, because no
neighbor ranks any label, or with ``model="plackett_luce"``, they are weighted
equally.

Convergence
-----------

The estimation of both models is iterative, and ``max_iter`` bounds the number of
iterations for each sample. By default, it is ``100`` with ``model="mallows"`` and
``1000`` with ``model="plackett_luce"``. If the estimation of some sample does not
converge, :meth:`~KNeighborsLabelRanker.predict` issues a single
:class:`~sklearn.exceptions.ConvergenceWarning` and returns the rankings reached.

With ``model="mallows"``, the estimation only iterates when some neighbor has an
incomplete ranking, and it does not converge when the center still changes after
``max_iter`` iterations or comes back to an earlier center, which the algorithm
would repeat forever.

With ``model="plackett_luce"``, the maximum likelihood estimate exists, and the MM
algorithm converges to it, if, for every split of the labels into two groups, some
label of the second group is ranked above some label of the first one
[hunter_mm_2004]_. Otherwise, as when the rankings of the neighbors agree or a label
is never ranked above another one, the parameters slowly approach a limit where
some of them are zero, and the warning is common when the neighbors are few. The
predicted ranking then sorts the labels by the parameters reached after
``max_iter`` iterations. A larger ``tol`` lets the estimation stop earlier, and a
larger ``max_iter`` lets it get closer to that limit.

Practical use
-------------

[cheng_decision_2009]_ and [cheng_label_2010]_ use the Euclidean distance after
normalizing the attributes, and choose the number of neighbors by cross-validation
on the training data. Normalizing the features keeps those with the largest range
from deciding the neighbors alone, and it is done with a
:class:`~sklearn.pipeline.Pipeline` that scales the features before the ranker.
The number of neighbors, and the model or the weights, are chosen with
:class:`~sklearn.model_selection.GridSearchCV` and the scorer of Kendall's
:math:`\tau` from :func:`sklr.metrics.get_scorer`, as described in
:ref:`model_evaluation`:

>>> from scipy.stats import rankdata
>>> from sklearn.model_selection import GridSearchCV
>>> from sklearn.pipeline import make_pipeline
>>> from sklearn.preprocessing import StandardScaler
>>> from sklr.metrics import get_scorer
>>> rng = np.random.RandomState(0)
>>> X = rng.normal(size=(100, 3)) * [1, 10, 100]
>>> y = rankdata(-(X / [1, 10, 100]) @ rng.normal(size=(3, 4)), axis=1)
>>> pipeline = make_pipeline(StandardScaler(), KNeighborsLabelRanker())
>>> param_grid = {"kneighborslabelranker__n_neighbors": [5, 10, 15, 20]}
>>> search = GridSearchCV(pipeline, param_grid, scoring=get_scorer("kendall_tau"))
>>> search.fit(X, y).best_params_
{'kneighborslabelranker__n_neighbors': 5}
>>> round(float(search.best_score_), 2)
0.85

Without the scaler, the third feature, whose range is the largest, decides the
neighbors, and the score drops:

>>> from sklearn.model_selection import cross_val_score
>>> scores = cross_val_score(
...     KNeighborsLabelRanker(), X, y, scoring=get_scorer("kendall_tau")
... )
>>> round(float(scores.mean()), 2)
0.33

The scorers only compare complete rankings, so cross-validation needs complete
rankings in the validation folds. The example
:ref:`sphx_glr_auto_examples_neighbors_plot_label_ranking_neighbors.py`
compares the models and the weights for several numbers of neighbors, also with
incomplete training rankings.

Complexity
----------

The cost of :meth:`~KNeighborsLabelRanker.fit` and of the neighbor search is that
of :class:`~sklearn.neighbors.NearestNeighbors`. Then, for each sample with
:math:`k` neighbors and :math:`n` labels, the Borda count takes :math:`O(kn)` time
plus a sort of the labels, each iteration of the Mallows model with incomplete
rankings takes at most :math:`O(kn^2)` time to complete them, and each iteration of
the MM algorithm takes :math:`O(kn)` time. The samples are estimated in parallel
by ``n_jobs`` threads, which also run the neighbor search.

.. rubric:: References

.. [cheng_decision_2009] W. Cheng, J. Hühn and E. Hüllermeier, "Decision tree and
   instance-based learning for label ranking", in Proceedings of the 26th
   International Conference on Machine Learning, Montreal, Canada, pp. 161-168,
   2009.

.. [cheng_label_2010] W. Cheng, K. Dembczyński and E. Hüllermeier, "Label ranking
   methods based on the Plackett-Luce model", in Proceedings of the 27th
   International Conference on Machine Learning, Haifa, Israel, pp. 215-222, 2010.

.. [dudani_distance_1976] S. A. Dudani, "The distance-weighted k-nearest-neighbor
   rule", IEEE Transactions on Systems, Man, and Cybernetics, vol. SMC-6, no. 4,
   pp. 325-327, 1976.

.. [ho_decision_1994] T. K. Ho, J. J. Hull and S. N. Srihari, "Decision combination
   in multiple classifier systems", IEEE Transactions on Pattern Analysis and
   Machine Intelligence, vol. 16, no. 1, pp. 66-75, 1994.

.. [hunter_mm_2004] D. R. Hunter, "MM algorithms for generalized Bradley-Terry
   models", The Annals of Statistics, vol. 32, no. 1, pp. 384-406, 2004.
