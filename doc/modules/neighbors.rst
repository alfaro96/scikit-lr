.. _neighbors:

Nearest neighbors
=================

.. currentmodule:: sklr.neighbors

The nearest neighbors rankers predict for each instance :math:`\boldsymbol{x}` a
ranking that aggregates the rankings :math:`\sigma_1, \dots, \sigma_k` of its
:math:`k` nearest training samples :math:`\boldsymbol{x}_1, \dots, \boldsymbol{x}_k`,
numbered from the nearest one, with the notation of :ref:`ranking_notation`. There is
one for each problem:

* :class:`KNeighborsLabelRanker`, for label ranking, estimates a probability model
  of the rankings from those of the neighbors, which have no ties, and predicts its
  most probable ranking, as described in :ref:`label_ranking_neighbors`.
* :class:`KNeighborsPartialLabelRanker`, for partial label ranking, aggregates the
  rankings of the neighbors, which may have ties, into a consensus bucket order, as
  described in :ref:`partial_label_ranking_neighbors`.

Both find and weigh the neighbors as the next section describes. The rankings follow
the representation described in :ref:`ranking_representation`; the training rankings
of both estimators may be incomplete, and their predictions are always complete.

.. _neighbor_search:

.. _neighbors_weights:

Neighbor search and weights
---------------------------

The neighbors are found by :class:`~sklearn.neighbors.NearestNeighbors`, which
receives the parameters ``n_neighbors``, ``algorithm``, ``leaf_size``, ``metric``,
``p``, ``metric_params`` and ``n_jobs``, so the search algorithms, the metrics,
``metric="precomputed"`` and the ``kneighbors`` and ``kneighbors_graph`` methods are
those described in the :ref:`nearest neighbors <sklearn:neighbors>` section of the
scikit-learn user guide. The ``n_jobs`` threads also aggregate the rankings of the
neighbors of the samples in parallel.

The ``weights`` parameter weighs the rankings of the neighbors of each sample in
their aggregation. ``"uniform"``, ``"distance"`` and a callable weigh them as in
:ref:`nearest neighbors classification <sklearn:classification>`. ``"dudani"``
follows equation (1) of [dudani_distance_1976]_, which [cheng_decision_2009]_ use:
if :math:`d_{(1)} \leq \dots \leq d_{(k)}` are the sorted distances
:math:`d(\boldsymbol{x}, \boldsymbol{x}_i)` of the neighbors :math:`\boldsymbol{x}_i`
to the instance :math:`\boldsymbol{x}`, the :math:`i`-th nearest neighbor weighs

.. math::

    w_i = \frac{d_{(k)} - d_{(i)}}{d_{(k)} - d_{(1)}},

from ``1`` for the nearest neighbor to ``0`` for the farthest one, or ``1`` if
:math:`d_{(k)} = d_{(1)}`.

A callable may give a zero weight to all the neighbors of a sample. Then, the
neighbors are weighted equally, except with the Mallows model of
:class:`KNeighborsLabelRanker`, which first tries the weights described in
:ref:`label_ranking_neighbors_weights`.

.. _label_ranking_neighbors:

Label ranking
-------------

:class:`KNeighborsLabelRanker` implements the instance-based label rankers of
[cheng_decision_2009]_ and [cheng_label_2010]_. They assume that the probability
distribution of the rankings is approximately constant around each instance
:math:`\boldsymbol{x}`, so the rankings of its :math:`k` nearest neighbors are taken
as a sample of a probability model of the rankings. The parameters of the model are
estimated from these rankings, and the predicted ranking is the mode of the model, its
most probable ranking:

>>> import numpy as np
>>> from sklr.neighbors import KNeighborsLabelRanker
>>> X = [[0], [1], [2], [3], [4], [5]]
>>> y = [[1, 2, 3], [2, 1, 3], [1, 3, 2], [np.nan, 1, 2], [2, np.nan, 1], [3, 2, 1]]
>>> ranker = KNeighborsLabelRanker(n_neighbors=3).fit(X, y)
>>> ranker.predict([[0.4], [2.6], [4.2]])
array([[1, 2, 3],
       [1, 3, 2],
       [3, 2, 1]])

The training rankings may be incomplete, even without any ranked label, but they
must not have ties, and the predicted rankings have no ties either. The ``model``
parameter chooses the probability model of the rankings of the neighbors,
``"mallows"`` for the Mallows model and ``"plackett_luce"`` for the Plackett-Luce
model. In both cases, ties between labels in the predicted ranking go to the label
with the smaller index. The neighbors are found and weighted as described in
:ref:`neighbor_search`.

.. _mallows_neighbors:

Mallows model
~~~~~~~~~~~~~

The Mallows model is a distance-based model whose mode is a center ranking
:math:`\pi`: the probability of a ranking decreases with its distance to
:math:`\pi`, at a rate given by a spread parameter :math:`\theta \geq 0`
[cheng_label_2010]_. With ``model="mallows"``, the predicted ranking is the
estimate :math:`\hat{\pi}` of :math:`\pi`, found as in Algorithm 1 of
[cheng_decision_2009]_.

When the rankings of the neighbors are complete, :math:`\hat{\pi}` is found by the
weighted Borda count [ho_decision_1994]_. Each ranking :math:`\sigma_i` gives
:math:`n - \sigma_i(j) + 1` votes to each label :math:`y_j`, that is, :math:`n` votes to
its first label, :math:`n - 1` to the second one and so on, each vote is multiplied by
the weight :math:`w_i` of the neighbor, and the labels are ranked by decreasing total
votes. When some rankings are incomplete, the initial center is found by the generalized
Borda count of Proposition 2 of [cheng_decision_2009]_: if :math:`\sigma_i` ranks
:math:`m = |\mathcal{Y}(\sigma_i)| \leq n` labels, each ranked label :math:`y_j`
receives :math:`(m - \sigma_i(j) + 1)(n + 1) / (m + 1)` votes and each unranked label
receives :math:`(n + 1) / 2`, its expected votes when the ranking is completed uniformly
at random. Then, each incomplete ranking is completed with its closest extension to the
center in Kendall distance, found as in Proposition 1 of [cheng_decision_2009]_, and the
center is found again by the Borda count of the completed rankings. These two steps are
repeated until the center no longer changes.

[cheng_decision_2009]_ also estimate :math:`\theta` as a measure of the reliability
of the prediction, which :class:`KNeighborsLabelRanker` does not compute.

Plackett-Luce model
~~~~~~~~~~~~~~~~~~~

The Plackett-Luce model has a positive parameter :math:`v_j` for each label
:math:`y_j`, which form the vector :math:`\boldsymbol{v} = (v_1, \dots, v_n)`. If a
ranking :math:`\pi` ranks :math:`m = |\mathcal{Y}(\pi)|` labels and
:math:`\pi^{-1}(i)` denotes the index of the label at position :math:`i`, its
probability is

.. math::

    P(\pi \mid \boldsymbol{v}) = \prod_{i=1}^{m}
    \frac{v_{\pi^{-1}(i)}}{\sum_{l=i}^{m} v_{\pi^{-1}(l)}},

which has the same form for complete and incomplete rankings [cheng_label_2010]_.
With ``model="plackett_luce"``, the parameters are the maximum likelihood estimate
from the rankings of the neighbors, which maximizes
:math:`\sum_{i=1}^{k} w_i \log P(\sigma_i \mid \boldsymbol{v})` for the rankings
:math:`\sigma_i` and the weights :math:`w_i` of the neighbors, the log-likelihood of
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

.. _label_ranking_neighbors_weights:

Weights
~~~~~~~

Weighting the neighbors changes the prediction of the second sample of the first
example:

>>> ranker = KNeighborsLabelRanker(n_neighbors=3, weights="distance").fit(X, y)
>>> ranker.predict([[0.4], [2.6], [4.2]])
array([[1, 2, 3],
       [1, 2, 3],
       [3, 2, 1]])

With ``model="mallows"``, the weight of each neighbor is also multiplied by the
fraction :math:`|\mathcal{Y}(\sigma_i)| / n` of the :math:`n` labels that its
ranking :math:`\sigma_i` ranks, where :math:`\mathcal{Y}(\sigma_i)` is the set of
those labels, as in [cheng_decision_2009]_: a ranking completed by the estimation is
less reliable than an observed one. If these products are zero for all the neighbors
of a sample, as with ``weights="dudani"`` when only the farthest neighbor ranks some
label, the neighbors are weighted by that fraction alone, and equally if it is also
zero, because no neighbor ranks any label. The Plackett-Luce model does not complete
the rankings, so its weights are not multiplied by that fraction.

Convergence
~~~~~~~~~~~

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

Complexity
~~~~~~~~~~

After the neighbor search, for each sample with :math:`k` neighbors and :math:`n`
labels, the Borda count takes :math:`O(kn)` time plus a sort of the labels, each
iteration of the Mallows model with incomplete rankings takes at most
:math:`O(kn^2)` time to complete them, and each iteration of the MM algorithm takes
:math:`O(kn)` time.

.. _partial_label_ranking_neighbors:

Partial label ranking
---------------------

:class:`KNeighborsPartialLabelRanker` implements the instance-based partial label
ranker of [alfaro_learning_2021]_. The training rankings may have ties, and the
prediction for an instance :math:`\boldsymbol{x}` is a bucket order, an ordered
partition of the labels into buckets of tied labels, that aggregates the rankings of
its :math:`k` nearest neighbors:

>>> import numpy as np
>>> from sklr.neighbors import KNeighborsPartialLabelRanker
>>> X = [[0], [1], [2], [3], [4], [5]]
>>> y = [
...     [1, 2, 2, np.nan],
...     [1, 1, 2, np.nan],
...     [1, 2, 3, np.nan],
...     [np.nan, 1, 1, 2],
...     [2, np.nan, 1, 1],
...     [3, 2, 1, np.nan],
... ]
>>> ranker = KNeighborsPartialLabelRanker(n_neighbors=3).fit(X, y)
>>> ranker.predict([[0.4], [4.2]])
array([[1, 2, 3, 4],
       [3, 1, 1, 2]])

No neighbor of the first sample ranks the fourth label, so it goes alone to a last
bucket, as explained in :ref:`labels_without_information`. The neighbors are found
and weighted as described in :ref:`neighbor_search`.

Consensus bucket order
~~~~~~~~~~~~~~~~~~~~~~

Here, the neighbors are indexed by :math:`l`, since :math:`i` and :math:`j` index the
labels. For each pair of labels :math:`y_i` and :math:`y_j`, the entry
:math:`C(y_i, y_j)` of the pair order matrix of the neighbors is the weighted
fraction of their rankings ranking both labels that put :math:`y_i` ahead of
:math:`y_j`, with a ranking that ties them counting one half, as in equation (3) of
[alfaro_learning_2021]_ with the weights :math:`w_l` of the neighbors:

.. math::

    C(y_i, y_j) = \frac{\sum_{l} w_l \, [\sigma_l(i) < \sigma_l(j)]
    + \frac{1}{2} \sum_{l} w_l \, [\sigma_l(i) = \sigma_l(j)]}{\sum_{l} w_l},

where the sums run over the neighbors whose rankings rank both labels, a condition
in brackets is ``1`` when it holds and ``0`` otherwise, and
:math:`C(y_i, y_j) = 1/2` if no neighbor of positive weight ranks both labels.
Thus, :math:`C(y_i, y_j) + C(y_j, y_i) = 1`. The weights are not multiplied by the
fraction of the labels that each ranking ranks, since the consensus does not
complete the rankings. A bucket order :math:`\pi` has the bucket matrix :math:`B`
with :math:`B(y_i, y_j)` equal to ``1``, ``1/2`` or ``0`` when :math:`\pi` puts
:math:`y_i` ahead of :math:`y_j`, ties them or puts :math:`y_j` ahead, and the
optimal bucket order problem looks for the bucket order whose bucket matrix
minimizes :math:`\sum_{i, j} |B(y_i, y_j) - C(y_i, y_j)|`, as in equation (1) of
[aledo_utopia_2017]_.

The predicted bucket order :math:`\hat\pi` is an approximate solution found by the
Bucket Pivot Algorithm. It chooses a pivot label :math:`y_p` and puts each other
label :math:`y_j` in a group of labels ahead of the pivot if
:math:`C(y_p, y_j) < 1/2 - \beta`, in the bucket of the pivot if
:math:`1/2 - \beta \leq C(y_p, y_j) \leq 1/2 + \beta`, and in a group of labels
behind the pivot otherwise, with the threshold :math:`\beta = 0.25`. Then, it
applies itself to the labels ahead and to the labels behind, and the bucket order is
the one of the labels ahead, followed by the bucket of the pivot and by the one of
the labels behind. :class:`KNeighborsPartialLabelRanker` uses the variant
:math:`\mathrm{LIA}_G^{\mathrm{MP2}}` of [aledo_utopia_2017]_, which
[alfaro_learning_2021]_ chose for its trade-off between accuracy and efficiency:

* With the least indecision assumption, the pivot is the label with the least DANI
  (Distance to Achieve the Nearest Idealization) index. The utopian matrix rounds
  each entry :math:`C(y_i, y_j)` to ``1`` if it is above ``0.75``, to ``0`` if it is
  below ``0.25`` and to ``1/2`` otherwise, and the DANI index of :math:`y_i` is the
  mean absolute difference between :math:`C(y_i, y_j)` and its rounded value over
  the other labels :math:`y_j`, which is small when the neighbors agree on how
  :math:`y_i` compares with the rest. The rounding follows Example 1 of
  [aledo_utopia_2017]_, since its Definition 1 rounds :math:`C(y_j, y_i)` instead.
* The DANI indices are computed once, among all the labels being ordered, and not
  again for the labels ahead and behind (the subscript :math:`G`, for global).
* With multiple pivots, each label placed in the bucket of the pivot joins it as a
  pivot, and the next labels are compared with the mean of :math:`C(y_l, y_j)` over
  the pivots :math:`y_l` instead of with :math:`C(y_p, y_j)`. In a second stage, the
  labels ahead and behind are placed again with the mean over the final pivots.

[aledo_utopia_2017]_ choose at random among the labels with the least DANI index.
:class:`KNeighborsPartialLabelRanker` chooses the one with the smaller index, and
places the other labels in the order of their indices, which matters because the
pivots grow, so its predictions are deterministic.

.. _labels_without_information:

Labels without information
~~~~~~~~~~~~~~~~~~~~~~~~~~

A label that no neighbor of positive weight ranks together with another label has
all its entries of :math:`C` equal to ``1/2``, hence the least DANI index, and as
the first pivot it would draw all the labels into its bucket. Neither
[alfaro_learning_2021]_ nor [aledo_utopia_2017]_ consider this case. Therefore, only
the other labels are ordered by the algorithm, with their DANI indices computed
among them, and the labels without information are tied in a last bucket. If no
neighbor of positive weight ranks two labels, all the labels are tied.

Complexity
~~~~~~~~~~

After the neighbor search, for each sample with :math:`k` neighbors and :math:`n`
labels, the pair order matrix takes :math:`O(kn^2)` time, and the DANI indices and
the Bucket Pivot Algorithm take :math:`O(n^2)` time, since each label becomes a
pivot once and then updates the sums of at most :math:`n` labels.

.. _neighbors_practical_use:

Practical use
-------------

Both estimators are evaluated and tuned with the tools of scikit-learn, as shown in
:ref:`getting_started`, with the scorers described in :ref:`model_evaluation`:
``"kendall_tau"`` for :class:`KNeighborsLabelRanker` and ``"tau_x"`` for
:class:`KNeighborsPartialLabelRanker`. [cheng_decision_2009]_,
[cheng_label_2010]_ and [alfaro_learning_2021]_ use the Euclidean distance and
choose the number of neighbors by cross-validation on the training data. The first
two normalize the attributes, as :ref:`feature scaling <sklearn:preprocessing_scaler>`
does, and the last one weighs the neighbors by the inverse of their distance, which
is ``weights="distance"``.

The examples
:ref:`sphx_glr_auto_examples_neighbors_plot_label_ranking_neighbors.py` and
:ref:`sphx_glr_auto_examples_neighbors_plot_partial_label_ranking_neighbors.py`
compare the models and the weights of each estimator for several numbers of
neighbors, also with incomplete training rankings, and
:ref:`sphx_glr_auto_examples_reproducibility_plot_paper_reproduction.py`
compares the results of :class:`KNeighborsLabelRanker` with those published in
[cheng_decision_2009]_ and [cheng_label_2010]_.

.. rubric:: References

.. [aledo_utopia_2017] J. A. Aledo, J. A. Gámez and A. Rosete, "Utopia in the
   solution of the bucket order problem", Decision Support Systems, vol. 97,
   pp. 69-80, 2017.

.. [alfaro_learning_2021] J. C. Alfaro, J. A. Aledo and J. A. Gámez, "Learning
   decision trees for the partial label ranking problem", International Journal of
   Intelligent Systems, vol. 36, no. 2, pp. 890-918, 2021.

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
