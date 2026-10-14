.. _model_evaluation:

Metrics and scoring: evaluating the rankers
===========================================

.. currentmodule:: sklr.metrics

:mod:`sklr.metrics` implements functions that compare the true rankings of some
samples with the rankings that an estimator predicts for them, and the scorers that
bring these functions to the model selection tools of scikit-learn. Each function
computes a correlation or a distance between the true and the predicted ranking of
each sample, and returns their average, which may be weighted with
``sample_weight``.

Each metric is written as a function of two rankings, such as
:math:`\tau(\hat{\pi}, \sigma)` for the predicted ranking :math:`\hat{\pi}` and the
true ranking :math:`\sigma` of a sample, with the notation of :ref:`ranking_notation`.
The rankings follow the representation described in :ref:`ranking_representation`,
and they must be complete: the metrics reject rankings with unranked labels, in
the true rankings as well as in the predicted ones.

.. _scoring:

Scoring
-------

The ``score`` method of the label rankers computes Kendall's :math:`\tau`, and
that of the partial label rankers computes the :math:`\tau_x` coefficient, both
described below. The model selection tools of scikit-learn use it unless they are
given a scorer with the :ref:`scoring <sklearn:scoring_parameter>` parameter.

scikit-learn only accepts the names of its own scorers as values of ``scoring``,
so the scorers of scikit-lr are passed as objects, which :func:`get_scorer` gets
from their names:

=============================  ===============================
Name                           Function
=============================  ===============================
``"kendall_tau"``              :func:`kendall_tau_score`
``"tau_x"``                    :func:`tau_x_score`
``"neg_kendall_distance"``     :func:`kendall_distance`
=============================  ===============================

As in scikit-learn, greater values of a scorer are better, so the scorer of a
distance returns it negated. :func:`get_scorer_names` lists the names:

>>> from sklr.metrics import get_scorer, get_scorer_names
>>> get_scorer_names()
['kendall_tau', 'neg_kendall_distance', 'tau_x']
>>> get_scorer("neg_kendall_distance")
make_scorer(kendall_distance, greater_is_better=False, response_method='predict')

The scorer is then passed as ``scoring``, for instance with
``cross_val_score(ranker, X, y, scoring=get_scorer("kendall_tau"))``. When metadata
routing is enabled, the scorers receive ``sample_weight`` if it is requested with
their ``set_score_request`` method.

.. _label_ranking_metrics:

Label ranking metrics
---------------------

The metrics of label ranking compare rankings without ties, and they reject the
rankings with tied labels.

.. _kendall_tau:

Kendall's :math:`\tau`
~~~~~~~~~~~~~~~~~~~~~~

:func:`kendall_tau_score` computes Kendall's :math:`\tau` [kendall_new_1938]_,
which scores each pair of labels with ``+1`` if the true and the predicted ranking
order it in the same way (a concordant pair) and with ``-1`` otherwise (a
discordant pair). Let :math:`a_{ij}(\pi)` be ``1`` if :math:`\pi(i) < \pi(j)` and
``-1`` if :math:`\pi(i) > \pi(j)`. Then

.. math::

    \tau(\hat{\pi}, \sigma) = \frac{2}{n(n - 1)} \sum_{i < j}
    a_{ij}(\hat{\pi}) \, a_{ij}(\sigma),

which ranges from ``-1``, when one ranking is the reverse of the other, to ``1``,
when both rankings are equal:

>>> from sklr.metrics import kendall_tau_score
>>> y_true = [[1, 2, 3], [3, 2, 1]]
>>> y_pred = [[1, 2, 3], [1, 3, 2]]
>>> kendall_tau_score(y_true, y_pred)
0.333...

Without ties, all the variants of Kendall's :math:`\tau`, such as :math:`\tau_a`
and :math:`\tau_b`, are the same. They differ in how they handle ties, which is
why the metrics of partial label ranking use the :math:`\tau_x` coefficient
instead, described in :ref:`partial_label_ranking_metrics`.

.. _kendall_distance:

Kendall distance
~~~~~~~~~~~~~~~~

:func:`kendall_distance` computes the Kendall distance, the number of discordant
pairs of labels,

.. math::

    D(\hat{\pi}, \sigma) = |\{(i, j) : i < j, \,
    (\hat{\pi}(i) - \hat{\pi}(j)) (\sigma(i) - \sigma(j)) < 0\}|,

which is also the minimum number of swaps of adjacent labels that turn one ranking
into the other. By default, it is divided by the number of pairs,
:math:`n(n - 1) / 2`, so that it ranges from ``0`` to ``1`` and is equal to
:math:`(1 - \tau(\hat{\pi}, \sigma)) / 2`:

>>> from sklr.metrics import kendall_distance
>>> kendall_distance(y_true, y_pred)
0.333...
>>> kendall_distance(y_true, y_pred, normalize=False)
1.0

.. _partial_label_ranking_metrics:

Partial label ranking metrics
-----------------------------

The metrics of partial label ranking compare rankings that may have ties.

.. _tau_x:

:math:`\tau_x`
~~~~~~~~~~~~~~

:func:`tau_x_score` computes the :math:`\tau_x` rank correlation coefficient
[emond_new_2002]_, which extends Kendall's :math:`\tau` to rankings with ties.
Let :math:`a'_{ij}(\pi)` be ``1`` if :math:`\pi(i) \leq \pi(j)`, that is, if label
:math:`y_i` is ahead of or tied with label :math:`y_j`, and ``-1`` if
:math:`\pi(i) > \pi(j)`, for :math:`i \neq j`. Then

.. math::

    \tau_x(\hat{\pi}, \sigma) = \frac{1}{n(n - 1)} \sum_{i \neq j}
    a'_{ij}(\hat{\pi}) \, a'_{ij}(\sigma).

Unlike Kendall's :math:`\tau_b`, its denominator does not depend on the ties, so
a ranking with all its labels tied has a coefficient of ``1`` with itself and of
``0`` with any ranking without ties. For rankings without ties, it is equal to
Kendall's :math:`\tau`:

>>> from sklr.metrics import tau_x_score
>>> tau_x_score([[1, 1, 1]], [[1, 1, 1]])
1.0
>>> tau_x_score([[1, 1, 1]], [[1, 2, 3]])
0.0
>>> tau_x_score(y_true, y_pred)
0.333...

The :math:`\tau_x` coefficient is also equivalent to the Kemeny-Snell distance
:math:`D_{\mathrm{KS}}` between rankings with ties, by

.. math::

    \tau_x(\hat{\pi}, \sigma)
    = 1 - \frac{2 D_{\mathrm{KS}}(\hat{\pi}, \sigma)}{n(n - 1)}.

.. rubric:: References

.. [kendall_new_1938] M. G. Kendall, "A new measure of rank correlation",
   Biometrika, vol. 30, no. 1/2, pp. 81-93, 1938.

.. [emond_new_2002] E. J. Emond and D. W. Mason, "A new rank correlation
   coefficient with application to the consensus ranking problem", Journal of
   Multi-Criteria Decision Analysis, vol. 11, pp. 17-28, 2002.
