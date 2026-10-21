.. _reduction:

Reduction to classification
===========================

.. currentmodule:: sklr.reduction

The rankers of :mod:`sklr.reduction` decompose label ranking into several
classification problems, solve each one with a classifier, and combine the
predictions of the classifiers into a ranking. The rankings follow the
representation described in :ref:`ranking_representation`, with the notation of
:ref:`ranking_notation`.

.. _labelwise_decomposition:

Labelwise decomposition
-----------------------

:class:`LabelwiseLabelRanker` implements the labelwise decomposition of
[cheng_labelwise_2013]_. The position of each label :math:`y_j` in the training
rankings is a class, from ``1`` to :math:`n`, so learning it is a classification
problem whose classes are ordered, an ordinal classification problem. The ranker fits
a probabilistic classifier to the positions of each label, which predicts the
probability :math:`P_j(r \mid \boldsymbol{x})` that :math:`y_j` is at position
:math:`r` for an instance :math:`\boldsymbol{x}`, and predicts the ranking that
minimizes the expected loss under these probabilities, as if the positions of the
labels were independent:

>>> import numpy as np
>>> from sklr.reduction import LabelwiseLabelRanker
>>> X = [[0], [1], [2], [3], [4], [5]]
>>> y = [[1, 2, 3], [2, 1, 3], [1, 3, 2], [np.nan, 1, 2], [2, np.nan, 1], [3, 2, 1]]
>>> ranker = LabelwiseLabelRanker().fit(X, y)
>>> ranker.predict([[0.4], [2.6], [4.2]])
array([[1, 2, 3],
       [3, 1, 2],
       [3, 2, 1]])

The training rankings may be incomplete, as described in
:ref:`labelwise_incomplete_rankings`, with the default estimator or any other of
:ref:`labelwise_other_estimators` that learns from them, but they must not have
ties, and the predicted rankings are complete and have no ties. The ``n_jobs``
parameter fits the classifiers of the labels in parallel.

Expected loss
~~~~~~~~~~~~~

The loss between the true ranking :math:`\sigma` and the predicted ranking
:math:`\hat\pi` is a sum over the labels of a loss
:math:`\ell(\sigma(j), \hat\pi(j))` between the true and the predicted position of
each label, chosen by the ``loss`` parameter:

* ``"absolute_error"``, the default, is :math:`\ell(r', r) = |r' - r|`, whose sum is
  an affine transformation of Spearman's footrule.
* ``"squared_error"`` is :math:`\ell(r', r) = (r' - r)^2`, whose sum is an affine
  transformation of Spearman's :math:`\rho`.

As the loss is a sum over the labels, its expected value is the sum of the expected
losses of the positions of the labels [cheng_labelwise_2013]_, where putting
:math:`y_j` at position :math:`r` costs

.. math::

    L_j(r) = \sum_{r'=1}^n \ell(r', r) P_j(r' \mid \boldsymbol{x}).

Choosing the best position of each label separately may put two labels at the same
position, so the predicted ranking is the permutation that minimizes the total cost,

.. math::

    \hat\pi = \arg\min_{\pi \in \Omega} \sum_{j=1}^n L_j(\pi(j)),

where :math:`\Omega` is the set of the rankings of the :math:`n` labels, the
solution of an assignment problem of the labels to the positions
[cheng_labelwise_2013]_. With ``loss="absolute_error"``, it is solved by
:func:`scipy.optimize.linear_sum_assignment`. With ``loss="squared_error"``, the
sum :math:`\sum_{j=1}^n \pi(j)^2` is the same for every permutation, so
:math:`\hat\pi` sorts the labels by increasing expected position
:math:`\sum_{r=1}^n r P_j(r \mid \boldsymbol{x})`, without solving the assignment
problem. The two losses may predict different rankings, as for the second sample of
the first example:

>>> ranker = LabelwiseLabelRanker(loss="squared_error").fit(X, y)
>>> ranker.predict([[0.4], [2.6], [4.2]])
array([[1, 2, 3],
       [2, 1, 3],
       [3, 2, 1]])

When several rankings have the minimum expected loss, the predicted one gives the
smallest position to the first label, then to the second one, and so on.
[cheng_labelwise_2013]_ note that the ranker cannot minimize the expected Kendall
distance, since the probabilities of the positions of each label do not determine
how likely each pair of labels is to be inverted. Kendall's :math:`\tau` is still
the default score of the ranker, as of every label ranker of scikit-lr, and the
one of the experiments of [cheng_labelwise_2013]_.

.. _labelwise_incomplete_rankings:

Incomplete rankings
~~~~~~~~~~~~~~~~~~~

An incomplete ranking :math:`\sigma` only ranks the :math:`m` labels of
:math:`\mathcal{Y}(\sigma)`, so the position of a ranked label :math:`y_j` in the
complete ranking is only known to be between :math:`\sigma(j)` and
:math:`\sigma(j) + n - m`. :class:`LabelwiseLabelRanker` assumes, as in Section 4.4
of [cheng_labelwise_2013]_, that the :math:`n - m` unranked labels were removed at
random from a complete ranking. If :math:`y_j` is at position :math:`r` of the
complete ranking, it is observed at position :math:`\sigma(j)` when
:math:`r - \sigma(j)` of the :math:`r - 1` labels ahead of it are removed, and the
other :math:`n - m - r + \sigma(j)` removed labels are among the :math:`n - r`
behind it. The binomial coefficients count these choices among the
:math:`\binom{n}{m}` choices of the ranked labels, so the probability of observing
:math:`y_j` at position :math:`\sigma(j)` is equation (16) of
[cheng_labelwise_2013]_,

.. math::

    \sum_{r=\sigma(j)}^{\sigma(j) + n - m} P_j(r \mid \boldsymbol{x})
    \frac{\binom{r - 1}{r - \sigma(j)}
    \binom{n - r}{n - m - r + \sigma(j)}}{\binom{n}{m}}.

The classifier of each label maximizes the log-likelihood of the rankings that rank
that label under this probability, equation (17) of [cheng_labelwise_2013]_, each
ranking weighted by its sample weight. With a complete ranking, :math:`m = n` and
the sum only has the term :math:`r = \sigma(j)`, with a binomial weight of ``1``, so
the likelihood is the usual one of a classifier. The rankings that do not rank the
label, or rank it alone, observe it with the same probability whatever the
classifier, so they are left out, as are those of zero weight.

A label that no ranking of positive weight ranks together with another label has
no classifier: its positions are predicted as equally likely, and
:meth:`~LabelwiseLabelRanker.fit` warns about it with a :class:`UserWarning`. Its
expected loss is then lowest in the middle positions, so such a label is not
necessarily predicted last.

[cheng_labelwise_2013]_ also consider rankings that keep the top labels of the
complete ranking, equations (18) and (19), which the representation of the rankings
does not tell apart from those with labels removed at random, and which
:class:`LabelwiseLabelRanker` does not model.

Default estimator
~~~~~~~~~~~~~~~~~

With ``estimator=None``, the classifier of each label is the cumulative logit model
of equation (20) of [cheng_labelwise_2013]_, an ordinal logistic regression. The
probability that the position of :math:`y_j` is at most :math:`r`, for
:math:`r = 1, \dots, n - 1`, is

.. math::

    \sum_{s=1}^r P_j(s \mid \boldsymbol{x}) =
    \frac{1}{1 + \exp(-\beta_r - \boldsymbol{w}^\top \boldsymbol{x})},

with coefficients :math:`\boldsymbol{w}` and thresholds
:math:`\beta_1 \leq \dots \leq \beta_{n - 1}` of its own, and the probability of
each position is the difference of two consecutive cumulative probabilities. It
learns from incomplete rankings by maximizing the likelihood of the previous
section directly, as described in :ref:`labelwise_other_estimators`.
[cheng_labelwise_2013]_ do not say how they fit it: here, the coefficients have the
:math:`L_2` penalty of :class:`~sklearn.linear_model.LogisticRegression` with
``C=1.0``, the thresholds are not penalized, and the likelihood is maximized by
L-BFGS-B, with a :class:`~sklearn.exceptions.ConvergenceWarning` from
:meth:`~LabelwiseLabelRanker.fit` if it does not converge within ``100``
iterations. Because of the penalty, the scale of the features matters, and they
should be standardized as for :class:`~sklearn.linear_model.LogisticRegression`, as
described in :ref:`sklearn_integration`.

.. _labelwise_other_estimators:

Other estimators
~~~~~~~~~~~~~~~~

The ``estimator`` parameter takes any classifier with :term:`fit` and
:term:`predict_proba` methods, which is cloned for each label. To learn from
incomplete rankings, the classifier also has to maximize the likelihood of
:ref:`labelwise_incomplete_rankings` with a method of this signature, as the
default estimator does:

.. code-block:: python

    def fit_likelihood(self, X, likelihood, classes, sample_weight=None):
        ...
        return self

``likelihood[i, k]`` is the probability of what is observed of sample ``i`` if its
class is ``classes[k]``, and the method maximizes the sum over the samples, weighted
by ``sample_weight``, of the logarithm of ``likelihood[i] @ proba[i]``, where
``proba[i]`` holds the probabilities of the classes of sample ``i`` that
:term:`predict_proba` predicts afterwards, with ``classes_`` set to ``classes``.
:class:`LabelwiseLabelRanker` calls it for each label with the positions from ``1``
to :math:`n` as ``classes``, and with the binomial weights of
:ref:`labelwise_incomplete_rankings` as ``likelihood``, which are one for the
observed position and zero for the others in a complete ranking. With complete
rankings only, the method maximizes the usual likelihood of a classifier.

A classifier without ``fit_likelihood`` is fitted with ``fit`` and only learns from
complete rankings, so :meth:`~LabelwiseLabelRanker.fit` raises an error with
incomplete ones:

>>> from sklearn.linear_model import LogisticRegression
>>> y = [[1, 2, 3], [2, 1, 3], [1, 3, 2], [3, 1, 2], [2, 3, 1], [3, 2, 1]]
>>> ranker = LabelwiseLabelRanker(LogisticRegression()).fit(X, y)
>>> ranker.predict([[0.4], [2.6], [4.2]])
array([[1, 2, 3],
       [3, 1, 2],
       [3, 2, 1]])

The classifier sees the positions as classes, and most classifiers ignore their
order, as :class:`~sklearn.linear_model.LogisticRegression` does. A position that a
label never takes in the training rankings is not a class of its classifier, so it
gets a zero probability, and a label that always takes the same position leaves its
classifier with a single class, which some classifiers, such as
:class:`~sklearn.linear_model.LogisticRegression`, reject. If ``sample_weight`` is
passed to :meth:`~LabelwiseLabelRanker.fit`, the classifier must accept it in the
method that fits it.

Complexity
~~~~~~~~~~

The ranker fits :math:`n` classifiers on at most :math:`N` samples each. With the
default estimator and :math:`d` features, each iteration of L-BFGS-B takes
:math:`O(N(d + n))` time for each label. For each predicted sample,
``loss="squared_error"`` takes :math:`O(n^2)` time after the classifiers predict the
probabilities, plus a sort of the labels. ``loss="absolute_error"`` takes
:math:`O(n^3)` time to compute the costs and solve the assignment problem, and
choosing the ranking of the smallest positions among those of minimum cost solves at
most :math:`O(n^2)` smaller assignment problems, which takes :math:`O(n^5)` time in
the worst case.

The example :ref:`sphx_glr_auto_examples_reduction_plot_labelwise_decomposition.py`
compares the losses and the estimators of :class:`LabelwiseLabelRanker`, also with
incomplete training rankings.

.. rubric:: References

.. [cheng_labelwise_2013] W. Cheng, S. Henzgen and E. Hüllermeier, "Labelwise
   versus pairwise decomposition in label ranking", in Proceedings of the Workshop
   on Lernen, Wissen & Adaptivität, pp. 129-136, 2013.
