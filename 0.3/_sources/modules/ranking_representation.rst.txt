.. _ranking_representation:

Ranking representation
======================

.. currentmodule:: sklr.utils

In label ranking, the target of each sample is a ranking of a fixed set of labels,
from the most preferred one to the least preferred one. scikit-lr represents the
rankings of ``n_samples`` samples with ``n_labels`` labels as a two-dimensional
array ``y`` of shape ``(n_samples, n_labels)``, with at least two labels, which
plays the role of the target ``y`` of the scikit-learn estimators.

Positions
---------

Each row of ``y`` is a ranking and each column a label: ``y[i, j]`` is the position
of label ``j`` in ranking ``i``, from ``1`` for the most preferred label. A row
is therefore not the list of the labels in order of preference, but the position
of each label. For instance, the following rankings rank the first label first,
the third label second and the second label last, and then the second label first,
the third label second and the first label last:

>>> import numpy as np
>>> y = np.array([[1, 3, 2], [3, 1, 2]])

The positions are integers, which may be stored with an integer or a floating
point dtype.

Ties
----

In partial label ranking, the rankings may have ties, which means that there is
no preference between some labels. Tied labels share their position, and the
positions are dense: the labels that follow a group of tied labels have the next
position, without gaps. For instance, ``[1, 1, 2]`` ranks the first two labels
first, tied, and the third label last, while ``[1, 1, 3]`` is not a valid ranking.

Incomplete rankings
-------------------

A ranking is incomplete when the position of some labels is not known. Their
position is ``np.nan``, the only value that marks a label without position, and
the positions of the other labels are dense from ``1``. For instance,
``[2, np.nan, 1]`` ranks the third label before the first one and says nothing
about the second one, and a row in which one label or none is ranked is a valid
incomplete ranking too, although it gives no preference between labels. The
estimators of scikit-lr learn from incomplete rankings, unless their documentation
says otherwise, while the :ref:`metrics <model_evaluation>` only compare complete
rankings.

.. _ranking_notation:

Notation
--------

The user guide and the documentation of the estimators and the metrics use the standard
notation of (partial) label ranking, common to both problems. An instance
:math:`\boldsymbol{x}` of the instance space :math:`\mathcal{X}` has the features
:math:`x_j`, and :math:`\boldsymbol{x}_i` is the :math:`i`-th of the :math:`N` training
samples. The labels are :math:`\mathcal{Y} = \{y_1, \dots, y_n\}`, with
:math:`n = |\mathcal{Y}|`, and a ranking :math:`\pi` of them gives the position
:math:`\pi(i)` of each label :math:`y_i`. In label ranking, the rankings have no ties,
so they are permutations of :math:`\{1, \dots, n\}`, while in partial label ranking they
may have ties, and tied labels share their dense position. The observed rankings, those
of the training samples and the true rankings of the test samples, are written
:math:`\sigma`, and :math:`\sigma_i` is the ranking of :math:`\boldsymbol{x}_i`, while
:math:`\pi` is a generic ranking or the center of a probability model, and the
predictions and the estimates carry a hat, such as :math:`\hat{\pi}`. Therefore,
counting rows and columns from one, row :math:`i` of the training rankings ``y`` is
:math:`\sigma_i`, and its column :math:`j` holds :math:`\sigma_i(j)`. An incomplete
ranking :math:`\sigma` only ranks the labels
:math:`\mathcal{Y}(\sigma) \subseteq \mathcal{Y}`, which are
:math:`m = |\mathcal{Y}(\sigma)| \leq n`.

Type of the rankings
--------------------

:func:`type_of_ranking` determines whether ``y`` holds label rankings, without
ties, partial label rankings, with ties in some ranking, or does not encode
rankings at all:

>>> from sklr.utils import type_of_ranking
>>> type_of_ranking(y)
'label_ranking'
>>> type_of_ranking([[1, 1, 2], [3, 1, 2]])
'partial_label_ranking'
>>> type_of_ranking([[2, np.nan, 1], [3, 1, 2]])
'label_ranking'
>>> type_of_ranking([[1, 1, 3], [3, 1, 2]])
'unknown'

Validation of the rankings
--------------------------

:func:`check_ranking` validates ``y`` and returns it as a ``float64`` array, the
representation used by the estimators. It raises an error that shows the first
invalid ranking if ``y`` does not encode rankings, and by default it also rejects
rankings with ties and incomplete rankings, which are accepted with
``allow_ties=True`` and ``allow_incomplete=True``:

>>> from sklr.utils import check_ranking
>>> check_ranking([[1, 1, 2], [3, 1, 2]], allow_ties=True)
array([[1., 1., 2.],
       [3., 1., 2.]])
>>> check_ranking([[1, 1, 2], [3, 1, 2]])
Traceback (most recent call last):
    ...
ValueError: Expected rankings without ties in y, got y[0] = [1 1 2].
>>> check_ranking([[2, np.nan, 1], [3, 1, 2]])
Traceback (most recent call last):
    ...
ValueError: Expected complete rankings in y, without unranked labels (NaN), got
y[0] = [ 2. nan  1.].
