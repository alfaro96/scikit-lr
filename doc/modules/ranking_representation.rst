.. _ranking_representation:

======================
Ranking representation
======================

.. currentmodule:: sklr.utils

In label ranking, the target of each sample is a ranking of a fixed set of labels,
from the most preferred one to the least preferred one. scikit-lr represents the
rankings of ``n_samples`` samples with ``n_labels`` labels as a two-dimensional
array ``y`` of shape ``(n_samples, n_labels)``, with at least two labels, which
plays the role of the target ``y`` of the scikit-learn estimators.

Positions
=========

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
====

In partial label ranking, the rankings may have ties, which means that there is
no preference between some labels. Tied labels share their position, and the
positions are dense: the labels that follow a group of tied labels have the next
position, without gaps. For instance, ``[1, 1, 2]`` ranks the first two labels
first, tied, and the third label last, while ``[1, 1, 3]`` is not a valid ranking.

Incomplete rankings
===================

A ranking is incomplete when the position of some labels is not known. Their
position is ``np.nan``, the only value that marks a label without position, and
the positions of the other labels are dense from ``1``. For instance,
``[2, np.nan, 1]`` ranks the third label before the first one and says nothing
about the second one, and a row in which one label or none is ranked is a valid
incomplete ranking too, although it gives no preference between labels.

Type of the rankings
====================

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
