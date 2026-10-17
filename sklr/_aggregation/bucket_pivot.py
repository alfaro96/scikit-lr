"""Consensus bucket order of possibly incomplete partial rankings.

The arguments are not checked: the rankings must be valid, possibly with ties, the
weights non-negative, and the other arguments within their range.
"""

import numpy as np

from sklr._aggregation import _bucket_pivot


def aggregate(y, sample_weight=None):
    """Find a consensus bucket order of partial rankings.

    The labels that some ranking of positive weight ranks together with another
    label are ordered by the Bucket Pivot Algorithm with the least indecision
    assumption, global DANI indices and two-stage multiple pivots,
    :math:`\\mathrm{LIA}_G^{\\mathrm{MP2}}` in [aledo_utopia_2017], applied to their
    pair order matrix as in [alfaro_learning_2021], and the others are tied in a last
    bucket. Returns the positions of the labels in that bucket order, dense from one.
    """
    if sample_weight is None:
        sample_weight = np.ones(len(y))
    return aggregate_batch(
        np.asarray(y)[np.newaxis], np.asarray(sample_weight)[np.newaxis]
    )[0]


def aggregate_batch(y, sample_weight, *, n_threads=1):
    """Find a consensus bucket order of each group of partial rankings.

    `y` has shape ``(n_groups, n_samples, n_labels)`` and `sample_weight` shape
    ``(n_groups, n_samples)``. The groups are split among `n_threads` threads.
    Returns the positions of the labels for each group, of shape
    ``(n_groups, n_labels)``.
    """
    y = np.ascontiguousarray(y, dtype=np.float64)
    sample_weight = np.ascontiguousarray(sample_weight, dtype=np.float64)
    return _bucket_pivot.aggregate_batch(
        y, sample_weight, max(1, min(n_threads, len(y)))
    )
