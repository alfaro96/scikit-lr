"""Center ranking and spread of the Mallows model of possibly incomplete rankings.

The arguments are not checked: the rankings must be valid and without ties, the
weights non-negative and not all zero, and the other arguments within their range.
"""

import numpy as np

from sklr._distributions import _mallows
from sklr._distributions._convergence import warn_if_not_converged


def estimate_center(y, sample_weight=None, *, max_iter=100):
    """Estimate the center ranking of the Mallows model from label rankings.

    The center is found by the Borda count, and that of incomplete rankings follows
    Algorithm 1 of [cheng_decision_2009], at most `max_iter` times. Ties between the
    votes go to the label with the smaller index, up to floating point rounding when
    the votes or the weights are not integers.
    """
    if sample_weight is None:
        sample_weight = np.ones(len(y))
    return estimate_center_batch(
        np.asarray(y)[np.newaxis],
        np.asarray(sample_weight)[np.newaxis],
        max_iter=max_iter,
    )[0]


def estimate_center_batch(y, sample_weight, *, max_iter=100, n_threads=1):
    """Estimate the center ranking of the Mallows model of each group of rankings.

    `y` has shape ``(n_groups, n_samples, n_labels)`` and `sample_weight` shape
    ``(n_groups, n_samples)``. The groups are split among `n_threads` threads.
    """
    y = np.ascontiguousarray(y, dtype=np.float64)
    sample_weight = np.ascontiguousarray(sample_weight, dtype=np.float64)
    center, _, converged = _mallows.estimate_center_batch(
        y, sample_weight, max_iter, max(1, min(n_threads, len(y)))
    )
    warn_if_not_converged(
        converged,
        "center ranking",
        f"it kept changing after max_iter={max_iter} iterations or started to "
        "repeat itself",
    )
    return center


def estimate_spread(y, center, sample_weight=None):
    """Estimate the spread of the Mallows model from label rankings and their center.

    As in Algorithm 1 of [cheng_decision_2009], incomplete rankings are completed
    with their closest extensions to `center`, and the spread is the maximum
    likelihood estimate given the completed rankings: ``0`` when their weighted
    mean Kendall distance to `center` is not below that of the uniform distribution,
    and ``np.inf`` when it is zero.
    """
    if sample_weight is None:
        sample_weight = np.ones(len(y))
    return estimate_spread_batch(
        np.asarray(y)[np.newaxis],
        np.asarray(center)[np.newaxis],
        np.asarray(sample_weight)[np.newaxis],
    )[0]


def estimate_spread_batch(y, center, sample_weight, *, n_threads=1):
    """Estimate the spread of the Mallows model of each group given its center.

    `y` has shape ``(n_groups, n_samples, n_labels)``, `center` shape
    ``(n_groups, n_labels)`` and `sample_weight` shape ``(n_groups, n_samples)``.
    The groups are split among `n_threads` threads.
    """
    y = np.ascontiguousarray(y, dtype=np.float64)
    center = np.ascontiguousarray(center, dtype=np.intp)
    sample_weight = np.ascontiguousarray(sample_weight, dtype=np.float64)
    return _mallows.estimate_spread_batch(
        y, sample_weight, center, max(1, min(n_threads, len(y)))
    )
