"""Center ranking and spread of the Mallows model of possibly incomplete rankings."""

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from sklr._distributions import _mallows
from sklr.utils import check_ranking
from sklr.utils._sklearn_compat import _check_sample_weight


def _check_max_iter(max_iter):
    """Check that the maximum number of iterations is not negative."""
    if max_iter < 0:
        raise ValueError(
            f"Expected max_iter to be non-negative, got max_iter={max_iter}."
        )


def _check_n_threads(n_threads):
    """Check that there is at least one thread."""
    if n_threads < 1:
        raise ValueError(f"Expected at least 1 thread, got n_threads={n_threads}.")


def _warn_if_not_converged(converged, max_iter):
    """Warn about the groups of rankings whose center did not converge."""
    n_not_converged = np.sum(~converged)
    if n_not_converged:
        warnings.warn(
            f"The center ranking did not converge for {n_not_converged} of "
            f"{len(converged)} groups of rankings: it kept changing after "
            f"max_iter={max_iter} iterations or started to repeat itself. The last "
            "center is returned.",
            ConvergenceWarning,
        )


def estimate_center(y, sample_weight=None, *, max_iter=100):
    """Estimate the center ranking of the Mallows model from label rankings.

    The center is found by the Borda count, and that of incomplete rankings follows
    Algorithm 1 of [cheng_decision_2009], at most `max_iter` times. Ties between the
    votes go to the label with the smaller index, up to floating point rounding when
    the votes or the weights are not integers.
    """
    y = check_ranking(y, allow_incomplete=True)
    sample_weight = _check_sample_weight(sample_weight, y, ensure_non_negative=True)
    _check_max_iter(max_iter)
    center, _, converged = _mallows.estimate_center_batch(
        np.ascontiguousarray(y[np.newaxis]),
        np.ascontiguousarray(sample_weight[np.newaxis]),
        max_iter,
        1,
    )
    _warn_if_not_converged(converged, max_iter)
    return center[0]


def estimate_center_batch(y, sample_weight, *, max_iter=100, n_threads=1):
    """Estimate the center ranking of the Mallows model of each group of rankings.

    `y` has shape ``(n_groups, n_samples, n_labels)`` and `sample_weight` shape
    ``(n_groups, n_samples)``. Their values must be valid, and the rankings must not
    have ties, since they are not checked. The groups are split among `n_threads`
    threads.
    """
    _check_max_iter(max_iter)
    _check_n_threads(n_threads)
    y = np.ascontiguousarray(y, dtype=np.float64)
    sample_weight = np.ascontiguousarray(sample_weight, dtype=np.float64)
    center, _, converged = _mallows.estimate_center_batch(
        y, sample_weight, max_iter, max(1, min(n_threads, len(y)))
    )
    _warn_if_not_converged(converged, max_iter)
    return center


def estimate_spread(y, center, sample_weight=None):
    """Estimate the spread of the Mallows model from label rankings and their center.

    As in Algorithm 1 of [cheng_decision_2009], incomplete rankings are completed
    with their closest extensions to `center`, and the spread is the maximum
    likelihood estimate given the completed rankings: ``0`` when their weighted
    mean Kendall distance to `center` is not below that of the uniform distribution,
    and ``np.inf`` when it is zero.
    """
    y = check_ranking(y, allow_incomplete=True)
    sample_weight = _check_sample_weight(sample_weight, y, ensure_non_negative=True)
    center = np.asarray(center)
    n_labels = y.shape[1]
    if center.shape != (n_labels,):
        raise ValueError(
            f"Expected a center of shape ({n_labels},), got {center.shape} instead."
        )
    if not np.array_equal(np.sort(center), np.arange(1, n_labels + 1)):
        raise ValueError(
            "Expected a center with the positions from 1 to the number of labels, "
            f"without ties or unranked labels, got center={center}."
        )
    spread = _mallows.estimate_spread_batch(
        np.ascontiguousarray(y[np.newaxis]),
        np.ascontiguousarray(sample_weight[np.newaxis]),
        np.ascontiguousarray(center[np.newaxis], dtype=np.intp),
        1,
    )
    return spread[0]


def estimate_spread_batch(y, center, sample_weight, *, n_threads=1):
    """Estimate the spread of the Mallows model of each group given its center.

    `y` has shape ``(n_groups, n_samples, n_labels)``, `center` shape
    ``(n_groups, n_labels)`` and `sample_weight` shape ``(n_groups, n_samples)``.
    Their values must be valid, the rankings must not have ties and the weights of
    a group must not all be zero, since they are not checked. The groups are split
    among `n_threads` threads.
    """
    _check_n_threads(n_threads)
    y = np.ascontiguousarray(y, dtype=np.float64)
    center = np.ascontiguousarray(center, dtype=np.intp)
    sample_weight = np.ascontiguousarray(sample_weight, dtype=np.float64)
    return _mallows.estimate_spread_batch(
        y, sample_weight, center, max(1, min(n_threads, len(y)))
    )
