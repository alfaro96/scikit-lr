"""Parameters of the Plackett-Luce model of possibly incomplete rankings.

The arguments are not checked: the rankings must be valid and without ties, the
weights non-negative and not all zero, and the other arguments within their range.
"""

import numpy as np

from sklr._distributions import _plackett_luce
from sklr._distributions._convergence import warn_if_not_converged


def estimate_parameters(y, sample_weight=None, *, tol=1e-6, max_iter=1000):
    """Estimate the parameters of the Plackett-Luce model from label rankings.

    They are the maximum likelihood estimate found by the MM algorithm of
    [hunter_mm_2004], which [cheng_label_2010] use for label ranking, and add up to
    one. The algorithm starts from equal parameters and stops when none changes by
    more than `tol`, or after `max_iter` iterations. The labels never ranked above
    another one get a zero parameter, and all the labels get the same one if no
    ranking has two labels. Without a maximum of the likelihood, as when all the
    rankings agree, the parameters converge slowly to a limit where some are zero.
    """
    if sample_weight is None:
        sample_weight = np.ones(len(y))
    return estimate_parameters_batch(
        np.asarray(y)[np.newaxis],
        np.asarray(sample_weight)[np.newaxis],
        tol=tol,
        max_iter=max_iter,
    )[0]


def estimate_parameters_batch(
    y, sample_weight, *, tol=1e-6, max_iter=1000, n_threads=1
):
    """Estimate the parameters of the Plackett-Luce model of each group of rankings.

    `y` has shape ``(n_groups, n_samples, n_labels)`` and `sample_weight` shape
    ``(n_groups, n_samples)``. The groups are split among `n_threads` threads.
    """
    y = np.ascontiguousarray(y, dtype=np.float64)
    sample_weight = np.ascontiguousarray(sample_weight, dtype=np.float64)
    parameters, _, converged = _plackett_luce.estimate_parameters_batch(
        y, sample_weight, tol, max_iter, max(1, min(n_threads, len(y)))
    )
    warn_if_not_converged(
        converged,
        "parameter vector of the Plackett-Luce model",
        f"some parameter still changed by more than tol={tol} after "
        f"max_iter={max_iter} iterations",
    )
    return parameters
