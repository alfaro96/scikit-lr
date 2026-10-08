"""Warning about the estimates that did not converge."""

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning


def warn_if_not_converged(converged, estimate, reason):
    """Warn about the groups of rankings whose estimate did not converge.

    The message says that `estimate` did not converge for those groups, because
    `reason`, and that the last one is returned.
    """
    n_not_converged = np.sum(~converged)
    if n_not_converged:
        warnings.warn(
            f"The {estimate} did not converge for {n_not_converged} of "
            f"{len(converged)} groups of rankings: {reason}. The last {estimate} is "
            "returned.",
            ConvergenceWarning,
        )
