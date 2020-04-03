"""Metrics to assess the performance on Partial Label Ranking task given
ranking prediction.

Functions named as ``*_score`` return a scalar value to maximize: the higher,
the better.

Functions named as ``*_distance`` return a scalar value to minimize: the lower,
the better.
"""


# =============================================================================
# Imports
# =============================================================================

# Third party
import numpy as np

# Local application
from ._partial_label_ranking_fast import tau_x_score_fast
from ..utils.ranking import check_partial_label_ranking_targets
from ..utils.validation import check_array, check_consistent_length


# =============================================================================
# Methods
# =============================================================================

def _check_targets(Y_true, Y_pred):
    """Check that ``Y_true`` and ``Y_pred`` belong to a Partial Label
    Ranking task.

    Checks ``Y_true`` and ``Y_pred`` for consistent length, enforces to be
    integer 2-D arrays and are checked to be non-empty and containing only
    finite values.
    """
    Y_true = check_array(Y_true, dtype=np.int64)
    Y_pred = check_array(Y_pred, dtype=np.int64)

    check_consistent_length(Y_true, Y_pred)

    check_partial_label_ranking_targets(Y_true)
    check_partial_label_ranking_targets(Y_pred)

    return (Y_true, Y_pred)


def tau_x_score(Y_true, Y_pred, sample_weight=None):
    """Kendall tau extension.

    The Kendall tau extension is a variation of the Kendall tau
    to handle ties, which gives a score of 1 to tied classes.

    Parameters
    ----------
    Y_true : ndarray of shape (n_samples, n_classes), dtype=np.int64
        The ground truth of (correct) rankings.

    Y_pred : ndarray of shape (n_samples, n_classes), dtype=np.int64
        The predicted rankings, as return by a Partial Label Ranker.

    sample_weight : ndarray of shape (n_samples,), dtype=np.float64, \
            default=None
        The sample weights. If ``None``, then samples are equally weighted.

    Returns
    -------
    score : float
        The Kendall tau extension.

    See also
    --------
    tau_score : Kendall tau.

    References
    ----------
    .. [1] `E. J. Emond and D. W. Mason, "A new rank correlation coefficient
            with application to the consensus ranking problem", Journal of
            Multi-Criteria Decision Analysis, vol. 11, pp. 17-28, 2002.`_

    Examples
    --------
    >>> import numpy as np
    >>> from sklr.metrics import tau_x_score
    >>> Y_true = np.array([[1, 2, 2], [2, 2, 1], [2, 1, 2]])
    >>> Y_pred = np.array([[2, 1, 2], [2, 2, 1], [1, 2, 2]])
    >>> tau_x_score(Y_true, Y_pred)
    0.11111111111111115
    """
    (Y_true, Y_pred) = _check_targets(Y_true, Y_pred)

    scores = np.zeros(Y_true.shape[0], dtype=np.float64)
    tau_x_score_fast(Y_true, Y_pred, scores)

    return np.average(a=scores, weights=sample_weight)
