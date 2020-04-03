"""Metrics to assess the performance on Label Ranking task given ranking
prediction.

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
from ..utils.ranking import check_label_ranking_targets
from ..utils.validation import check_array, check_consistent_length
from ._label_ranking_fast import kendall_distance_fast, tau_score_fast


# =============================================================================
# Methods
# =============================================================================

def _check_targets(Y_true, Y_pred):
    """Check that ``Y_true`` and ``Y_pred`` belong to a Label Ranking task.

    Checks ``Y_true`` and ``Y_pred`` for consistent length, enforces to be
    integer 2-D arrays and are checked to be non-empty and containing only
    finite values.
    """
    Y_true = check_array(Y_true, dtype=np.int64)
    Y_pred = check_array(Y_pred, dtype=np.int64)

    check_consistent_length(Y_true, Y_pred)

    check_label_ranking_targets(Y_true)
    check_label_ranking_targets(Y_pred)

    return (Y_true, Y_pred)


def kendall_distance(Y_true, Y_pred, normalize=True, sample_weight=None):
    """Kendall distance.

    The Kendall distance is a metric that counts the number of pairwise
    disagreements between two rankings. The larger the distance, the more
    dissimilar the two rankings are.

    Parameters
    ----------
    Y_true : ndarray of shape (n_samples, n_classes), dtype=np.int64
        The ground truth of (correct) rankings.

    Y_pred : ndarray of shape (n_samples, n_classes), dtype=np.int64
        The predicted rankings, as return by a Label Ranker.

    normalize : bool, default=True
        If ``False``, return the number of pairwise disagreements.
        Otherwise, return the fraction of pairwise disagreements.

    sample_weight : ndarray of shape (n_samples,), dtype=np.float64, \
            default=None
        The sample weights. If ``None``, then samples are equally weighted.

    Returns
    -------
    dist : float
         If ``normalize=True``, return the fraction of pairwise disagrements,
         else returns the number of pairwise disagreements.

    See also
    --------
    tau_score : Kendall tau.

    References
    ----------
    .. [1] `M. G. Kendall, "A new measure on Rank Correlation", Biometrika,
            vol. 30, pp. 81-89, 1938.`_

    Examples
    --------
    >>> import numpy as np
    >>> from sklr.metrics import kendall_distance
    >>> Y_true = np.array([[1, 2, 3], [3, 2, 1], [2, 1, 3]])
    >>> Y_pred = np.array([[2, 1, 3], [3, 2, 1], [1, 3, 2]])
    >>> kendall_distance(Y_true, Y_pred)
    0.3333333333333333
    >>> kendall_distance(Y_true, Y_pred, normalize=False)
    1.0
    """
    (Y_true, Y_pred) = _check_targets(Y_true, Y_pred)

    dists = np.zeros(Y_true.shape[0], dtype=np.float64)
    kendall_distance_fast(Y_true, Y_pred, normalize, dists)

    return np.average(a=dists, weights=sample_weight)


def tau_score(Y_true, Y_pred, sample_weight=None):
    """Kendall tau.

    The Kendall tau is a metric of the correspondence between two rankings.
    Values closer to 1 indicate strong agreement, and values closer to -1
    indicate strong disagreement.

    Parameters
    ----------
    Y_true : ndarray of shape (n_samples, n_classes), dtype=np.int64
        The ground truth of (correct) rankings.

    Y_pred : ndarray of shape (n_samples, n_classes), dtype=np.int64
        The predicted rankings, as return by a Label Ranker.

    sample_weight : ndarray of shape (n_samples,), dtype=np.float64, \
            default=None
        The sample weights. If ``None``, then samples are equally weighted.

    Returns
    -------
    score : float
        The Kendall tau.

    See also
    --------
    kendall_distance : Kendall distance.
    tau_x_score : Kendall tau extension.

    References
    ----------
    .. [1] `M. G. Kendall, "A new measure on Rank Correlation", Biometrika,
            vol. 30, pp. 81-89, 1938.`_

    Examples
    --------
    >>> import numpy as np
    >>> from sklr.metrics import tau_score
    >>> Y_true = np.array([[1, 2, 3], [3, 2, 1], [2, 1, 3]])
    >>> Y_pred = np.array([[2, 1, 3], [3, 2, 1], [1, 3, 2]])
    >>> tau_score(Y_true, Y_pred)
    0.3333333333333333
    """
    (Y_true, Y_pred) = _check_targets(Y_true, Y_pred)

    scores = np.zeros(Y_true.shape[0], dtype=np.float64)
    tau_score_fast(Y_true, Y_pred, scores)

    return np.average(a=scores, weights=sample_weight)
