"""Metrics to assess the performance on Partial Label Ranking
task given scores.

Functions named as ``*_score`` return a scalar value to maximize: the higher
the better.

Functions named as ``*_distance`` return a scalar value to minimize: the lower
the better.
"""


# =============================================================================
# Imports
# =============================================================================

# Third party
import numpy as np

# Local application
from ._partial_label_ranking_fast import (
    penalized_kendall_distance_fast, tau_x_score_fast)
from ..utils.ranking import check_partial_label_ranking_targets
from ..utils.validation import check_array, check_consistent_length


# =============================================================================
# Methods
# =============================================================================

def _check_targets(Y_true, Y_pred):
    """Check that Y_true and Y_pred belong to the same Label Ranking task."""
    # Check the format of the input rankings,
    # ensuring that they are of an integer type
    Y_true = check_array(Y_true, dtype=np.int64)
    Y_pred = check_array(Y_pred, dtype=np.int64)

    # Check that the input rankings are
    # Partial Label Ranking targets
    check_partial_label_ranking_targets(Y_true)
    check_partial_label_ranking_targets(Y_pred)

    # Check the input rankings for consistent
    # length (i.e., same number of samples)
    check_consistent_length(Y_true, Y_pred)

    # Return the converted and validated rankings
    return (Y_true, Y_pred)


def penalized_kendall_distance(Y_true, Y_pred, normalize=True,
                               sample_weight=None, return_dists=False):
    """Compute penalized Kendall distance between rankings.

    Parameters
    ----------
    Y_true : np.ndarray of shape (n_samples, n_classes)
        Ground truth of (correct) rankings.

    Y_pred : np.ndarray of shape (n_samples, n_classes)
        Predicted rankings, as return by a Partial Label Ranker.

    normalize : bool, optional (default=True)
        If ``False``, return the penalized Kendall tau distance.
        Otherwise, return the normalized and penalized Kendall tau distance.

    sample_weight : {None, np.ndarray} of shape (n_samples,),
            optional (default=None)
        Sample weights.

    return_dists : bool, optional (default=False)
        If ``True``, also return the array of distances.
        Otherwise, just return the mean of the distances.

    Returns
    -------
    dist : float
        If ``normalize``, return the normalized and penalized
        Kendall tau distance between the rankings, else return the
        penalized Kendall tau distance.

        The best performance is 0.

    dists : np.ndarray of shape (n_samples,)
        If ``returns_dists and normalize``, return the array of normalized
        and penalized Kendall tau distances between the rankings,
        else return the array of penalized Kendall tau distances.

    See also
    --------
    tau_x_score, kendall_distance

    References
    ----------
    .. [1] `R. Fagin and R. Kumar and M. Mahdian and D. Sivakumar and E. Vee,
            "Comparing and aggregating rankings with ties", In Proceedings of
            the twenty-third ACM SIGMOD-SIGACT-SIGART symposium on Principles
            of database systems, 2004, pp. 47-58.`_

    Examples
    --------
    >>> import numpy as np
    >>> from sklr.metrics import penalized_kendall_distance
    >>> Y_true = np.array([[1, 2, 2], [2, 2, 1], [2, 1, 2]])
    >>> Y_pred = np.array([[2, 1, 2], [2, 2, 1], [1, 2, 2]])
    >>> penalized_kendall_distance(Y_true, Y_pred)
    0.4444444444444444
    >>> penalized_kendall_distance(Y_true, Y_pred, normalize=False)
    1.3333333333333333
    """
    # Check the input rankings
    (Y_true, Y_pred) = _check_targets(Y_true, Y_pred)

    # Initialize the penalized Kendall distances
    dists = np.zeros(Y_true.shape[0], dtype=np.float64)

    # Compute the penalized Kendall distances using
    # the fast version of the Cython extension module
    penalized_kendall_distance_fast(Y_true, Y_pred, normalize, dists)

    # Return the mean penalized Kendall distance according to the
    # sample weights (also return the array of distances if specified)
    if return_dists:
        return (np.average(a=dists, weights=sample_weight), dists)
    else:
        return np.average(a=dists, weights=sample_weight)


def tau_x_score(Y_true, Y_pred, sample_weight=None):
    """Compute Tau-x score between rankings.

    Parameters
    ----------
    Y_true : np.ndarray of shape (n_samples, n_classes)
        Ground truth of (correct) rankings.

    Y_pred : np.ndarray of shape (n_samples, n_classes)
        Predicted rankings, as return by a Label Ranker.

    sample_weight : {None, np.ndarray} of shape (n_samples,)
            optional (default=None)
        Sample weights.

    Returns
    -------
    score : float
        The Tau-x score.

        The best performance is 1.

    See also
    --------
    penalized_kendall_distance, tau_score

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
    # Check the input rankings
    (Y_true, Y_pred) = _check_targets(Y_true, Y_pred)

    # Initialize the Tau-x scores
    scores = np.zeros(Y_true.shape[0], dtype=np.float64)

    # Compute the Tau-x scores using the fast
    # version of the Cython extension module
    tau_x_score_fast(Y_true, Y_pred, scores)

    # Return the mean Tau-x score according to the sample weights
    return np.average(a=scores, weights=sample_weight)
