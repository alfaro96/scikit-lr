"""Utilities for rankings."""


# =============================================================================
# Imports
# =============================================================================

# Third party
import numpy as np

# Local application
from .validation import check_array
from .._types import RANK_TYPE
from ._ranking import is_label_ranking, is_partial_label_ranking


# =============================================================================
# Methods
# =============================================================================

def num_buckets(Y):
    """Find the mean number of buckets."""
    return np.mean([np.unique(y).shape[0] for y in Y])


def unique_rankings(Y):
    """Find the number of unique rankings."""
    return np.unique(Y, axis=0)


def check_label_ranking_targets(Y):
    """Ensure that target ``Y`` is of a Label Ranking type.

    Parameters
    ----------
    Y : ndarray of shape (n_samples, n_classes), dtype=np.int64 or \
            dtype=np.float64
        The input rankings.
    """
    Y_type = type_of_target(Y)

    if Y_type not in {"label_ranking"}:
        raise ValueError("Unknown label type: {0}.".format(Y_type))


def check_partial_label_ranking_targets(Y):
    """Ensure that target ``Y`` is of a Partial Label Ranking type.

    Parameters
    ----------
    Y : ndarray of shape (n_samples, n_classes), dtype=np.int64 or \
            dtype=np.float64
        The input rankings.
    """
    Y_type = type_of_target(Y)

    if Y_type not in {"label_ranking", "partial_label_ranking"}:
        raise ValueError("Unknown label type: {0}.".format(Y_type))


def type_of_target(Y):
    """Determine the type of data indicated by the target.

    Parameters
    ----------
    Y : ndarray of shape (n_samples, n_classes), dtype=np.int64 or \
            dtype=np.float64
        The input rankings.

    Returns
    -------
    target_types : {"label_ranking", "partial_label_ranking", "unknown"}
        The target type.

    Examples
    --------
    >>> import numpy as np
    >>> from sklr.utils import type_of_target
    >>> type_of_target(np.array([[1, 2, 3], [3, 2, 1]]))
    'label_ranking'
    >>> type_of_target(np.array([[1, 2, 2], [2, 2, 1]]))
    'partial_label_ranking'
    >>> type_of_target(np.array([[1, 2, 4], [2, 2, 1]]))
    'unknown'
    """
    # Transform the input rankings to the integer representation
    # since it is the format used by the underlying fast methods
    Y = _transform_rankings(check_array(Y, force_all_finite=False))

    if is_label_ranking(Y):
        return "label_ranking"
    elif is_partial_label_ranking(Y):
        return "partial_label_ranking"
    else:
        return "unknown"


def _transform_rankings(Y):
    """Transform the rankings to integer."""
    Yt = np.zeros(Y.shape, dtype=np.int64)

    Yt[np.isfinite(Y)] = Y[np.isfinite(Y)]
    Yt[np.isnan(Y)] = RANK_TYPE.RANDOM.value
    Yt[np.isinf(Y)] = RANK_TYPE.TOP.value

    return Yt
