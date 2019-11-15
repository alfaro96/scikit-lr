"""Utilities for rankings."""


# =============================================================================
# Imports
# =============================================================================

# Third party
import numpy as np

# Local application
from ._ranking_fast import RANK_METHODS
from ._ranking_fast import (
    are_label_ranking_targets, are_partial_label_ranking_targets,
    is_ranking_without_ties_fast, is_ranking_with_ties_fast, rank_data_view)
from .validation import check_array


# =============================================================================
# Constants
# =============================================================================

# Type of rankings
_RANK_TYPES = {
    "random": 2147483646,
}


# =============================================================================
# Methods
# =============================================================================

def unique_rankings(Y):
    """Find the unique rankings of an array.

    Returns the sorted unique rankings of an array.

    Note that np.nan values are equally treated since
    that means that the class have been randomly missed.

    Parameters
    ----------
    Y : np.ndarray of shape (n_samples, n_classes)
        Input rankings.

    Returns
    -------
    Y_unique : np.ndarray of shape (n_unique_samples, n_classes)
        The sorted unique rankings.

    Raises
    ------
    ValueError
        If the input rankings are not for neither the
        Label Ranking problem nor the Partial Label problem.

    Examples
    --------
    >>> import numpy as np
    >>> from sklr.utils import unique_rankings
    >>> Y = np.array([[1, 2, 3], [2, 3, 1], [1, 2, 3]])
    >>> unique_rankings(Y)
    array([[1, 2, 3],
           [2, 3, 1]])
    """
    # Obtain the target types of the input rankings to
    # ensure that they are either for the Label Ranking
    # problem or the Partial Label Ranking problem
    target_types = type_of_targets(Y)

    # If the target types of the rankings are
    # unknown, raise the corresponding error
    if "unknown" in target_types:
        raise ValueError("Unknown label type: {}."
                         .format(sorted(list(target_types))))

    # Lexically sort the rankings to find the
    # unique ones in an efficient manner
    Y_sorted = Y[np.lexsort(np.transpose(Y)[::-1])]

    # Obtain the unique rankings
    # (previously extracting the proper masks)
    mask1 = Y_sorted[1:] != Y_sorted[:-1]
    mask2 = np.isnan(Y_sorted[1:]) & np.isnan(Y_sorted[:-1])
    mask = np.hstack(([True], np.any(mask1 & ~mask2, axis=1)))
    Y_unique = Y_sorted[mask]

    # Return the sorted unique rankings
    return Y_unique


def check_label_ranking_targets(Y):
    """Ensure that target Y is of a Label Ranking type.

    Parameters
    ----------
    Y : np.ndarray of shape (n_samples, n_classes)
        Input rankings.

    Raises
    ------
    ValueError
        If the input rankings are not for the Label Ranking problem.
    """
    # Obtain the target types
    target_types = type_of_targets(Y)

    # Raise the corresponding error if they
    # are not for the Label Ranking problem
    if "label_ranking" not in target_types:
        raise ValueError("Unknown label type: {}."
                         .format(sorted(list(target_types))))


def check_partial_label_ranking_targets(Y):
    """Ensure that target Y is of a Partial Label Ranking type.

    Parameters
    ----------
    Y : np.ndarray of shape (n_samples, n_classes)
        Input rankings.

    Raises
    ------
    ValueError
        If the input rankings are not for the Partial Label Ranking problem.
    """
    # Obtain the target types
    target_types = type_of_targets(Y)

    # Raise the corresponding error if they are
    # not for the Partial Label Ranking problem
    if "partial_label_ranking" not in target_types:
        raise ValueError("Unknown label type: {}."
                         .format(sorted(list(target_types))))


def type_of_targets(Y):
    """Determine the types of data indicated by the target.

    Parameters
    ----------
    Y : np.ndarray of shape (n_samples, n_classes)
        Input rankings.

    Returns
    -------
    target_types : str
        Some of:

            - "label_ranking": Y are all rankings without ties.
            - "partial_label_ranking": Y are all rankings with ties.
            - "unknown": Y is an array but none of the above.

    Examples
    --------
    >>> import numpy as np
    >>> from sklr.utils import type_of_targets
    >>> type_of_targets(np.array([[1, 2, 3], [3, 2, 1]]))
    {'partial_label_ranking', 'label_ranking'}
    >>> type_of_targets(np.array([[1, 2, 2], [2, 2, 1]]))
    {'partial_label_ranking'}
    >>> type_of_targets(np.array([[1, 2, 4], [2, 2, 1]]))
    {'unknown'}
    """
    # Check the format of the input rankings
    # (taking into account that some classes
    # can be missed so that it is not possible
    # to force that all the values are finite)
    Y = check_array(Y, force_all_finite=False)

    # Transform the rankings to properly manage them in Cython
    Y = _transform_rankings(Y)

    # Initialize an empty set with the target types
    target_types = set()

    # Even if it is possible to use "all" function to
    # check whether all the rankings in Y are of a
    # corresponding type, this method must be efficiently
    # executed, so that optimized Cython versions are employed

    # Label Ranking
    if are_label_ranking_targets(Y):
        target_types.add("label_ranking")
    # Partial Label Ranking
    if are_partial_label_ranking_targets(Y):
        target_types.add("partial_label_ranking")
    # Unknown
    if not target_types:
        target_types.add("unknown")

    # Return the target types
    return target_types


def is_ranking_without_ties(y):
    """Check if y is a ranking without ties.

    Parameters
    ----------
    y : np.ndarray of shape (n_classes,)
        Input ranking.

    Returns
    -------
    out : bool
        Return True if y is a ranking without ties, else False.

    Examples
    --------
    >>> import numpy as np
    >>> from sklr.utils import is_ranking_without_ties
    >>> is_ranking_without_ties(np.array([1, 3, 2]))
    True
    >>> is_ranking_without_ties(np.array([1, 2, 2]))
    False
    >>> is_ranking_without_ties(np.array([1, 2, 4]))
    False
    """
    # Check the format of the input ranking
    # (taking into account that some classes
    # can be missed so that it is not possible
    # to force that all the values are finite)
    y = check_array(y, force_all_finite=False, ensure_2d=False)

    # If it is not 1-D, then, it is not a ranking
    if y.ndim != 1:
        out = False
    else:
        # Transform the ranking to properly manage it in Cython
        y = _transform_rankings(y[None, :])[0]
        # Check if it is a ranking without ties using the fast version
        out = is_ranking_without_ties_fast(y)

    # Return whether the array is a ranking without ties
    return out


def is_ranking_with_ties(y):
    """Check if y is a ranking with ties.

    Parameters
    ----------
    y : np.ndarray of shape (n_classes,)
        Input ranking.

    Returns
    -------
    out: bool
        Return True if y is a ranking with ties, else False.

    Examples
    --------
    >>> import numpy as np
    >>> from sklr.utils import is_ranking_with_ties
    >>> is_ranking_with_ties(np.array([1, 2, 2]))
    True
    >>> is_ranking_with_ties(np.array([1, 2, 3]))
    True
    >>> is_ranking_with_ties(np.array([1, 2, 4]))
    False
    """
    # Check the format of the input ranking
    # (taking into account that some classes
    # can be missed so that it is not possible
    # to force that all the values are finite)
    y = check_array(y, force_all_finite=False, ensure_2d=False)

    # If it is not 1-D, then, it is not a ranking
    if y.ndim != 1:
        out = False
    else:
        # Transform the ranking to properly manage it in Cython
        y = _transform_rankings(y[None, :])[0]
        # Check if it is a ranking with ties using the fast version
        out = is_ranking_with_ties_fast(y)

    # Return whether the array is a ranking with ties
    return out


def rank_data(data, method="ordinal", check_input=True):
    """Assign ranks to data, dealing with ties appropriately.

    Ranks begin at 1. The `method` argument controls how ranks
    are assigned to equal values.

    Parameters
    ----------
    data : np.ndarray of shape (n_classes,)
        The array of values to be ranked. The array is first flattened.

    method : str, optional (default="ordinal")
        The method used to assign ranks to tied elements.
        The options are "ordinal" and "dense".

            - "ordinal": All values are given a distinct rank,
                         corresponding to the order that the
                         values occur in `data`.

            - "dense": The rank of the next highest element
                       is assigned the rank immediately after
                       those assigned to the tied elements.

    Returns
    -------
    y : np.ndarray of shape (n_classes,)
        An array of length equal to the size of `data`,
        containing rank scores.

    Raises
    ------
    ValueError
        If `method` is not available.

    References
    ----------
    .. [1] `MathWorks. "ranknum",
            https://uk.mathworks.com/matlabcentral/fileexchange/70301-ranknum,
            2019.`_

    Examples
    --------
    >>> import numpy as np
    >>> from sklr.utils import rank_data
    >>> data = np.array([0, 2, 3, 2])
    >>> rank_data(data, method="ordinal")
    array([1, 2, 4, 3])
    >>> rank_data(data, method="dense")
    array([1, 2, 3, 2])
    """
    # Check the format of the input data. In fact, it is
    # possible that some of the values are missed and, so
    # it is not possible to force all them to be finite. Also,
    # the input array can be of any dimension (since it will
    # flattened) and force the data type to be floating to
    # properly manage this array in the Cython version
    data = check_array(
        data, force_all_finite=False, ensure_2d=False, dtype=np.float64)

    # Flatten the data
    data = data.flatten()

    # Initialize the output ranking
    y = np.zeros(data.shape[0], dtype=np.int64)

    # Rank the data using the appropiate method
    try:
        rank_data_view(
            data=data, y=y, method=RANK_METHODS[method])
    except KeyError:
        raise ValueError("Unknown method '{}'.".format(method))

    # Return the ranking
    return y


def _transform_rankings(Y):
    """Transform the rankings for properly handled them in Cython."""
    # Initialize the transformed rankings. Even if
    # the transformation can be directly done over
    # the input rankings, it is not desired that
    # they are modified, since they may be used
    Yt = np.zeros(Y.shape, dtype=np.int64)

    # Copy the finite classes from the input
    # rankings to the transformed rankings
    Yt[~np.isnan(Y)] = Y[~np.isnan(Y)]

    # Copy the randomly missed classes from the input
    # rankings to the transformed, taking into account
    # the value internally used by Cython
    Yt[np.isnan(Y)] = _RANK_TYPES["random"]

    # Return the transformed rankings
    return Yt
