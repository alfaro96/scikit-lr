"""Utilities to work with the rankings of the labels."""

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import issparse
from sklearn.utils import check_array


def type_of_ranking(
    y: ArrayLike,
) -> Literal["label_ranking", "partial_label_ranking", "unknown"]:
    """Determine the type of rankings indicated by the target.

    Each row of `y` is the ranking of a sample and each column a label, so that
    ``y[i, j]`` is the position of label ``j`` in ranking ``i``, from ``1`` for the
    most preferred one. Tied labels share their position and the positions must be
    dense, without gaps between them. ``np.nan`` marks a label that is not ranked,
    and the positions of the ranked labels of each row must be dense from ``1``,
    whatever their number. See :ref:`ranking_representation` for the details.

    Parameters
    ----------
    y : array-like of shape (n_samples, n_labels)
        The rankings, with at least one sample and two labels.

    Returns
    -------
    target_type : {"label_ranking", "partial_label_ranking", "unknown"}
        ``"label_ranking"`` if no ranking has ties, ``"partial_label_ranking"``
        if some ranking has ties, and ``"unknown"`` if `y` does not encode
        rankings, because it is not a two-dimensional array of numbers with at
        least one sample and two labels or because the positions of some ranking
        are not dense from ``1``.

    See Also
    --------
    sklearn.utils.multiclass.type_of_target : Determine the type of data indicated
        by the target of a classifier or a regressor.

    Notes
    -----
    A label ranking, without ties, is also a valid partial label ranking.

    Examples
    --------
    >>> import numpy as np
    >>> from sklr.utils import type_of_ranking
    >>> type_of_ranking([[1, 2, 3], [3, 1, 2]])
    'label_ranking'
    >>> type_of_ranking([[1, 1, 2], [3, 1, 2]])
    'partial_label_ranking'
    >>> type_of_ranking([[2, np.nan, 1], [3, 1, 2]])
    'label_ranking'
    >>> type_of_ranking([[1, 1, 3], [3, 1, 2]])
    'unknown'
    """
    # Positions start at one, so rankings have no zeros to leave implicit and
    # are not accepted as sparse matrices
    if issparse(y):
        return "unknown"
    try:
        ranks = check_array(
            y,
            dtype="numeric",
            ensure_all_finite=False,
            ensure_2d=False,
            allow_nd=True,
            ensure_min_samples=0,
        )
    except (TypeError, ValueError):
        # Strings, complex numbers and ragged sequences
        return "unknown"
    if (
        ranks.ndim != 2
        or ranks.shape[0] == 0
        or ranks.shape[1] < 2
        or ranks.dtype.kind not in {"i", "u", "f"}
    ):
        return "unknown"

    # NaN is sorted last, so the ranked labels of each row come first and in
    # order of preference. Their positions are dense from one if the first one
    # is one and each of the others is equal to the previous one or the next
    # integer, which also rules out positions that are not integers and
    # infinite ones
    ranks = np.sort(np.asarray(ranks, dtype=np.float64), axis=1)
    is_ranked = ~np.isnan(ranks)
    steps = np.diff(ranks, axis=1)
    # The differences that involve an unranked label are NaN, which is neither
    # zero nor one, so they must be left out of the check of the steps. Since
    # NaN is sorted last, a difference involves an unranked label if and only
    # if the second label is not ranked
    is_between_ranked = is_ranked[:, 1:]
    is_dense = np.all((ranks[:, 0] == 1) | ~is_ranked[:, 0]) and np.all(
        (steps == 0) | (steps == 1) | ~is_between_ranked
    )
    if not is_dense:
        return "unknown"
    # Tied labels share their position, and a difference of zero is always
    # between two ranked labels, because the others are NaN
    if np.any(steps == 0):
        return "partial_label_ranking"
    return "label_ranking"
