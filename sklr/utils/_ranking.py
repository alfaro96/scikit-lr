"""Utilities to work with the rankings of the labels."""

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import issparse
from sklearn.utils import check_array


def _check_positions(ranks):
    """Find the rankings whose positions are dense from one and those with ties.

    ``ranks`` is a two-dimensional ``float64`` array, and two boolean arrays of
    shape ``(n_samples,)`` are returned.
    """
    # NaN is sorted last, so the ranked labels of each row come first and in
    # order of preference. Their positions are dense from one if the first one
    # is one and each of the others is equal to the previous one or the next
    # integer, which also rules out positions that are not integers and
    # infinite ones
    ranks = np.sort(ranks, axis=1)
    is_ranked = ~np.isnan(ranks)
    steps = np.diff(ranks, axis=1)
    # The differences that involve an unranked label are NaN, which is neither
    # zero nor one, so they must be left out of the check of the steps. Since
    # NaN is sorted last, a difference involves an unranked label if and only
    # if the second label is not ranked
    is_between_ranked = is_ranked[:, 1:]
    is_dense = ((ranks[:, 0] == 1) | ~is_ranked[:, 0]) & np.all(
        (steps == 0) | (steps == 1) | ~is_between_ranked, axis=1
    )
    # Tied labels share their position, and a difference of zero is always
    # between two ranked labels, because the others are NaN
    has_ties = np.any(steps == 0, axis=1)
    return is_dense, has_ties


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

    is_dense, has_ties = _check_positions(np.asarray(ranks, dtype=np.float64))
    if not np.all(is_dense):
        return "unknown"
    if np.any(has_ties):
        return "partial_label_ranking"
    return "label_ranking"


def check_ranking(
    y: ArrayLike,
    *,
    allow_ties: bool = False,
    allow_incomplete: bool = False,
    input_name: str = "y",
) -> np.ndarray:
    """Input validation on the rankings of a target.

    Each row of `y` is the ranking of a sample and each column a label, so that
    ``y[i, j]`` is the position of label ``j`` in ranking ``i``, from ``1`` for the
    most preferred one. Tied labels share their position and the positions must be
    dense, without gaps between them. ``np.nan`` marks a label that is not ranked,
    and the positions of the ranked labels of each row must be dense from ``1``,
    whatever their number. See :ref:`ranking_representation` for the details.

    Parameters
    ----------
    y : array-like of shape (n_samples, n_labels)
        The rankings to check, with at least one sample and two labels.

    allow_ties : bool, default=False
        Whether to accept rankings with tied labels, that is, partial label
        rankings.

    allow_incomplete : bool, default=False
        Whether to accept incomplete rankings, with labels that are not ranked.

    input_name : str, default="y"
        The name of the rankings in the error messages.

    Returns
    -------
    y_converted : ndarray of shape (n_samples, n_labels)
        The validated rankings, as a ``float64`` array. It may be `y` itself if
        `y` is already a ``float64`` array.

    Raises
    ------
    TypeError
        If `y` is a sparse matrix.

    ValueError
        If `y` does not encode rankings, because it is not a two-dimensional array
        of numbers with at least one sample and two labels or because the
        positions of some ranking are not dense from ``1``, or if `y` holds
        rankings with ties or incomplete rankings that are not allowed.

    See Also
    --------
    type_of_ranking : Determine the type of rankings indicated by the target.
    sklearn.utils.check_array : Input validation on an array, list, sparse matrix
        or similar.

    Examples
    --------
    >>> import numpy as np
    >>> from sklr.utils import check_ranking
    >>> check_ranking([[1, 2, 3], [3, 1, 2]])
    array([[1., 2., 3.],
           [3., 1., 2.]])
    >>> check_ranking([[1, 1, 2], [3, 1, 2]], allow_ties=True)
    array([[1., 1., 2.],
           [3., 1., 2.]])
    >>> check_ranking([[2, np.nan, 1], [3, 1, 2]], allow_incomplete=True)
    array([[ 2., nan,  1.],
           [ 3.,  1.,  2.]])
    >>> check_ranking([[1, 1, 2], [3, 1, 2]])
    Traceback (most recent call last):
        ...
    ValueError: Expected rankings without ties in y, got y[0] = [1 1 2].
    """
    ranks = check_array(
        y,
        dtype="numeric",
        ensure_all_finite="allow-nan",
        ensure_2d=False,
        allow_nd=True,
        # The number of samples and labels is checked below, with messages
        # that speak of labels instead of features
        ensure_min_samples=0,
        ensure_min_features=0,
        input_name=input_name,
    )
    if ranks.ndim != 2:
        raise ValueError(
            f"Expected a 2D array of shape (n_samples, n_labels) for {input_name}, "
            f"got a {ranks.ndim}D array instead."
        )
    n_samples, n_labels = ranks.shape
    if n_samples < 1:
        raise ValueError(
            f"Found {input_name} with {n_samples} sample(s) (shape={ranks.shape}) "
            "while a minimum of 1 is required."
        )
    if n_labels < 2:
        raise ValueError(
            f"Found {input_name} with {n_labels} label(s) (shape={ranks.shape}) "
            "while a minimum of 2 is required."
        )
    # Booleans are numeric for check_array, but they are not positions
    if ranks.dtype.kind not in {"i", "u", "f"}:
        raise ValueError(
            f"Expected the positions of the labels in {input_name} to be integer "
            f"or floating point numbers, got dtype {ranks.dtype} instead."
        )

    ranks_converted = np.asarray(ranks, dtype=np.float64)
    is_dense, has_ties = _check_positions(ranks_converted)
    if not np.all(is_dense):
        # The row is shown with the dtype of the input, so that integer
        # positions are not shown as floating point numbers
        sample = np.flatnonzero(~is_dense)[0]
        raise ValueError(
            "Expected the positions of the ranked labels of each ranking in "
            f"{input_name} to be dense from 1, got {input_name}[{sample}] = "
            f"{ranks[sample]}."
        )
    is_incomplete = np.any(np.isnan(ranks_converted), axis=1)
    if not allow_incomplete and np.any(is_incomplete):
        sample = np.flatnonzero(is_incomplete)[0]
        raise ValueError(
            f"Expected complete rankings in {input_name}, without unranked labels "
            f"(NaN), got {input_name}[{sample}] = {ranks[sample]}."
        )
    if not allow_ties and np.any(has_ties):
        sample = np.flatnonzero(has_ties)[0]
        raise ValueError(
            f"Expected rankings without ties in {input_name}, got "
            f"{input_name}[{sample}] = {ranks[sample]}."
        )
    return ranks_converted
