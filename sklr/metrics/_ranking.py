"""Metrics to assess the performance on label ranking and partial label ranking."""

import numpy as np
from numpy.typing import ArrayLike
from sklearn.utils import check_consistent_length

from sklr.utils import check_ranking
from sklr.utils._sklearn_compat import _check_sample_weight, validate_params


def _check_targets(y_true, y_pred, sample_weight, *, allow_ties):
    """Validate the true and the predicted rankings, and the sample weights.

    The rankings must be complete and have the same shape, and ties are only
    accepted if `allow_ties` is ``True``. The sample weights are returned as an
    array of non-negative numbers, not all of them zero, filled with ones if
    `sample_weight` is ``None``.
    """
    y_true = check_ranking(y_true, allow_ties=allow_ties, input_name="y_true")
    y_pred = check_ranking(y_pred, allow_ties=allow_ties, input_name="y_pred")
    check_consistent_length(y_true, y_pred, sample_weight)
    if y_true.shape[1] != y_pred.shape[1]:
        raise ValueError(
            "y_true and y_pred have different number of labels "
            f"({y_true.shape[1]} != {y_pred.shape[1]})."
        )
    sample_weight = _check_sample_weight(
        sample_weight, y_true, ensure_non_negative=True
    )
    return y_true, y_pred, sample_weight


def _is_ahead(y, *, or_tied):
    """Tell, for each sample and pair of labels ``i < j``, if ``i`` is ahead of ``j``.

    If `or_tied` is ``True``, a label tied with the other one also counts as ahead
    of it. The pairs are ordered as :func:`numpy.triu_indices` gives them.
    """
    i, j = np.triu_indices(y.shape[1], k=1)
    if or_tied:
        return y[:, i] <= y[:, j]
    return y[:, i] < y[:, j]


def _discordant_pairs(y_true, y_pred):
    """Count, for each sample, the pairs of labels ordered differently by the rankings.

    The rankings have no ties, so each pair is either concordant or discordant.
    """
    is_discordant = _is_ahead(y_true, or_tied=False) != _is_ahead(y_pred, or_tied=False)
    return is_discordant.sum(axis=1)


@validate_params(
    {
        "y_true": ["array-like"],
        "y_pred": ["array-like"],
        "sample_weight": ["array-like", None],
    },
    prefer_skip_nested_validation=True,
)
def kendall_tau_score(
    y_true: ArrayLike, y_pred: ArrayLike, *, sample_weight: ArrayLike | None = None
) -> float:
    """Compute Kendall's :math:`\\tau` between label rankings.

    Kendall's :math:`\\tau` [kendall_new_1938]_ compares two rankings of the same
    labels by scoring each pair of labels with ``+1`` if both rankings order it in
    the same way (a concordant pair) and with ``-1`` otherwise (a discordant pair),
    and dividing the total score by the number of pairs. It ranges from ``-1``, when
    one ranking is the reverse of the other, to ``1``, when both rankings are equal.
    The score of the rankings is the average of the coefficients of their samples.

    Read more in the :ref:`User Guide <kendall_tau>`.

    Parameters
    ----------
    y_true : array-like of shape (n_samples, n_labels)
        The true rankings, complete and without ties, as described in
        :ref:`ranking_representation`.

    y_pred : array-like of shape (n_samples, n_labels)
        The predicted rankings, complete and without ties.

    sample_weight : array-like of shape (n_samples,), default=None
        Non-negative weights of the samples in the average. If ``None``, the
        samples are equally weighted.

    Returns
    -------
    score : float
        The average Kendall's :math:`\\tau` of the samples, between ``-1`` and
        ``1``.

    See Also
    --------
    tau_x_score : Compute the :math:`\\tau_x` coefficient between partial label
        rankings.
    kendall_distance : Compute the Kendall distance between label rankings.
    scipy.stats.kendalltau : Kendall's :math:`\\tau` between two variables.

    Notes
    -----
    Given a ranking :math:`y` of :math:`m` labels, where :math:`y_i` is the
    position of label :math:`i`, let :math:`a_{ij}` be ``1`` if :math:`y_i < y_j`
    and ``-1`` if :math:`y_i > y_j`. Kendall's :math:`\\tau` between the rankings
    :math:`y` and :math:`\\hat{y}` is

    .. math::

        \\tau(y, \\hat{y}) = \\frac{2}{m(m - 1)} \\sum_{i < j} a_{ij} \\hat{a}_{ij}.

    Without ties, it is equal to Kendall's :math:`\\tau_a` and :math:`\\tau_b`, to
    the coefficient computed by :func:`scipy.stats.kendalltau` and to the
    coefficient computed by :func:`tau_x_score`. Rankings with ties are rejected,
    because the variants of Kendall's :math:`\\tau` handle ties in different ways;
    use :func:`tau_x_score` for them.

    References
    ----------
    .. [kendall_new_1938] M. G. Kendall, "A new measure of rank correlation",
       Biometrika, vol. 30, no. 1/2, pp. 81-93, 1938.

    Examples
    --------
    >>> from sklr.metrics import kendall_tau_score
    >>> y_true = [[1, 2, 3], [3, 2, 1]]
    >>> y_pred = [[1, 2, 3], [1, 3, 2]]
    >>> kendall_tau_score(y_true, y_pred)
    0.333...
    >>> kendall_tau_score(y_true, y_pred, sample_weight=[3, 1])
    0.666...
    """
    y_true, y_pred, sample_weight = _check_targets(
        y_true, y_pred, sample_weight, allow_ties=False
    )
    n_labels = y_true.shape[1]
    n_pairs = n_labels * (n_labels - 1) / 2
    scores = 1 - 2 * _discordant_pairs(y_true, y_pred) / n_pairs
    return float(np.average(scores, weights=sample_weight))


@validate_params(
    {
        "y_true": ["array-like"],
        "y_pred": ["array-like"],
        "sample_weight": ["array-like", None],
    },
    prefer_skip_nested_validation=True,
)
def tau_x_score(
    y_true: ArrayLike, y_pred: ArrayLike, *, sample_weight: ArrayLike | None = None
) -> float:
    """Compute the :math:`\\tau_x` coefficient between partial label rankings.

    The :math:`\\tau_x` rank correlation coefficient [emond_new_2002]_ extends
    Kendall's :math:`\\tau` to rankings with ties. It scores each ordered pair of
    different labels with ``+1`` if both rankings agree on whether the first label is
    ahead of or tied with the second one, and with ``-1`` otherwise, and divides the
    total score by the number of ordered pairs. It ranges from ``-1`` to ``1``, which
    it reaches when both rankings are equal, and it is equal to Kendall's
    :math:`\\tau` for rankings without ties. It is only ``-1`` when one ranking is
    the reverse of the other and neither has ties, because two tied labels are
    each ahead of or tied with the other, so the pairs that they form cannot
    disagree in both orders. The score of the rankings is the average of the
    coefficients of their samples.

    Read more in the :ref:`User Guide <tau_x>`.

    Parameters
    ----------
    y_true : array-like of shape (n_samples, n_labels)
        The true rankings, complete and possibly with ties, as described in
        :ref:`ranking_representation`.

    y_pred : array-like of shape (n_samples, n_labels)
        The predicted rankings, complete and possibly with ties.

    sample_weight : array-like of shape (n_samples,), default=None
        Non-negative weights of the samples in the average. If ``None``, the
        samples are equally weighted.

    Returns
    -------
    score : float
        The average :math:`\\tau_x` of the samples, between ``-1`` and ``1``.

    See Also
    --------
    kendall_tau_score : Compute Kendall's :math:`\\tau` between label rankings.

    Notes
    -----
    Given a ranking :math:`y` of :math:`m` labels, where :math:`y_i` is the
    position of label :math:`i`, let :math:`a'_{ij}` be ``1`` if
    :math:`y_i \\leq y_j` and ``-1`` if :math:`y_i > y_j`, for :math:`i \\neq j`.
    The :math:`\\tau_x` coefficient between the rankings :math:`y` and
    :math:`\\hat{y}` is

    .. math::

        \\tau_x(y, \\hat{y}) = \\frac{1}{m(m - 1)} \\sum_{i \\neq j} a'_{ij}
        \\hat{a}'_{ij}.

    Unlike Kendall's :math:`\\tau_b`, its denominator does not depend on the ties,
    so a ranking with all its labels tied has a coefficient of ``1`` with itself.
    It is related to the Kemeny-Snell distance :math:`d` by
    :math:`\\tau_x = 1 - 2 d / (m(m - 1))`.

    References
    ----------
    .. [emond_new_2002] E. J. Emond and D. W. Mason, "A new rank correlation
       coefficient with application to the consensus ranking problem", Journal of
       Multi-Criteria Decision Analysis, vol. 11, pp. 17-28, 2002.

    Examples
    --------
    >>> from sklr.metrics import tau_x_score
    >>> y_true = [[1, 1, 2], [1, 2, 3]]
    >>> y_pred = [[1, 1, 2], [1, 1, 1]]
    >>> tau_x_score(y_true, y_pred)
    0.5
    """
    y_true, y_pred, sample_weight = _check_targets(
        y_true, y_pred, sample_weight, allow_ties=True
    )
    n_labels = y_true.shape[1]
    n_pairs = n_labels * (n_labels - 1) / 2
    # Each pair of labels i < j stands for the ordered pairs (i, j) and (j, i),
    # and negating the positions gives the comparisons of the pairs (j, i).
    # With each agreement scoring +1 and each disagreement -1 over the m(m - 1)
    # ordered pairs, tau_x is one minus the disagreements divided by m(m - 1) / 2
    disagreements = (
        _is_ahead(y_true, or_tied=True) != _is_ahead(y_pred, or_tied=True)
    ).sum(axis=1) + (
        _is_ahead(-y_true, or_tied=True) != _is_ahead(-y_pred, or_tied=True)
    ).sum(axis=1)
    scores = 1 - disagreements / n_pairs
    return float(np.average(scores, weights=sample_weight))


@validate_params(
    {
        "y_true": ["array-like"],
        "y_pred": ["array-like"],
        "normalize": ["boolean"],
        "sample_weight": ["array-like", None],
    },
    prefer_skip_nested_validation=True,
)
def kendall_distance(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    normalize: bool = True,
    sample_weight: ArrayLike | None = None,
) -> float:
    """Compute the Kendall distance between label rankings.

    The Kendall distance between two rankings of the same labels is the number of
    pairs of labels that they order in different ways (the discordant pairs)
    [kendall_new_1938]_, which is also the minimum number of swaps of adjacent
    labels that turn one ranking into the other. The distance of the rankings is the
    average of the distances of their samples.

    Read more in the :ref:`User Guide <kendall_distance>`.

    Parameters
    ----------
    y_true : array-like of shape (n_samples, n_labels)
        The true rankings, complete and without ties, as described in
        :ref:`ranking_representation`.

    y_pred : array-like of shape (n_samples, n_labels)
        The predicted rankings, complete and without ties.

    normalize : bool, default=True
        If ``True``, divide the distance of each sample by the number of pairs of
        labels, ``n_labels * (n_labels - 1) / 2``, so that it ranges from ``0`` to
        ``1``. Otherwise, it is the number of discordant pairs.

    sample_weight : array-like of shape (n_samples,), default=None
        Non-negative weights of the samples in the average. If ``None``, the
        samples are equally weighted.

    Returns
    -------
    distance : float
        The average Kendall distance of the samples. It is ``0`` when the rankings
        are equal.

    See Also
    --------
    kendall_tau_score : Compute Kendall's :math:`\\tau` between label rankings.

    Notes
    -----
    The normalized distance :math:`d` is related to Kendall's :math:`\\tau` by
    :math:`d = (1 - \\tau) / 2`, sample by sample. Rankings with ties are rejected,
    because a pair of labels that is tied in one ranking is neither concordant nor
    discordant.

    References
    ----------
    .. [kendall_new_1938] M. G. Kendall, "A new measure of rank correlation",
       Biometrika, vol. 30, no. 1/2, pp. 81-93, 1938.

    Examples
    --------
    >>> from sklr.metrics import kendall_distance
    >>> y_true = [[1, 2, 3], [3, 2, 1]]
    >>> y_pred = [[1, 2, 3], [1, 3, 2]]
    >>> kendall_distance(y_true, y_pred)
    0.333...
    >>> kendall_distance(y_true, y_pred, normalize=False)
    1.0
    """
    y_true, y_pred, sample_weight = _check_targets(
        y_true, y_pred, sample_weight, allow_ties=False
    )
    distances = _discordant_pairs(y_true, y_pred)
    if normalize:
        n_labels = y_true.shape[1]
        distances = distances / (n_labels * (n_labels - 1) / 2)
    return float(np.average(distances, weights=sample_weight))
