"""Scorers of the label ranking and partial label ranking metrics.

scikit-learn only resolves the names of its own scorers, so the scorers of
:mod:`sklr.metrics` are looked up with :func:`get_scorer` and passed as objects
to the ``scoring`` parameter of the model selection tools.
"""

import copy
from collections.abc import Callable
from typing import Any

from sklearn.metrics import make_scorer

from sklr.metrics._ranking import (
    kendall_distance,
    kendall_tau_score,
    kendall_tau_x_score,
)
from sklr.utils._sklearn_compat import validate_params

_SCORERS = {
    "kendall_tau": make_scorer(kendall_tau_score),
    "kendall_tau_x": make_scorer(kendall_tau_x_score),
    "neg_kendall_distance": make_scorer(kendall_distance, greater_is_better=False),
}


@validate_params(
    {"scoring": [str, callable, None]},
    prefer_skip_nested_validation=True,
)
def get_scorer(scoring: str | Callable | None) -> Any:
    """Get a scorer of label ranking or partial label ranking from its name.

    The scorers can be passed to the ``scoring`` parameter of the model selection
    tools of scikit-learn, such as :func:`sklearn.model_selection.cross_val_score`
    and :class:`sklearn.model_selection.GridSearchCV`, which do not accept their
    names.

    Read more in the :ref:`User Guide <scoring>`.

    Parameters
    ----------
    scoring : str, callable or None
        The name of the scorer, among those returned by :func:`get_scorer_names`.
        If it is a callable or ``None``, it is returned as is.

    Returns
    -------
    scorer : callable or None
        The scorer, which is called as ``scorer(estimator, X, y)``.

    Raises
    ------
    ValueError
        If `scoring` is not the name of a scorer.

    See Also
    --------
    get_scorer_names : Get the names of all the scorers.
    sklearn.metrics.make_scorer : Make a scorer from a metric.

    Notes
    -----
    When passed a string, this function always returns a copy of the scorer, so
    calling it twice with the same name gives two separate scorers.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.base import BaseEstimator
    >>> from sklr.base import LabelRankerMixin
    >>> from sklr.metrics import get_scorer
    >>> class FirstRanker(LabelRankerMixin, BaseEstimator):
    ...     def fit(self, X, y):
    ...         self.ranking_ = np.asarray(y)[0]
    ...         return self
    ...     def predict(self, X):
    ...         return np.tile(self.ranking_, (len(X), 1))
    >>> X = [[0], [1]]
    >>> y = [[1, 2, 3], [3, 2, 1]]
    >>> ranker = FirstRanker().fit(X, y)
    >>> scorer = get_scorer("kendall_tau")
    >>> scorer(ranker, X, y)
    0.0
    """
    if isinstance(scoring, str):
        try:
            return copy.deepcopy(_SCORERS[scoring])
        except KeyError:
            raise ValueError(
                f"{scoring!r} is not a valid scoring value. Use "
                "sklr.metrics.get_scorer_names() to get valid options."
            ) from None
    return scoring


def get_scorer_names() -> list[str]:
    """Get the names of all the scorers of label ranking and partial label ranking.

    Each name can be passed to :func:`get_scorer` to get its scorer. They are not
    valid values of the ``scoring`` parameter of scikit-learn, which only knows the
    names of its own scorers.

    Returns
    -------
    list of str
        The names of the scorers, in alphabetical order.

    See Also
    --------
    get_scorer : Get a scorer from its name.

    Examples
    --------
    >>> from sklr.metrics import get_scorer_names
    >>> get_scorer_names()
    ['kendall_tau', 'kendall_tau_x', 'neg_kendall_distance']
    """
    return sorted(_SCORERS)
