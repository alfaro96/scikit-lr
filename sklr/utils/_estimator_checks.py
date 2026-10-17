"""Common checks of scikit-learn adapted to the rankers of scikit-lr."""

from functools import partial

import numpy as np
import pytest
from scipy.stats import rankdata
from sklearn.utils.estimator_checks import (
    parametrize_with_checks as sklearn_parametrize_with_checks,
)

from sklr.base import is_label_ranker, is_partial_label_ranker
from sklr.utils._sklearn_compat import _patch_enforce_estimator_tags_y

# The number of labels of the rankings built from the targets of the checks
N_LABELS = 3


def _enforce_estimator_tags_y(enforce_estimator_tags_y, estimator, y):
    """Turn the target of a check into rankings if the estimator is a ranker.

    The targets of the checks are class labels, mostly from zero to two, or real
    values. The labels of each ranking are sorted by the distance from the target
    of the sample to their index, so that each class gets a different ranking. The
    labels at the same distance are tied for the partial label rankers, and the
    ties are broken in favour of the lowest index for the label rankers, which do
    not accept them. A target with several outputs is reduced to its first one
    before.
    """
    is_partial = is_partial_label_ranker(estimator)
    if not (is_label_ranker(estimator) or is_partial):
        return enforce_estimator_tags_y(estimator, y)
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 2:
        y = y[:, 0]
    distances = np.abs(y[:, np.newaxis] - np.arange(N_LABELS))
    method = "dense" if is_partial else "ordinal"
    return rankdata(distances, method=method, axis=1).astype(np.intp)


class _RankingCheck(partial):
    """Check of scikit-learn that runs with the targets turned into rankings.

    It is a :class:`functools.partial`, as the checks of scikit-learn are, so that
    the tests keep the identifiers that scikit-learn gives them.
    """

    def __call__(self, *args, **kwargs):
        with _patch_enforce_estimator_tags_y(_enforce_estimator_tags_y):
            return super().__call__(*args, **kwargs)


def parametrize_with_checks(estimators, *, expected_failed_checks=None):
    """Pytest specific decorator for parametrizing the checks of the rankers.

    It is :func:`sklearn.utils.estimator_checks.parametrize_with_checks`, with the
    targets of the checks of the rankers turned into rankings, so that their ``fit``
    can succeed. `expected_failed_checks` is a callable that returns, for each
    estimator, a dictionary from the name of each check that it is expected to fail
    to the reason.
    """
    mark = sklearn_parametrize_with_checks(
        estimators, expected_failed_checks=expected_failed_checks
    )
    argnames, argvalues = mark.args
    params = []
    for argvalue in argvalues:
        # The checks expected to fail come in a pytest.param with their xfail
        # mark, and the others as an (estimator, check) tuple
        param = argvalue if hasattr(argvalue, "marks") else pytest.param(*argvalue)
        estimator, check = param.values
        # The checks are partials of the check functions with the name of the
        # estimator, which the identifiers of the tests are built from
        assert isinstance(check, partial)
        check = _RankingCheck(check.func, *check.args, **check.keywords)
        params.append(pytest.param(estimator, check, marks=param.marks))
    return pytest.mark.parametrize(argnames, params, **mark.kwargs)
