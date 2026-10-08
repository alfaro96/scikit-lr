"""Private API of scikit-learn used by scikit-lr.

The private API of scikit-learn may change without deprecation, so the rest of
:mod:`sklr` imports it only from this module, and the ``no-private-sklearn-imports``
pre-commit hook enforces it. scikit-lr is built and run against a single minor
release of scikit-learn, so each name is imported here as that release defines it.
When moving to a new minor release, the names that have changed are adapted here,
keeping the interface that the rest of the package uses.
"""

from contextlib import contextmanager
from functools import partial
from unittest import mock

from sklearn.utils._param_validation import validate_params
from sklearn.utils.validation import _check_sample_weight

__all__ = ["_check_sample_weight", "_patch_enforce_estimator_tags_y", "validate_params"]


@contextmanager
def _patch_enforce_estimator_tags_y(enforce_estimator_tags_y):
    """Make the common checks of scikit-learn adapt their targets with another function.

    The common checks build their targets for classifiers and regressors, and adapt
    them to the tags of the estimator with a private function that they look up in
    their module at each call. While the context is active, it is replaced by
    `enforce_estimator_tags_y`, which receives the replaced function as its first
    argument, followed by the estimator and the target.
    """
    # The module of the checks imports pytest, so it is only imported when the
    # checks run, instead of whenever this module is imported
    from sklearn.utils import estimator_checks

    replaced = estimator_checks._enforce_estimator_tags_y
    with mock.patch.object(
        estimator_checks,
        "_enforce_estimator_tags_y",
        partial(enforce_estimator_tags_y, replaced),
    ):
        yield
