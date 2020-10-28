"""Utilities for input validation."""


# =============================================================================
# Imports
# =============================================================================

# Standard
from inspect import isclass, signature
from numbers import Integral

# Third party
import numpy as np
from sklearn.exceptions import NotFittedError


# =============================================================================
# Methods
# =============================================================================

def check_consistent_length(*arrays):
    """Check that all ``arrays`` have consistent length.

    Checks whether all objects in ``arrays`` have the same length.

    Parameters
    ----------
    *arrays : list of ndarray
        The arrays that will be checked for consistent length.
    """
    # Remove "None" values from the arrays (not interesting)
    arrays = [array for array in arrays if array is not None]

    if any(map(lambda array: not isinstance(array, np.ndarray), arrays)):
        raise TypeError("All the input objects must be of an array type.")

    if any(map(lambda array: array.shape[0] != arrays[0].shape[0], arrays)):
        raise ValueError("The input arrays have different number of samples.")


def check_array(array, dtype=None, force_all_finite=True, ensure_2d=True):
    """Input validation on an array.

    By default, the input is checked to be a non-empty numeric 2-D array
    containing only finite values.

    Parameters
    ----------
    array : ndarray
        The input array to check and convert.

    dtype : dtype, default=None
        The data type of result. If ``None``, the data type of the
        input is preserved.

    force_all_finite : bool, default=True
        Whether to raise an error on ``np.inf`` and ``np.nan`` in ``array``.

    ensure_2d : bool, default=True
        Whether to raise an error if ``array`` is not 2-D.

    Returns
    -------
    array_converted : ndarray
        The converted and validated array.
    """
    if not isinstance(array, np.ndarray):
        raise TypeError("The input object is not an array. "
                        "Got {0}.".format(type(array).__name__))

    if (not np.issubdtype(array.dtype, np.int64) and
            not np.issubdtype(array.dtype, np.float64)):
        raise TypeError("The data type of the array is not integer "
                        "or floating. Got {0}.".format(array.dtype))

    if force_all_finite and np.any(~np.isfinite(array)):
        raise ValueError("The provided array contain infinite values.")

    if ensure_2d:
        if array.ndim != 2:
            raise ValueError("Expected 2-D array. Got {0}-D "
                             "array instead.".format(array.ndim))

    return np.array(array, dtype=dtype)


def check_is_fitted(estimator):
    """Perform ``is_fitted`` validation for estimator.

    Checks if the estimator is fitted by verifying the presence of
    fitted attributes (ending with a trailing underscore) and otherwise
    raises a ``NotFittedError``.

    Parameters
    ----------
    estimator : BaseEstimator
        The estimator instance for which the check is performed.
    """
    if isclass(estimator):
        raise TypeError("{0} is a class, not an instance.".format(estimator))

    if not hasattr(estimator, "fit"):
        raise TypeError("'{0}' not an estimator instance.".format(estimator))

    if not any(map(lambda var: var.endswith("_"), vars(estimator))):
        raise NotFittedError("This {0} instance is not fitted yet. "
                             "Call 'fit' with appropriate arguments "
                             "before using this estimator."
                             .format(type(estimator).__name__))


def check_sample_weight(sample_weight, X):
    """Validate sample weights.

    Note that passing ``sample_weight=None`` will output an array of ones.

    Parameters
    ----------
    sample_weight : None or ndarray of shape (n_samples,), dtype=np.float64
        The input sample weights.

    X : ndarray of shape (n_samples, n_features), dtype=np.float64
        The input data.

    Returns
    -------
    sample_weight : ndarray of shape (n_samples,), dtype=np.float64
        The validated sample weights.
    """
    if sample_weight is None:
        sample_weight = np.ones(X.shape[0], dtype=np.float64)
    else:
        sample_weight = check_array(
            sample_weight, ensure_2d=False, dtype=np.float64)

        if sample_weight.ndim != 1:
            raise ValueError("Sample weights must be 1-D array. Got "
                             "{0}-D array.".format(sample_weight.ndim))

        check_consistent_length(X, sample_weight)

    return sample_weight


def check_random_state(seed):
    """Turn seed into a ``RandomState`` instance.

    Parameters
    ----------
    seed : None, int or RandomState instance
        - If ``None``, return the ``RandomState`` singleton of ``np.random``.
        - If ``int``, return the ``RandomState`` instance seeded with seed.
        - If already a ``RandomState`` instance, return it.

    Returns
    -------
    random_state : RandomState instance
        The ``RandomState`` instance seeded with seed.
    """
    if seed is None or seed is np.random:
        random_state = np.random.mtrand._rand
    elif isinstance(seed, (Integral, np.integer)):
        random_state = np.random.RandomState(seed)
    elif isinstance(seed, np.random.RandomState):
        random_state = seed
    else:
        raise ValueError("{0} cannot be used to seed a "
                         "RandomState instance.".format(seed))

    return random_state


def has_fit_parameter(estimator, parameter):
    """Check whether the estimator's fit method supports the given parameter.

    Parameters
    ----------
    estimator : BaseEstimator
        The estimator to inspect.

    parameter : str
        The searched parameter.

    Returns
    -------
    is_parameter : bool
        Whether the parameter was found to be a named parameter of the
        estimator's fit method.

    Examples
    --------
    >>> from sklr.utils import has_fit_parameter
    >>> from sklr.neighbors import KNeighborsLabelRanker
    >>> from sklr.tree import DecisionTreeLabelRanker
    >>> knn = KNeighborsLabelRanker(n_neighbors=1)
    >>> tree = DecisionTreeLabelRanker(random_state=0)
    >>> has_fit_parameter(knn, "sample_weight")
    False
    >>> has_fit_parameter(tree, "sample_weight")
    True
    """
    return parameter in signature(estimator.fit).parameters


def check_X_Y(X, Y):
    """Input validation for standard estimators.

    Checks ``X`` and ``Y`` for consistent length, enforces ``X`` to be a
    floating 2-D array and ``Y`` to be a 2-D array mantaining the data
    type. ``X`` is checked to be non-empty and containing only finite values
    while ``Y`` is checked to be non-empty and can contain infinite values.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features), dtype=np.float64
        The input data.

    Y : ndarray of shape (n_samples, n_classes), dtype=np.int64 or \
            dtype=np.float64
        The input rankings.

    Returns
    -------
    X_converted : ndarray of shape (n_samples, n_features), dtype=np.float64
        The converted and validated ``X``.

    Y_converted : ndarray of shape (n_samples, n_classes), dtype=np.int64 or \
            dtype=np.float64
        The converted and validated ``Y``.
    """
    X = check_array(X, dtype=np.float64)
    Y = check_array(Y, force_all_finite=False)

    check_consistent_length(X, Y)

    return (X, Y)
