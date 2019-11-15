"""Utilities for input validation."""


# =============================================================================
# Imports
# =============================================================================

# Standard
from inspect import isclass, signature
from numbers import Integral

# Third party
import numpy as np

# Local application
from ..exceptions import NotFittedError


# =============================================================================
# Methods
# =============================================================================

def check_consistent_length(*arrays):
    """Check that all arrays have consistent first dimensions.

    Checks whether all objects in arrays have the same shape or length.

    Parameters
    ----------
    *arrays : {list, tuple} of input arrays.
        Arrays that will be checked for consistent length.

    Raises
    ------
    TypeError
        If any of the input objects is not an array.

    ValueError
        If the input arrays have inconsistent first dimension.
    """
    # Remove None values from the arrays (they are not interesting)
    arrays = [array for array in arrays if array is not None]

    # Check that all the input objects are arrays
    if any(map(lambda array: not isinstance(array, np.ndarray), arrays)):
        raise TypeError("The input objects are not arrays. Got {}."
                        .format(list(map(
                            lambda array: type(array).__name__, arrays))))

    # Check for consistent first dimensions
    if any(map(lambda array: array.shape[0] != arrays[0].shape[0], arrays)):
        raise ValueError("Found input variables with inconsistent numbers of "
                         "samples: {}."
                         .format(list(map(
                             lambda array: array.shape[0], arrays))))


def check_array(array, dtype=None, force_all_finite=True, ensure_2d=True):
    """Input validation on an array.

    By default, the input is checked to be a non-empty
    numeric 2-D array containing only finite values.

    Parameters
    ----------
    array : np.ndarray
        Input array to check / convert.

    dtype : {None, np.dtype}, optional (default=None)
        Data type of result. If None, the dtype of the input is preserved.

    force_all_finite : boolean, optional (default=True)
        Whether to raise an error on np.inf and np.nan in array. The
        possibilities are:

        - True: Force all values of array to be finite.
        - False: accept both np.inf and np.nan in array.

    ensure_2d : boolean, optional (default=True)
        Whether to raise an error if array is not 2-D.

    Returns
    -------
    array_converted : np.ndarray
        The converted and validated array.

    Raises
    ------
    TypeError
        If the input object is not an array.
        If the data type of the provided array is not integer or float.

    ValueError
        If ``force_all_finite`` and the input array contains infinite values.
        If ``ensure_2d`` and the input array is not 2-D.
    """
    # Check that the provided object is an array
    if not isinstance(array, np.ndarray):
        raise TypeError("The input object is not an array. "
                        "Got {}.".format(type(array).__name__))

    # Check that the data type of the input array is either integer or float
    # (other data types are not allowed for efficiency purposes)
    if (not np.issubdtype(array.dtype, np.int64) and
            not np.issubdtype(array.dtype, np.float64)):
        raise TypeError("The data type of the array is not integer "
                        "or floating. Got {}.".format(array.dtype))

    # Check whether the array contain infinite
    # values when they are not allowed
    if force_all_finite and np.any(~np.isfinite(array)):
        raise ValueError("The provided array contain infinite values.")

    # Check if the array is 2-D as far as
    # it is forced to be of such dimension
    if ensure_2d:
        if array.ndim != 2:
            raise ValueError("Expected 2-D array. Got {}-D array instead."
                             .format(array.ndim))

    # Convert the array to the desired data type
    array = np.array(array, dtype=dtype)

    # Return the converted array
    return array


def check_is_fitted(estimator):
    """Perform is_fitted validation for estimator.

    Checks if the estimator is fitted by verifying the presence of
    fitted attributes (ending with a trailing underscore) and otherwise
    raises a NotFittedError.

    Parameters
    ----------
    estimator : object.
        Estimator instance for which the check is performed.

    Raises
    ------
    TypeError
        If the input object is not an estimator instance.

    NotFittedError
        If the attributes are not found.
    """
    # Check if the estimator is an object, raising
    # the corresponding error when it is a class
    if isclass(estimator):
        raise TypeError("{} is a class, not an instance.".format(estimator))

    # Check if the given object is an estimator by
    # veryfing the presence of a "fit" method
    if not hasattr(estimator, "fit"):
        raise TypeError("'{}' is not an estimator instance.".format(estimator))

    # Obtain the attributes of the estimator
    attrs = [
        v for v in vars(estimator)
        if (v.endswith("_") or v.startswith("_")) and not v.startswith("__")
    ]

    # Check if the estimator contains the corresponding attributes
    if not attrs:
        raise NotFittedError("This {} instance is not fitted yet. "
                             "Call 'fit' with appropriate arguments "
                             "before using this estimator."
                             .format(type(estimator).__name__))


def _check_sample_weight(sample_weight, X):
    """Validate sample weights.

    Note that passing sample_weight=None will output an array of ones.

    Therefore, in some cases, you may want to protect the call with:
    if sample_weight is not None:
        sample_weight = _check_sample_weight(sample_weight)."""
    # Initialize the number of samples from the input data
    n_samples = X.shape[0]

    # If the sample weights are not provided
    # (None value), output an array of ones
    if sample_weight is None:
        sample_weight = np.ones(n_samples, dtype=np.float64)
    # Otherwise, check their format, ensuring that they are
    # a 1-D array with the same number of samples than the data
    else:
        # Check the format of the sample weights
        sample_weight = check_array(
            sample_weight, ensure_2d=False, dtype=np.float64)
        # Ensure that they are a 1-D array
        if sample_weight.ndim != 1:
            raise ValueError("Sample weights must be 1-D array. "
                             "Got {}-D array."
                             .format(sample_weight.ndim))
        # Ensure that the data and the sample
        # weights have the number of samples
        check_consistent_length(X, sample_weight)

    # Return the validated sample weights
    return sample_weight


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance.

    Parameters
    ----------
    seed : {None, int, instance of RandomState}
        - If seed is None, return the RandomState singleton used by np.random.
        - If seed is an int, return a RandomState instance seeded with seed.
        - If seed is already a RandomState instance, return it.

    Returns
    -------
    random_state : RandomState
        The RandomState instance seeded with seed.

    Raises
    ------
    ValueError
        If the seed cannot be used for seeding.
    """
    # If the seed is None, use the RandomState singleton used by np.random
    if seed is None or seed is np.random:
        random_state = np.random.mtrand._rand
    # If seed is an int, use a new RandomState instance seeded with seed"
    elif isinstance(seed, (Integral, np.integer)):
        random_state = np.random.RandomState(seed)
    # If seed is already a RandomState instance, return it
    elif isinstance(seed, np.random.RandomState):
        random_state = seed
    else:
        raise ValueError("{} cannot be used to seed a np.random.RandomState "
                         "instance.".format(seed))

    # Return the RandomState instance
    return random_state


def has_fit_parameter(estimator, parameter):
    """Check whether the estimator's fit method supports the given parameter.

    Parameters
    ----------
    estimator : object
        An estimator to inspect.

    parameter : str
        The searched parameter.

    Returns
    -------
    is_parameter: bool
        Whether the parameter was found to be a named parameter of the
        estimator's fit method.

    Examples
    --------
    >>> from sklr.neighbors import KNeighborsLabelRanker
    >>> from sklr.tree import DecisionTreeLabelRanker
    >>> knn = KNeighborsLabelRanker()
    >>> tree = DecisionTreeLabelRanker(random_state=0)
    >>> has_fit_parameter(knn, "sample_weight")
    False
    >>> has_fit_parameter(tree, "sample_weight")
    True
    """
    # Return whether the fit method supports the given parameter
    return parameter in signature(estimator.fit).parameters


def check_X_Y(X, Y):
    """Input validation for standard estimators.

    Check X and Y for consistent length, enforces X to be a
    floating 2-D array and Y to be a 2-D array mantaining
    the data type. X is checked to be non-empty and containing only
    finite values while Y is checked to be non-empty
    and can contain infinite values.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Input data.

    Y : np.ndarray of shape (n_samples, n_classes)
        Rankings.

    Returns
    -------
    X_converted : np.ndarray of shape (n_samples, n_features)
        The converted and validated X.

    Y_converted : np.ndarray of shape (n_samples, n_classes)
        The converted and validated Y.
    """
    # Check the format input data and rankings
    # (allowing infinite values in the rankings
    # since they may contain missed classes)
    X = check_array(X, dtype=np.float64)
    Y = check_array(Y, force_all_finite=False)

    # Check that the input data and rankings have the same length
    check_consistent_length(X, Y)

    # Return the converted and validated X and Y arrays
    return (X, Y)
