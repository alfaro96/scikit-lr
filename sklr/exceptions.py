"""
The :mod:`sklr.exceptions` module includes all custom warnings and error
classes used across scikit-lr.
"""


# =============================================================================
# Public objects
# =============================================================================

# Define the list of public classes and methods that
# will be exported from this module when importing it
__all__ = ["NotFittedError"]


# =============================================================================
# Classes
# =============================================================================

# =============================================================================
# Not fitted error
# =============================================================================
class NotFittedError(ValueError, AttributeError):
    """Exception class to raise if estimator is used before fitting.

    This class inherits from both ValueError and AttributeError to help with
    exception handling and backward compatibility.

    Examples
    --------
    >>> import numpy as np
    >>> from sklr.neighbors import KNeighborsLabelRanker
    >>> from sklr.exceptions import NotFittedError
    >>> try:
    ...     KNeighborsLabelRanker().predict(np.array([[1, 2]]))
    ... except NotFittedError as e:
    ...     print(repr(e))
    NotFittedError("This KNeighborsLabelRanker instance is not fitted yet.
    Call 'fit' with appropriate arguments before using this estimator.",)
    """
