"""
The :mod:`sklr.exceptions` module includes all custom warnings and error
classes used across scikit-lr.
"""


# =============================================================================
# Module public objects
# =============================================================================
__all__ = ["NotFittedError"]


# =============================================================================
# Classes
# =============================================================================

class NotFittedError(ValueError, AttributeError):
    """Exception class to raise if estimator is used before fitting.

    This class inherits from both ``ValueError`` and ``AttributeError`` to help
    with exception handling and backward compatibility.

    Examples
    --------
    >>> import numpy as np
    >>> from sklr.neighbors import KNeighborsLabelRanker
    >>> from sklr.exceptions import NotFittedError
    >>> try:
    ...     KNeighborsLabelRanker(n_neighbors=1).predict(np.array([[1]]))
    ... except NotFittedError as e:
    ...     print(repr(e))
    NotFittedError("This KNeighborsLabelRanker instance is not fitted yet.
    Call 'fit' with appropriate arguments before using this estimator.",)
    """
