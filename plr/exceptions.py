"""
The :mod:`plr.exceptions` module includes all custom warnings and error
classes used across plr.
"""


# =============================================================================
# Public objects
# =============================================================================

# All error classes
__all__ = ["NotFittedError"]


# =============================================================================
# Classes
# =============================================================================

# =============================================================================
# Not fitted error
# =============================================================================
class NotFittedError(ValueError, AttributeError):
    """Exception class to raise if an estimator is used before fitting.

    This class inherits from both ValueError and AttributeError to help with
    exception handling and backward compatibility.

    Examples
    --------
    >>> from plr.neighbors import KNeighborsLabelRanker
    >>> from plr.exceptions import NotFittedError
    >>> try:
    ...     KNeighborsLabelRanker().predict([[1, 2], [2, 3], [3, 4]])
    ... except NotFittedError as e:
    ...     print(repr(e))
    NotFittedError("This KNeighborsLabelRanker instance is not fitted yet.
    Call fit with appropriate arguments before using this method.",)
    """
    pass
