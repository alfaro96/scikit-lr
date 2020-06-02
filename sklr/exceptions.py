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
    """Exception to raise if :term:`estimator` is used before :term:`fitting`.

    This class inherits from :class:`ValueError` and :class:`AttributeError`
    to help with exception handling and :term:`backwards compatibility`.

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
