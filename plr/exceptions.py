"""
    This module gathers all the exceptions for the "plr" package.
"""

# =============================================================================
# Public objects
# =============================================================================

# Classes
__all__ = ["NotFittedError"]

# =============================================================================
# Not fitted error
# =============================================================================
class NotFittedError(ValueError, AttributeError):
    """
        Exception class to raise if an object is used before fitting.
    """
    pass
