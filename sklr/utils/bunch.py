"""Bunch class providing attribute-style access."""


# =============================================================================
# Classes
# =============================================================================

# =============================================================================
# Bunch
# =============================================================================
class Bunch(dict):
    """Container object for datasets.

    Dictionary-like object that exposes its keys as attributes.

    >>> from sklr.utils import Bunch
    >>> b = Bunch(a=1, b=2)
    >>> b["b"]
    2
    >>> b.b
    2
    >>> b.a = 3
    >>> b["a"]
    3
    >>> b.c = 6
    >>> b["c"]
    6
    """

    def __init__(self, **kwargs):
        """Constructor."""
        # Call to the constructor the parent
        # that will build the inner dictionary
        super().__init__(kwargs)

    def __getattr__(self, key):
        """Called when an attribute lookup has not found the attribute."""
        # Try to obtain the value of the key
        try:
            return self[key]
        # If not found, raise the corresponding error
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        """Called when an attribute assignment is attempted."""
        # Set the value into the inner dictionary
        self[key] = value

    def __dir__(self):
        """Return the list of names in the current local scope."""
        # Return the keys of the inner dictionary as local scope
        return self.keys()
