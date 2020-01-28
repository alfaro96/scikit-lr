"""
The :mod:`sklr.miss` module includes
transformers to miss classes from rankings.
"""


# =============================================================================
# Imports
# =============================================================================

# Local application
from ._base import SimpleMisser


# =============================================================================
# Public objects
# =============================================================================

# Define the list of public classes and methods that
# will be exported from this module when importing it
__all__ = ["SimpleMisser"]
