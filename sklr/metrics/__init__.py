"""
The :mod:`sklr.metrics` module includes score functions, performance metrics
and distance computations.
"""


# =============================================================================
# Imports
# =============================================================================

# Local application
from .label_ranking import kendall_distance, tau_score
from .partial_label_ranking import penalized_kendall_distance, tau_x_score


# =============================================================================
# Public objects
# =============================================================================

# set the methods that are accessible
# from the module sklr.metrics
__all__ = [
    "kendall_distance", "tau_score",
    "penalized_kendall_distance", "tau_x_score"
]
